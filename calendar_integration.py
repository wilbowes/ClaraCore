"""
Apple iCloud Calendar integration for Clara.
Uses CalDAV protocol directly via httpx.
Handles recurring events via python-dateutil rrule expansion.

Fetches events from the family shared calendar.
Requires CALDAV_USERNAME and CALDAV_PASSWORD environment variables.
"""

import os
import re
import httpx
from datetime import datetime, timedelta, timezone, date
from dateutil import rrule as rrulelib
from dateutil.parser import parse as dateutil_parse

try:
    from zoneinfo import ZoneInfo
    MELBOURNE_TZ = ZoneInfo("Australia/Melbourne")
except ImportError:
    MELBOURNE_TZ = None

CALDAV_USERNAME = os.environ.get("CALDAV_USERNAME", "")
CALDAV_PASSWORD = os.environ.get("CALDAV_PASSWORD", "")

CALENDAR_URL = "https://p152-caldav.icloud.com/102450828/calendars/cb594d88-fe69-4c53-99c8-cbae3d23aca0/"

REPORT_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<c:calendar-query xmlns:d="DAV:" xmlns:c="urn:ietf:params:xml:ns:caldav">
  <d:prop>
    <d:getetag/>
    <c:calendar-data/>
  </d:prop>
  <c:filter>
    <c:comp-filter name="VCALENDAR">
      <c:comp-filter name="VEVENT">
        <c:time-range start="{start}" end="{end}"/>
      </c:comp-filter>
    </c:comp-filter>
  </c:filter>
</c:calendar-query>"""


# ─── Timezone helpers ─────────────────────────────────────────────────

def _melb_now() -> datetime:
    if MELBOURNE_TZ:
        return datetime.now(MELBOURNE_TZ)
    return datetime.now(timezone.utc) + timedelta(hours=10)


def _to_melb(dt: datetime) -> datetime:
    if MELBOURNE_TZ:
        return dt.astimezone(MELBOURNE_TZ)
    return dt + timedelta(hours=10)


def _melb_tz():
    return MELBOURNE_TZ or timezone(timedelta(hours=10))


# ─── iCal parsing ─────────────────────────────────────────────────────

def _parse_ical_value(line: str) -> str:
    """Extract value from an iCal property line."""
    if ":" in line:
        return line.split(":", 1)[1].strip()
    return line.strip()


def _parse_ical_dt(value: str, tzid: str | None = None) -> tuple[datetime | None, bool]:
    """
    Parse iCal DATE or DATETIME string.
    Returns (datetime in Melbourne tz, is_all_day).
    tzid: TZID parameter value if present (e.g. 'Australia/Melbourne')
    """
    value = value.strip()
    try:
        if "T" in value:
            if value.endswith("Z"):
                dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = datetime.strptime(value[:15], "%Y%m%dT%H%M%S")
                if tzid:
                    try:
                        from zoneinfo import ZoneInfo
                        tz = ZoneInfo(tzid)
                        dt = dt.replace(tzinfo=tz)
                    except Exception:
                        dt = dt.replace(tzinfo=_melb_tz())
                else:
                    dt = dt.replace(tzinfo=_melb_tz())
            return _to_melb(dt), False
        else:
            dt = datetime.strptime(value[:8], "%Y%m%d")
            dt = dt.replace(tzinfo=_melb_tz())
            return dt, True
    except ValueError:
        return None, False


def _extract_tzid(line: str) -> str | None:
    """Extract TZID parameter from a property line e.g. DTSTART;TZID=Australia/Melbourne:..."""
    m = re.search(r'TZID=([^:;]+)', line)
    return m.group(1).strip() if m else None


def _unfold(lines: list[str]) -> list[str]:
    """Unfold iCal continuation lines."""
    unfolded = []
    for line in lines:
        if line.startswith((" ", "\t")) and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def _parse_vevent(vevent_text: str) -> dict | None:
    """Parse a VEVENT block into a dict including raw RRULE and EXDATEs."""
    event = {}
    lines = _unfold(vevent_text.splitlines())

    for line in lines:
        if line.startswith("SUMMARY"):
            event["title"] = _parse_ical_value(line)

        elif line.startswith("DTSTART"):
            tzid = _extract_tzid(line)
            raw = _parse_ical_value(line)
            dt, all_day = _parse_ical_dt(raw, tzid)
            event["start"] = dt
            event["all_day"] = all_day
            event["start_tzid"] = tzid

        elif line.startswith("DTEND"):
            tzid = _extract_tzid(line)
            raw = _parse_ical_value(line)
            dt, _ = _parse_ical_dt(raw, tzid)
            event["end"] = dt

        elif line.startswith("DURATION"):
            event["duration_raw"] = _parse_ical_value(line)

        elif line.startswith("RRULE"):
            event["rrule_raw"] = _parse_ical_value(line)

        elif line.startswith("EXDATE"):
            tzid = _extract_tzid(line)
            raw_dates = _parse_ical_value(line)
            exdates = event.setdefault("exdates", [])
            for raw in raw_dates.split(","):
                dt, _ = _parse_ical_dt(raw.strip(), tzid)
                if dt:
                    exdates.append(dt)

        elif line.startswith("LOCATION"):
            event["location"] = _parse_ical_value(line)

        elif line.startswith("DESCRIPTION"):
            event["description"] = _parse_ical_value(line).replace("\\n", " ").strip()

        elif line.startswith("ORGANIZER"):
            cn_match = re.search(r'CN="?([^":;]+)"?', line, re.IGNORECASE)
            if cn_match:
                event["organizer"] = cn_match.group(1).strip()
            else:
                email_match = re.search(r'mailto:(.+)', line, re.IGNORECASE)
                if email_match:
                    event["organizer"] = email_match.group(1).strip()

    if not event.get("title") or not event.get("start"):
        return None

    return event


def _expand_recurrences(event: dict, range_start: datetime, range_end: datetime) -> list[dict]:
    """
    Expand a (possibly recurring) event into concrete occurrences within range.
    Returns list of event dicts, one per occurrence.
    """
    start = event["start"]
    rrule_raw = event.get("rrule_raw")

    # Non-recurring — just check if it falls in range
    if not rrule_raw:
        if start and range_start <= start < range_end:
            return [event]
        return []

    # Calculate duration for recurring events
    if event.get("end") and start:
        duration = event["end"] - start
    else:
        duration = timedelta(hours=1)

    # Build rrule — dateutil needs a naive or tz-aware dtstart
    try:
        rule = rrulelib.rrulestr(
            f"DTSTART:{start.strftime('%Y%m%dT%H%M%S')}\nRRULE:{rrule_raw}",
            ignoretz=True,
        )
    except Exception:
        # Fallback — just return the original if we can't parse the rrule
        if range_start <= start < range_end:
            return [event]
        return []

    # Build exdate set for comparison (naive, Melbourne local)
    exdates = set()
    for ex in event.get("exdates", []):
        exdates.add(ex.replace(tzinfo=None).replace(second=0, microsecond=0))

    # Expand occurrences within range (use naive for rrule, reattach tz after)
    range_start_naive = range_start.replace(tzinfo=None)
    range_end_naive = range_end.replace(tzinfo=None)

    occurrences = []
    for occ_naive in rule.between(range_start_naive, range_end_naive, inc=True):
        # Check against exdates
        occ_cmp = occ_naive.replace(second=0, microsecond=0)
        if occ_cmp in exdates:
            continue

        # Reattach Melbourne timezone
        occ_melb = occ_naive.replace(tzinfo=_melb_tz())
        if MELBOURNE_TZ:
            occ_melb = occ_naive.replace(tzinfo=MELBOURNE_TZ)

        occ_event = dict(event)
        occ_event["start"] = occ_melb
        occ_event["end"] = occ_melb + duration
        occurrences.append(occ_event)

    return occurrences


def _extract_vevents(ical_data: str) -> list[dict]:
    """Extract all raw VEVENT dicts from iCal data."""
    events = []
    for match in re.finditer(r"BEGIN:VEVENT(.*?)END:VEVENT", ical_data, re.DOTALL):
        event = _parse_vevent(match.group(1))
        if event:
            events.append(event)
    return events


# ─── Formatting ───────────────────────────────────────────────────────

def _format_event(event: dict) -> str:
    start = event["start"]
    if start is None:
        return ""

    title = event.get("title", "Untitled")
    location = event.get("location", "")

    if event.get("all_day"):
        date_str = start.strftime("%a %d %b")
        time_str = "all day"
    else:
        date_str = start.strftime("%a %d %b")
        time_str = start.strftime("%I:%M %p").lstrip("0")

    location_str = f" @ {location}" if location else ""
    organizer = event.get("organizer", "")
    organizer_str = f" (added by {organizer})" if organizer else ""
    return f"  {date_str}, {time_str} — {title}{location_str}{organizer_str}"


# ─── Main fetch function ───────────────────────────────────────────────

async def get_family_events(days_ahead: int = 7) -> str:
    """
    Fetch upcoming events from the Family calendar, including recurring events.
    Returns a formatted string suitable for Clara's context.
    """
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=days_ahead)

    # Fetch a wider window from CalDAV to catch recurring event masters
    # that started before our range but have occurrences within it
    fetch_start = now - timedelta(days=365)
    start_str = fetch_start.strftime("%Y%m%dT%H%M%SZ")
    end_str = end.strftime("%Y%m%dT%H%M%SZ")

    body = REPORT_TEMPLATE.format(start=start_str, end=end_str)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.request(
                method="REPORT",
                url=CALENDAR_URL,
                headers={
                    "Content-Type": "application/xml; charset=utf-8",
                    "Depth": "1",
                },
                auth=(CALDAV_USERNAME, CALDAV_PASSWORD),
                content=body.encode("utf-8"),
            )
            response.raise_for_status()
    except Exception as e:
        return f"[Calendar unavailable: {e}]"

    ical_blocks = re.findall(
        r"<.*?calendar-data.*?>(.*?)</.*?calendar-data.*?>",
        response.text,
        re.DOTALL,
    )

    if not ical_blocks:
        return "[No upcoming events in Family calendar]"

    # Convert range to Melbourne time for expansion
    now_melb = _melb_now()
    end_melb = now_melb + timedelta(days=days_ahead)

    all_occurrences = []
    for block in ical_blocks:
        block = block.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
        for raw_event in _extract_vevents(block):
            occurrences = _expand_recurrences(raw_event, now_melb, end_melb)
            all_occurrences.extend(occurrences)

    if not all_occurrences:
        return "[No upcoming events in Family calendar]"

    # Sort by start time
    all_occurrences.sort(key=lambda e: e["start"])

    lines = [f"== Family calendar — next {days_ahead} days =="]
    for event in all_occurrences:
        line = _format_event(event)
        if line:
            lines.append(line)

    return "\n".join(lines)
