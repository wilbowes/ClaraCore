"""
PTV Timetable API v3 integration for Clara.
Provides next departures and disruption info for the family household.

Requires PTV_DEV_ID and PTV_API_KEY environment variables.
API key request: email APIKeyRequest@ptv.vic.gov.au

Route types:
  0 = Train
  1 = Tram
  2 = Bus
  3 = V/Line
  4 = Night Bus

Direction IDs for Werribee line:
  5  = City (Flinders Street)
  16 = Werribee
"""

import os
import hmac
import hashlib
import httpx
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
    _MELB_TZ = ZoneInfo("Australia/Melbourne")
    def _now_melb() -> datetime:
        return datetime.now(_MELB_TZ)
    def _melb_tz():
        return _MELB_TZ
except ImportError:
    def _now_melb() -> datetime:
        return datetime.now(timezone.utc) + timedelta(hours=10)
    def _melb_tz():
        return timezone(timedelta(hours=10))

PTV_DEV_ID = os.environ.get("PTV_DEV_ID", "")
PTV_API_KEY = os.environ.get("PTV_API_KEY", "")
PTV_BASE_URL = "https://timetableapi.ptv.vic.gov.au"

# ─── Known stops ──────────────────────────────────────────────────────
# Stop IDs confirmed from PTV website URLs (/stop/{id}/name/route_type/mode)

STOPS = {
    "laverton":          {"id": 1113, "route_type": 0, "name": "Laverton Station"},
    "laverton station":  {"id": 1113, "route_type": 0, "name": "Laverton Station"},
}

ROUTE_TYPE_TRAIN = 0
ROUTE_TYPE_BUS = 2

WERRIBEE_ROUTE_ID = 16   # Werribee line route ID

# Werribee line direction IDs
DIRECTION_CITY = 5       # towards Flinders Street
DIRECTION_WERRIBEE = 16  # towards Werribee

# Direction aliases Clara can use in the tool call
DIRECTION_ALIASES = {
    "city":            DIRECTION_CITY,
    "flinders":        DIRECTION_CITY,
    "flinders street": DIRECTION_CITY,
    "werribee":        DIRECTION_WERRIBEE,
    "outbound":        DIRECTION_WERRIBEE,
    "inbound":         DIRECTION_CITY,
}


# ─── Auth ─────────────────────────────────────────────────────────────

def _sign(endpoint_with_query: str) -> str:
    """
    Generate a signed URL for the PTV API.
    Appends devid to query params, then HMAC-SHA1 signs the path+query.
    Returns the full signed URL.
    """
    separator = "&" if "?" in endpoint_with_query else "?"
    request = f"{endpoint_with_query}{separator}devid={PTV_DEV_ID}"
    key = PTV_API_KEY.encode("utf-8")
    sig = hmac.new(key, request.encode("utf-8"), hashlib.sha1).hexdigest()
    return f"{PTV_BASE_URL}{request}&signature={sig}"


# ─── API calls ────────────────────────────────────────────────────────

async def _ptv_get(endpoint: str, params: dict | None = None) -> dict:
    """Make a signed GET request to the PTV API."""
    if not PTV_DEV_ID or not PTV_API_KEY:
        raise ValueError("PTV_DEV_ID and PTV_API_KEY environment variables required")

    if params:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        endpoint_with_query = f"{endpoint}?{query}"
    else:
        endpoint_with_query = endpoint

    url = _sign(endpoint_with_query)

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


# ─── Departures ───────────────────────────────────────────────────────

def _format_departure(dep: dict, routes: dict, directions: dict, future: bool = False) -> str | None:
    """
    Format a single departure into a readable line.
    future=True: show scheduled time only (countdown meaningless for tomorrow etc)
    future=False: show time + countdown from now
    """
    sched = dep.get("scheduled_departure_utc")
    est = dep.get("estimated_departure_utc")  # real-time if available

    time_str_utc = est or sched
    if not time_str_utc:
        return None

    try:
        dt_utc = datetime.fromisoformat(time_str_utc.replace("Z", "+00:00"))
        dt_melb = dt_utc.astimezone(_melb_tz())
    except Exception:
        return None

    now = _now_melb()
    diff_mins = int((dt_melb - now).total_seconds() / 60)

    if diff_mins < -1:
        return None  # already departed

    time_label = dt_melb.strftime("%I:%M %p").lstrip("0")
    rt_note = " (live)" if est else ""

    if future:
        timing = time_label
    else:
        if diff_mins <= 0:
            countdown = "now"
        elif diff_mins == 1:
            countdown = "1 min"
        else:
            countdown = f"{diff_mins} mins"
        timing = f"{time_label} — {countdown}{rt_note}"

    # Route name
    route_id = str(dep.get("route_id", ""))
    route_name = routes.get(route_id, {}).get("route_name", "")
    route_number = routes.get(route_id, {}).get("route_number", "")
    route_label = route_number or route_name

    # Direction
    direction_id = str(dep.get("direction_id", ""))
    direction_name = directions.get(direction_id, {}).get("direction_name", "")

    # Platform
    platform = dep.get("platform_number")
    platform_str = f" (Platform {platform})" if platform else ""

    parts = [f"  {timing}"]
    if direction_name:
        parts.append(f"towards {direction_name}")
    if route_label:
        parts.append(f"[{route_label}]")
    if platform_str:
        parts.append(platform_str)

    return " ".join(parts)


async def get_departures(
    stop_name: str,
    max_results: int = 8,
    direction: str | None = None,
    arrive_by: str | None = None,
) -> str:
    """
    Get departures from a named stop.

    Args:
        stop_name:   key in STOPS dict
        max_results: number of services to return
        direction:   optional direction filter e.g. 'city', 'werribee', 'flinders'
        arrive_by:   optional ISO 8601 datetime string (Melbourne local or UTC).
                     Returns services departing before this time.
                     Use for "trains to arrive by 9am tomorrow" type queries.
                     Clara should pass the arrival target time at destination;
                     the Werribee line takes ~38 mins to Flinders St.
    """
    stop_key = stop_name.lower().strip()
    stop = STOPS.get(stop_key)

    if not stop:
        known = ", ".join(STOPS.keys())
        return f"[Unknown stop: '{stop_name}'. Known stops: {known}]"

    stop_id = stop["id"]
    route_type = stop["route_type"]
    stop_label = stop["name"]

    # Resolve direction alias
    direction_id = None
    if direction:
        direction_id = DIRECTION_ALIASES.get(direction.lower().strip())
        if direction_id is None:
            return f"[Unknown direction: '{direction}'. Try: city, flinders, werribee]"

    # Build params
    params: dict = {
        "max_results": max_results,
        "expand": "route,direction,stop",
        "include_cancelled": "false",
    }

    future = False
    target_dt = None

    if arrive_by:
        try:
            target_dt = datetime.fromisoformat(arrive_by)
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=_melb_tz())

            # Start the window from midnight on the target day so we get
            # the full morning picture, not just from the current moment
            window_start = target_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            now = _now_melb()

            if window_start > now:
                future = True
                params["date_utc"] = window_start.astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                # Fetch more to cover the morning window before filtering
                params["max_results"] = max(max_results, 20)
        except ValueError as e:
            return f"[Invalid arrive_by format: {e}. Use ISO 8601 e.g. '2026-04-13T09:00:00']"

    if direction_id is not None:
        params["direction_id"] = direction_id

    try:
        data = await _ptv_get(
            f"/v3/departures/route_type/{route_type}/stop/{stop_id}",
            params=params,
        )
    except Exception as e:
        return f"[PTV API error: {e}]"

    departures = data.get("departures", [])
    if not departures:
        return f"[No departures found from {stop_label}]"

    # Build lookup dicts from expanded objects
    routes = {
        str(r["route_id"]): r
        for r in data.get("routes", {}).values()
    } if isinstance(data.get("routes"), dict) else {}

    directions = {
        str(d["direction_id"]): d
        for d in data.get("directions", {}).values()
    } if isinstance(data.get("directions"), dict) else {}

    # Filter to services departing before arrive_by target
    if target_dt:
        departures = [
            d for d in departures
            if d.get("scheduled_departure_utc") and
            datetime.fromisoformat(
                d["scheduled_departure_utc"].replace("Z", "+00:00")
            ) <= target_dt.astimezone(timezone.utc)
        ]

    if not departures:
        return f"[No departures from {stop_label} before the requested time]"

    # Header
    if arrive_by and target_dt:
        arrive_label = target_dt.strftime("%I:%M %p, %a %-d %b").lstrip("0")
        header = f"== Departures from {stop_label} arriving by {arrive_label} =="
    else:
        now_str = _now_melb().strftime("%I:%M %p").lstrip("0")
        header = f"== Departures from {stop_label} (as of {now_str}) =="

    lines = [header]
    formatted = []
    for dep in departures:
        line = _format_departure(dep, routes, directions, future=future)
        if line:
            formatted.append(line)

    if not formatted:
        return f"[No departures from {stop_label}]"

    lines.extend(formatted)

    if future:
        lines.append("\n  (Scheduled times — check for disruptions on the day)")

    return "\n".join(lines)


# ─── Disruptions ──────────────────────────────────────────────────────

async def get_disruptions(route_id: int = WERRIBEE_ROUTE_ID) -> str:
    """
    Get current and upcoming planned disruptions for a specific route.
    Defaults to Werribee line (route_id=16).
    Queries both current and planned disruptions, filters to next 14 days.
    """
    now = _now_melb()
    lookahead = now + timedelta(days=14)
    all_disruptions = []

    for status in ("current", "planned"):
        try:
            data = await _ptv_get(
                f"/v3/disruptions/route/{route_id}",
                params={"disruption_status": status},
            )
        except Exception as e:
            continue

        disruptions = data.get("disruptions", {})
        for key, items in disruptions.items():
            if isinstance(items, list):
                all_disruptions.extend(items)

    if not all_disruptions:
        return "[No current or upcoming disruptions on Werribee line]"

    # Filter to disruptions active within next 14 days
    relevant = []
    for d in all_disruptions:
        from_dt = d.get("from_date")
        to_dt = d.get("to_date")
        try:
            if from_dt:
                f = datetime.fromisoformat(from_dt.replace("Z", "+00:00")).astimezone(_melb_tz())
                if f > lookahead:
                    continue  # too far away
            if to_dt:
                t = datetime.fromisoformat(to_dt.replace("Z", "+00:00")).astimezone(_melb_tz())
                if t < now:
                    continue  # already ended
        except Exception:
            pass
        relevant.append(d)

    # Deduplicate by disruption_id
    seen = set()
    deduped = []
    for d in relevant:
        did = d.get("disruption_id")
        if did not in seen:
            seen.add(did)
            deduped.append(d)

    if not deduped:
        return "[No current or upcoming disruptions on Werribee line]"

    lines = ["== Werribee line disruptions =="]
    for d in deduped[:8]:
        title = d.get("title", "").strip()
        description = d.get("description", "").strip()
        from_dt = d.get("from_date")
        to_dt = d.get("to_date")

        date_range = ""
        if from_dt and to_dt:
            try:
                f = datetime.fromisoformat(from_dt.replace("Z", "+00:00")).astimezone(_melb_tz())
                t = datetime.fromisoformat(to_dt.replace("Z", "+00:00")).astimezone(_melb_tz())
                date_range = f" ({f.strftime('%-d %b')} – {t.strftime('%-d %b')})"
            except Exception:
                pass

        lines.append(f"\n  {title}{date_range}")
        if description and description != title:
            desc = description[:200] + "..." if len(description) > 200 else description
            lines.append(f"  {desc}")

    return "\n".join(lines)


# ─── Combined query ───────────────────────────────────────────────────

async def get_transport_info(
    stop_name: str,
    include_disruptions: bool = True,
    direction: str | None = None,
    arrive_by: str | None = None,
) -> str:
    """
    Combined departures + disruptions for a stop.
    Main entry point for Clara's tool.
    """
    parts = []

    departures = await get_departures(
        stop_name,
        direction=direction,
        arrive_by=arrive_by,
    )
    parts.append(departures)

    if include_disruptions:
        disruptions = await get_disruptions(route_id=WERRIBEE_ROUTE_ID)
        if "No current or upcoming disruptions" not in disruptions:
            parts.append(disruptions)

    return "\n\n".join(parts)