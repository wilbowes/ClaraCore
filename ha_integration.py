"""
Home Assistant integration for Clara.

Provides:
- get_home_state(domains, filter): fetch current state for specific domains/entities (tool use)
- discover_home(): pulls all HA entity states, returns a readable summary (startup/debug)
- ha_get(endpoint): GET request to HA API
- ha_post(endpoint, payload): POST request to HA API (for service calls)
- call_service(domain, service, data): convenience wrapper for service calls

Add HA_URL and HA_TOKEN to your environment or config.json.
"""

import os
import httpx
from datetime import datetime, timezone, timedelta

try:
    from zoneinfo import ZoneInfo
    _MELB_TZ = ZoneInfo("Australia/Melbourne")
    def _now_melb() -> datetime:
        return datetime.now(_MELB_TZ)
except ImportError:
    def _now_melb() -> datetime:
        return datetime.now(timezone.utc) + timedelta(hours=10)

HA_URL = os.environ.get("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.environ.get("HA_TOKEN", "")

HA_HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

# Entity domains for discover_home() — actionable devices only.
# sensor, binary_sensor, automation, input_boolean excluded — too noisy,
# available on-demand via get_home_state() tool if needed.
RELEVANT_DOMAINS = {
    "light",
    "switch",
    "climate",
    "media_player",
    "cover",
    "lock",
    "alarm_control_panel",
    "fan",
}


async def ha_get(endpoint: str) -> dict | list:
    """GET request to Home Assistant API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(
            f"{HA_URL}/api/{endpoint}",
            headers=HA_HEADERS,
        )
        r.raise_for_status()
        return r.json()


async def ha_post(endpoint: str, payload: dict = None) -> dict:
    """POST request to Home Assistant API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(
            f"{HA_URL}/api/{endpoint}",
            headers=HA_HEADERS,
            json=payload or {},
        )
        r.raise_for_status()
        return r.json()


async def call_service(domain: str, service: str, data: dict = None) -> dict:
    """Call a Home Assistant service. e.g. call_service('light', 'turn_on', {'entity_id': 'light.kitchen'})"""
    return await ha_post(f"services/{domain}/{service}", data or {})


DOMAIN_LABELS = {
    "light": "Lights",
    "switch": "Switches",
    "climate": "Climate",
    "sensor": "Sensors",
    "binary_sensor": "Sensors (binary)",
    "media_player": "Media",
    "cover": "Covers / Blinds",
    "lock": "Locks",
    "alarm_control_panel": "Alarm",
    "fan": "Fans",
    "input_boolean": "Toggles",
    "automation": "Automations",
}


async def get_home_state(domains: list[str], filter: str | None = None) -> str:
    """
    Fetch current HA state for specific domains, with optional keyword filter.
    Used by Clara's tool call handler — returns a compact, current snapshot.

    Args:
        domains: list of HA domains e.g. ['light', 'climate']
        filter: optional keyword to filter entity names e.g. 'bedroom', 'kitchen'
    """
    try:
        states = await ha_get("states")
    except Exception as e:
        return f"[Home Assistant unavailable: {e}]"

    domain_set = set(domains)
    filter_lower = filter.lower() if filter else None

    by_domain: dict[str, list] = {}
    for state in states:
        domain = state["entity_id"].split(".")[0]
        if domain not in domain_set:
            continue
        if filter_lower:
            name = state.get("attributes", {}).get("friendly_name", state["entity_id"]).lower()
            entity_id = state["entity_id"].lower()
            if filter_lower not in name and filter_lower not in entity_id:
                continue
        by_domain.setdefault(domain, []).append(state)

    # Deduplicate media_player entities with the same friendly name —
    # MA creates multiple entities per speaker (Sonos, AirPlay, queue).
    # Prefer the one with actual track metadata (media_title/media_artist),
    # then the one with the richest state, then fall back to first seen.
    if "media_player" in by_domain:
        seen_names: dict[str, dict] = {}
        for state in by_domain["media_player"]:
            name = state.get("attributes", {}).get("friendly_name", state["entity_id"])
            attrs = state.get("attributes", {})
            has_track = "media_title" in attrs or "media_artist" in attrs
            existing = seen_names.get(name)
            if existing is None:
                seen_names[name] = state
            else:
                ex_attrs = existing.get("attributes", {})
                ex_has_track = "media_title" in ex_attrs or "media_artist" in ex_attrs
                # Prefer track metadata; then active state; then keep existing
                if has_track and not ex_has_track:
                    seen_names[name] = state
                elif not ex_has_track and state["state"] == "playing" and existing["state"] != "playing":
                    seen_names[name] = state
        by_domain["media_player"] = list(seen_names.values())

    if not by_domain:
        filter_note = f" matching '{filter}'" if filter else ""
        return f"[No entities found for {', '.join(domains)}{filter_note}]"

    now_melb = _now_melb()
    time_str = now_melb.strftime("%d %b %Y, %I:%M %p")

    lines = [f"[Home state as of {time_str}]"]
    for domain in sorted(by_domain.keys()):
        label = DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())
        lines.append(f"\n{label}:")
        for state in sorted(by_domain[domain], key=lambda s: s["entity_id"]):
            lines.append(_friendly(state))

    return "\n".join(lines)


def _friendly(state: dict) -> str:
    """Format a single entity state into a readable line."""
    attrs = state.get("attributes", {})
    name = attrs.get("friendly_name", state["entity_id"])
    entity_id = state["entity_id"]
    s = state["state"]
    domain = entity_id.split(".")[0]

    extras = []

    if domain == "sensor":
        # For sensors, state IS the reading — format it with unit inline
        unit = attrs.get("unit_of_measurement", "")
        return f"  {name} [{entity_id}]: {s}{unit}"

    if domain == "device_tracker":
        # Translate home/not_home to readable form
        if s == "home":
            location = "home"
        elif s == "not_home":
            location = "away"
        else:
            location = s  # zone name or GPS location
        battery = attrs.get("battery_level")
        battery_str = f", battery {battery}%" if battery is not None else ""
        return f"  {name} [{entity_id}]: {location}{battery_str}"

    if "brightness" in attrs and attrs["brightness"] is not None:
        pct = round(attrs["brightness"] / 255 * 100)
        extras.append(f"{pct}% brightness")
    if "color_temp" in attrs:
        extras.append(f"color temp {attrs['color_temp']}")
    if "temperature" in attrs and attrs["temperature"] is not None:
        extras.append(f"{attrs['temperature']}°")
    if "unit_of_measurement" in attrs:
        extras.append(attrs["unit_of_measurement"])
    if "current_temperature" in attrs and attrs["current_temperature"] is not None:
        extras.append(f"current {attrs['current_temperature']}°")
    if "media_title" in attrs or "media_artist" in attrs:
        title = attrs.get("media_title", "")
        artist = attrs.get("media_artist", "")
        album = attrs.get("media_album_name", "")
        if title and artist:
            playing = f"{title} — {artist}"
        elif title:
            playing = title
        elif artist:
            playing = artist
        else:
            playing = "unknown"
        if album:
            playing += f" ({album})"
        extras.append(f"playing: {playing}")

    extra_str = f" ({', '.join(extras)})" if extras else ""
    return f"  {name} [{entity_id}]: {s}{extra_str}"


async def discover_home() -> str:
    """
    Pull all entity states from Home Assistant and return a readable summary
    suitable for injection into Clara's context.
    """
    try:
        states = await ha_get("states")
    except Exception as e:
        return f"[Home Assistant unavailable: {e}]"

    # Group by domain
    by_domain: dict[str, list] = {}
    for state in states:
        domain = state["entity_id"].split(".")[0]
        if domain in RELEVANT_DOMAINS:
            by_domain.setdefault(domain, []).append(state)

    if not by_domain:
        return "[Home Assistant: no relevant entities found]"

    now_melb = _now_melb()
    time_str = now_melb.strftime("%A %d %B %Y, %I:%M %p AEDT")

    lines = [f"== The house (as of {time_str}) =="]

    for domain in sorted(by_domain.keys()):
        label = DOMAIN_LABELS.get(domain, domain.replace("_", " ").title())
        lines.append(f"\n{label}:")
        for state in sorted(by_domain[domain], key=lambda s: s["entity_id"]):
            lines.append(_friendly(state))

    return "\n".join(lines)