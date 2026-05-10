"""
Music Assistant integration for Clara.

Uses the MA HTTP RPC API at /api endpoint.
Requires MA_URL and MA_TOKEN environment variables.

Provides:
- ma_search(query, media_type): search MA for artists/albums/tracks/playlists
- ma_play(player_id, uri, enqueue): play a URI on a player
- get_players(): list available players
"""

import os
import httpx

from ha_integration import call_service

MA_URL = os.environ.get("MA_URL", "http://homeassistant.local:8095")
MA_TOKEN = os.environ.get("MA_TOKEN", "")

MA_HEADERS = {
    "Authorization": f"Bearer {MA_TOKEN}",
    "Content-Type": "application/json",
}

# The Sonos speakers via hass_players provider — these are the right targets.
# AirPlay+, RINCON_*, and universal_player duplicates are excluded.
SPEAKERS = {
    "lounge":        "media_player.lounge",
    "kitchen":       "media_player.kitchen",
    "office":        "media_player.office",
    "main bedroom":  "media_player.main_bedroom",
    "bedroom":       "media_player.main_bedroom",
    "sam":           "e76d58e8-e442-aac9-1912-da8224c4d56c",
    "sam's room":    "e76d58e8-e442-aac9-1912-da8224c4d56c",
    "everywhere":    None,  # handled specially — play on all speakers
}

ALL_SPEAKERS = [
    "media_player.lounge",
    "media_player.kitchen",
    "media_player.office",
    "media_player.main_bedroom",
    "e76d58e8-e442-aac9-1912-da8224c4d56c",
]


async def _ma_post(command: str, args: dict) -> dict:
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.post(
            f"{MA_URL}/api",
            headers=MA_HEADERS,
            json={"command": command, "args": args},
        )
        r.raise_for_status()
        return r.json()


async def ma_search(query: str, media_type: str = "artist") -> list[dict]:
    """
    Search Music Assistant for a query.
    media_type: artist | album | track | playlist
    Returns list of results with name, uri, and type.
    """
    result = await _ma_post("music/search", {
        "search_query": query,
        "media_types": [media_type],
        "limit": 5,
    })

    # Extract results from the appropriate key
    key = f"{media_type}s"
    items = result.get(key, [])

    return [
        {
            "name": item["name"],
            "uri": item["uri"],
            "media_type": media_type,
            "provider": item.get("provider", ""),
        }
        for item in items
        if item.get("uri")
    ]


async def ma_play(
    player_id: str,
    uri: str,
    enqueue: str = "replace",
    radio_mode: bool = False,
) -> str:
    """
    Play a URI on a player.
    enqueue: replace | next | add
    Returns status string.
    """
    args = {
        "queue_id": player_id,
        "media": [uri],
        "option": enqueue,
    }
    if radio_mode:
        args["radio_mode"] = True

    await _ma_post("player_queues/play_media", args)
    return "ok"


async def ma_stop(player_id: str) -> str:
    """Stop playback and clear the queue."""
    await _ma_post("player_queues/stop", {"queue_id": player_id})
    return "ok"


async def ma_pause(player_id: str) -> str:
    """Pause playback (toggle — resumes if already paused)."""
    await _ma_post("player_queues/play_pause", {"queue_id": player_id})
    return "ok"


async def ma_next(player_id: str) -> str:
    """Skip to next track."""
    await _ma_post("player_queues/next", {"queue_id": player_id})
    return "ok"


async def ma_previous(player_id: str) -> str:
    """Go back to previous track."""
    await _ma_post("player_queues/previous", {"queue_id": player_id})
    return "ok"


async def ma_set_volume(player_id: str, volume: int) -> str:
    """Set volume 0-100."""
    await _ma_post("players/cmd/volume_set", {
        "player_id": player_id,
        "volume_level": max(0, min(100, volume)),
    })
    return "ok"


async def ma_join(lead_player_id: str, child_player_ids: list[str]) -> str:
    """
    Group players for synchronised playback via HA media_player.join.
    Lead player holds the queue.
    """
    await call_service("media_player", "join", {
        "entity_id": lead_player_id,
        "group_members": child_player_ids,
    })
    return "ok"


async def ma_unjoin(player_id: str) -> str:
    """Remove a player from its current group via HA media_player.unjoin."""
    await call_service("media_player", "unjoin", {
        "entity_id": player_id,
    })
    return "ok"


def resolve_speaker(room: str) -> list[str] | None:
    """
    Resolve a room name to a list of player_ids.
    Returns None if room not recognised.
    Returns list of all speaker IDs for 'everywhere'.
    """
    key = room.lower().strip()
    if key not in SPEAKERS:
        return None
    if SPEAKERS[key] is None:
        return ALL_SPEAKERS
    return [SPEAKERS[key]]
