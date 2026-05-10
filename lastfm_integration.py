"""
Last.fm integration for Clara.
Read-only endpoints — no auth required beyond API key.

Requires LASTFM_API_KEY environment variable.
"""

import os
import httpx
import logging

log = logging.getLogger("clara-lastfm")

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")
LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
LASTFM_TIMEOUT = 8.0


async def _lastfm_get(method: str, params: dict) -> dict:
    """Make a GET request to the Last.fm API."""
    if not LASTFM_API_KEY:
        return {"error": 1, "message": "LASTFM_API_KEY not set"}
    try:
        async with httpx.AsyncClient(timeout=LASTFM_TIMEOUT) as client:
            r = await client.get(
                LASTFM_BASE,
                params={
                    "method": method,
                    "api_key": LASTFM_API_KEY,
                    "format": "json",
                    **params,
                },
            )
            r.raise_for_status()
            return r.json()
    except Exception as e:
        log.warning(f"Last.fm {method} failed: {e}")
        return {"error": 1, "message": str(e)}


# ─── Similar artists ──────────────────────────────────────────────────

async def get_similar_artists(artist: str, limit: int = 8) -> list[dict]:
    """
    Get artists similar to a given artist.
    Returns list of dicts with 'name' and 'match' (0.0-1.0 similarity score).
    """
    data = await _lastfm_get("artist.getsimilar", {"artist": artist, "limit": limit + 4})
    if "error" in data:
        log.warning(f"Last.fm similar artists error for '{artist}': {data.get('message')}")
        return []
    similar = data.get("similarartists", {}).get("artist", [])
    return [
        {"name": a["name"], "match": float(a.get("match", 0))}
        for a in similar if a.get("name")
    ][:limit]


async def format_similar_artists(artist: str, limit: int = 6) -> str:
    """Format similar artists for Clara's tool response."""
    similar = await get_similar_artists(artist, limit=limit)
    if not similar:
        return f"[Last.fm: no similar artists found for '{artist}']"
    lines = [f"== Artists similar to {artist} (Last.fm) =="]
    for a in similar:
        score_pct = int(a["match"] * 100)
        lines.append(f"  {a['name']} ({score_pct}% match)")
    return "\n".join(lines)


# ─── Top tracks ───────────────────────────────────────────────────────

async def get_top_tracks(artist: str, limit: int = 5) -> list[dict]:
    """
    Get an artist's most popular tracks.
    Returns list of dicts with 'name', 'playcount', 'rank'.
    """
    data = await _lastfm_get("artist.getTopTracks", {"artist": artist, "limit": limit})
    if "error" in data:
        log.warning(f"Last.fm top tracks error for '{artist}': {data.get('message')}")
        return []
    tracks = data.get("toptracks", {}).get("track", [])
    return [
        {
            "name": t["name"],
            "playcount": int(t.get("playcount", 0)),
            "rank": int(t.get("@attr", {}).get("rank", i + 1)),
        }
        for i, t in enumerate(tracks) if t.get("name")
    ]


async def format_top_tracks(artist: str, limit: int = 5) -> str:
    """Format top tracks for Clara's tool response."""
    tracks = await get_top_tracks(artist, limit=limit)
    if not tracks:
        return f"[Last.fm: no top tracks found for '{artist}']"
    lines = [f"== Top tracks — {artist} (Last.fm) =="]
    for t in tracks:
        lines.append(f"  {t['rank']}. {t['name']} ({t['playcount']:,} plays)")
    return "\n".join(lines)


# ─── Genre/tag artists ────────────────────────────────────────────────

async def get_artists_by_tag(tag: str, limit: int = 10) -> list[dict]:
    """
    Get top artists for a genre/mood tag.
    Returns list of dicts with 'name' and 'rank'.
    """
    data = await _lastfm_get("tag.getTopArtists", {"tag": tag, "limit": limit})
    if "error" in data:
        log.warning(f"Last.fm tag artists error for '{tag}': {data.get('message')}")
        return []
    artists = data.get("topartists", {}).get("artist", [])
    return [
        {
            "name": a["name"],
            "rank": int(a.get("@attr", {}).get("rank", i + 1)),
        }
        for i, a in enumerate(artists) if a.get("name")
    ]


async def format_tag_artists(tag: str, limit: int = 8) -> str:
    """Format genre/tag artists for Clara's tool response."""
    artists = await get_artists_by_tag(tag, limit=limit)
    if not artists:
        return f"[Last.fm: no artists found for tag '{tag}']"
    lines = [f"== Top artists tagged '{tag}' (Last.fm) =="]
    for a in artists:
        lines.append(f"  {a['rank']}. {a['name']}")
    return "\n".join(lines)


# ─── Health check helper ──────────────────────────────────────────────

async def check_connection() -> tuple[bool, int, str | None]:
    """
    Lightweight connectivity check.
    Returns (ok, latency_ms, error_message).
    """
    import time
    start = time.monotonic()
    data = await _lastfm_get("artist.getsimilar", {"artist": "Blur", "limit": 1})
    ms = int((time.monotonic() - start) * 1000)
    if "error" in data:
        return False, ms, data.get("message", "unknown error")
    return True, ms, None