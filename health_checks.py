"""
Health checks for Clara's integrations.
Each check makes a real but lightweight API call to verify connectivity and auth.
Returns a list of HealthResult dicts suitable for storing and displaying.
"""

import os
import time
import httpx
import asyncio
from datetime import datetime, timezone

# ─── Result type ──────────────────────────────────────────────────────

def _result(service: str, ok: bool, latency_ms: int, error: str | None = None) -> dict:
    return {
        "service":    service,
        "status":     "ok" if ok else "fail",
        "latency_ms": latency_ms,
        "error":      error,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# ─── Individual checks ────────────────────────────────────────────────

async def check_anthropic() -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        if r.status_code == 200:
            return _result("Anthropic API", True, _ms(start))
        else:
            return _result("Anthropic API", False, _ms(start), f"HTTP {r.status_code}")
    except Exception as e:
        return _result("Anthropic API", False, _ms(start), str(e))


async def check_home_assistant() -> dict:
    ha_url = os.environ.get("HA_URL", "http://homeassistant.local:8123")
    ha_token = os.environ.get("HA_TOKEN", "")
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{ha_url}/api/",
                headers={"Authorization": f"Bearer {ha_token}"},
            )
        if r.status_code == 200:
            return _result("Home Assistant", True, _ms(start))
        else:
            return _result("Home Assistant", False, _ms(start), f"HTTP {r.status_code}")
    except Exception as e:
        return _result("Home Assistant", False, _ms(start), str(e))


async def check_music_assistant() -> dict:
    ma_url = os.environ.get("MA_URL", "http://homeassistant.local:8095")
    ma_token = os.environ.get("MA_TOKEN", "")
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{ma_url}/api",
                headers={
                    "Authorization": f"Bearer {ma_token}",
                    "Content-Type": "application/json",
                },
                json={"command": "players/all", "args": {}},
            )
        if r.status_code == 200:
            return _result("Music Assistant", True, _ms(start))
        else:
            return _result("Music Assistant", False, _ms(start), f"HTTP {r.status_code}")
    except Exception as e:
        return _result("Music Assistant", False, _ms(start), str(e))


async def check_ptv() -> dict:
    import hmac
    import hashlib
    dev_id = os.environ.get("PTV_DEV_ID", "")
    api_key = os.environ.get("PTV_API_KEY", "")
    base_url = "https://timetableapi.ptv.vic.gov.au"
    start = time.monotonic()
    try:
        endpoint = f"/v3/route_types?devid={dev_id}"
        key = api_key.encode("utf-8")
        sig = hmac.new(key, endpoint.encode("utf-8"), hashlib.sha1).hexdigest()
        url = f"{base_url}{endpoint}&signature={sig}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(url)
        if r.status_code == 200:
            return _result("PTV", True, _ms(start))
        else:
            return _result("PTV", False, _ms(start), f"HTTP {r.status_code}")
    except Exception as e:
        return _result("PTV", False, _ms(start), str(e))


async def check_brave() -> dict:
    api_key = os.environ.get("BRAVE_API_KEY", "")
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                },
                params={"q": "test", "count": 1},
            )
        if r.status_code == 200:
            return _result("Brave Search", True, _ms(start))
        else:
            return _result("Brave Search", False, _ms(start), f"HTTP {r.status_code}")
    except Exception as e:
        return _result("Brave Search", False, _ms(start), str(e))


async def check_caldav() -> dict:
    caldav_username = os.environ.get("CALDAV_USERNAME", "")
    caldav_password = os.environ.get("CALDAV_PASSWORD", "")
    # Just check iCloud CalDAV principal endpoint
    url = "https://caldav.icloud.com/"
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.request(
                "PROPFIND",
                url,
                auth=(caldav_username, caldav_password),
                headers={"Depth": "0"},
            )
        # iCloud returns 207 Multi-Status for valid PROPFIND
        if r.status_code in (200, 207):
            return _result("iCloud CalDAV", True, _ms(start))
        elif r.status_code == 401:
            return _result("iCloud CalDAV", False, _ms(start), "Authentication failed")
        else:
            return _result("iCloud CalDAV", False, _ms(start), f"HTTP {r.status_code}")
    except Exception as e:
        return _result("iCloud CalDAV", False, _ms(start), str(e))


async def check_database(db_path: str) -> dict:
    import sqlite3
    start = time.monotonic()
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT COUNT(*) FROM messages")
        conn.execute("CREATE TABLE IF NOT EXISTS _health_check_probe (id INTEGER)")
        conn.execute("DROP TABLE _health_check_probe")
        conn.close()
        return _result("Database", True, _ms(start))
    except Exception as e:
        return _result("Database", False, _ms(start), str(e))


async def check_lastfm() -> dict:
    from lastfm_integration import check_connection
    start = time.monotonic()
    ok, ms, error = await check_connection()
    return _result("Last.fm", ok, ms, error)


async def check_llama() -> dict:
    llama_url = os.environ.get("LLAMA_URL", "http://localhost:8080")
    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{llama_url}/health")
        if r.status_code == 200:
            return _result("llama.cpp", True, _ms(start))
        else:
            return _result("llama.cpp", False, _ms(start), f"HTTP {r.status_code}")
    except Exception as e:
        # llama.cpp being down is non-critical — mark as warn not fail
        return _result("llama.cpp", False, _ms(start), f"Unavailable (non-critical): {e}")


# ─── Run all checks ───────────────────────────────────────────────────

async def run_all(db_path: str) -> list[dict]:
    """Run all health checks concurrently. Returns list of HealthResult dicts."""
    results = await asyncio.gather(
        check_anthropic(),
        check_home_assistant(),
        check_music_assistant(),
        check_ptv(),
        check_brave(),
        check_caldav(),
        check_lastfm(),
        check_database(db_path),
        check_llama(),
        return_exceptions=False,
    )
    return list(results)


# ─── Formatting ───────────────────────────────────────────────────────

def format_results(results: list[dict], run_time: str | None = None) -> str:
    """Format health check results for Telegram."""
    lines = []
    header = f"== Health Check"
    if run_time:
        header += f" — {run_time}"
    header += " =="
    lines.append(header)

    all_ok = all(r["status"] == "ok" for r in results)
    for r in results:
        icon = "✅" if r["status"] == "ok" else "❌"
        line = f"{icon} {r['service']} — {r['latency_ms']}ms"
        if r["error"]:
            line += f"\n   {r['error']}"
        lines.append(line)

    if all_ok:
        lines.append("\nAll systems operational.")
    else:
        failed = [r["service"] for r in results if r["status"] != "ok"]
        lines.append(f"\n⚠️ Issues: {', '.join(failed)}")

    return "\n".join(lines)