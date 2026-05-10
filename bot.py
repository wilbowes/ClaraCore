# Clara Telegram Bot v2 - Conversation-Boundary Memory Architecture
#
# Memory model mirrors human episodic memory:
#
#   ACTIVE CONVERSATION  — verbatim, everything, while gap < 2hrs
#   TODAY'S CONVERSATIONS — per-conversation rich summaries, same day
#   PAST 3 DAYS          — one day summary per day, written overnight
#   CORE MEMORY          — enduring facts/patterns, updated weekly
#
# Consolidation triggers:
#   Conversation close  → summarise that conversation (background, after response)
#   First msg of day    → consolidate yesterday's conversation summaries → day summary
#   Weekly              → distil day summaries → core memory update
#
# Context assembly per API call:
#   system prompt → core memory → day summaries (3 days) → today's conversation summaries
#   → active conversation (verbatim) → current message

import os
import json
import sqlite3
import logging
import random
import base64
import httpx
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from aiohttp import web

try:
    from zoneinfo import ZoneInfo
    _MELB_TZ = ZoneInfo("Australia/Melbourne")
    def _now_melb() -> datetime:
        return datetime.now(_MELB_TZ)
except ImportError:
    _MELB_TZ = None
    def _now_melb() -> datetime:
        # Fallback without zoneinfo — won't handle DST transitions
        import time
        offset = -time.timezone if time.daylight == 0 else -time.altzone
        return datetime.now(timezone(timedelta(seconds=offset)))

from ha_integration import discover_home, get_home_state, call_service
from ma_integration import (
    ma_search, ma_play, ma_stop, ma_pause, ma_next, ma_previous,
    ma_set_volume, ma_join, ma_unjoin, resolve_speaker, ALL_SPEAKERS, SPEAKERS
)
from calendar_integration import get_family_events
from search_integration import brave_search, fetch_page
from ptv_integration import get_transport_info
from health_checks import run_all as run_health_checks, format_results as format_health_results
from music_graph import MusicGraph
from lastfm_integration import format_similar_artists, format_top_tracks, format_tag_artists
from memory_validator import (
    validate_dossier, validate_conversation_summary,
    validate_day_summary, validate_core_memory, validate_rhythms,
)

# ─── Config ───────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ALLOWED_USER_IDS = os.environ.get("ALLOWED_USER_IDS", "").split(",")

CONFIG_PATH = os.environ.get("CLARA_CONFIG_PATH", "/config/config.json")
DB_PATH = os.environ.get("CLARA_DB_PATH", "/data/clara.db")

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
ANTHROPIC_API = "https://api.anthropic.com/v1/messages"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_prompt(path: str) -> str:
    config_dir = Path(CONFIG_PATH).parent
    prompt_path = config_dir / path
    with open(prompt_path, "r") as f:
        return f.read().strip()


cfg = load_config()

CLAUDE_MODEL = cfg["models"]["conversation"]
CLAUDE_MODEL_SUMMARISE = cfg["models"]["summarisation"]
CLAUDE_MODEL_DOSSIER = cfg["models"].get("dossier", cfg["models"]["summarisation"])
MAX_TOKENS = cfg["max_tokens"]

CONVERSATION_GAP_SECONDS = cfg["memory"]["conversation_gap_seconds"]    # 7200 = 2hrs
SLEEP_GAP_SECONDS = cfg["memory"]["sleep_gap_seconds"]                  # 21600 = 6hrs
DAY_SUMMARY_KEEP_DAYS = cfg["memory"]["day_summary_keep_days"]          # 3
CORE_UPDATE_INTERVAL_DAYS = cfg["memory"]["core_update_interval_days"]  # 7

USERS = cfg.get("users", {})  # telegram username → display name
PRICING = cfg.get("pricing", {})

THINK_PAUSE_MIN = cfg["pacing"]["think_pause_min"]
THINK_PAUSE_MAX = cfg["pacing"]["think_pause_max"]
TYPING_CHARS_PER_SEC = cfg["pacing"]["typing_chars_per_sec"]
TYPING_MAX_DELAY = cfg["pacing"]["typing_max_delay"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("clara-bot")


# ─── Prompts ──────────────────────────────────────────────────────────

CONVERSATION_SUMMARY_PROMPT = load_prompt(cfg["prompts"]["conversation_summarise"])
DAY_SUMMARY_PROMPT = load_prompt(cfg["prompts"]["day_summarise"])
CORE_MEMORY_PROMPT = load_prompt(cfg["prompts"]["core_summarise"])
RHYTHMS_PROMPT = load_prompt(cfg["prompts"]["rhythms_summarise"])
DOSSIER_PROMPT = load_prompt(cfg["prompts"]["dossier"])
SYSTEM_STATIC = load_prompt(cfg["prompts"]["system_static"])
SYSTEM_DYNAMIC_TEMPLATE = load_prompt(cfg["prompts"]["system_dynamic"])


# ─── Tool Definitions ─────────────────────────────────────────────────

HA_TOOLS = [
    {
        "name": "get_home_state",
        "description": (
            "Get current state of house devices — lights, temperature, media, locks, who's home. "
            "Fetch only the domains you need. For temperature try ['climate'] first, "
            "then ['sensor'] with filter='temperature'. Never claim device state without calling this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "domains": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "light", "climate", "media_player", "lock",
                            "cover", "switch", "binary_sensor", "sensor", "fan",
                            "device_tracker"
                        ],
                    },
                    "description": (
                        "Device types to fetch. Be specific — "
                        "e.g. ['light'] for lighting questions, ['climate'] for temperature."
                    ),
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Optional keyword to filter by entity name or area, "
                        "e.g. 'bedroom', 'kitchen', 'lounge'."
                    ),
                },
            },
            "required": ["domains"],
        },
    },
    {
        "name": "play_music",
        "description": (
            "Search and play music on a house speaker via Music Assistant. "
            "Rooms: lounge, kitchen, office, main bedroom, sam's room, everywhere. "
            "Ask which room if unclear."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for — artist, album, track, or playlist name.",
                },
                "media_type": {
                    "type": "string",
                    "enum": ["artist", "album", "track", "playlist"],
                    "description": "Type of content. Default to 'artist' unless a specific album or track is requested.",
                },
                "room": {
                    "type": "string",
                    "description": (
                        "Where to play. One of: lounge, kitchen, office, main bedroom, "
                        "sam's room, everywhere. Ask if unclear."
                    ),
                },
                "enqueue": {
                    "type": "string",
                    "enum": ["replace", "next", "add"],
                    "description": "replace = play now (default), next = play after current, add = add to queue.",
                },
                "radio_mode": {
                    "type": "boolean",
                    "description": "Enable radio mode to auto-generate a continuous playlist based on the selection.",
                },
            },
            "required": ["query", "room"],
        },
    },
    {
        "name": "control_music",
        "description": (
            "Control music playback on house speakers — stop, pause, resume, skip, "
            "set volume, or group/ungroup speakers. "
            "stop = fully stop ('turn it off', 'kill it'). "
            "pause/resume: always check get_home_state domains=['media_player'] first — "
            "pause only if playing, resume only if paused (they hit the same toggle)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["stop", "pause", "resume", "next", "previous", "volume", "group", "ungroup"],
                    "description": "stop/pause/resume/next/previous/volume/group/ungroup.",
                },
                "room": {
                    "type": "string",
                    "description": "Which speaker/room to control. One of: lounge, kitchen, office, main bedroom, sam's room, everywhere.",
                },
                "volume_level": {
                    "type": "integer",
                    "description": "Volume 0-100. Required for volume action.",
                },
                "rooms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of rooms to group together for synchronised playback. First room becomes the lead. Required for group action.",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "get_calendar",
        "description": (
            "Fetch upcoming events from the family shared calendar. "
            "Use when someone asks what's on, what's coming up, whether anyone's busy, "
            "or anything involving family schedule or plans. "
            "Default is 7 days ahead — request more if asked about further ahead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {
                    "type": "integer",
                    "description": "How many days ahead to fetch. Default 7, max 30.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_phone",
        "description": (
            "Ring a family member's phone to help find it in the house. "
            "Use when someone asks to find, ring, or locate their phone. "
            "Requires the Home Assistant iOS companion app on the target device. "
            "If the app isn't installed the notification won't arrive — let the user know."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "person": {
                    "type": "string",
                    "enum": ["wil", "kirsty", "sam", "ezra"],
                    "description": "Who to ring.",
                },
            },
            "required": ["person"],
        },
    },
    {
        "name": "get_transport",
        "description": (
            "Get public transport departures and disruption info. "
            "Use when someone asks about trains, buses, next service, or disruptions. "
            "For 'arrive by' queries (e.g. 'need to be at Flinders by 9am tomorrow'), "
            "pass arrive_by as ISO 8601 and direction='city'. "
            "The Werribee line takes ~38 mins from Laverton to Flinders Street. "
            "Currently supports Laverton Station (train)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stop": {
                    "type": "string",
                    "description": "Stop name. Supported: 'laverton', 'laverton station'.",
                },
                "direction": {
                    "type": "string",
                    "description": (
                        "Optional direction filter. "
                        "Use 'city' or 'flinders' for city-bound, 'werribee' for outbound. "
                        "Omit to get both directions."
                    ),
                },
                "arrive_by": {
                    "type": "string",
                    "description": (
                        "Optional target arrival time in ISO 8601 format e.g. '2026-04-13T09:00:00'. "
                        "Returns all departures that would arrive before this time. "
                        "Use Melbourne local time. Werribee line is ~38 mins to Flinders St."
                    ),
                },
                "include_disruptions": {
                    "type": "boolean",
                    "description": "Include current disruption info. Default true.",
                },
            },
            "required": ["stop"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Search the web for current information. "
            "Use for movie times, news, local business hours, event info, "
            "or anything requiring up-to-date data not covered by other tools. "
            "Returns results with titles, URLs, and descriptions. "
            "Follow up with fetch_page to read full page content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Be specific — include location where relevant e.g. 'cinema times Melbourne tonight'.",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results to return. Default 5, max 10.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_page",
        "description": (
            "Fetch and read the content of a web page. "
            "Use when a search result snippet isn't enough — e.g. to read actual session times, "
            "timetable details, or article content. "
            "Pass the URL from a web_search result."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL of the page to fetch.",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "rate_music",
        "description": (
            "Record a music rating for a person. "
            "You have no internal memory for music preferences — this tool is the ONLY way to save them. "
            "Never confirm a rating in text without calling this tool first. "
            "Call this for any expressed preference or instruction to save one, including: "
            "direct statements ('I love Telenova', 'Big Pants are great', 'not a fan of X'), "
            "imperative requests ('add AC/DC as a band I like', 'save Blur as a favourite', "
            "'remember that I like X', 'log The Beths for me'), "
            "lists ('AC/DC, Blur — bands I really like'), "
            "and reactions while music is playing ('this is brilliant', 'skip this'). "
            "Use 'inferred' source when interpreting a reaction, 'explicit' when they directly state a preference. "
            "When someone lists multiple artists, call this tool once per artist. "
            "Also use to record artist discoveries."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "person": {
                    "type": "string",
                    "description": "Who is being rated for. Use their display name e.g. 'Alex', 'Jordan'.",
                },
                "type": {
                    "type": "string",
                    "enum": ["track", "album", "artist"],
                    "description": "Whether rating a track, album, or artist.",
                },
                "artist": {
                    "type": "string",
                    "description": "Artist name. Required for all rating types.",
                },
                "track": {
                    "type": "string",
                    "description": "Track title. Required for track ratings.",
                },
                "album": {
                    "type": "string",
                    "description": "Album title. Required for album ratings. Optional for track ratings to help disambiguate.",
                },
                "year": {
                    "type": "integer",
                    "description": "Album release year. Optional, for album ratings.",
                },
                "rating": {
                    "type": "integer",
                    "enum": [1, -1, 0],
                    "description": "1 = thumbs up, -1 = thumbs down, 0 = remove rating.",
                },
                "source": {
                    "type": "string",
                    "enum": ["explicit", "inferred"],
                    "description": "explicit = person directly rated it, inferred = Clara interpreted their reaction.",
                },
                "discovered_via": {
                    "type": "string",
                    "description": "Optional. How the person discovered this artist e.g. 'assistant recommendation', 'radio', 'friend'.",
                },
            },
            "required": ["person", "type", "artist", "rating"],
        },
    },
    {
        "name": "get_music_taste",
        "description": (
            "Get a person's music taste from the music graph — their liked and disliked artists, "
            "albums, and tracks. "
            "ALWAYS call this before choosing music to play for someone, recommending artists, "
            "or answering questions about their taste. "
            "Never rely on your own assumptions about what someone likes — use this tool. "
            "Also use when asked 'play something I'd like', 'play something good', "
            "'surprise me', or any request where you need to infer taste."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "person": {
                    "type": "string",
                    "description": "Who to get taste for. Use their display name e.g. 'Alex', 'Jordan'.",
                },
            },
            "required": ["person"],
        },
    },
    {
        "name": "get_similar_artists",
        "description": (
            "Get artists similar to a given artist via Last.fm. "
            "Use when someone asks to play something similar to an artist they like, "
            "wants a recommendation based on a specific artist, or asks 'what else is like X?'. "
            "Returns a ranked list of similar artists with similarity scores. "
            "After calling this, pick one of the returned artists and call play_music. "
            "Prefer artists the person hasn't already rated — cross-reference with get_music_taste."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "artist": {
                    "type": "string",
                    "description": "Artist to find similar artists for.",
                },
            },
            "required": ["artist"],
        },
    },
    {
        "name": "get_artist_top_tracks",
        "description": (
            "Get an artist's most popular tracks from Last.fm. "
            "Use this before playing an artist to find their best-known track — "
            "gives MA something specific to search for rather than guessing. "
            "Use when asked to play an artist's 'best', 'most popular', or 'biggest' tracks, "
            "or any time you want to play a specific well-known track rather than a random one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "artist": {
                    "type": "string",
                    "description": "Artist name to get top tracks for.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of tracks to return. Default 5.",
                },
            },
            "required": ["artist"],
        },
    },
    {
        "name": "get_artists_by_genre",
        "description": (
            "Get top artists for a genre or mood tag from Last.fm. "
            "Use when someone asks to play music by genre or mood — "
            "e.g. 'play some shoegaze', 'put on something post-punk', 'play indie pop'. "
            "Returns a ranked list of artists for that genre. "
            "Cross-reference with get_music_taste to prefer artists the person already likes, "
            "then pick one and call play_music."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "Genre or mood tag e.g. 'shoegaze', 'post-punk', 'indie pop', 'lo-fi'.",
                },
            },
            "required": ["tag"],
        },
        "cache_control": {"type": "ephemeral"},
    },
]


# ─── Anthropic API with retry ─────────────────────────────────────────

async def _call_anthropic(
    model: str,
    max_tokens: int,
    messages: list[dict],
    system: str | list = "",
    tools: list[dict] | None = None,
    max_retries: int = 4,
    base_delay: float = 15.0,
    call_type: str = "conversation",
    db: "MemoryDB | None" = None,
) -> dict:
    """POST to Anthropic API with exponential backoff on 429.
    system can be a plain string or a list of cache-control blocks.
    """
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools

    for attempt in range(max_retries):
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                ANTHROPIC_API,
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "prompt-caching-2024-07-31",
                    "content-type": "application/json",
                },
                json=payload,
            )

        if response.status_code == 429:
            wait = base_delay * (2 ** attempt)
            log.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {wait:.0f}s")
            await asyncio.sleep(wait)
            continue

        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_created = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        if cache_read or cache_created:
            log.info(f"Tokens — input: {input_tokens}, cache hit: {cache_read}, cache write: {cache_created}")
        if db:
            db.log_api_call(
                model=model,
                call_type=call_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
                cache_write_tokens=cache_created,
            )
        return data

    raise Exception(f"Anthropic API rate limit exceeded after {max_retries} retries")


# ─── Database ─────────────────────────────────────────────────────────

class MemoryDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id     INTEGER NOT NULL,
                conv_id     INTEGER,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL,
                timestamp   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id         INTEGER NOT NULL,
                started_at      TEXT NOT NULL,
                ended_at        TEXT,
                summary         TEXT,
                summarised_at   TEXT
            );

            CREATE TABLE IF NOT EXISTS day_summaries (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id             INTEGER NOT NULL,
                date                TEXT NOT NULL,
                summary             TEXT NOT NULL,
                conversation_count  INTEGER,
                created_at          TEXT NOT NULL,
                UNIQUE(chat_id, date)
            );

            CREATE TABLE IF NOT EXISTS core_memory (
                chat_id     INTEGER PRIMARY KEY,
                memory      TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS household_rhythms (
                chat_id     INTEGER PRIMARY KEY,
                rhythms     TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS person_dossiers (
                person_name TEXT PRIMARY KEY,
                dossier     TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS consolidation_state (
                chat_id                 INTEGER PRIMARY KEY,
                last_conv_summarised_id INTEGER DEFAULT 0,
                last_day_consolidation  TEXT,
                last_core_update        TEXT,
                last_rhythms_update     TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_messages_chat_conv
                ON messages(chat_id, conv_id);
            CREATE INDEX IF NOT EXISTS idx_messages_chat_time
                ON messages(chat_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_conversations_chat
                ON conversations(chat_id, started_at);

            CREATE TABLE IF NOT EXISTS api_calls (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp           TEXT NOT NULL,
                model               TEXT NOT NULL,
                call_type           TEXT NOT NULL,
                input_tokens        INTEGER DEFAULT 0,
                output_tokens       INTEGER DEFAULT 0,
                cache_read_tokens   INTEGER DEFAULT 0,
                cache_write_tokens  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS brave_calls (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                query       TEXT NOT NULL,
                result_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS health_check_results (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at      TEXT NOT NULL,
                service     TEXT NOT NULL,
                status      TEXT NOT NULL,
                latency_ms  INTEGER,
                error       TEXT
            );
        """)
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(consolidation_state)").fetchall()]
        if "last_rhythms_update" not in cols:
            self.conn.execute("ALTER TABLE consolidation_state ADD COLUMN last_rhythms_update TEXT")
        self.conn.commit()
        self.music = MusicGraph(self.conn)

    # ── Messages ──────────────────────────────────────────────────────

    def store_message(self, chat_id: int, role: str, content: str,
                      conv_id: int | None = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO messages (chat_id, conv_id, role, content, timestamp) VALUES (?, ?, ?, ?, ?)",
            (chat_id, conv_id, role, content, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_last_message_time(self, chat_id: int) -> datetime | None:
        row = self.conn.execute(
            "SELECT timestamp FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
            (chat_id,)
        ).fetchone()
        if row:
            return datetime.fromisoformat(row["timestamp"])
        return None

    def get_conversation_messages(self, chat_id: int, conv_id: int) -> list[dict]:
        rows = self.conn.execute(
            """SELECT role, content, timestamp FROM messages
               WHERE chat_id = ? AND conv_id = ?
               ORDER BY id ASC""",
            (chat_id, conv_id)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Conversations ─────────────────────────────────────────────────

    def get_active_conversation(self, chat_id: int) -> dict | None:
        row = self.conn.execute(
            """SELECT * FROM conversations
               WHERE chat_id = ? AND ended_at IS NULL
               ORDER BY id DESC LIMIT 1""",
            (chat_id,)
        ).fetchone()
        return dict(row) if row else None

    def open_conversation(self, chat_id: int) -> int:
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO conversations (chat_id, started_at) VALUES (?, ?)",
            (chat_id, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def close_conversation(self, conv_id: int):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE conversations SET ended_at = ? WHERE id = ?",
            (now, conv_id)
        )
        self.conn.commit()

    def store_conversation_summary(self, conv_id: int, summary: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "UPDATE conversations SET summary = ?, summarised_at = ? WHERE id = ?",
            (summary, now, conv_id)
        )
        self.conn.commit()

    def get_todays_conversation_summaries(self, chat_id: int) -> list[dict]:
        """Closed, summarised conversations from today (Melbourne time)."""
        now_melb = _now_melb()
        midnight_melb = now_melb.replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = midnight_melb.astimezone(timezone.utc).isoformat()
        rows = self.conn.execute(
            """SELECT id, started_at, ended_at, summary FROM conversations
               WHERE chat_id = ? AND ended_at IS NOT NULL
               AND summary IS NOT NULL AND started_at >= ?
               ORDER BY started_at ASC""",
            (chat_id, cutoff)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_unsummarised_closed_conversations(self, chat_id: int) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM conversations
               WHERE chat_id = ? AND ended_at IS NOT NULL AND summary IS NULL
               ORDER BY id ASC""",
            (chat_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_yesterday_conversations(self, chat_id: int) -> list[dict]:
        """Summarised conversations from yesterday (Melbourne time)."""
        now_melb = _now_melb()
        yesterday_melb = now_melb - timedelta(days=1)
        day_start_utc = (yesterday_melb.replace(hour=0, minute=0, second=0, microsecond=0)
                         .astimezone(timezone.utc).isoformat())
        day_end_utc = (yesterday_melb.replace(hour=23, minute=59, second=59, microsecond=0)
                       .astimezone(timezone.utc).isoformat())
        rows = self.conn.execute(
            """SELECT * FROM conversations
               WHERE chat_id = ? AND summary IS NOT NULL
               AND started_at >= ? AND started_at <= ?
               ORDER BY started_at ASC""",
            (chat_id, day_start_utc, day_end_utc)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Day summaries ─────────────────────────────────────────────────

    def store_day_summary(self, chat_id: int, date_str: str, summary: str,
                          conversation_count: int):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO day_summaries (chat_id, date, summary, conversation_count, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(chat_id, date) DO UPDATE SET
                   summary = excluded.summary,
                   conversation_count = excluded.conversation_count,
                   created_at = excluded.created_at""",
            (chat_id, date_str, summary, conversation_count, now)
        )
        self.conn.commit()

    def get_day_summaries(self, chat_id: int, days: int = DAY_SUMMARY_KEEP_DAYS) -> list[dict]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self.conn.execute(
            """SELECT date, summary FROM day_summaries
               WHERE chat_id = ? AND date >= ?
               ORDER BY date ASC""",
            (chat_id, cutoff)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Core memory ───────────────────────────────────────────────────

    def get_core_memory(self, chat_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT memory FROM core_memory WHERE chat_id = ?",
            (chat_id,)
        ).fetchone()
        return row["memory"] if row else None

    def store_core_memory(self, chat_id: int, memory: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO core_memory (chat_id, memory, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(chat_id) DO UPDATE SET
                   memory = excluded.memory,
                   updated_at = excluded.updated_at""",
            (chat_id, memory, now)
        )
        self.conn.commit()

    def get_household_rhythms(self, chat_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT rhythms FROM household_rhythms WHERE chat_id = ?",
            (chat_id,)
        ).fetchone()
        return row["rhythms"] if row else None

    def store_household_rhythms(self, chat_id: int, rhythms: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO household_rhythms (chat_id, rhythms, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(chat_id) DO UPDATE SET
                   rhythms = excluded.rhythms,
                   updated_at = excluded.updated_at""",
            (chat_id, rhythms, now)
        )
        self.conn.commit()

    # ── Person dossiers ───────────────────────────────────────────────

    def get_dossier(self, person_name: str) -> str | None:
        row = self.conn.execute(
            "SELECT dossier FROM person_dossiers WHERE person_name = ?",
            (person_name,)
        ).fetchone()
        return row["dossier"] if row else None

    def store_dossier(self, person_name: str, dossier: str):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO person_dossiers (person_name, dossier, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(person_name) DO UPDATE SET
                   dossier = excluded.dossier,
                   updated_at = excluded.updated_at""",
            (person_name, dossier, now)
        )
        self.conn.commit()

    def get_all_dossiers(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT person_name, dossier, updated_at FROM person_dossiers ORDER BY person_name"
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Consolidation state ───────────────────────────────────────────

    def get_consolidation_state(self, chat_id: int) -> dict:
        row = self.conn.execute(
            "SELECT * FROM consolidation_state WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        if row:
            return dict(row)
        self.conn.execute(
            "INSERT INTO consolidation_state (chat_id) VALUES (?)", (chat_id,)
        )
        self.conn.commit()
        return {
            "chat_id": chat_id,
            "last_conv_summarised_id": 0,
            "last_day_consolidation": None,
            "last_core_update": None,
            "last_rhythms_update": None,
        }

    def update_consolidation_state(self, chat_id: int, **kwargs):
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values()) + [chat_id]
        self.conn.execute(
            f"UPDATE consolidation_state SET {sets} WHERE chat_id = ?", vals
        )
        self.conn.commit()

    def inject_memory_note(self, chat_id: int, note: str):
        """Inject a note directly as a user message (for /note and /discover)."""
        self.store_message(chat_id, "user", f"[MEMORY NOTE] {note}")

    # ── API call tracking ─────────────────────────────────────────────

    def log_api_call(self, model: str, call_type: str, input_tokens: int,
                     output_tokens: int, cache_read_tokens: int, cache_write_tokens: int):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO api_calls
               (timestamp, model, call_type, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (now, model, call_type, input_tokens, output_tokens,
             cache_read_tokens, cache_write_tokens)
        )
        self.conn.commit()

    def log_brave_call(self, query: str, result_count: int):
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO brave_calls (timestamp, query, result_count) VALUES (?, ?, ?)",
            (now, query, result_count)
        )
        self.conn.commit()

    def get_api_usage(self, since_iso: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT model, call_type,
                      COUNT(*) as calls,
                      SUM(input_tokens) as input_tokens,
                      SUM(output_tokens) as output_tokens,
                      SUM(cache_read_tokens) as cache_read_tokens,
                      SUM(cache_write_tokens) as cache_write_tokens
               FROM api_calls WHERE timestamp >= ?
               GROUP BY model, call_type
               ORDER BY model, call_type""",
            (since_iso,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_brave_usage(self, since_iso: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as calls FROM brave_calls WHERE timestamp >= ?",
            (since_iso,)
        ).fetchone()
        return row["calls"] if row else 0

    # ── Health checks ─────────────────────────────────────────────────

    def store_health_results(self, results: list[dict]):
        run_at = datetime.now(timezone.utc).isoformat()
        # Clear previous results
        self.conn.execute("DELETE FROM health_check_results")
        for r in results:
            self.conn.execute(
                """INSERT INTO health_check_results
                   (run_at, service, status, latency_ms, error)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_at, r["service"], r["status"], r["latency_ms"], r.get("error"))
            )
        self.conn.commit()

    def get_health_results(self) -> tuple[list[dict], str | None]:
        """Returns (results, run_at) or ([], None) if never run."""
        rows = self.conn.execute(
            "SELECT * FROM health_check_results ORDER BY service"
        ).fetchall()
        if not rows:
            return [], None
        results = [dict(r) for r in rows]
        run_at = results[0]["run_at"] if results else None
        return results, run_at


# ─── Memory Engine ────────────────────────────────────────────────────

class MemoryEngine:
    def __init__(self, db: MemoryDB):
        self.db = db

    async def _summarise(self, system: str, content: str, call_type: str = "summarisation") -> str:
        data = await _call_anthropic(
            model=CLAUDE_MODEL_SUMMARISE,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": content}],
            call_type=call_type,
            db=self.db,
        )
        return data["content"][0]["text"]

    # ── Conversation boundary detection ───────────────────────────────

    def get_or_create_conversation(self, chat_id: int) -> tuple[int, bool]:
        """
        Returns (conv_id, is_new).
        Closes stale conversation and opens a new one if gap exceeded.
        """
        now = datetime.now(timezone.utc)
        active = self.db.get_active_conversation(chat_id)

        if active:
            last_msg_time = self.db.get_last_message_time(chat_id)
            if last_msg_time:
                gap = (now - last_msg_time).total_seconds()
                if gap > CONVERSATION_GAP_SECONDS:
                    log.info(f"Conversation gap {gap:.0f}s — closing conv {active['id']}")
                    self.db.close_conversation(active["id"])
                    return self.db.open_conversation(chat_id), True
            return active["id"], False

        return self.db.open_conversation(chat_id), True

    def is_morning_wakeup(self, chat_id: int) -> bool:
        """True if last message was more than SLEEP_GAP_SECONDS ago."""
        last = self.db.get_last_message_time(chat_id)
        if not last:
            return False
        gap = (datetime.now(timezone.utc) - last).total_seconds()
        return gap > SLEEP_GAP_SECONDS

    # ── Summarisation ─────────────────────────────────────────────────

    async def summarise_conversation(self, chat_id: int, conv_id: int):
        messages = self.db.get_conversation_messages(chat_id, conv_id)
        if len(messages) < 2:
            return  # not worth summarising

        def _fmt_timestamp(ts: str) -> str:
            try:
                dt = datetime.fromisoformat(ts).astimezone(_MELB_TZ or timezone(timedelta(hours=10)))
                return dt.strftime("%A %d %B, %I:%M %p")
            except Exception:
                return ts

        formatted = "\n".join(
            f"[{_fmt_timestamp(m['timestamp'])}] {m['role']}: {m['content']}"
            for m in messages
        )
        summary = await self._summarise(CONVERSATION_SUMMARY_PROMPT, formatted)
        validation = validate_conversation_summary(summary)
        if not validation:
            log.warning(f"Conversation {conv_id} summary failed validation: {validation.reason}")
            log.warning(f"Rejected summary preview: {summary[:200]!r}")
            return
        self.db.store_conversation_summary(conv_id, summary)
        log.info(f"Conversation {conv_id} summarised ({len(messages)} messages)")

    async def consolidate_yesterday(self, chat_id: int):
        """Consolidate yesterday's conversation summaries into a single day summary."""
        convs = self.db.get_yesterday_conversations(chat_id)
        if not convs:
            log.info("No conversations from yesterday to consolidate")
            return

        now_melb = _now_melb()
        yesterday_str = (now_melb - timedelta(days=1)).strftime("%Y-%m-%d")

        state = self.db.get_consolidation_state(chat_id)
        if state.get("last_day_consolidation") == yesterday_str:
            log.info(f"Yesterday ({yesterday_str}) already consolidated")
            return

        log.info(f"Consolidating {len(convs)} conversations for {yesterday_str}")

        formatted = "\n\n---\n\n".join(
            f"[{c['started_at']} to {c['ended_at']}]\n{c['summary']}"
            for c in convs
        )
        summary = await self._summarise(DAY_SUMMARY_PROMPT, formatted)
        validation = validate_day_summary(summary)
        if not validation:
            log.warning(f"Day summary for {yesterday_str} failed validation: {validation.reason}")
            log.warning(f"Rejected summary preview: {summary[:200]!r}")
            return
        self.db.store_day_summary(chat_id, yesterday_str, summary, len(convs))
        self.db.update_consolidation_state(chat_id, last_day_consolidation=yesterday_str)
        log.info(f"Day summary stored for {yesterday_str}")

    async def maybe_update_core_memory(self, chat_id: int):
        """Weekly distillation of day summaries into core memory."""
        state = self.db.get_consolidation_state(chat_id)
        last = state.get("last_core_update")

        if last:
            last_dt = datetime.fromisoformat(last)
            if datetime.now(timezone.utc) - last_dt < timedelta(days=CORE_UPDATE_INTERVAL_DAYS):
                return

        day_summaries = self.db.get_day_summaries(chat_id, days=30)
        if not day_summaries:
            return

        log.info(f"Updating core memory from {len(day_summaries)} day summaries")

        existing_core = self.db.get_core_memory(chat_id) or "No existing core memory."
        formatted = "\n\n---\n\n".join(
            f"[{s['date']}]\n{s['summary']}" for s in day_summaries
        )
        prompt = CORE_MEMORY_PROMPT.format(existing_core=existing_core)
        memory = await self._summarise(prompt, formatted)

        validation = validate_core_memory(memory, existing_core)
        if not validation:
            log.warning(f"Core memory update failed validation: {validation.reason}")
            log.warning(f"Rejected memory preview: {memory[:200]!r}")
            return
        self.db.store_core_memory(chat_id, memory)
        self.db.update_consolidation_state(
            chat_id, last_core_update=datetime.now(timezone.utc).isoformat()
        )
        log.info("Core memory updated")

    async def maybe_update_rhythms(self, chat_id: int):
        """Weekly extraction of household rhythms from day summaries."""
        state = self.db.get_consolidation_state(chat_id)
        last = state.get("last_rhythms_update")

        if last:
            last_dt = datetime.fromisoformat(last)
            if datetime.now(timezone.utc) - last_dt < timedelta(days=CORE_UPDATE_INTERVAL_DAYS):
                return

        day_summaries = self.db.get_day_summaries(chat_id, days=30)
        if not day_summaries:
            return

        log.info(f"Updating household rhythms from {len(day_summaries)} day summaries")

        existing_rhythms = self.db.get_household_rhythms(chat_id) or "No existing rhythm notes."
        formatted = "\n\n---\n\n".join(
            f"[{s['date']}]\n{s['summary']}" for s in day_summaries
        )
        prompt = RHYTHMS_PROMPT.format(existing_rhythms=existing_rhythms)
        rhythms = await self._summarise(prompt, formatted)

        validation = validate_rhythms(rhythms, existing_rhythms)
        if not validation:
            log.warning(f"Household rhythms update failed validation: {validation.reason}")
            log.warning(f"Rejected rhythms preview: {rhythms[:200]!r}")
            return
        self.db.store_household_rhythms(chat_id, rhythms)
        self.db.update_consolidation_state(
            chat_id, last_rhythms_update=datetime.now(timezone.utc).isoformat()
        )
        log.info("Household rhythms updated")

    async def update_dossier(self, person_name: str, conv_id: int, chat_id: int):
        """
        Update a person's dossier from their conversation.
        Only called when the conversation is attributable to a known person.
        Only feeds user messages — assistant content is irrelevant to the dossier.
        Requires at least 3 meaningful user messages before updating.
        """
        messages = self.db.get_conversation_messages(chat_id, conv_id)

        # Only consider user messages — dossier is built from what the person said
        user_messages = [m for m in messages if m["role"] == "user"]

        # Require meaningful user content — skip short/thin conversations
        meaningful = [m for m in user_messages if len(m["content"].strip()) > 20]
        if len(meaningful) < 3:
            log.info(f"Skipping dossier update for {person_name} — insufficient user content ({len(meaningful)} meaningful messages)")
            return

        def _fmt_timestamp(ts: str) -> str:
            try:
                dt = datetime.fromisoformat(ts).astimezone(_MELB_TZ or timezone(timedelta(hours=10)))
                return dt.strftime("%A %d %B, %I:%M %p")
            except Exception:
                return ts

        formatted = "\n".join(
            f"[{_fmt_timestamp(m['timestamp'])}] {m['content']}"
            for m in user_messages
        )

        existing_dossier = self.db.get_dossier(person_name) or "No existing dossier."
        prompt = DOSSIER_PROMPT.format(existing_dossier=existing_dossier)

        data = await _call_anthropic(
            model=CLAUDE_MODEL_DOSSIER,
            max_tokens=1024,
            system=prompt,
            messages=[{"role": "user", "content": formatted}],
            call_type="dossier",
            db=self.db,
        )
        dossier = data["content"][0]["text"]
        existing_dossier_text = self.db.get_dossier(person_name)
        validation = validate_dossier(dossier, existing_dossier_text)
        if not validation:
            log.warning(f"Dossier update for {person_name} failed validation: {validation.reason}")
            log.warning(f"Rejected dossier preview: {dossier[:200]!r}")
            return
        self.db.store_dossier(person_name, dossier)
        log.info(f"Dossier updated for {person_name}")

    # ── Context assembly ──────────────────────────────────────────────

    def build_context(self, chat_id: int, conv_id: int, person_name: str | None = None) -> tuple[list, list[dict]]:
        """
        Assemble system prompt blocks and message list for a Claude API call.
        Returns (system_blocks, messages).

        System blocks use prompt caching:
          Block 1 — static identity/personality/family  [cache ← never changes]
          Block 2 — core memory                         [cache ← changes weekly]
          Block 3 — household rhythms                   [cache ← changes weekly]
          Block 4 — person dossier                      [cache ← changes per conversation]
          Block 5 — dynamic: time + day/today summaries [fresh every call]
        """
        core = self.db.get_core_memory(chat_id)
        day_summaries = self.db.get_day_summaries(chat_id)
        todays_convs = self.db.get_todays_conversation_summaries(chat_id)
        active_messages = self.db.get_conversation_messages(chat_id, conv_id)
        system_blocks = [
            {
                "type": "text",
                "text": SYSTEM_STATIC,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # ── Block 2: core memory — cached, changes weekly ──
        if core:
            system_blocks.append({
                "type": "text",
                "text": f"== What I know about the family ==\n{core}",
                "cache_control": {"type": "ephemeral"},
            })

        # ── Block 3: household rhythms — cached, changes weekly ──
        rhythms = self.db.get_household_rhythms(chat_id)
        if rhythms:
            system_blocks.append({
                "type": "text",
                "text": f"== Household rhythms ==\n{rhythms}",
                "cache_control": {"type": "ephemeral"},
            })

        # ── Block 4: person dossier + music taste — cached, changes per conversation ──
        if person_name:
            dossier = self.db.get_dossier(person_name)
            music_summary = self.db.music.get_person_summary(person_name)
            has_music = "[No music preferences" not in music_summary

            if dossier or has_music:
                block_text = ""
                if dossier:
                    block_text += f"== What I know about {person_name} ==\n{dossier}"
                if has_music:
                    block_text += f"\n\n{music_summary}" if block_text else music_summary
                system_blocks.append({
                    "type": "text",
                    "text": block_text,
                    # No cache_control — dossier changes per conversation,
                    # and we're at the 4-block cache limit (static/core/rhythms/tools)
                })

        # ── Block 5: dynamic context — fresh every call ──
        now_melb = _now_melb()
        time_str = now_melb.strftime("%A %d %B %Y, %I:%M %p AEDT")
        today_str = now_melb.strftime("%Y-%m-%d")

        days_block = ""
        if day_summaries:
            past_days = [s for s in day_summaries if s["date"] != today_str]
            if past_days:
                days_text = "\n\n".join(
                    f"[{s['date']}]\n{s['summary']}" for s in past_days
                )
                days_block = f"== Past few days ==\n{days_text}"

        today_block = ""
        if todays_convs:
            today_text = "\n\n".join(s["summary"] for s in todays_convs)
            today_block = f"== Earlier today ==\n{today_text}"

        dynamic_text = SYSTEM_DYNAMIC_TEMPLATE.format(
            current_time=time_str,
            day_summaries=days_block,
            today_summaries=today_block,
        )
        system_blocks.append({
            "type": "text",
            "text": dynamic_text,
        })

        # ── Message list from active conversation ──
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in active_messages
        ]

        return system_blocks, messages


async def _handle_play_music(tool_input: dict) -> str:
    """
    Handle a play_music tool call.
    Searches MA, picks the best result, plays on the specified speaker(s).
    Returns a status string for Clara to relay.
    """
    query = tool_input["query"]
    media_type = tool_input.get("media_type", "artist")
    room = tool_input["room"]
    enqueue = tool_input.get("enqueue", "replace")
    radio_mode = tool_input.get("radio_mode", False)

    # Resolve room to player IDs
    player_ids = resolve_speaker(room)
    if player_ids is None:
        return f"[Unknown room: '{room}'. Available: lounge, kitchen, office, main bedroom, sam's room, everywhere]"

    # Search MA
    try:
        results = await ma_search(query, media_type)
    except Exception as e:
        return f"[Music Assistant search failed: {e}]"

    if not results:
        return f"[No results found for '{query}' ({media_type})]"

    best = results[0]
    uri = best["uri"]
    name = best["name"]

    # Play on each target speaker
    errors = []
    for player_id in player_ids:
        try:
            await ma_play(player_id, uri, enqueue=enqueue, radio_mode=radio_mode)
        except Exception as e:
            errors.append(f"{player_id}: {e}")

    if errors:
        return f"[Playback errors: {'; '.join(errors)}]"

    speaker_label = room if room != "everywhere" else "all speakers"
    mode_note = " (radio mode)" if radio_mode else ""
    return f"Playing {media_type} '{name}' on {speaker_label}{mode_note}."


async def _handle_find_phone(tool_input: dict) -> str:
    """Ring a family member's phone via HA iOS companion app notify service."""
    person = tool_input["person"].lower()

    # Map person to HA notify service
    # These match the iOS companion app device names — update if different
    notify_services = {
        "wil":    "notify.mobile_app_wils_iphone",
        "kirsty": "notify.mobile_app_kirstys_iphone",
        "sam":    "notify.mobile_app_sams_iphone",
        "ezra":   "notify.mobile_app_ezras_iphone",
    }

    service = notify_services.get(person)
    if not service:
        return f"[Unknown person: {person}]"

    try:
        await call_service("notify", service.replace("notify.", ""), {
            "message": "Find my phone!",
            "title": "Assistant is trying to find you 📱",
            "data": {
                "push": {
                    "sound": {
                        "name": "default",
                        "critical": 1,
                        "volume": 1.0,
                    }
                }
            }
        })
        return f"Ringing {person.title()}'s phone now."
    except Exception as e:
        return (
            f"[Couldn't ring {person.title()}'s phone: {e}. "
            f"Make sure the HA iOS companion app is installed on their device.]"
        )


async def _handle_control_music(tool_input: dict) -> str:
    """Handle a control_music tool call."""
    action = tool_input["action"]
    room = tool_input.get("room", "lounge")

    # Group action is special — needs a list of rooms
    if action == "group":
        rooms = tool_input.get("rooms", [])
        if len(rooms) < 2:
            return "[group action requires at least 2 rooms]"
        player_ids = []
        for r in rooms:
            ids = resolve_speaker(r)
            if not ids:
                return f"[Unknown room: '{r}']"
            player_ids.extend(ids)
        lead = player_ids[0]
        children = player_ids[1:]
        try:
            await ma_join(lead, children)
            return f"Grouped {', '.join(rooms)} — playing in sync from {rooms[0]}."
        except Exception as e:
            return f"[Group failed: {e}]"

    # Ungroup
    if action == "ungroup":
        player_ids = resolve_speaker(room)
        if not player_ids:
            return f"[Unknown room: '{room}']"
        try:
            await ma_unjoin(player_ids[0])
            return f"Ungrouped {room}."
        except Exception as e:
            return f"[Ungroup failed: {e}]"

    # Single-player actions
    player_ids = resolve_speaker(room) if room else None
    if not player_ids:
        return f"[Unknown room: '{room}']"

    # For 'everywhere', apply to all speakers
    errors = []
    for player_id in player_ids:
        try:
            if action == "stop":
                await ma_stop(player_id)
            elif action in ("pause", "resume"):
                await ma_pause(player_id)
            elif action == "next":
                await ma_next(player_id)
            elif action == "previous":
                await ma_previous(player_id)
            elif action == "volume":
                vol = tool_input.get("volume_level")
                if vol is None:
                    return "[volume action requires volume_level]"
                await ma_set_volume(player_id, vol)
            else:
                return f"[Unknown action: {action}]"
        except Exception as e:
            errors.append(f"{player_id}: {e}")

    if errors:
        return f"[Errors: {'; '.join(errors)}]"

    labels = {"stop": "Stopped", "pause": "Paused", "resume": "Resumed",
               "next": "Skipped to next", "previous": "Back to previous",
               "volume": f"Volume set to {tool_input.get('volume_level')}%"}
    return f"{labels.get(action, action.title())} on {room}."


# ─── Bot ──────────────────────────────────────────────────────────────

class ClaraBot:
    def __init__(self):
        self.db = MemoryDB(DB_PATH)
        self.memory = MemoryEngine(self.db)

    async def _download_image(self, file_id: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(
                f"{TELEGRAM_API}/getFile",
                params={"file_id": file_id}
            )
            r.raise_for_status()
            file_path = r.json()["result"]["file_path"]
            img_r = await client.get(
                f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
            )
            img_r.raise_for_status()
            return base64.b64encode(img_r.content).decode()

    async def handle_message(
        self,
        chat_id: int,
        user_text: str,
        channel: str = "telegram",
        username: str = "unknown",
        image_b64: str | None = None,
    ) -> str:

        # Resolve display name from username map
        display_name = USERS.get(username, username)

        # ── Morning wakeup — consolidate before responding ──
        if self.memory.is_morning_wakeup(chat_id):
            log.info("Morning wakeup detected — consolidating yesterday before responding")
            try:
                for conv in self.db.get_unsummarised_closed_conversations(chat_id):
                    await self.memory.summarise_conversation(chat_id, conv["id"])
                await self.memory.consolidate_yesterday(chat_id)
            except Exception as e:
                log.error(f"Morning consolidation error: {e}")

        # ── Conversation boundary ──
        conv_id, is_new = self.memory.get_or_create_conversation(chat_id)
        if is_new:
            log.info(f"New conversation opened: {conv_id}")

        # ── Store incoming message ──
        now_melb = _now_melb()
        time_str = now_melb.strftime("%A %d %B, %I:%M %p")
        store_text = user_text if user_text else "[image]"
        self.db.store_message(chat_id, "user", store_text, conv_id)

        # Build context
        system_prompt, messages = self.memory.build_context(chat_id, conv_id, person_name=display_name)

        # Current message content — identity is explicit and authoritative
        timestamped = f"[{time_str} — message from {display_name} via {channel}]\n{user_text}"
        if image_b64:
            current_content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {
                    "type": "text",
                    "text": timestamped if user_text else f"[{time_str} — message from {display_name} via {channel}]\n[image sent with no caption]",
                },
            ]
        else:
            current_content = timestamped

        # Replace last user message with timestamped version
        if messages and messages[-1]["role"] == "user":
            messages = messages[:-1] + [{"role": "user", "content": current_content}]
        else:
            messages = messages + [{"role": "user", "content": current_content}]

        try:
            # ── Agentic loop ──
            max_iterations = 10
            for _ in range(max_iterations):
                data = await _call_anthropic(
                    model=CLAUDE_MODEL,
                    max_tokens=MAX_TOKENS,
                    system=system_prompt,
                    tools=HA_TOOLS,
                    messages=messages,
                    call_type="conversation",
                    db=self.db,
                )

                stop_reason = data.get("stop_reason")
                log.info(f"Model stop_reason: {stop_reason} | content types: {[b['type'] for b in data.get('content', [])]}")

                if stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": data["content"]})
                    tool_results = []

                    for block in data["content"]:
                        if block["type"] != "tool_use":
                            continue
                        log.info(f"Tool call: {block['name']} {block['input']}")

                        if block["name"] == "get_home_state":
                            result = await get_home_state(
                                domains=block["input"]["domains"],
                                filter=block["input"].get("filter"),
                            )

                        elif block["name"] == "play_music":
                            result = await _handle_play_music(block["input"])

                        elif block["name"] == "control_music":
                            result = await _handle_control_music(block["input"])

                        elif block["name"] == "get_calendar":
                            days = block["input"].get("days_ahead", 7)
                            result = await get_family_events(days_ahead=min(days, 30))

                        elif block["name"] == "find_phone":
                            result = await _handle_find_phone(block["input"])

                        elif block["name"] == "get_transport":
                            result = await get_transport_info(
                                stop_name=block["input"]["stop"],
                                include_disruptions=block["input"].get("include_disruptions", True),
                                direction=block["input"].get("direction"),
                                arrive_by=block["input"].get("arrive_by"),
                            )

                        elif block["name"] == "web_search":
                            query = block["input"]["query"]
                            count = min(block["input"].get("count", 5), 10)
                            result = await brave_search(query=query, count=count)
                            # Log the Brave call for cost tracking
                            try:
                                self.db.log_brave_call(query=query, result_count=count)
                            except Exception:
                                pass

                        elif block["name"] == "fetch_page":
                            result = await fetch_page(block["input"]["url"])

                        elif block["name"] == "rate_music":
                            inp = block["input"]
                            person = inp["person"]
                            rating = inp["rating"]
                            source = inp.get("source", "explicit")
                            if inp["type"] == "track":
                                r = await self.db.music.rate_track(
                                    person=person,
                                    title=inp["track"],
                                    artist_name=inp["artist"],
                                    rating=rating,
                                    source=source,
                                    album_title=inp.get("album"),
                                )
                            elif inp["type"] == "album":
                                r = await self.db.music.rate_album(
                                    person=person,
                                    album_title=inp["album"],
                                    artist_name=inp["artist"],
                                    rating=rating,
                                    source=source,
                                    year=inp.get("year"),
                                )
                            else:
                                r = await self.db.music.rate_artist(
                                    person=person,
                                    artist_name=inp["artist"],
                                    rating=rating,
                                    source=source,
                                )
                            if inp.get("discovered_via"):
                                self.db.music.record_discovery(
                                    person=person,
                                    artist_name=inp["artist"],
                                    discovered_via=inp["discovered_via"],
                                )
                            result = self.db.music.format_rating_confirmation(r)

                        elif block["name"] == "get_music_taste":
                            person = block["input"]["person"]
                            result = self.db.music.get_person_summary(person)

                        elif block["name"] == "get_similar_artists":
                            artist = block["input"]["artist"]
                            result = await format_similar_artists(artist)
                            log.info(f"Last.fm similar artists for '{artist}'")

                        elif block["name"] == "get_artist_top_tracks":
                            artist = block["input"]["artist"]
                            limit = block["input"].get("limit", 5)
                            result = await format_top_tracks(artist, limit=limit)
                            log.info(f"Last.fm top tracks for '{artist}'")

                        elif block["name"] == "get_artists_by_genre":
                            tag = block["input"]["tag"]
                            result = await format_tag_artists(tag)
                            log.info(f"Last.fm genre lookup: '{tag}'")

                        else:
                            result = f"[Unknown tool: {block['name']}]"

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": result,
                        })

                    messages.append({"role": "user", "content": tool_results})
                    continue

                assistant_message = next(
                    (b["text"] for b in data["content"] if b["type"] == "text"), ""
                )
                self.db.store_message(chat_id, "assistant", assistant_message, conv_id)
                asyncio.create_task(self._background_tasks(chat_id, conv_id, person_name=display_name))
                return assistant_message

            log.error("Agentic loop exhausted max iterations")
            return "Something got tangled up. Try again."

        except httpx.HTTPStatusError as e:
            log.error(f"Claude API error: {e.response.status_code} - {e.response.text}")
            return "Sorry, having trouble connecting right now. Give me a sec."
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            return "Something went sideways. Try again in a minute."

    async def _background_tasks(self, chat_id: int, conv_id: int, person_name: str | None = None):
        """Post-response background work, staggered clear of conversation token budget."""
        try:
            await asyncio.sleep(random.uniform(20, 30))

            for conv in self.db.get_unsummarised_closed_conversations(chat_id):
                await self.memory.summarise_conversation(chat_id, conv["id"])
                await asyncio.sleep(15)
                # Update dossier from this conversation now that it's summarised
                if person_name and person_name.lower() not in ("unknown",):
                    await self.memory.update_dossier(person_name, conv["id"], chat_id)
                    await asyncio.sleep(10)

            await self.memory.maybe_update_core_memory(chat_id)
            await asyncio.sleep(15)
            await self.memory.maybe_update_rhythms(chat_id)

        except Exception as e:
            log.error(f"Background task error: {e}")


# ─── Telegram Polling ─────────────────────────────────────────────────

async def send_message(chat_id: int, text: str):
    max_len = 4096
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)]
    async with httpx.AsyncClient(timeout=30.0) as client:
        for chunk in chunks:
            await client.post(
                f"{TELEGRAM_API}/sendMessage",
                json={"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"},
            )


async def send_typing(chat_id: int):
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"{TELEGRAM_API}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
        )


async def get_updates(offset: int = 0) -> tuple[list, int]:
    async with httpx.AsyncClient(timeout=35.0) as client:
        response = await client.get(
            f"{TELEGRAM_API}/getUpdates",
            params={"offset": offset, "timeout": 30},
        )
        data = response.json()

    if not data.get("ok"):
        log.warning(f"Telegram API error: {data}")
        return [], offset

    updates = data.get("result", [])
    if updates:
        offset = updates[-1]["update_id"] + 1
    return updates, offset


async def human_pacing(chat_id: int, reply: str):
    """
    Simulate human reading and typing behaviour.
    Phase 1 — silence: she's read the message, thinking (no typing indicator)
    Phase 2 — typing: indicator shown, duration proportional to response length
    Two thumbs on a phone — slower than desktop typing.
    """
    # Phase 1: silent read/think pause
    think_time = random.uniform(THINK_PAUSE_MIN, THINK_PAUSE_MAX)
    await asyncio.sleep(think_time)

    # Phase 2: typing indicator proportional to response length
    type_time = min(len(reply) / TYPING_CHARS_PER_SEC, TYPING_MAX_DELAY)
    type_time *= random.uniform(0.8, 1.2)

    elapsed = 0.0
    while elapsed < type_time:
        await send_typing(chat_id)
        chunk = min(4.0, type_time - elapsed)
        await asyncio.sleep(chunk)
        elapsed += chunk


def _format_costs(db: "MemoryDB") -> str:
    """Format API usage and cost summary for /status and /run_tests."""
    from datetime import date

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

    lines = ["== API Usage =="]

    for label, since in [("Today", today_start), ("This month", month_start)]:
        usage = db.get_api_usage(since)
        brave_calls = db.get_brave_usage(since)

        if not usage and brave_calls == 0:
            lines.append(f"{label}: no calls")
            continue

        total_cost = 0.0
        call_lines = []

        for row in usage:
            model = row["model"]
            pricing = PRICING.get(model, {})
            if not pricing:
                continue

            input_cost  = row["input_tokens"]       / 1_000_000 * pricing.get("input_per_1m", 0)
            output_cost = row["output_tokens"]      / 1_000_000 * pricing.get("output_per_1m", 0)
            cr_cost     = row["cache_read_tokens"]  / 1_000_000 * pricing.get("cache_read_per_1m", 0)
            cw_cost     = row["cache_write_tokens"] / 1_000_000 * pricing.get("cache_write_per_1m", 0)
            cost        = input_cost + output_cost + cr_cost + cw_cost
            total_cost += cost

            total_tokens = row["input_tokens"] + row["output_tokens"]
            call_lines.append(
                f"  {row['call_type']} ({model.split('-')[1]}): "
                f"{row['calls']} calls, {total_tokens:,} tokens — ${cost:.4f}"
            )

        # Brave
        brave_pricing = PRICING.get("brave_search", {})
        free_tier = brave_pricing.get("free_tier_monthly", 2000)
        brave_cost = 0.0
        if brave_calls > free_tier and label == "This month":
            brave_cost = (brave_calls - free_tier) / 1000 * brave_pricing.get("per_1k_calls", 3.0)
        total_cost += brave_cost
        call_lines.append(
            f"  brave search: {brave_calls} calls"
            + (f" — ${brave_cost:.4f}" if brave_cost > 0 else " (free tier)")
        )

        lines.append(f"\n{label}: ${total_cost:.4f} total")
        lines.extend(call_lines)

        # Cache savings — what cache_read_tokens would have cost at full input rate
        total_cache_read = sum(row["cache_read_tokens"] for row in usage)
        total_cache_savings = 0.0
        for row in usage:
            model = row["model"]
            pricing = PRICING.get(model, {})
            if not pricing:
                continue
            full_rate = pricing.get("input_per_1m", 0)
            cache_rate = pricing.get("cache_read_per_1m", 0)
            saving = row["cache_read_tokens"] / 1_000_000 * (full_rate - cache_rate)
            total_cache_savings += saving

        if total_cache_read > 0:
            lines.append(
                f"  cache savings: {total_cache_read:,} tokens read from cache"
                f" — saved ~${total_cache_savings:.4f}"
            )

    return "\n".join(lines)

# ─── Voice Request Handling ─────────────────────────────────────────────────

async def handle_voice_request(request, bot):
    data = await request.json()
    text = data.get("text", "")
    username = data.get("username", "unknown")
    chat_id = data.get("chat_id", 0)
    
    if not text or not chat_id:
        return web.json_response({"error": "missing text or chat_id"}, status=400)
    
    response = await bot.handle_message(
        chat_id=chat_id,
        user_text=text,
        channel="voice",
        username=username,
    )
    return web.json_response({"response": response})

async def main():
    log.info("Clara is waking up (v2 — conversation-boundary memory)...")
    log.info(f"Config: {CONFIG_PATH}")
    log.info(f"Conversation model: {CLAUDE_MODEL}")
    log.info(f"Summarisation model: {CLAUDE_MODEL_SUMMARISE}")
    log.info(f"Database: {DB_PATH}")

    log.info("Discovering home...")
    house_state = await discover_home()
    log.info(f"Home Assistant: {len(house_state)} chars of state discovered")

    bot = ClaraBot()

    # Internal HTTP API for voice server
    app = web.Application()

    async def handle_voice_request_bound(request):
        return await handle_voice_request(request, bot)

    app.router.add_post("/message", handle_voice_request_bound)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8766)
    await site.start()
    log.info("Voice API listening on port 8766")

    offset = 0

    while True:
        try:
            updates, offset = await get_updates(offset)

            for update in updates:
                message = update.get("message")
                if not message:
                    continue

                chat_id = message["chat"]["id"]
                user_id = str(message["from"]["id"])
                username = message["from"].get("username", "unknown")

                if ALLOWED_USER_IDS[0] and user_id not in ALLOWED_USER_IDS:
                    log.warning(f"Unauthorized: {username} ({user_id})")
                    await send_message(chat_id, "I don't know you. Sorry.")
                    continue

                has_text = "text" in message
                has_photo = "photo" in message

                if not has_text and not has_photo:
                    continue

                user_text = message.get("text") or message.get("caption") or ""
                log.info(f"[{username}] {user_text[:80]}")

                # ── Commands ──
                if user_text == "/start":
                    await send_message(chat_id, "Hey. I'm here.")
                    continue

                if user_text == "/memory":
                    core = bot.db.get_core_memory(chat_id)
                    day_sums = bot.db.get_day_summaries(chat_id)
                    today_convs = bot.db.get_todays_conversation_summaries(chat_id)
                    active = bot.db.get_active_conversation(chat_id)
                    unsummarised = bot.db.get_unsummarised_closed_conversations(chat_id)
                    state = bot.db.get_consolidation_state(chat_id)
                    dossiers = bot.db.get_all_dossiers()
                    status = (
                        f"Core memory: {'yes' if core else 'not yet'}\n"
                        f"Day summaries: {len(day_sums)}\n"
                        f"Today's conversations: {len(today_convs)} summarised\n"
                        f"Active conversation: {'yes, id=' + str(active['id']) if active else 'none'}\n"
                        f"Unsummarised closed convs: {len(unsummarised)}\n"
                        f"Last day consolidation: {state.get('last_day_consolidation') or 'never'}\n"
                        f"Last core update: {state.get('last_core_update') or 'never'}\n"
                        f"Last rhythms update: {state.get('last_rhythms_update') or 'never'}\n"
                        f"Dossiers: {', '.join(d['person_name'] for d in dossiers) or 'none'}"
                    )
                    await send_message(chat_id, status)
                    continue

                if user_text.startswith("/dossier"):
                    parts = user_text.split(maxsplit=1)
                    name = parts[1].strip().title() if len(parts) > 1 else USERS.get(username, username)
                    dossier = bot.db.get_dossier(name)
                    if dossier:
                        await send_message(chat_id, f"*Dossier: {name}*\n\n{dossier}")
                    else:
                        await send_message(chat_id, f"No dossier yet for {name}.")
                    continue

                if user_text.startswith("/note "):
                    note = user_text[6:].strip()
                    if note:
                        bot.db.inject_memory_note(chat_id, note)
                        await send_message(chat_id, "Got it, I'll remember that.")
                    continue

                if user_text == "/discover":
                    house_state = await discover_home()
                    bot.db.inject_memory_note(chat_id, house_state)
                    await send_message(chat_id, "Done. I've had a look around.")
                    continue

                if user_text == "/status":
                    results, run_at = bot.db.get_health_results()
                    if not results:
                        msg = "No health checks run yet. Use /run_tests to check."
                    else:
                        try:
                            run_dt = datetime.fromisoformat(run_at).astimezone(
                                _MELB_TZ or timezone(timedelta(hours=10))
                            )
                            run_label = run_dt.strftime("%-d %b, %I:%M %p")
                        except Exception:
                            run_label = run_at
                        msg = format_health_results(results, run_label)
                        msg += "\n\n" + _format_costs(bot.db)
                    await send_message(chat_id, msg)
                    continue

                if user_text == "/run_tests":
                    await send_message(chat_id, "Running health checks...")
                    results = await run_health_checks(DB_PATH)
                    bot.db.store_health_results(results)
                    now_label = _now_melb().strftime("%-d %b, %I:%M %p")
                    msg = format_health_results(results, now_label)
                    msg += "\n\n" + _format_costs(bot.db)
                    await send_message(chat_id, msg)
                    continue

                # ── Image download ──
                image_b64 = None
                if has_photo:
                    try:
                        photo = message["photo"][-1]
                        image_b64 = await bot._download_image(photo["file_id"])
                    except Exception as e:
                        log.error(f"Image download failed: {e}")

                reply = await bot.handle_message(
                    chat_id,
                    user_text,
                    channel="telegram",
                    username=username,
                    image_b64=image_b64,
                )
                await human_pacing(chat_id, reply)
                await send_message(chat_id, reply)

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Loop error: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())