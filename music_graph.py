"""
Music graph for Clara.
Tracks artists, albums, tracks, per-person ratings (thumbs up/down), and discoveries.
SQLite-backed, same database as the rest of Clara's memory.

Artist canonicalisation:
  1. Fuzzy match against existing artists (catches typos/near-misses in DB)
  2. MusicBrainz lookup for unknown artists (canonical name + MBID)
  3. Alias table records all alternate spellings seen

Rating values:
   1 = thumbs up
  -1 = thumbs down
   0 = neutral/explicit unrate

Source values:
  'explicit'  — person directly rated it
  'inferred'  — Clara inferred from conversation
"""

import httpx
import asyncio
import logging
from datetime import datetime, timezone
from rapidfuzz import fuzz, process as fuzz_process

log = logging.getLogger("clara-music")

# Fuzzy match threshold — below this, treat as a new artist
FUZZY_THRESHOLD = 85  # 85% similarity required to match existing

# Last.fm (used for artist name correction)
LASTFM_BASE = "https://ws.audioscrobbler.com/2.0/"
LASTFM_TIMEOUT = 5.0


# ─── Schema ───────────────────────────────────────────────────────────

MUSIC_GRAPH_SCHEMA = """
    CREATE TABLE IF NOT EXISTS mg_artists (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL COLLATE NOCASE,
        genre       TEXT,
        mbid        TEXT UNIQUE,
        created_at  TEXT NOT NULL,
        UNIQUE(name)
    );

    CREATE TABLE IF NOT EXISTS mg_artist_aliases (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        artist_id   INTEGER NOT NULL REFERENCES mg_artists(id),
        alias       TEXT NOT NULL COLLATE NOCASE,
        created_at  TEXT NOT NULL,
        UNIQUE(alias)
    );

    CREATE TABLE IF NOT EXISTS mg_albums (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        artist_id   INTEGER NOT NULL REFERENCES mg_artists(id),
        title       TEXT NOT NULL COLLATE NOCASE,
        year        INTEGER,
        mbid        TEXT,
        created_at  TEXT NOT NULL,
        UNIQUE(artist_id, title)
    );

    CREATE TABLE IF NOT EXISTS mg_tracks (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        artist_id   INTEGER NOT NULL REFERENCES mg_artists(id),
        album_id    INTEGER REFERENCES mg_albums(id),
        title       TEXT NOT NULL COLLATE NOCASE,
        mbid        TEXT,
        created_at  TEXT NOT NULL,
        UNIQUE(artist_id, title)
    );

    CREATE TABLE IF NOT EXISTS mg_artist_ratings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person      TEXT NOT NULL,
        artist_id   INTEGER NOT NULL REFERENCES mg_artists(id),
        rating      INTEGER NOT NULL CHECK(rating IN (-1, 0, 1)),
        source      TEXT NOT NULL DEFAULT 'explicit',
        noted_at    TEXT NOT NULL,
        UNIQUE(person, artist_id)
    );

    CREATE TABLE IF NOT EXISTS mg_album_ratings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person      TEXT NOT NULL,
        album_id    INTEGER NOT NULL REFERENCES mg_albums(id),
        rating      INTEGER NOT NULL CHECK(rating IN (-1, 0, 1)),
        source      TEXT NOT NULL DEFAULT 'explicit',
        noted_at    TEXT NOT NULL,
        UNIQUE(person, album_id)
    );

    CREATE TABLE IF NOT EXISTS mg_track_ratings (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        person      TEXT NOT NULL,
        track_id    INTEGER NOT NULL REFERENCES mg_tracks(id),
        rating      INTEGER NOT NULL CHECK(rating IN (-1, 0, 1)),
        source      TEXT NOT NULL DEFAULT 'explicit',
        noted_at    TEXT NOT NULL,
        UNIQUE(person, track_id)
    );

    CREATE TABLE IF NOT EXISTS mg_discoveries (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        person          TEXT NOT NULL,
        artist_id       INTEGER NOT NULL REFERENCES mg_artists(id),
        discovered_via  TEXT,
        discovered_at   TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS mg_artist_links (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        artist_id       INTEGER NOT NULL REFERENCES mg_artists(id),
        related_id      INTEGER NOT NULL REFERENCES mg_artists(id),
        relationship    TEXT NOT NULL DEFAULT 'similar',
        UNIQUE(artist_id, related_id)
    );

    CREATE INDEX IF NOT EXISTS idx_mg_track_ratings_person
        ON mg_track_ratings(person);
    CREATE INDEX IF NOT EXISTS idx_mg_album_ratings_person
        ON mg_album_ratings(person);
    CREATE INDEX IF NOT EXISTS idx_mg_artist_ratings_person
        ON mg_artist_ratings(person);
    CREATE INDEX IF NOT EXISTS idx_mg_tracks_artist
        ON mg_tracks(artist_id);
    CREATE INDEX IF NOT EXISTS idx_mg_albums_artist
        ON mg_albums(artist_id);
    CREATE INDEX IF NOT EXISTS idx_mg_artist_aliases_alias
        ON mg_artist_aliases(alias);
"""


# ─── MusicBrainz lookup ───────────────────────────────────────────────

async def _lastfm_correct(name: str) -> str:
    """
    Correct an artist name via Last.fm artist.getCorrection.
    Returns the canonical name, or the original if no correction found or on error.
    Note: mbid column is kept in schema but not populated (was MusicBrainz-specific).
    """
    import os
    api_key = os.environ.get("LASTFM_API_KEY", "")
    if not api_key:
        return name

    try:
        async with httpx.AsyncClient(timeout=LASTFM_TIMEOUT) as client:
            r = await client.get(
                LASTFM_BASE,
                params={
                    "method": "artist.getCorrection",
                    "artist": name,
                    "api_key": api_key,
                    "format": "json",
                },
            )
            r.raise_for_status()
            data = r.json()

        corrections = data.get("corrections", {})
        if not corrections:
            return name

        correction = corrections.get("correction", {})
        if not correction:
            return name

        canonical = correction.get("artist", {}).get("name", name)
        if canonical.lower() != name.lower():
            log.info(f"Last.fm correction: '{name}' → '{canonical}'")
        return canonical

    except Exception as e:
        log.warning(f"Last.fm correction failed for '{name}': {e}")
        return name


# ─── MusicGraph class ─────────────────────────────────────────────────

class MusicGraph:
    def __init__(self, conn):
        self.conn = conn
        self._init_schema()
        self._artist_name_cache: dict[int, str] = {}  # id → canonical name

    def _init_schema(self):
        self.conn.executescript(MUSIC_GRAPH_SCHEMA)
        self.conn.commit()

        # Migrations for existing DBs
        cols = [r[1] for r in self.conn.execute("PRAGMA table_info(mg_tracks)").fetchall()]
        if "album_id" not in cols:
            self.conn.execute("ALTER TABLE mg_tracks ADD COLUMN album_id INTEGER REFERENCES mg_albums(id)")
            self.conn.commit()

        # Check alias table exists (new addition)
        tables = [r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if "mg_artist_aliases" not in tables:
            self.conn.execute("""
                CREATE TABLE mg_artist_aliases (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    artist_id   INTEGER NOT NULL REFERENCES mg_artists(id),
                    alias       TEXT NOT NULL COLLATE NOCASE,
                    created_at  TEXT NOT NULL,
                    UNIQUE(alias)
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mg_artist_aliases_alias ON mg_artist_aliases(alias)")
            self.conn.commit()

    def _all_artist_names(self) -> dict[str, int]:
        """Return {lowercase_name: id} for all artists — used for fuzzy matching."""
        rows = self.conn.execute("SELECT id, name FROM mg_artists").fetchall()
        return {r["name"].lower(): r["id"] for r in rows}

    def _fuzzy_match(self, name: str) -> int | None:
        """
        Try to find an existing artist by fuzzy name match.
        Returns artist_id if confident match found, else None.
        """
        existing = self._all_artist_names()
        if not existing:
            return None

        # Check aliases too
        alias_rows = self.conn.execute(
            "SELECT artist_id, alias FROM mg_artist_aliases"
        ).fetchall()
        for row in alias_rows:
            existing[row["alias"].lower()] = row["artist_id"]

        result = fuzz_process.extractOne(
            name.lower(),
            existing.keys(),
            scorer=fuzz.ratio,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if result:
            matched_name, score, _ = result
            artist_id = existing[matched_name]
            log.info(f"Fuzzy match: '{name}' → '{matched_name}' (score={score:.0f}, id={artist_id})")
            return artist_id

        return None

    def _store_alias(self, artist_id: int, alias: str):
        """Record an alternate spelling for an artist."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO mg_artist_aliases (artist_id, alias, created_at) VALUES (?, ?, ?)",
                (artist_id, alias, now)
            )
            self.conn.commit()
        except Exception:
            pass  # alias already exists, no-op

    async def get_or_create_artist_async(self, name: str, genre: str | None = None) -> tuple[int, str]:
        """
        Async version — does fuzzy match, then MusicBrainz if needed.
        Returns (artist_id, canonical_name).
        """
        # 1. Check alias table first (exact match on previously seen spellings)
        alias_row = self.conn.execute(
            "SELECT artist_id FROM mg_artist_aliases WHERE alias = ?", (name,)
        ).fetchone()
        if alias_row:
            artist_id = alias_row["artist_id"]
            canonical = self.conn.execute(
                "SELECT name FROM mg_artists WHERE id = ?", (artist_id,)
            ).fetchone()["name"]
            log.info(f"Alias match: '{name}' → '{canonical}'")
            return artist_id, canonical

        # 2. Exact match (case-insensitive)
        row = self.conn.execute(
            "SELECT id, name FROM mg_artists WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return row["id"], row["name"]

        # 3. Fuzzy match against existing artists
        fuzzy_id = self._fuzzy_match(name)
        if fuzzy_id is not None:
            canonical = self.conn.execute(
                "SELECT name FROM mg_artists WHERE id = ?", (fuzzy_id,)
            ).fetchone()["name"]
            # Record this spelling as an alias
            if name.lower() != canonical.lower():
                self._store_alias(fuzzy_id, name)
            return fuzzy_id, canonical

        # 4. New artist — Last.fm correction for canonical name
        canonical = await _lastfm_correct(name)
        mbid = None  # mbid column retained in schema but not populated

        # After MB lookup, check again in case canonical differs from input
        if canonical.lower() != name.lower():
            row = self.conn.execute(
                "SELECT id, name FROM mg_artists WHERE name = ?", (canonical,)
            ).fetchone()
            if row:
                # MB gave us a canonical name that already exists — alias the input
                self._store_alias(row["id"], name)
                return row["id"], row["name"]

            # Also fuzzy-check the canonical name
            fuzzy_id = self._fuzzy_match(canonical)
            if fuzzy_id is not None:
                canonical_stored = self.conn.execute(
                    "SELECT name FROM mg_artists WHERE id = ?", (fuzzy_id,)
                ).fetchone()["name"]
                self._store_alias(fuzzy_id, name)
                if canonical.lower() != canonical_stored.lower():
                    self._store_alias(fuzzy_id, canonical)
                return fuzzy_id, canonical_stored

        # 5. Genuinely new — insert
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO mg_artists (name, genre, mbid, created_at) VALUES (?, ?, ?, ?)",
            (canonical, genre, mbid, now)
        )
        self.conn.commit()
        artist_id = cursor.lastrowid

        # If input differed from canonical, store input as alias
        if name.lower() != canonical.lower():
            self._store_alias(artist_id, name)

        return artist_id, canonical

    def get_or_create_artist(self, name: str, genre: str | None = None) -> int:
        """
        Sync version — fuzzy match only, no MusicBrainz.
        Used where async isn't available (album/track lookups).
        Returns artist_id.
        """
        # Alias check
        alias_row = self.conn.execute(
            "SELECT artist_id FROM mg_artist_aliases WHERE alias = ?", (name,)
        ).fetchone()
        if alias_row:
            return alias_row["artist_id"]

        # Exact match
        row = self.conn.execute(
            "SELECT id FROM mg_artists WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return row["id"]

        # Fuzzy match
        fuzzy_id = self._fuzzy_match(name)
        if fuzzy_id is not None:
            canonical = self.conn.execute(
                "SELECT name FROM mg_artists WHERE id = ?", (fuzzy_id,)
            ).fetchone()["name"]
            if name.lower() != canonical.lower():
                self._store_alias(fuzzy_id, name)
            return fuzzy_id

        # Insert as-is (no MB in sync path)
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO mg_artists (name, genre, mbid, created_at) VALUES (?, ?, ?, ?)",
            (name, genre, None, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_artist(self, name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM mg_artists WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    # ── Albums ────────────────────────────────────────────────────────

    def get_or_create_album(self, title: str, artist_name: str,
                            year: int | None = None) -> int:
        artist_id = self.get_or_create_artist(artist_name)
        row = self.conn.execute(
            "SELECT id FROM mg_albums WHERE artist_id = ? AND title = ?",
            (artist_id, title)
        ).fetchone()
        if row:
            return row["id"]
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO mg_albums (artist_id, title, year, created_at) VALUES (?, ?, ?, ?)",
            (artist_id, title, year, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_album(self, title: str, artist_name: str) -> dict | None:
        artist = self.get_artist(artist_name)
        if not artist:
            return None
        row = self.conn.execute(
            """SELECT al.*, a.name as artist_name FROM mg_albums al
               JOIN mg_artists a ON a.id = al.artist_id
               WHERE al.artist_id = ? AND al.title = ?""",
            (artist["id"], title)
        ).fetchone()
        return dict(row) if row else None

    # ── Tracks ────────────────────────────────────────────────────────

    def get_or_create_track(self, title: str, artist_name: str,
                            album_title: str | None = None) -> int:
        artist_id = self.get_or_create_artist(artist_name)
        album_id = None
        if album_title:
            album_id = self.get_or_create_album(album_title, artist_name)

        row = self.conn.execute(
            "SELECT id FROM mg_tracks WHERE artist_id = ? AND title = ?",
            (artist_id, title)
        ).fetchone()
        if row:
            if album_id:
                self.conn.execute(
                    "UPDATE mg_tracks SET album_id = ? WHERE id = ?",
                    (album_id, row["id"])
                )
                self.conn.commit()
            return row["id"]

        now = datetime.now(timezone.utc).isoformat()
        cursor = self.conn.execute(
            "INSERT INTO mg_tracks (artist_id, album_id, title, created_at) VALUES (?, ?, ?, ?)",
            (artist_id, album_id, title, now)
        )
        self.conn.commit()
        return cursor.lastrowid

    # ── Ratings ───────────────────────────────────────────────────────

    async def rate_artist(self, person: str, artist_name: str,
                          rating: int, source: str = "explicit") -> dict:
        artist_id, canonical = await self.get_or_create_artist_async(artist_name)
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO mg_artist_ratings (person, artist_id, rating, source, noted_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(person, artist_id) DO UPDATE SET
                   rating = excluded.rating,
                   source = excluded.source,
                   noted_at = excluded.noted_at""",
            (person, artist_id, rating, source, now)
        )
        self.conn.commit()
        return {"person": person, "type": "artist", "artist": canonical, "rating": rating, "source": source}

    async def rate_album(self, person: str, album_title: str, artist_name: str,
                         rating: int, source: str = "explicit",
                         year: int | None = None) -> dict:
        artist_id, canonical = await self.get_or_create_artist_async(artist_name)
        album_id = self.get_or_create_album(album_title, canonical, year)
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO mg_album_ratings (person, album_id, rating, source, noted_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(person, album_id) DO UPDATE SET
                   rating = excluded.rating,
                   source = excluded.source,
                   noted_at = excluded.noted_at""",
            (person, album_id, rating, source, now)
        )
        self.conn.commit()
        return {"person": person, "type": "album", "album": album_title, "artist": canonical, "rating": rating, "source": source}

    async def rate_track(self, person: str, title: str, artist_name: str,
                         rating: int, source: str = "explicit",
                         album_title: str | None = None) -> dict:
        artist_id, canonical = await self.get_or_create_artist_async(artist_name)
        track_id = self.get_or_create_track(title, canonical, album_title)
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO mg_track_ratings (person, track_id, rating, source, noted_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(person, track_id) DO UPDATE SET
                   rating = excluded.rating,
                   source = excluded.source,
                   noted_at = excluded.noted_at""",
            (person, track_id, rating, source, now)
        )
        self.conn.commit()
        return {"person": person, "type": "track", "track": title, "artist": canonical, "rating": rating, "source": source}

    # ── Discoveries ───────────────────────────────────────────────────

    def record_discovery(self, person: str, artist_name: str,
                         discovered_via: str | None = None):
        artist_id = self.get_or_create_artist(artist_name)
        existing = self.conn.execute(
            "SELECT id FROM mg_discoveries WHERE person = ? AND artist_id = ?",
            (person, artist_id)
        ).fetchone()
        if existing:
            return
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT INTO mg_discoveries (person, artist_id, discovered_via, discovered_at) VALUES (?, ?, ?, ?)",
            (person, artist_id, discovered_via, now)
        )
        self.conn.commit()

    # ── Artist links ──────────────────────────────────────────────────

    def link_artists(self, artist_name: str, related_name: str,
                     relationship: str = "similar"):
        a1 = self.get_or_create_artist(artist_name)
        a2 = self.get_or_create_artist(related_name)
        self.conn.execute(
            "INSERT OR IGNORE INTO mg_artist_links (artist_id, related_id, relationship) VALUES (?, ?, ?)",
            (a1, a2, relationship)
        )
        self.conn.commit()

    # ── Queries ───────────────────────────────────────────────────────

    def get_liked_artists(self, person: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT a.name, a.genre, a.mbid, r.source, r.noted_at
               FROM mg_artist_ratings r
               JOIN mg_artists a ON a.id = r.artist_id
               WHERE r.person = ? AND r.rating = 1
               ORDER BY r.noted_at DESC""",
            (person,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_liked_albums(self, person: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT al.title, a.name as artist, al.year, r.source, r.noted_at
               FROM mg_album_ratings r
               JOIN mg_albums al ON al.id = r.album_id
               JOIN mg_artists a ON a.id = al.artist_id
               WHERE r.person = ? AND r.rating = 1
               ORDER BY r.noted_at DESC""",
            (person,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_liked_tracks(self, person: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT t.title, a.name as artist, al.title as album, r.source, r.noted_at
               FROM mg_track_ratings r
               JOIN mg_tracks t ON t.id = r.track_id
               JOIN mg_artists a ON a.id = t.artist_id
               LEFT JOIN mg_albums al ON al.id = t.album_id
               WHERE r.person = ? AND r.rating = 1
               ORDER BY r.noted_at DESC""",
            (person,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_disliked_tracks(self, person: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT t.title, a.name as artist, al.title as album, r.source, r.noted_at
               FROM mg_track_ratings r
               JOIN mg_tracks t ON t.id = r.track_id
               JOIN mg_artists a ON a.id = t.artist_id
               LEFT JOIN mg_albums al ON al.id = t.album_id
               WHERE r.person = ? AND r.rating = -1
               ORDER BY r.noted_at DESC""",
            (person,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_person_summary(self, person: str) -> str:
        liked_artists = self.get_liked_artists(person)
        liked_albums = self.get_liked_albums(person)
        liked_tracks = self.get_liked_tracks(person)
        disliked_tracks = self.get_disliked_tracks(person)

        if not liked_artists and not liked_albums and not liked_tracks:
            return f"[No music preferences recorded for {person} yet]"

        lines = [f"== {person}'s music taste =="]

        if liked_artists:
            names = ", ".join(a["name"] for a in liked_artists[:10])
            lines.append(f"Liked artists: {names}")

        if liked_albums:
            albums = ", ".join(
                f"{a['title']} ({a['artist']})" for a in liked_albums[:10]
            )
            lines.append(f"Liked albums: {albums}")

        if liked_tracks:
            tracks = ", ".join(
                f"{t['title']} ({t['artist']})" for t in liked_tracks[:10]
            )
            lines.append(f"Liked tracks: {tracks}")

        if disliked_tracks:
            tracks = ", ".join(
                f"{t['title']} ({t['artist']})" for t in disliked_tracks[:10]
            )
            lines.append(f"Disliked tracks: {tracks}")

        return "\n".join(lines)

    def format_rating_confirmation(self, result: dict) -> str:
        symbol = "👍" if result["rating"] == 1 else "👎" if result["rating"] == -1 else "↩️"
        t = result["type"]
        if t == "track":
            return f"{symbol} Noted — {result['track']} by {result['artist']} for {result['person']}."
        elif t == "album":
            return f"{symbol} Noted — {result['album']} by {result['artist']} for {result['person']}."
        else:
            return f"{symbol} Noted — {result['artist']} for {result['person']}."