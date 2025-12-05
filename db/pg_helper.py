# db/pg_helper.py
import psycopg
from typing import Iterable, Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
import difflib
from psycopg.rows import dict_row

DDL = """
CREATE TABLE IF NOT EXISTS gifs (
    id          SERIAL PRIMARY KEY,
    title       TEXT NOT NULL,
    url         TEXT NOT NULL UNIQUE,
    nsfw        BOOLEAN NOT NULL,
    anime       TEXT,
    created_by  INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Nachrüsten falls gifs bereits existiert
ALTER TABLE gifs
    ADD COLUMN IF NOT EXISTS created_by INTEGER;

-- Falls deine PG-Version "CREATE INDEX IF NOT EXISTS" nicht kennt,
-- werden wir die Indizes weiter unten per try/except anlegen.
CREATE INDEX IF NOT EXISTS idx_gifs_title       ON gifs(title);
CREATE INDEX IF NOT EXISTS idx_gifs_anime       ON gifs(anime);
CREATE INDEX IF NOT EXISTS idx_gifs_nsfw        ON gifs(nsfw);
CREATE INDEX IF NOT EXISTS idx_gifs_created_by  ON gifs(created_by);
CREATE INDEX IF NOT EXISTS idx_gifs_created_at  ON gifs(created_at);

CREATE TABLE IF NOT EXISTS characters (
    id    SERIAL PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS tags (
    id    SERIAL PRIMARY KEY,
    name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS gif_characters (
    gif_id        INTEGER NOT NULL REFERENCES gifs(id) ON DELETE CASCADE,
    character_id  INTEGER NOT NULL REFERENCES characters(id) ON DELETE RESTRICT,
    PRIMARY KEY (gif_id, character_id)
);
CREATE TABLE IF NOT EXISTS gif_tags (
    gif_id  INTEGER NOT NULL REFERENCES gifs(id) ON DELETE CASCADE,
    tag_id  INTEGER NOT NULL REFERENCES tags(id) ON DELETE RESTRICT,
    PRIMARY KEY (gif_id, tag_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at  TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS counter (
    id    integer PRIMARY KEY CHECK (id = 1),
    total integer NOT NULL
);
INSERT INTO counter (id, total) VALUES (1, 0)
ON CONFLICT (id) DO NOTHING;

CREATE TABLE IF NOT EXISTS users (
    id                SERIAL PRIMARY KEY,
    username          TEXT NOT NULL,
    password          TEXT NOT NULL,
    linktree_id       INTEGER,
    profile_picture   TEXT,
    Admin             BOOLEAN NOT NULL DEFAULT FALSE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS linktrees (
    id                  SERIAL PRIMARY KEY,
    user_id             INTEGER NOT NULL
                        REFERENCES users(id) ON DELETE CASCADE,
    device_type         TEXT NOT NULL DEFAULT 'pc'
                        CHECK (device_type IN ('pc','mobile')),
    slug                TEXT NOT NULL,            -- wird zu /tree/{slug}
    location            TEXT,
    quote               TEXT,
    song_url            TEXT,                     -- Audio-URL
    background_url      TEXT,                     -- Bild/GIF/Video
    background_is_video BOOLEAN NOT NULL DEFAULT FALSE,
    transparency        SMALLINT NOT NULL DEFAULT 0
                        CHECK (transparency BETWEEN 0 AND 100),
    name_effect         TEXT NOT NULL DEFAULT 'none'
                        CHECK (name_effect IN ('none','glow','neon','rainbow')),
    background_effect   TEXT NOT NULL DEFAULT 'none'
                        CHECK (background_effect IN ('none','night','rain','snow')),
    display_name_mode   TEXT NOT NULL DEFAULT 'slug'
                        CHECK (display_name_mode IN ('slug','username','custom')),
    custom_display_name TEXT,
    link_color          TEXT,
    link_bg_color       TEXT,
    link_bg_alpha       SMALLINT NOT NULL DEFAULT 100
                        CHECK (link_bg_alpha BETWEEN 0 AND 100),
    text_color          TEXT,
    muted_color         TEXT,
    name_color          TEXT,
    location_color      TEXT,
    quote_color         TEXT,
    cursor_url          TEXT,
    discord_frame_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ
);

-- Einzelne Links im Linktree mit optional eigenem Icon (Upload)
CREATE TABLE IF NOT EXISTS linktree_links (
    id           SERIAL PRIMARY KEY,
    linktree_id  INTEGER NOT NULL REFERENCES linktrees(id) ON DELETE CASCADE,
    url          TEXT NOT NULL,
    label        TEXT,                 -- Anzeige-Text / Name des Links
    icon_url     TEXT,                 -- optional: benutzerhochgeladenes Icon
    position     INTEGER NOT NULL DEFAULT 0,
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_ltree_links_tree_pos
    ON linktree_links(linktree_id, position);

-- Verdienbare / freischaltbare Icons (Katalog)
CREATE TABLE IF NOT EXISTS icons (
    id          SERIAL PRIMARY KEY,
    code        TEXT NOT NULL UNIQUE,    -- z.B. 'veteran', 'founder'
    image_url   TEXT NOT NULL,
    description TEXT
);

-- Welche Icons ein User hat und ob sie angezeigt werden
CREATE TABLE IF NOT EXISTS user_icons (
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    icon_id     INTEGER NOT NULL REFERENCES icons(id) ON DELETE CASCADE,
    displayed   BOOLEAN NOT NULL DEFAULT FALSE,
    acquired_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, icon_id)
);

CREATE TABLE IF NOT EXISTS gif_blacklist (
    user_id     INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    reason      TEXT
);

CREATE TABLE IF NOT EXISTS health_data (
    id              SERIAL PRIMARY KEY,
    day             DATE NOT NULL,
    borg            INTEGER NOT NULL,
    erschöpfung     INTEGER NOT NULL DEFAULT 0,
    muskelschwäche  INTEGER NOT NULL DEFAULT 0,
    schmerzen       INTEGER NOT NULL DEFAULT 0,
    angst           INTEGER NOT NULL DEFAULT 0,
    konzentration   INTEGER NOT NULL DEFAULT 0,
    husten          INTEGER NOT NULL DEFAULT 0,
    atemnot         INTEGER NOT NULL DEFAULT 0,
    temperatur      REAL NULL,
    mens            BOOLEAN NOT NULL DEFAULT FALSE,
    notizen         TEXT
);  --  <<<<<<<<<<<<<<  Semikolon!

CREATE TABLE IF NOT EXISTS discord_accounts (
    id                  SERIAL PRIMARY KEY,
    user_id             INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    discord_user_id     VARCHAR(32) NOT NULL, -- Snowflake als String
    discord_username    VARCHAR(100),         -- historisch name#0000
    discord_global_name VARCHAR(100),         -- neuer "Global Name"
    avatar_hash         VARCHAR(128),
    avatar_decoration   JSONB,                -- Rohdaten, falls du sie speichern willst
    access_token        TEXT NOT NULL,
    refresh_token       TEXT NOT NULL,
    token_expires_at    TIMESTAMP NOT NULL,
    scopes              TEXT NOT NULL,        -- z.B. "identify"
    linked_at           TIMESTAMP NOT NULL DEFAULT NOW(),

    UNIQUE (user_id),
    UNIQUE (discord_user_id)
);

-- Saubere FK für users.linktree_id -> linktrees.id
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'fk_users_linktree_id'
    ) THEN
        ALTER TABLE users
        ADD CONSTRAINT fk_users_linktree_id
        FOREIGN KEY (linktree_id) REFERENCES linktrees(id)
        ON DELETE SET NULL;
    END IF;
END
$$ LANGUAGE plpgsql
"""


class PgGifDB:
    def __init__(self, dsn: str):
        self.dsn = dsn
        # Schema beim Start sicherstellen
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                # 1) Grundschema (inkl. GIF-Indizes via IF NOT EXISTS)
                cur.execute(DDL)
                cur.execute("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name='health_data'
                        AND column_name='temperatur'
                        AND data_type NOT IN ('real','double precision')
                    ) THEN
                        ALTER TABLE health_data
                        ALTER COLUMN temperatur TYPE REAL USING temperatur::REAL;
                    END IF;
                END$$;
            """)
                cur.execute("""
                            ALTER TABLE health_data
                            ALTER COLUMN temperatur DROP NOT NULL;
                            """)
                # (Optional, aber empfehlenswert)
                # Ein Eintrag pro Tag:
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_indexes
                            WHERE tablename='health_data'
                            AND indexname='ux_health_data_day'
                        ) THEN
                            -- UNIQUE geht nur, wenn keine Duplikate existieren
                            BEGIN
                                ALTER TABLE health_data
                                ADD CONSTRAINT ux_health_data_day UNIQUE(day);
                            EXCEPTION WHEN duplicate_table THEN
                                -- ignorieren, falls schon vorhanden
                            END;
                        END IF;
                    END$$;
                """)

                # (Optional) Index für schnelle by-day Abfragen
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_health_data_day
                    ON health_data(day);
                """)

                cur.execute("""ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS display_name_mode TEXT NOT NULL DEFAULT 'slug'
    CHECK (display_name_mode IN ('slug','username','custom'));""")
                cur.execute("""
  ALTER TABLE linktrees
  DROP CONSTRAINT IF EXISTS linktrees_display_name_mode_check;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD CONSTRAINT linktrees_display_name_mode_check
  CHECK (display_name_mode IN ('slug','username','custom'));
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS device_type TEXT NOT NULL DEFAULT 'pc';
                """)
                cur.execute("""
  ALTER TABLE linktrees
  DROP CONSTRAINT IF EXISTS linktrees_device_type_check;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD CONSTRAINT linktrees_device_type_check
  CHECK (device_type IN ('pc','mobile'));
                """)
                cur.execute("""
    ALTER TABLE health_data
    ADD COLUMN IF NOT EXISTS other TEXT;                     
""")
                cur.execute("""
  DO $$
  BEGIN
    IF EXISTS (
      SELECT 1 FROM information_schema.table_constraints
      WHERE table_name='linktrees' AND constraint_name='linktrees_slug_key'
    ) THEN
      ALTER TABLE linktrees DROP CONSTRAINT linktrees_slug_key;
    END IF;
  END$$;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS custom_display_name TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS link_color TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS link_bg_color TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS link_bg_alpha SMALLINT;
                """)
                cur.execute("""
  UPDATE linktrees SET link_bg_alpha = 100 WHERE link_bg_alpha IS NULL;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ALTER COLUMN link_bg_alpha SET DEFAULT 100;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ALTER COLUMN link_bg_alpha SET NOT NULL;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS discord_frame_enabled BOOLEAN NOT NULL DEFAULT FALSE;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS text_color TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS muted_color TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS name_color TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS location_color TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS quote_color TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD COLUMN IF NOT EXISTS cursor_url TEXT;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  DROP CONSTRAINT IF EXISTS chk_link_bg_alpha_range;
                """)
                cur.execute("""
  ALTER TABLE linktrees
  ADD CONSTRAINT chk_link_bg_alpha_range
  CHECK (link_bg_alpha BETWEEN 0 AND 100);
                """)
                cur.execute("""
  DO $$
  BEGIN
    IF EXISTS (
      SELECT 1 FROM information_schema.table_constraints
      WHERE table_name='linktrees' AND constraint_type='UNIQUE' AND constraint_name='linktrees_user_id_key'
    ) THEN
      ALTER TABLE linktrees DROP CONSTRAINT linktrees_user_id_key;
    END IF;
  END$$;
                """)
                cur.execute("""
  CREATE UNIQUE INDEX IF NOT EXISTS ux_linktrees_user_device
  ON linktrees(user_id, device_type);
                """)
                cur.execute("""
  DO $$
  BEGIN
    IF EXISTS (
      SELECT 1 FROM pg_indexes WHERE tablename='linktrees' AND indexname='ux_linktrees_slug'
    ) THEN
      DROP INDEX ux_linktrees_slug;
    END IF;
  END$$;
                """)
                cur.execute("""
  CREATE UNIQUE INDEX IF NOT EXISTS ux_linktrees_slug_device
  ON linktrees (lower(slug), device_type);
                """)
                # 2) Nachrüst-Änderungen idempotent
                cur.execute(
                    """
                    ALTER TABLE sessions
                    ADD COLUMN IF NOT EXISTS user_id INTEGER
                    REFERENCES users(id) ON DELETE CASCADE
                """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON sessions(user_id)
                """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS ux_users_username_ci
                    ON users (lower(username))
                """
                )
                cur.execute(
                    """
                    CREATE UNIQUE INDEX IF NOT EXISTS ux_linktrees_user_device
                    ON linktrees (user_id, device_type);
                """
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.table_constraints
                            WHERE table_name='linktrees' AND constraint_name='linktrees_slug_key'
                        ) THEN
                            ALTER TABLE linktrees DROP CONSTRAINT linktrees_slug_key;
                        END IF;
                    END$$;
                    """
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM pg_indexes
                            WHERE tablename='linktrees'
                            AND indexname='ux_linktrees_slug_device'
                        ) THEN
                            DROP INDEX ux_linktrees_slug_device;
                        END IF;
                    END$$;
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_linktrees_slug_device
                    ON linktrees (lower(slug), device_type);
                """
                )
                cur.execute(
                    """
                    ALTER TABLE gifs
                    ADD COLUMN IF NOT EXISTS created_by INTEGER
                """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_gifs_created_by
                    ON gifs(created_by)
                """
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.table_constraints
                            WHERE table_name='gifs' AND constraint_name='fk_gifs_created_by'
                        ) THEN
                            ALTER TABLE gifs
                            ADD CONSTRAINT fk_gifs_created_by
                            FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL;
                        END IF;
                    END$$;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gif_blacklist (
                        user_id    INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        reason     TEXT
                    )
                """
                )
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM pg_indexes
                            WHERE tablename='linktrees'
                            AND indexname='ux_linktrees_slug_user'
                        ) THEN
                            DROP INDEX ux_linktrees_slug_user;
                        END IF;
                    END$$;
                """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_linktrees_slug_user
                    ON linktrees (lower(slug), user_id);
                """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_user_icons_user ON user_icons(user_id);
                """
                )

            conn.commit()

    # ---------- intern ----------
    def _get_or_create(self, cur, table: str, name: str) -> int:
        name = (name or "").strip()
        if not name:
            raise ValueError("empty name")
        cur.execute(f"SELECT id FROM {table} WHERE name = %s", (name,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            f"INSERT INTO {table}(name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
            (name,),
        )
        r = cur.fetchone()
        if r:
            return r[0]
        # Race: nochmal lesen
        cur.execute(f"SELECT id FROM {table} WHERE name = %s", (name,))
        return cur.fetchone()[0]

    def _replace_links(
        self,
        cur,
        link_table: str,
        main_id_col: str,
        ref_id_col: str,
        gif_id: int,
        names: Iterable[str],
        ref_table: str,
    ) -> None:
        cur.execute(f"DELETE FROM {link_table} WHERE {main_id_col} = %s", (gif_id,))
        for n in names or []:
            if not n or not str(n).strip():
                continue
            rid = self._get_or_create(cur, ref_table, str(n))
            cur.execute(
                f"INSERT INTO {link_table}({main_id_col},{ref_id_col}) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (gif_id, rid),
            )

    def _compose_gif(self, cur, row) -> Dict[str, Any]:
        gif_id = row[0]
        title = row[1]
        url = row[2]
        nsfw = row[3]
        anime = row[4]
        created_at = row[5]
        created_by = row[6] if len(row) > 6 else None
        cur.execute(
            """
            SELECT c.name FROM characters c
            JOIN gif_characters gc ON gc.character_id = c.id
            WHERE gc.gif_id = %s ORDER BY c.name
        """,
            (gif_id,),
        )
        chars = [r[0] for r in cur.fetchall()]

        cur.execute(
            """
            SELECT t.name FROM tags t
            JOIN gif_tags gt ON gt.tag_id = t.id
            WHERE gt.gif_id = %s ORDER BY t.name
        """,
            (gif_id,),
        )
        tags = [r[0] for r in cur.fetchall()]

        return {
            "id": gif_id,
            "title": title,
            "url": url,
            "nsfw": bool(nsfw),
            "anime": anime,
            "created_at": created_at.isoformat(),
            "created_by": created_by,
            "characters": chars,
            "tags": tags,
        }

    # ---------- öffentliche API (gleich wie bei SQLite) ----------
    def insert_gif(
        self,
        *,
        title: str,
        url: str,
        nsfw: bool,
        anime: Optional[str] = None,
        characters: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
        created_at: Optional[str] = None,
        created_by: Optional[int] = None,
    ) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            fields = ["title", "url", "nsfw", "anime", "created_by"]
            vals = [title, url, nsfw, anime, created_by]
            if created_at is not None:
                fields.append("created_at")
                vals.append(created_at)  # ISO-8601; Postgres parst TIMESTAMPTZ

            ph = ",".join(["%s"] * len(fields))
            cur.execute(
                f"INSERT INTO gifs({','.join(fields)}) VALUES ({ph}) RETURNING id",
                tuple(vals),
            )
            gif_id = cur.fetchone()[0]

            self._replace_links(
                cur,
                "gif_characters",
                "gif_id",
                "character_id",
                gif_id,
                characters,
                "characters",
            )
            self._replace_links(
                cur, "gif_tags", "gif_id", "tag_id", gif_id, tags, "tags"
            )
            conn.commit()
            return gif_id

    def update_gif(
        self,
        gif_id: int,
        *,
        title=None,
        url=None,
        nsfw=None,
        anime=None,
        characters: Optional[Iterable[str]] = None,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        sets, vals = [], []
        if title is not None:
            sets.append("title=%s")
            vals.append(title)
        if url is not None:
            sets.append("url=%s")
            vals.append(url)
        if nsfw is not None:
            sets.append("nsfw=%s")
            vals.append(bool(nsfw))
        if anime is not None:
            sets.append("anime=%s")
            vals.append(anime)

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            if sets:
                vals.append(gif_id)
                cur.execute(
                    f"UPDATE gifs SET {', '.join(sets)} WHERE id=%s", tuple(vals)
                )
            if characters is not None:
                self._replace_links(
                    cur,
                    "gif_characters",
                    "gif_id",
                    "character_id",
                    gif_id,
                    characters,
                    "characters",
                )
            if tags is not None:
                self._replace_links(
                    cur, "gif_tags", "gif_id", "tag_id", gif_id, tags, "tags"
                )
            conn.commit()

    def delete_gif(self, gif_id: int) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM gifs WHERE id=%s", (gif_id,))
            conn.commit()

    def get_gif(self, gif_id: int) -> Dict[str, Any]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,title,url,nsfw,anime,created_at,created_by FROM gifs WHERE id=%s",
                (gif_id,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"gif id {gif_id} not found")
            return self._compose_gif(cur, row)

    def get_gif_by_url(self, gif_url: str) -> Dict[str, Any]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,title,url,nsfw,anime,created_at,created_by FROM gifs WHERE url=%s",
                (gif_url,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"gif url {gif_url} not found")
            return self._compose_gif(cur, row)

    def search_by_title(
        self, query: str, nsfw_mode: str = "false", limit: int = 50, offset: int = 0
    ):
        where = []
        params: List[Any] = []
        if query:
            where.append("title ILIKE %s")
            params.append(f"%{query}%")
        m = (nsfw_mode or "false").lower()
        if m == "only":
            where.append("nsfw = TRUE")
        elif m == "false":
            where.append("nsfw = FALSE")

        sql = "SELECT id,title,url,nsfw,anime,created_at,created_by FROM gifs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            return [self._compose_gif(cur, r) for r in rows]

    def search_user_gifs(
        self,
        user_id: int,
        query: str = "",
        nsfw_mode: str = "false",
        limit: int = 50,
        offset: int = 0,
    ):
        where = ["created_by = %s"]
        params: List[Any] = [user_id]
        if query:
            where.append("title ILIKE %s")
            params.append(f"%{query}%")
        m = (nsfw_mode or "false").lower()
        if m == "only":
            where.append("nsfw = TRUE")
        elif m == "false":
            where.append("nsfw = FALSE")

        sql = "SELECT id,title,url,nsfw,anime,created_at,created_by FROM gifs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            return [self._compose_gif(cur, r) for r in rows]

    def get_random(self, nsfw_mode: str = "false"):
        cond = {"only": "nsfw=TRUE", "false": "nsfw=FALSE"}.get(
            (nsfw_mode or "false").lower(), "TRUE"
        )
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id,title,url,nsfw,anime,created_at,created_by FROM gifs WHERE {cond} ORDER BY random() LIMIT 1"
            )
            row = cur.fetchone()
            if not row:
                raise KeyError("no gifs found")
            return self._compose_gif(cur, row)

    def get_random_by_tag(self, tag: str, nsfw_mode: str = "false"):
        cond = {"only": "g.nsfw=TRUE", "false": "g.nsfw=FALSE"}.get(
            (nsfw_mode or "false").lower(), "TRUE"
        )
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT g.id,g.title,g.url,g.nsfw,g.anime,g.created_at,g.created_by
                  FROM gifs g
                  JOIN gif_tags gt ON gt.gif_id=g.id
                  JOIN tags t ON t.id=gt.tag_id
                 WHERE t.name = %s AND {cond}
              ORDER BY random() LIMIT 1
            """,
                (tag,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"no gifs found for tag '{tag}'")
            return self._compose_gif(cur, row)

    def get_random_by_anime(self, anime: str, nsfw_mode: str = "false"):
        cond = {"only": "nsfw=TRUE", "false": "nsfw=FALSE"}.get(
            (nsfw_mode or "false").lower(), "TRUE"
        )
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id,title,url,nsfw,anime,created_at,created_by
                  FROM gifs
                 WHERE anime ILIKE %s AND {cond}
              ORDER BY random() LIMIT 1
            """,
                (anime,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"no gifs found for anime '{anime}'")
            return self._compose_gif(cur, row)

    def get_random_by_character(self, character: str, nsfw_mode: str = "false"):
        cond = {"only": "g.nsfw=TRUE", "false": "g.nsfw=FALSE"}.get(
            (nsfw_mode or "false").lower(), "TRUE"
        )
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT g.id,g.title,g.url,g.nsfw,g.anime,g.created_at,g.created_by
                  FROM gifs g
                  JOIN gif_characters gc ON gc.gif_id=g.id
                  JOIN characters c ON c.id=gc.character_id
                 WHERE c.name ILIKE %s AND {cond}
              ORDER BY random() LIMIT 1
            """,
                (character,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"no gifs found for character '{character}'")
            return self._compose_gif(cur, row)

    def get_all_tags(self, nsfw_mode: str = "false"):
        cond = {"only": "g.nsfw=TRUE", "false": "g.nsfw=FALSE"}.get(
            (nsfw_mode or "false").lower(), "TRUE"
        )
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT t.name
                  FROM tags t
                  JOIN gif_tags gt ON gt.tag_id = t.id
                  JOIN gifs g ON g.id = gt.gif_id
                 WHERE {cond}
              ORDER BY t.name
            """
            )
            return [r[0] for r in cur.fetchall()]

    def count_user_gifs(self, user_id: int) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM gifs WHERE created_by=%s", (user_id,))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def is_user_blacklisted(self, user_id: int) -> bool:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM gif_blacklist WHERE user_id=%s LIMIT 1", (user_id,)
            )
            return cur.fetchone() is not None

    def add_to_gif_blacklist(self, user_id: int, reason: str | None = None) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO gif_blacklist(user_id, reason)
                VALUES (%s,%s)
                ON CONFLICT (user_id)
                DO UPDATE SET reason=EXCLUDED.reason, created_at=now()
                """,
                (user_id, reason),
            )
            conn.commit()

    def remove_from_gif_blacklist(self, user_id: int) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM gif_blacklist WHERE user_id=%s", (user_id,))
            conn.commit()

    def list_gif_blacklist(self) -> list[dict]:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute("SELECT user_id, created_at, reason FROM gif_blacklist ORDER BY created_at DESC")
            return cur.fetchall() or []

    # --- Sessions (Tokens) ---
    def create_token(self, hours_valid: int = 24, user_id: int | None = None) -> str:
        import secrets

        token = secrets.token_hex(32)
        expires = datetime.now(timezone.utc) + timedelta(hours=hours_valid)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions(token, expires_at, user_id) VALUES (%s,%s,%s)",
                (token, expires, user_id),
            )
            conn.commit()
        return token

    def get_token_user(self, token: str) -> dict | None:
        if not token:
            return None
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.*, l.slug AS linktree_slug
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                LEFT JOIN linktrees l ON l.id = u.linktree_id
                WHERE s.token = %s
                AND (s.expires_at IS NULL OR s.expires_at > now())
                """,
                (token,),
            )
            return cur.fetchone()

    def validate_admin_token(self, token: str) -> bool:
        u = self.get_token_user(token)
        return bool(u and u.get("admin"))

    def validate_token(self, token: str) -> bool:
        if not token:
            return False
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT expires_at FROM sessions WHERE token=%s", (token,))
            row = cur.fetchone()
            if not row:
                return False
            exp = row[0]  # TIMESTAMPTZ → datetime
            return (exp is None) or (datetime.now(timezone.utc) < exp)

    def revoke_token(self, token: str) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE token=%s", (token,))
            conn.commit()

    # ---- Suggest Helpers (identisch zu SQLite-Variante) ----
    def list_all_anime(self) -> list[str]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT anime FROM gifs WHERE anime IS NOT NULL AND TRIM(anime) <> '' ORDER BY anime"
            )
            return [r[0] for r in cur.fetchall()]

    def list_all_characters(self) -> list[str]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT c.name
                  FROM characters c
                  JOIN gif_characters gc ON gc.character_id = c.id
                 ORDER BY c.name
            """
            )
            return [r[0] for r in cur.fetchall()]

    def _suggest_from_list(
        self, items: list[str], query: str, limit: int = 5
    ) -> list[str]:
        q = (query or "").strip()
        if not q:
            return []
        qcf = q.casefold()
        subs = [name for name in items if qcf in name.casefold()]
        fuzzy = difflib.get_close_matches(q, items, n=limit * 2, cutoff=0.5)
        out, seen = [], set()
        for name in subs + fuzzy:
            key = name.casefold()
            if key not in seen:
                out.append(name)
                seen.add(key)
            if len(out) >= limit:
                break
        return out

    def suggest_anime(self, query: str, limit: int = 5) -> list[str]:
        return self._suggest_from_list(self.list_all_anime(), query, limit)

    def suggest_character(self, query: str, limit: int = 5) -> list[str]:
        return self._suggest_from_list(self.list_all_characters(), query, limit)

    def get_token_expiry(self, token: str) -> str | None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT expires_at FROM sessions WHERE token=%s", (token,))
            row = cur.fetchone()
            if not row:
                return None
            exp = row[0]
            return exp.isoformat() if exp is not None else None

    def incrementCounter(self):
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("UPDATE counter SET total = total + 1 WHERE id = 1;")
            conn.commit()

    def getCounterValue(self) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT total FROM counter WHERE id = 1;")
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def createUser(
        self,
        username: str,
        hashed_password: str,
        linktree_id: int | None = None,
        profile_picture: str | None = None,
        admin: bool | None = False,
    ) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO users (username, password, linktree_id, profile_picture, admin)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """,
                (username, hashed_password, linktree_id, profile_picture, admin),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            return new_id

    def updateUser(
        self,
        user_id: int,
        *,
        username: str | None = None,
        password: str | None = None,
        linktree_id: int | None = None,
        profile_picture: str | None = None,
        admin: bool | None = None,  # <- wichtig: None statt False
    ) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            if username is not None:
                cur.execute(
                    "UPDATE users SET username = %s WHERE id = %s", (username, user_id)
                )
            if password is not None:
                cur.execute(
                    "UPDATE users SET password = %s WHERE id = %s", (password, user_id)
                )
            if linktree_id is not None:
                cur.execute(
                    "UPDATE users SET linktree_id = %s WHERE id = %s",
                    (linktree_id, user_id),
                )
            if profile_picture is not None:
                cur.execute(
                    "UPDATE users SET profile_picture = %s WHERE id = %s",
                    (profile_picture, user_id),
                )
            if admin is not None:
                cur.execute(
                    "UPDATE users SET admin = %s WHERE id = %s", (admin, user_id)
                )
            cur.execute(
                "UPDATE users SET updated_at = %s WHERE id = %s",
                (datetime.now(timezone.utc), user_id),
            )
            conn.commit()

    def getUser(self, user_id: int) -> dict | None:
        with psycopg.connect(
            self.dsn, row_factory=dict_row
        ) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT u.*, l.slug AS linktree_slug
                FROM users u
                LEFT JOIN linktrees l ON u.linktree_id = l.id
                WHERE u.id = %s
            """,
                (user_id,),
            )
            return cur.fetchone()

    def getUserByUsername(self, username: str) -> dict | None:
        if not username:
            return None
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    u.id,
                    u.username,
                    u.password,
                    u.admin,
                    u.profile_picture,
                    u.linktree_id,
                    u.created_at,
                    u.updated_at,
                    l.slug AS linktree_slug
                FROM users AS u
                LEFT JOIN linktrees AS l ON l.id = u.linktree_id
                WHERE LOWER(u.username) = LOWER(%s)
                LIMIT 1
                """,
                (username,),
            )
            return cur.fetchone()

    # ---------------- Discord Accounts ----------------

    def upsert_discord_account(
        self,
        user_id: int,
        *,
        discord_user_id: str,
        discord_username: str | None = None,
        discord_global_name: str | None = None,
        avatar_hash: str | None = None,
        avatar_decoration: Any | None = None,
        access_token: str,
        refresh_token: str,
        token_expires_at,
        scopes: str,
    ) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO discord_accounts (
                    user_id, discord_user_id, discord_username, discord_global_name,
                    avatar_hash, avatar_decoration,
                    access_token, refresh_token, token_expires_at, scopes
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (user_id) DO UPDATE SET
                    discord_user_id = EXCLUDED.discord_user_id,
                    discord_username = EXCLUDED.discord_username,
                    discord_global_name = EXCLUDED.discord_global_name,
                    avatar_hash = EXCLUDED.avatar_hash,
                    avatar_decoration = EXCLUDED.avatar_decoration,
                    access_token = EXCLUDED.access_token,
                    refresh_token = EXCLUDED.refresh_token,
                    token_expires_at = EXCLUDED.token_expires_at,
                    scopes = EXCLUDED.scopes
                RETURNING id
                """,
                (
                    user_id,
                    discord_user_id,
                    discord_username,
                    discord_global_name,
                    avatar_hash,
                    avatar_decoration,
                    access_token,
                    refresh_token,
                    token_expires_at,
                    scopes,
                ),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            return new_id

    def get_discord_account(self, user_id: int) -> dict | None:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM discord_accounts WHERE user_id = %s LIMIT 1", (user_id,)
            )
            return cur.fetchone()

    def delete_discord_account(self, user_id: int) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM discord_accounts WHERE user_id = %s", (user_id,))
            conn.commit()

    # ---------------- Linktrees ----------------


    def create_linktree(
        self,
        user_id: int,
        slug: str,
        *,
        location: str | None = None,
        quote: str | None = None,
        song_url: str | None = None,
        background_url: str | None = None,
        background_is_video: bool = False,
        transparency: int = 0,
        name_effect: str = "none",
        background_effect: str = "none",
        display_name_mode: str = "slug",  # <-- NEU
        device_type: str = "pc",
        custom_display_name: str | None = None,
        link_color: str | None = None,
        link_bg_color: str | None = None,
        link_bg_alpha: int = 100,
        text_color: str | None = None,
        muted_color: str | None = None,
        name_color: str | None = None,
        location_color: str | None = None,
        quote_color: str | None = None,
        cursor_url: str | None = None,
        discord_frame_enabled: bool = False,
    ) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO linktrees (
                    user_id, slug, device_type, location, quote, song_url,
                    background_url, background_is_video,
                    transparency, name_effect, background_effect,
                    display_name_mode,          -- <-- NEU
                    custom_display_name,
                    link_color,
                    link_bg_color,
                    link_bg_alpha,
                    text_color,
                    muted_color,
                    name_color,
                    location_color,
                    quote_color,
                    cursor_url,
                    discord_frame_enabled
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (
                    user_id, slug, device_type, location, quote, song_url,
                    background_url, background_is_video,
                    transparency, name_effect, background_effect,
                    display_name_mode,
                    custom_display_name,
                    link_color,
                    link_bg_color,
                    link_bg_alpha,
                    text_color,
                    muted_color,
                    name_color,
                    location_color,
                    quote_color,
                    cursor_url,
                    discord_frame_enabled,
                ),
            )
            linktree_id = cur.fetchone()[0]
            if device_type == "pc":
                cur.execute(
                    "UPDATE users SET linktree_id = %s, updated_at = now() WHERE id = %s",
                    (linktree_id, user_id),
                )
            conn.commit()
            return linktree_id


    def update_linktree(self, linktree_id: int, **fields) -> None:
        allowed = {
            "slug",
            "device_type",
            "location",
            "quote",
            "song_url",
            "background_url",
            "background_is_video",
            "transparency",
            "name_effect",
            "background_effect",
            "display_name_mode",
            "custom_display_name",
            "link_color",
            "link_bg_color",
            "link_bg_alpha",
            "text_color",
            "muted_color",
            "name_color",
            "location_color",
            "quote_color",
            "cursor_url",
            "discord_frame_enabled",
        }
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=%s")
                vals.append(v)
        if not sets:
            return
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            vals.append(linktree_id)
            cur.execute(
                f"UPDATE linktrees SET {', '.join(sets)}, updated_at = now() WHERE id=%s",
                tuple(vals),
            )
            conn.commit()


    def clone_linktree_variant(self, source_slug: str, source_device: str, target_device: str, user_id: int) -> int:
        """Clone an existing linktree (including links) to another device variant."""
        if target_device not in {"pc", "mobile"}:
            raise ValueError("invalid target device")
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM linktrees WHERE lower(slug)=lower(%s) AND device_type=%s AND user_id=%s",
                (source_slug, source_device, user_id),
            )
            src = cur.fetchone()
            if not src:
                raise KeyError("source linktree not found")
            cur.execute(
                "SELECT id FROM linktrees WHERE lower(slug)=lower(%s) AND device_type=%s AND user_id=%s",
                (source_slug, target_device, user_id),
            )
            if cur.fetchone():
                raise ValueError("target variant already exists")
            cur.execute(
                """
                INSERT INTO linktrees (
                    user_id, slug, device_type, location, quote, song_url,
                    background_url, background_is_video,
                    transparency, name_effect, background_effect,
                    display_name_mode, custom_display_name,
                    link_color, link_bg_color, link_bg_alpha, text_color, muted_color,
                    name_color, location_color, quote_color, cursor_url, discord_frame_enabled
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING id
                """,
                (
                    src["user_id"],
                    src["slug"],
                    target_device,
                    src.get("location"),
                    src.get("quote"),
                    src.get("song_url"),
                    src.get("background_url"),
                    src.get("background_is_video"),
                    src.get("transparency"),
                    src.get("name_effect"),
                    src.get("background_effect"),
                    src.get("display_name_mode"),
                    src.get("custom_display_name"),
                    src.get("link_color"),
                    src.get("link_bg_color"),
                    src.get("link_bg_alpha", 100),
                    src.get("text_color"),
                    src.get("muted_color"),
                    src.get("name_color"),
                    src.get("location_color"),
                    src.get("quote_color"),
                    src.get("cursor_url"),
                    src.get("discord_frame_enabled", False),
                ),
            )
            new_id = cur.fetchone()["id"]
            cur.execute(
                "SELECT url, label, icon_url, position, is_active FROM linktree_links WHERE linktree_id=%s",
                (src["id"],),
            )
            for link in cur.fetchall() or []:
                cur.execute(
                    """
                    INSERT INTO linktree_links(linktree_id, url, label, icon_url, position, is_active)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        new_id,
                        link["url"],
                        link.get("label"),
                        link.get("icon_url"),
                        link.get("position", 0),
                        link.get("is_active", True),
                    ),
                )
            conn.commit()
            return new_id


    def get_linktree_by_slug(self, slug: str, device_type: str = "pc") -> dict | None:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM linktrees WHERE lower(slug)=lower(%s) AND device_type=%s", (slug, device_type))
            lt = cur.fetchone()
            if not lt:
                return None
            cur.execute(
                """
                SELECT id, url, label, icon_url, position, is_active
                FROM linktree_links
                WHERE linktree_id=%s
            ORDER BY position, id
            """,
                (lt["id"],),
            )
            links = cur.fetchall()
            # NEU: per icon_id joinen
            cur.execute("""
                SELECT i.id, i.code, i.image_url, i.description,
                    ui.displayed, ui.acquired_at
                FROM user_icons ui
                JOIN icons i ON i.id = ui.icon_id
                WHERE ui.user_id = %s
                ORDER BY i.code
            """, (lt["user_id"],))
            rows = cur.fetchall()

            def _to_iso(v):
                return v.isoformat() if hasattr(v, "isoformat") else v

            icons = [
                {
                    "id": i["id"],
                    "code": i["code"],
                    "image_url": i["image_url"],
                    "description": i.get("description"),
                    "displayed": i.get("displayed", False),
                    "acquired_at": _to_iso(i.get("acquired_at")),
                }
                for i in rows
            ]

            lt["links"] = links
            lt["icons"] = icons
            return lt


    # ---------------- Links ----------------


    def add_link(
        self,
        linktree_id: int,
        url: str,
        *,
        label: str | None = None,
        icon_url: str | None = None,
        position: int | None = None,
        is_active: bool = True,
    ) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            if position is None:
                cur.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM linktree_links WHERE linktree_id=%s",
                    (linktree_id,),
                )
                position = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO linktree_links(linktree_id, url, label, icon_url, position, is_active)
                VALUES (%s,%s,%s,%s,%s,%s)
                RETURNING id
            """,
                (linktree_id, url, label, icon_url, position, is_active),
            )
            new_id = cur.fetchone()[0]
            conn.commit()
            return new_id


    def update_link(self, link_id: int, **fields) -> None:
        allowed = {"url", "label", "icon_url", "position", "is_active"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=%s")
                vals.append(v)
        if not sets:
            return
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            vals.append(link_id)
            cur.execute(
                f"UPDATE linktree_links SET {', '.join(sets)}, updated_at=now() WHERE id=%s",
                tuple(vals),
            )
            conn.commit()


    def delete_link(self, link_id: int) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM linktree_links WHERE id=%s", (link_id,))
            conn.commit()


    # ---------------- Icons (Katalog + Besitz) ----------------


    def upsert_icon(self, code: str, image_url: str, description: str | None = None) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO icons(code, image_url, description)
                VALUES (%s,%s,%s)
                ON CONFLICT (code) DO UPDATE SET image_url=EXCLUDED.image_url,
                                                description=EXCLUDED.description
                RETURNING id
            """,
                (code, image_url, description),
            )
            icon_id = cur.fetchone()[0]
            conn.commit()
            return icon_id


    def grant_icon(self, user_id: int, icon_code: str, *, displayed: bool = False) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT id FROM icons WHERE code=%s", (icon_code,))
            row = cur.fetchone()
            if not row:
                raise KeyError(f"icon '{icon_code}' not found")
            icon_id = row[0]
            cur.execute(
                """
                INSERT INTO user_icons(user_id, icon_id, displayed)
                VALUES (%s,%s,%s)
                ON CONFLICT (user_id, icon_id) DO UPDATE SET displayed=EXCLUDED.displayed
            """,
                (user_id, icon_id, displayed),
            )
            conn.commit()

    def revoke_icon(self, user_id: int, icon_code: str) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM user_icons ui
                USING icons i
                WHERE ui.user_id=%s AND ui.icon_id=i.id AND i.code=%s
                """,
                (user_id, icon_code),
            )
            conn.commit()


    def set_icon_displayed(self, user_id: int, icon_code: str, displayed: bool) -> None:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE user_icons ui
                SET displayed=%s
                FROM icons i
                WHERE ui.user_id=%s AND ui.icon_id=i.id AND i.code=%s
            """,
                (displayed, user_id, icon_code),
            )
            conn.commit()

    def update_linktree_by_user(self, user_id: int, **fields):
        """Update the linktree of a given user with provided fields."""
        if not fields:
            return
        sets = [f"{k} = %s" for k in fields.keys()]
        values = list(fields.values())
        values.append(user_id)

        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE linktrees
                SET {", ".join(sets)}, updated_at = now()
                WHERE id = (
                    SELECT linktree_id FROM users WHERE id = %s
                )
                """,
                values,
            )
            conn.commit()

    def update_linktree_by_user_and_device(
        self, user_id: int, device_type: str, **fields
    ) -> bool:
        """Update the user's linktree for a specific device. Returns True if a row changed."""
        if not fields:
            return False
        sets = [f"{k} = %s" for k in fields.keys()]
        values = list(fields.values())
        values.extend([user_id, device_type])
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE linktrees
                   SET {", ".join(sets)}, updated_at = now()
                 WHERE user_id = %s AND device_type = %s
                """,
                values,
            )
            conn.commit()
            return cur.rowcount > 0



    # ---------------- Health Data ----------------

    def insert_health_data(
        self,
        *,
        day: str,
        borg: int,
        temperatur: Optional[float] = None,   # <-- jetzt optional
        erschöpfung: int = 0,
        muskelschwäche: int = 0,
        schmerzen: int = 0,
        angst: int = 0,
        konzentration: int = 0,
        husten: int = 0,
        atemnot: int = 0,
        mens: bool = False,
        notizen: str | None = None,
        other: str | None = None,
    ) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cols = [
                "day",
                "borg",
                "erschöpfung",
                "muskelschwäche",
                "schmerzen",
                "angst",
                "konzentration",
                "husten",
                "atemnot",
                # "temperatur" kommt ggf. dynamisch
                "mens",
                "notizen",
                "other",
            ]
            vals = [
                day,
                borg,
                erschöpfung,
                muskelschwäche,
                schmerzen,
                angst,
                konzentration,
                husten,
                atemnot,
                # temperatur ggf. weiter unten,
                mens,
                notizen,
                other,
            ]

            # Temperatur ggf. an passender Stelle einschieben (vor mens/notizen)
            if temperatur is not None:
                # Spalte VOR 'mens' einfügen (d. h. Index -2 in unserer Liste)
                insert_pos = len(cols) - 2
                cols.insert(insert_pos, "temperatur")
                vals.insert(insert_pos, temperatur)

            placeholders = ", ".join(["%s"] * len(cols))
            sql = f"""
                INSERT INTO health_data ({", ".join(cols)})
                VALUES ({placeholders})
                RETURNING id
            """
            cur.execute(sql, tuple(vals))
            new_id = cur.fetchone()[0]
            conn.commit()
            return new_id


    def get_health_data(self, data_id: int) -> dict | None:
        """Liest einen Health-Datensatz anhand seiner ID."""
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM health_data WHERE id = %s", (data_id,))
            return cur.fetchone()


    def get_health_data_by_day(self, day: str) -> dict | None:
        """Liest den Health-Datensatz für ein bestimmtes Datum (falls vorhanden)."""
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM health_data WHERE day = %s", (day,))
            return cur.fetchone()


    def list_health_data(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Liefert eine Liste aller Health-Datensätze, sortiert nach Datum (absteigend)."""
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM health_data
                ORDER BY day DESC, id DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
            return cur.fetchall()


    def update_health_data(self, data_id: int, **fields) -> None:
        """
        Aktualisiert bestimmte Felder eines Health-Datensatzes.
        Wichtig: Ein explizit übergebener Key mit Wert None (z. B. temperatur=None)
        setzt die Spalte in der DB auf NULL.
        """
        allowed = {
            "day", "borg", "erschöpfung", "muskelschwäche", "schmerzen",
            "angst", "konzentration", "husten", "atemnot", "temperatur",
            "mens", "notizen", "other",
        }
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k} = %s")
                vals.append(v)
        if not sets:
            return  # nichts zu tun

        vals.append(data_id)
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f"UPDATE health_data SET {', '.join(sets)} WHERE id = %s",
                tuple(vals),
            )
            conn.commit()


    def delete_health_data(self, data_id: int) -> None:
        """Löscht einen Health-Datensatz."""
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM health_data WHERE id = %s", (data_id,))
            conn.commit()
