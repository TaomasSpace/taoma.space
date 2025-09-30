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
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Falls deine PG-Version "CREATE INDEX IF NOT EXISTS" nicht kennt,
-- werden wir die Indizes weiter unten per try/except anlegen.
CREATE INDEX IF NOT EXISTS idx_gifs_title       ON gifs(title);
CREATE INDEX IF NOT EXISTS idx_gifs_anime       ON gifs(anime);
CREATE INDEX IF NOT EXISTS idx_gifs_nsfw        ON gifs(nsfw);
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
    user_id             INTEGER NOT NULL UNIQUE
                        REFERENCES users(id) ON DELETE CASCADE,
    slug                TEXT NOT NULL UNIQUE,     -- wird zu /tree/{slug}
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
END$$;
"""


class PgGifDB:
    def __init__(self, dsn: str):
        self.dsn = dsn
        # Schema beim Start sicherstellen
        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                # 1) Grundschema (inkl. GIF-Indizes via IF NOT EXISTS)
                cur.execute(DDL)

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
                    CREATE UNIQUE INDEX IF NOT EXISTS ux_linktrees_slug ON linktrees (lower(slug));
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
        gif_id, title, url, nsfw, anime, created_at = row
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
    ) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            fields = ["title", "url", "nsfw", "anime"]
            vals = [title, url, nsfw, anime]
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
                "SELECT id,title,url,nsfw,anime,created_at FROM gifs WHERE id=%s",
                (gif_id,),
            )
            row = cur.fetchone()
            if not row:
                raise KeyError(f"gif id {gif_id} not found")
            return self._compose_gif(cur, row)

    def get_gif_by_url(self, gif_url: str) -> Dict[str, Any]:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id,title,url,nsfw,anime,created_at FROM gifs WHERE url=%s",
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

        sql = "SELECT id,title,url,nsfw,anime,created_at FROM gifs"
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
                f"SELECT id,title,url,nsfw,anime,created_at FROM gifs WHERE {cond} ORDER BY random() LIMIT 1"
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
                SELECT g.id,g.title,g.url,g.nsfw,g.anime,g.created_at
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
                SELECT id,title,url,nsfw,anime,created_at
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
                SELECT g.id,g.title,g.url,g.nsfw,g.anime,g.created_at
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
    ) -> int:
        with psycopg.connect(self.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO linktrees (user_id, slug, location, quote, song_url,
                                    background_url, background_is_video,
                                    transparency, name_effect, background_effect)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """,
                (
                    user_id,
                    slug,
                    location,
                    quote,
                    song_url,
                    background_url,
                    background_is_video,
                    transparency,
                    name_effect,
                    background_effect,
                ),
            )
            linktree_id = cur.fetchone()[0]
            # Optional: users.linktree_id für schnellen Join setzen
            cur.execute(
                "UPDATE users SET linktree_id = %s, updated_at = now() WHERE id = %s",
                (linktree_id, user_id),
            )
            conn.commit()
            return linktree_id


    def update_linktree(self, linktree_id: int, **fields) -> None:
        allowed = {
            "slug",
            "location",
            "quote",
            "song_url",
            "background_url",
            "background_is_video",
            "transparency",
            "name_effect",
            "background_effect",
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


    def get_linktree_by_slug(self, slug: str) -> dict | None:
        with psycopg.connect(self.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM linktrees WHERE lower(slug)=lower(%s)", (slug,))
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
            cur.execute(
                """
                SELECT i.id, i.code, i.image_url, i.description, ui.displayed, ui.acquired_at
                FROM user_icons ui
                JOIN icons i ON i.id = ui.icon_id
                WHERE ui.user_id = (SELECT user_id FROM linktrees WHERE id=%s)
            ORDER BY i.code
            """,
                (lt["id"],),
            )
            icons = cur.fetchall()
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
