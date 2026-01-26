from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
import os, httpx
from dotenv import load_dotenv
import logging
import json, html
from datetime import datetime, timezone, timedelta
from fastapi import BackgroundTasks
import asyncio, uuid
import random
import time
try:
    import websockets
except Exception:
    websockets = None
from pydantic import BaseModel, Field
from typing import Any
from fastapi import FastAPI, HTTPException, Query, Header, Depends
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from pydantic import BaseModel
import os
from fastapi import Depends, Header
from db.db_helper import GifDB as SqliteGifDB
from db.pg_helper import PgGifDB
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import Response, Depends
from fastapi.middleware.cors import CORSMiddleware
import threading
from psycopg import errors as pg_errors
import os, base64, secrets
from urllib.parse import quote_plus, urlparse, urlunparse
from fastapi import Path as PathParam
from fastapi import UploadFile, File
import pathlib
import io
from PIL import Image, UnidentifiedImageError
from typing import Literal
import psycopg
from typing import Literal
import bcrypt
from psycopg.rows import dict_row
from fastapi.responses import JSONResponse
from fastapi import Request
from datetime import date
from pydantic import AliasChoices
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
from .firepbase import get_firestore_client
from google.cloud import firestore


Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    username = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    discord_account = relationship(
        "DiscordAccount",
        uselist=False,
        back_populates="user"
    )


class DiscordAccount(Base):
    __tablename__ = "discord_accounts"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    discord_user_id = Column(String(32), nullable=False)
    discord_username = Column(String(100))
    discord_global_name = Column(String(100))
    avatar_hash = Column(String(128))
    avatar_decoration = Column(Text)  # oder JSON, je nach DB/Driver

    presence_status = Column(String(16))
    status_text = Column(Text)
    presence_updated_at = Column(DateTime)

    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    token_expires_at = Column(DateTime, nullable=False)
    scopes = Column(Text, nullable=False)
    linked_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="discord_account")

    __table_args__ = (
        UniqueConstraint("user_id"),
        UniqueConstraint("discord_user_id"),
    )

DisplayNameMode = Literal["slug", "username", "custom"]
DeviceType = Literal["pc", "mobile"]
load_dotenv()
app = FastAPI(title="Anime GIF API", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
DATABASE_URL = os.getenv("DATABASE_URL")  # von Render
db = PgGifDB(DATABASE_URL) if DATABASE_URL else SqliteGifDB("gifs.db")

ADMIN_PASSWORD = os.getenv("GIFAPI_ADMIN_PASSWORD", "")
lock = threading.Lock()

ALG = "pbkdf2_sha256"
ITER = 200_000
SALT_LEN = 16

UPLOAD_DIR = pathlib.Path(os.getenv("UPLOAD_DIR", "/var/data/avatars"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MEDIA_ROOT = UPLOAD_DIR.parent  # -> /var/data
app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")

ALLOWED_AUDIO = {"audio/mpeg", "audio/ogg", "audio/wav"}
ALLOWED_IMAGE_CT = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "image/heic",
    "image/heif",
}
ALLOWED_VIDEO_CT = {"video/mp4", "video/quicktime", "video/webm"}
VIDEO_EXTENSIONS = (".mp4", ".webm", ".m4v", ".mov")
MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15 MB
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB (Avatare/Icons)
MAX_BACKGROUND_BYTES = 50 * 1024 * 1024  # 50 MB (Hintergrund-Video/Bild)

ALLOWED_USER_IDS = {
    int(x)
    for x in os.getenv("ALLOWED_USER_IDS", "1").replace(" ", "").split(",")
    if x
}

# GIF badge thresholds: (min_gifs, icon_code)
GIF_BADGE_THRESHOLDS: list[tuple[int, str]] = [
    (5, "753248"),
    (100, "753249"),
    (500, "753250"),
    (1000, "753251"),
]
def _ensure_pg():
    if not isinstance(db, PgGifDB):
        raise HTTPException(501, "Linktree features require PostgreSQL.")


def _get_linktree_owner(linktree_id: int) -> int:
    # minimaler Read nur für Owner-Check
    _ensure_pg()
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT user_id FROM linktrees WHERE id=%s", (linktree_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Linktree not found")
        return int(row[0])


def _require_tree_owner_or_admin(linktree_id: int, user: dict):
    owner_id = _get_linktree_owner(linktree_id)
    if not (user.get("admin") or user["id"] == owner_id):
        raise HTTPException(403, "Forbidden (owner or admin only)")


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def require_token(
    request: Request,
    x_auth_token: str | None = Header(None, alias="X-Auth-Token"),
    authorization: str | None = Header(None, alias="Authorization"),
):
    token = _extract_token(x_auth_token, authorization, request)
    if not token or not db.validate_token(token):
        raise HTTPException(401, "Unauthorized")
    return token


def require_user(x_auth_token: str = Depends(require_token)):
    user = db.get_token_user(x_auth_token)
    if not user:
        raise HTTPException(401, "Unauthorized")
    return user

def require_specific_user(user: dict = Depends(require_user)):
    if user.get("id") not in ALLOWED_USER_IDS:
        raise HTTPException(403, "Forbidden")
    return user

def require_admin(user: dict = Depends(require_user)):
    if not bool(user.get("admin", False)):
        raise HTTPException(403, "Forbidden: admin only")
    return user


def hashPassword(password: str) -> str:
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("password must be a string with length >= 8")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def decodePassword(hashedPassword: str) -> str:
    # Nicht möglich: Hashes sind one-way. Wir lassen das bewusst scheitern:
    raise NotImplementedError(
        "Passwords cannot be decoded; compare with CheckPassword()."
    )


def CheckPassword(hashedPassword: str, unhashedPassword: str) -> bool:
    if not hashedPassword or not unhashedPassword:
        return False
    try:
        return bcrypt.checkpw(
            unhashedPassword.encode("utf-8"), hashedPassword.encode("utf-8")
        )
    except Exception:
        return False


def _is_local_media_url(url: str) -> bool:
    if not isinstance(url, str) or not url:
        return False
    # Erlaube nur genau deinen /media/<dirname>/... Pfad
    prefix = f"/media/{UPLOAD_DIR.name}/"
    return url.startswith(prefix)


def _path_from_media_url(url: str) -> pathlib.Path | None:
    if not _is_local_media_url(url):
        return None
    # Mappe /media/<dir>/<file> -> UPLOAD_DIR/<file>
    name = url.rsplit("/", 1)[-1]
    p = UPLOAD_DIR / name
    # Finaler Schutz: Stelle sicher, dass die Datei im Upload-Verzeichnis liegt
    try:
        p.resolve().relative_to(UPLOAD_DIR.resolve())
    except Exception:
        return None
    return p


def _add_local_media_url(url: str | None, bucket: set[str]) -> None:
    if not url:
        return
    if _is_local_media_url(str(url)):
        bucket.add(str(url))


def _count_db_references(url: str) -> int:
    if not url:
        return 0
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        total = 0
        cur.execute("SELECT COUNT(*) FROM users WHERE profile_picture = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM linktrees WHERE song_url = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute(
            "SELECT COUNT(*) FROM linktrees WHERE song_icon_url = %s", (url,)
        )
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM linktrees WHERE background_url = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM linktrees WHERE cursor_url = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM linktree_links WHERE icon_url = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM icons WHERE image_url = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM gifs WHERE url = %s", (url,))
        total += int(cur.fetchone()[0])
        return total


def _has_media_references(url: str, cache: set[str] | None = None) -> bool:
    if not _is_local_media_url(url):
        return False
    if cache is not None:
        return url in cache
    try:
        if _count_db_references(url) > 0:
            return True
    except Exception as exc:
        logger.warning("DB reference check failed for %s: %s", url, exc)
        return True
    try:
        return _is_referenced_in_templates(url)
    except Exception as exc:
        logger.warning("Template reference check failed for %s: %s", url, exc)
        return True


def _delete_if_unreferenced(url: str) -> bool:
    """Löscht die Datei, wenn lokales Media und DB-Referenzen == 0.
    Gibt True zurück, wenn gelöscht wurde."""
    if not _is_local_media_url(url):
        return False
    if _has_media_references(url):
        return False
    p = _path_from_media_url(url)
    if not p or not p.exists():
        return False
    try:
        p.unlink(missing_ok=True)
        return True
    except Exception as e:
        logger.warning("Failed to delete unreferenced media %s: %s", url, e)
        return False


def _template_doc_has_url(data: dict, url: str) -> bool:
    if not data or not isinstance(data, dict):
        return False
    if data.get("preview_image_url") == url:
        return True
    if data.get("owner_profile_picture") == url:
        return True
    variants = data.get("variants") or []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        if any(
            variant.get(k) == url
            for k in ("song_url", "song_icon_url", "background_url", "cursor_url")
        ):
            return True
        links = variant.get("links") or []
        for link in links:
            if isinstance(link, dict) and link.get("icon_url") == url:
                return True
    legacy = data.get("data")
    if isinstance(legacy, dict):
        if any(
            legacy.get(k) == url
            for k in ("song_url", "song_icon_url", "background_url", "cursor_url")
        ):
            return True
        links = legacy.get("links") or []
        for link in links:
            if isinstance(link, dict) and link.get("icon_url") == url:
                return True
    return False


def _collect_template_media_urls() -> tuple[set[str], bool, list[str]]:
    urls: set[str] = set()
    warnings: list[str] = []
    try:
        fs = _fs()
    except HTTPException as exc:
        if getattr(exc, "status_code", None) == 503:
            warnings.append("Templates disabled (Firestore not configured); template references not scanned.")
            return urls, True, warnings
        warnings.append(str(exc.detail if hasattr(exc, "detail") else exc))
        return urls, False, warnings
    except Exception as exc:
        warnings.append(str(exc))
        return urls, False, warnings
    try:
        docs = fs.collection(TEMPLATE_COLLECTION).stream()
    except Exception as exc:
        warnings.append(str(exc))
        return urls, False, warnings

    for doc in docs:
        data = doc.to_dict() or {}
        _add_local_media_url(data.get("preview_image_url"), urls)
        _add_local_media_url(data.get("owner_profile_picture"), urls)
        variants = data.get("variants") or []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            _add_local_media_url(variant.get("song_url"), urls)
            _add_local_media_url(variant.get("song_icon_url"), urls)
            _add_local_media_url(variant.get("background_url"), urls)
            _add_local_media_url(variant.get("cursor_url"), urls)
            links = variant.get("links") or []
            for link in links:
                if isinstance(link, dict):
                    _add_local_media_url(link.get("icon_url"), urls)
        legacy = data.get("data")
        if isinstance(legacy, dict):
            _add_local_media_url(legacy.get("song_url"), urls)
            _add_local_media_url(legacy.get("song_icon_url"), urls)
            _add_local_media_url(legacy.get("background_url"), urls)
            _add_local_media_url(legacy.get("cursor_url"), urls)
            links = legacy.get("links") or []
            for link in links:
                if isinstance(link, dict):
                    _add_local_media_url(link.get("icon_url"), urls)
    return urls, True, warnings


def _collect_media_references() -> tuple[set[str], bool, list[str]]:
    urls: set[str] = set()
    warnings: list[str] = []
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            queries = (
                "SELECT profile_picture FROM users WHERE profile_picture IS NOT NULL",
                "SELECT song_url FROM linktrees WHERE song_url IS NOT NULL",
                "SELECT song_icon_url FROM linktrees WHERE song_icon_url IS NOT NULL",
                "SELECT background_url FROM linktrees WHERE background_url IS NOT NULL",
                "SELECT cursor_url FROM linktrees WHERE cursor_url IS NOT NULL",
                "SELECT icon_url FROM linktree_links WHERE icon_url IS NOT NULL",
                "SELECT image_url FROM icons WHERE image_url IS NOT NULL",
                "SELECT url FROM gifs WHERE url IS NOT NULL",
            )
            for sql in queries:
                cur.execute(sql)
                rows = cur.fetchall()
                for row in rows:
                    _add_local_media_url(row[0], urls)
    except Exception as exc:
        raise HTTPException(500, "Could not collect media references") from exc

    template_urls, templates_ok, template_warnings = _collect_template_media_urls()
    urls |= template_urls
    warnings.extend(template_warnings)
    return urls, templates_ok, warnings


def _perform_media_gc(min_age_seconds: int, refs: set[str]) -> tuple[list[str], list[str]]:
    deleted: list[str] = []
    skipped: list[str] = []
    now = datetime.now().timestamp()

    for p in UPLOAD_DIR.iterdir():
        if not p.is_file():
            continue
        url = f"/media/{UPLOAD_DIR.name}/{p.name}"
        try:
            age = now - p.stat().st_mtime
        except Exception as exc:
            logger.warning("GC stat failed for %s: %s", p, exc)
            skipped.append(p.name)
            continue
        if age < min_age_seconds:
            skipped.append(p.name)  # zu frisch
            continue
        try:
            if not _has_media_references(url, cache=refs):
                p.unlink(missing_ok=True)
                deleted.append(p.name)
            else:
                skipped.append(p.name)
        except Exception as e:
            logger.warning("GC failed for %s: %s", p, e)
            skipped.append(p.name)

    return deleted, skipped


def _run_media_gc(min_age_seconds: int, *, require_templates: bool) -> dict:
    refs, templates_ok, warnings = _collect_media_references()
    if not templates_ok:
        msg = "Template reference scan failed; skipped GC to avoid false positives."
        if msg not in warnings:
            warnings.append(msg)
        if require_templates:
            detail = {"message": "Template reference scan failed. No files were deleted.", "warnings": warnings}
            raise HTTPException(503, detail)
        return {
            "ok": False,
            "deleted": [],
            "skipped": [],
            "warnings": warnings,
            "skipped_due_to_templates": True,
        }

    deleted, skipped = _perform_media_gc(min_age_seconds, refs)
    return {
        "ok": True,
        "deleted": deleted,
        "skipped": skipped,
        "warnings": warnings,
        "skipped_due_to_templates": False,
    }


def _is_referenced_in_templates(url: str) -> bool:
    if not _is_local_media_url(url):
        return False
    try:
        fs = _fs()
    except HTTPException as exc:
        if getattr(exc, "status_code", None) == 503:
            return False
        raise
    docs = fs.collection(TEMPLATE_COLLECTION).stream()
    for doc in docs:
        data = doc.to_dict() or {}
        if _template_doc_has_url(data, url):
            return True
    return False


def _session_response(
    payload: dict, token: str, max_age: int = 24 * 3600
) -> JSONResponse:
    resp = JSONResponse(payload)
    resp.set_cookie(
        key="taoma_token",
        value=token,
        max_age=max_age,
        httponly=True,  # schützt vor JS-Zugriff
        secure=True,  # nur über HTTPS
        samesite="lax",  # Navigation-Links funktionieren
    )
    return resp


def _extract_token(
    x_auth_token: str | None, authorization: str | None, request: Request
) -> str | None:
    # 1) Header X-Auth-Token
    if x_auth_token:
        return x_auth_token
    # 2) Authorization: Bearer <token>
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]
    # 3) Cookie
    tok = request.cookies.get("taoma_token")
    if tok:
        return tok
    return None


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
DISCORD_CLIENT_ID = os.getenv("DISCORD_CLIENT_ID")
DISCORD_CLIENT_SECRET = os.getenv("DISCORD_CLIENT_SECRET")
DISCORD_REDIRECT_URI = os.getenv("DISCORD_REDIRECT_URI")
DISCORD_OAUTH_SCOPES = os.getenv("DISCORD_OAUTH_SCOPES", "identify")
DISCORD_STATE_TTL = int(os.getenv("DISCORD_STATE_TTL", "600"))
DISCORD_STATE_STORE: dict[str, dict] = {}
logger = logging.getLogger("uvicorn.error")


def _ensure_discord_config():
    if not (DISCORD_CLIENT_ID and DISCORD_CLIENT_SECRET and DISCORD_REDIRECT_URI):
        raise HTTPException(
            503,
            "Discord linking is not configured (set DISCORD_CLIENT_ID, DISCORD_CLIENT_SECRET, DISCORD_REDIRECT_URI)",
        )


def _discord_configured() -> bool:
    return bool(DISCORD_CLIENT_ID and DISCORD_CLIENT_SECRET and DISCORD_REDIRECT_URI)


def _new_discord_state(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    DISCORD_STATE_STORE[token] = {
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc),
    }
    return token


def _pop_discord_state(token: str) -> int | None:
    data = DISCORD_STATE_STORE.pop(token, None)
    if not data:
        return None
    created = data.get("created_at")
    if not created or created < datetime.now(timezone.utc) - timedelta(
        seconds=DISCORD_STATE_TTL
    ):
        return None
    return data.get("user_id")


def _decoration_to_url(decoration: Any) -> str | None:
    if not decoration:
        return None
    data = decoration
    if isinstance(decoration, str):
        try:
            data = json.loads(decoration)
        except Exception:
            data = {"asset": decoration}
    if isinstance(data, dict):
        asset = data.get("asset") or data.get("avatar_decoration") or data.get("hash")
    elif isinstance(data, str):
        asset = data
    else:
        return None
    if not asset:
        return None
    return f"https://cdn.discordapp.com/avatar-decoration-presets/{asset}.png?size=320"


def _load_discord_account(user_id: int) -> dict:
    if not isinstance(db, PgGifDB):
        return {}
    try:
        acct = db.get_discord_account(user_id)
    except Exception:
        return {}
    if not acct:
        return {}
    acct["decoration_url"] = _decoration_to_url(acct.get("avatar_decoration"))
    return acct


DISCORD_BADGE_MAP = [
    {"bit": 1 << 0, "code": "staff", "label": "Discord Staff"},
    {"bit": 1 << 1, "code": "partner", "label": "Partner"},
    {"bit": 1 << 2, "code": "hypesquad", "label": "HypeSquad Events"},
    {"bit": 1 << 3, "code": "bug_hunter_lvl1", "label": "Bug Hunter Level 1"},
    {"bit": 1 << 6, "code": "bravery", "label": "HypeSquad Bravery"},
    {"bit": 1 << 7, "code": "brilliance", "label": "HypeSquad Brilliance"},
    {"bit": 1 << 8, "code": "balance", "label": "HypeSquad Balance"},
    {"bit": 1 << 9, "code": "early_supporter", "label": "Early Supporter"},
    {"bit": 1 << 14, "code": "bug_hunter_lvl2", "label": "Bug Hunter Level 2"},
    {"bit": 1 << 16, "code": "verified_bot", "label": "Verified Bot"},
    {"bit": 1 << 17, "code": "verified_dev", "label": "Early Verified Bot Developer"},
    {"bit": 1 << 18, "code": "certified_mod", "label": "Certified Moderator"},
    {"bit": 1 << 19, "code": "bot_http", "label": "Bot HTTP Interactions"},
    {"bit": 1 << 22, "code": "active_dev", "label": "Active Developer"},
]


def _discord_badges_from_account(acct: dict) -> list[dict]:
    if not acct:
        return []
    flags = int(acct.get("public_flags") or 0)
    badges = []
    for badge in DISCORD_BADGE_MAP:
        if flags & int(badge["bit"]):
            badges.append(
                {
                    "code": badge["code"],
                    "label": badge["label"],
                    "icon_url": badge.get("icon_url"),
                }
            )
    premium = int(acct.get("premium_type") or 0)
    if premium > 0:
        badges.append({"code": "nitro", "label": "Discord Nitro", "icon_url": None})
    return badges


DISCORD_BOT_TOKEN = (os.getenv("DISCORD_BOT_TOKEN") or "").strip()
DISCORD_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"
DISCORD_PRESENCE_VALUES = {"online", "idle", "dnd", "offline"}
DISCORD_STATUS_TEXT_MAX = 140
DISCORD_INTENTS = (1 << 1) | (1 << 8)  # GUILD_MEMBERS + GUILD_PRESENCES


def _normalize_presence_status(raw: str | None) -> str:
    val = (raw or "").lower().strip()
    return val if val in DISCORD_PRESENCE_VALUES else "offline"


def _extract_custom_status(activities: list[dict] | None) -> str | None:
    for act in activities or []:
        if act.get("type") == 4:
            text = (act.get("state") or "").strip()
            if text:
                return text[:DISCORD_STATUS_TEXT_MAX]
            return None
    return None


def _resolve_discord_presence(
    lt: dict, acct: dict | None
) -> tuple[str, str | None]:
    if not acct:
        return ("offline", None)
    presence = _normalize_presence_status(acct.get("presence_status"))
    status_text = (acct.get("status_text") or "").strip() or None
    return (presence, status_text)


class DiscordPresenceGateway:
    def __init__(self, token: str, db_ref: PgGifDB):
        self._token = token
        self._db = db_ref
        self._task: asyncio.Task | None = None
        self._seq: int | None = None
        self._session_id: str | None = None
        self._resume_url: str | None = None
        self._known_ids: set[str] = set()
        self._presence_cache: dict[str, tuple[str, str | None]] = {}
        self._last_refresh = 0.0

    def start(self) -> None:
        if not self._token:
            logger.warning("Discord presence sync disabled: DISCORD_BOT_TOKEN missing")
            return
        if websockets is None:
            logger.warning("Discord presence sync disabled: websockets not available")
            return
        if not isinstance(self._db, PgGifDB):
            logger.warning("Discord presence sync disabled: DB not available")
            return
        if self._task:
            return
        self._task = asyncio.create_task(self._run_forever())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _run_forever(self) -> None:
        backoff = 2.0
        while True:
            try:
                await self._run_gateway()
                backoff = 2.0
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Discord presence gateway error: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

    def _refresh_known_ids(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_refresh < 120:
            return
        try:
            ids = self._db.list_discord_user_ids()
            self._known_ids = set(ids)
            self._last_refresh = now
        except Exception as exc:
            logger.warning("Discord presence: failed to refresh linked IDs: %s", exc)

    async def _send_json(self, ws, payload: dict) -> None:
        await ws.send(json.dumps(payload))

    async def _run_gateway(self) -> None:
        url = self._resume_url or DISCORD_GATEWAY_URL
        self._refresh_known_ids(force=True)
        async with websockets.connect(url, max_size=2**20) as ws:
            hello_raw = await ws.recv()
            hello = json.loads(hello_raw)
            if hello.get("op") != 10:
                raise RuntimeError("Discord gateway did not send HELLO")
            interval = (hello.get("d") or {}).get("heartbeat_interval", 45000) / 1000
            heartbeat_task = asyncio.create_task(self._heartbeat(ws, interval))
            try:
                if self._session_id and self._seq:
                    await self._send_json(
                        ws,
                        {
                            "op": 6,
                            "d": {
                                "token": self._token,
                                "session_id": self._session_id,
                                "seq": self._seq,
                            },
                        },
                    )
                else:
                    await self._send_json(
                        ws,
                        {
                            "op": 2,
                            "d": {
                                "token": self._token,
                                "intents": DISCORD_INTENTS,
                                "properties": {
                                    "$os": "linux",
                                    "$browser": "taoma",
                                    "$device": "taoma",
                                },
                            },
                        },
                    )
                async for raw in ws:
                    payload = json.loads(raw)
                    op = payload.get("op")
                    t = payload.get("t")
                    d = payload.get("d")
                    s = payload.get("s")
                    if s is not None:
                        self._seq = s

                    if op == 0:
                        await self._handle_dispatch(t, d)
                    elif op == 1:
                        await self._send_json(ws, {"op": 1, "d": self._seq})
                    elif op == 7:
                        logger.info("Discord gateway requested reconnect")
                        break
                    elif op == 9:
                        self._session_id = None
                        self._seq = None
                        await asyncio.sleep(random.uniform(1, 5))
                        break
            finally:
                heartbeat_task.cancel()

    async def _heartbeat(self, ws, interval: float) -> None:
        while True:
            await asyncio.sleep(interval)
            try:
                await self._send_json(ws, {"op": 1, "d": self._seq})
            except Exception:
                break

    async def _handle_dispatch(self, event: str | None, data: dict | None) -> None:
        if not event:
            return
        if event == "READY":
            if isinstance(data, dict):
                self._session_id = data.get("session_id") or self._session_id
                self._resume_url = data.get("resume_gateway_url") or self._resume_url
            self._refresh_known_ids(force=True)
            return
        if event == "RESUMED":
            self._refresh_known_ids(force=True)
            return
        if event != "PRESENCE_UPDATE":
            return
        if not isinstance(data, dict):
            return
        user = data.get("user") or {}
        discord_user_id = user.get("id")
        if not discord_user_id:
            return
        self._refresh_known_ids()
        if self._known_ids and discord_user_id not in self._known_ids:
            return
        presence = _normalize_presence_status(data.get("status"))
        status_text = _extract_custom_status(data.get("activities") or [])
        cache_key = self._presence_cache.get(discord_user_id)
        new_key = (presence, status_text)
        if cache_key == new_key:
            return
        try:
            updated = self._db.update_discord_presence(
                discord_user_id,
                presence_status=presence,
                status_text=status_text,
            )
            if updated:
                self._presence_cache[discord_user_id] = new_key
        except Exception as exc:
            logger.warning("Discord presence update failed: %s", exc)

ALLOWED_ORIGINS = [
    o.strip()
    for o in (
        os.getenv("ALLOWED_ORIGINS") or "*,http://127.0.0.1:8000,http://localhost:8000"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Auth-Token"],
)

ALLOWED_MIME = ALLOWED_IMAGE_CT


EffectName = Literal["none", "glow", "neon", "rainbow"]
BgEffectName = Literal["none", "night", "rain", "snow"]
DiscordPresence = Literal["online", "idle", "dnd", "offline"]
HEX_COLOR_RE = r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"


class LinkOut(BaseModel):
    id: int
    url: HttpUrl
    label: Optional[str] = None
    icon_url: Optional[str] = None
    position: int
    is_active: bool


class IconOut(BaseModel):
    id: int
    code: str
    image_url: str
    description: Optional[str] = None
    displayed: Optional[bool] = None
    acquired_at: Optional[str] = None


class DiscordBadgeOut(BaseModel):
    code: str
    label: str
    icon_url: Optional[str] = None


class LinktreeCreateIn(BaseModel):
    slug: str = Field(..., min_length=2, max_length=48, pattern=r"^[a-zA-Z0-9_-]+$")
    device_type: DeviceType = "pc"
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    song_icon_url: Optional[str] = None
    show_audio_player: bool = False
    audio_player_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    audio_player_bg_alpha: int = Field(60, ge=0, le=100)
    audio_player_text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    audio_player_accent_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    background_url: Optional[str] = None
    background_is_video: bool = False
    transparency: int = Field(0, ge=0, le=100)
    name_effect: EffectName = "none"
    background_effect: BgEffectName = "none"
    display_name_mode: DisplayNameMode = "slug"
    custom_display_name: Optional[str] = Field(None, min_length=1, max_length=64)
    link_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_alpha: int = Field(100, ge=0, le=100)
    card_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    name_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    location_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    quote_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_url: Optional[str] = None
    discord_frame_enabled: bool = False
    discord_presence_enabled: bool = False
    discord_presence: DiscordPresence = "online"
    discord_status_enabled: bool = False
    discord_status_text: Optional[str] = Field(None, max_length=140)
    discord_badges_enabled: bool = False
    show_visit_counter: bool = False
    visit_counter_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    visit_counter_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    visit_counter_bg_alpha: int = Field(20, ge=0, le=100)


class LinktreeUpdateIn(BaseModel):
    slug: Optional[str] = Field(
        None, min_length=2, max_length=48, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    device_type: Optional[DeviceType] = None
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    song_icon_url: Optional[str] = None
    show_audio_player: Optional[bool] = None
    audio_player_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    audio_player_bg_alpha: Optional[int] = Field(None, ge=0, le=100)
    audio_player_text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    audio_player_accent_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    background_url: Optional[str] = None
    background_is_video: Optional[bool] = None
    transparency: Optional[int] = Field(None, ge=0, le=100)
    name_effect: Optional[EffectName] = None
    background_effect: Optional[BgEffectName] = None
    display_name_mode: Optional[DisplayNameMode] = None
    custom_display_name: Optional[str] = Field(None, min_length=1, max_length=64)
    link_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_alpha: Optional[int] = Field(None, ge=0, le=100)
    card_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    name_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    location_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    quote_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_url: Optional[str] = None
    discord_frame_enabled: Optional[bool] = None
    discord_presence_enabled: Optional[bool] = None
    discord_presence: Optional[DiscordPresence] = None
    discord_status_enabled: Optional[bool] = None
    discord_status_text: Optional[str] = Field(None, max_length=140)
    discord_badges_enabled: Optional[bool] = None
    show_visit_counter: Optional[bool] = None
    visit_counter_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    visit_counter_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    visit_counter_bg_alpha: Optional[int] = Field(None, ge=0, le=100)


class LinkCreateIn(BaseModel):
    url: HttpUrl
    label: Optional[str] = None
    icon_url: Optional[str] = None  # erlaubt benutzerhochgeladenes Icon
    position: Optional[int] = None  # None = append
    is_active: bool = True


class LinkUpdateIn(BaseModel):
    url: Optional[HttpUrl] = None
    label: Optional[str] = None
    icon_url: Optional[str] = None
    position: Optional[int] = None
    is_active: Optional[bool] = None


class LinktreeOut(BaseModel):
    id: int
    user_id: int
    slug: str
    device_type: DeviceType
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    song_icon_url: Optional[str] = None
    show_audio_player: bool = False
    audio_player_bg_color: Optional[str] = None
    audio_player_bg_alpha: int = 60
    audio_player_text_color: Optional[str] = None
    audio_player_accent_color: Optional[str] = None
    background_url: Optional[str] = None
    background_is_video: bool
    transparency: int
    name_effect: EffectName
    background_effect: BgEffectName
    display_name_mode: DisplayNameMode  # NEU
    custom_display_name: Optional[str] = None
    link_color: Optional[str] = None
    link_bg_color: Optional[str] = None
    link_bg_alpha: int = 100
    card_color: Optional[str] = None
    text_color: Optional[str] = None
    name_color: Optional[str] = None
    location_color: Optional[str] = None
    quote_color: Optional[str] = None
    cursor_url: Optional[str] = None
    discord_frame_enabled: bool = False
    discord_decoration_url: Optional[str] = None
    discord_presence_enabled: bool = False
    discord_presence: DiscordPresence = "online"
    discord_status_enabled: bool = False
    discord_status_text: Optional[str] = None
    discord_badges_enabled: bool = False
    discord_badges: Optional[List[DiscordBadgeOut]] = None
    discord_linked: bool = False
    profile_picture: Optional[str] = None  # NEU - fuer Avatar
    user_username: Optional[str] = None  # NEU - fuer "username"-Modus
    show_visit_counter: bool = False
    visit_count: int = 0
    visit_counter_color: Optional[str] = None
    visit_counter_bg_color: Optional[str] = None
    visit_counter_bg_alpha: int = 20
    links: List[LinkOut]
    icons: List[IconOut]

VISIT_COOKIE_MAX_AGE = 365 * 24 * 3600  # 1 Jahr


def _visit_cookie_name(linktree_id: int) -> str:
    return f"ltv_{linktree_id}"


def _get_canonical_linktree_id(slug: str) -> int | None:
    if not isinstance(db, PgGifDB) or not slug:
        return None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(id) FROM linktrees WHERE lower(slug)=lower(%s)",
                (slug,),
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else None
    except Exception as exc:
        logger.warning("Failed to resolve canonical linktree id for %s: %s", slug, exc)
        return None


def _record_linktree_visit(lt: dict, request: Request, response: Response) -> int:
    """Counts unique visits per linktree using a cookie-based token."""
    if not isinstance(db, PgGifDB):
        return 0
    canonical_id = _get_canonical_linktree_id(str(lt.get("slug", ""))) or int(lt["id"])
    cookie_name = _visit_cookie_name(canonical_id)
    token = request.cookies.get(cookie_name) or ""
    is_new_token = False
    if not token:
        token = uuid.uuid4().hex
        is_new_token = True

    try:
        db.record_linktree_visit(canonical_id, token)
    except Exception as exc:
        logger.warning("Failed to record linktree visit: %s", exc)
    else:
        if is_new_token:
            response.set_cookie(
                cookie_name,
                token,
                max_age=VISIT_COOKIE_MAX_AGE,
                httponly=False,
                secure=True,
                samesite="lax",
            )

    try:
        return db.get_linktree_visit_count(canonical_id)
    except Exception as exc:
        logger.warning("Failed to fetch linktree visit count: %s", exc)
        return 0

class IconUpsertIn(BaseModel):
    code: str = Field(..., min_length=2, max_length=64, pattern=r"^[a-z0-9_\-]+$")
    image_url: str
    description: Optional[str] = None


class GrantIconIn(BaseModel):
    displayed: bool = False


class ToggleDisplayedIn(BaseModel):
    displayed: bool


class MeUpdateIn(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=32)
    profile_picture: Optional[str] = None
    linktree_id: Optional[int] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = Field(None, min_length=8)


class CounterOut(BaseModel):
    total: int


class GifIn(BaseModel):
    title: str
    url: HttpUrl
    nsfw: bool
    anime: Optional[str] = None
    characters: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class GifUpdate(BaseModel):
    title: Optional[str] = None
    url: Optional[HttpUrl] = None
    nsfw: Optional[bool] = None
    anime: Optional[str] = None
    characters: Optional[List[str]] = None  # None = nicht ändern, [] = leeren
    tags: Optional[List[str]] = None  # None = nicht ändern, [] = leeren


class GifOut(BaseModel):
    id: int
    title: str
    url: HttpUrl
    nsfw: bool
    anime: Optional[str]
    created_at: str
    characters: List[str]
    tags: List[str]
    created_by: Optional[int] = None


class GifBlacklistIn(BaseModel):
    reason: Optional[str] = None


class LoginIn(BaseModel):
    password: str


class LoginOut(BaseModel):
    token: str
    expires_at: str | None = None


class JobIn(BaseModel):
    seconds: int = Field(3, ge=1, le=20)


class Contact(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    subject: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=10, max_length=5000)
    website: str | None = None  # Honeypot


class UserOut(BaseModel):
    id: int
    username: str
    linktree_id: Optional[int] = None
    linktree_slug: Optional[str] = None  # <- add this
    profile_picture: Optional[str] = None
    admin: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class UserCreateIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8)
    linktree_id: Optional[int] = None
    profile_picture: Optional[str] = None


class UserUpdateIn(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=32)
    password: Optional[str] = Field(None, min_length=8)
    linktree_id: Optional[int] = None
    profile_picture: Optional[str] = None
    admin: Optional[bool] = None


JOB_STORE: dict[str, dict] = {}


class LoginUserIn(BaseModel):
    username: str
    password: str


class RegisterIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    password: str = Field(..., min_length=8)
    linktree_id: Optional[int] = None
    profile_picture: Optional[str] = None


class RegisterOut(BaseModel):
    token: str
    expires_at: Optional[str] = None
    user: UserOut



class HealthDataBase(BaseModel):
    day: date
    borg: int = Field(..., ge=0, le=20)                         # <- 10 → 20
    temperatur: Optional[float] = Field(None, ge=20, le=45)               # <- int → float
    erschoepfung: int = Field(
        0, ge=0, le=10,
        alias="erschöpfung",
        validation_alias=AliasChoices("erschoepfung","erschöpfung")
    )
    muskelschwaeche: int = Field(
        0, ge=0, le=10,
        alias="muskelschwäche",
        validation_alias=AliasChoices("muskelschwaeche","muskelschwäche")
    )
    schmerzen: int = Field(0, ge=0, le=10)
    angst: int = Field(0, ge=0, le=10)
    konzentration: int = Field(0, ge=0, le=10)
    husten: int = Field(0, ge=0, le=10)
    atemnot: int = Field(0, ge=0, le=10)
    mens: bool = False
    notizen: Optional[str] = None
    other: Optional[str] = None
    model_config = {"populate_by_name": True}


class HealthDataIn(HealthDataBase):
    pass


class HealthDataUpdate(BaseModel):
    day: Optional[date] = None
    borg: Optional[int] = Field(None, ge=0, le=20)              # <- 20
    temperatur: Optional[float] = Field(None, ge=20, le=45)
    erschoepfung: Optional[int] = Field(
        None, ge=0, le=10,
        alias="erschöpfung",
        validation_alias=AliasChoices("erschoepfung","erschöpfung")
    )
    muskelschwaeche: Optional[int] = Field(
        None, ge=0, le=10,
        alias="muskelschwäche",
        validation_alias=AliasChoices("muskelschwaeche","muskelschwäche")
    )
    schmerzen: Optional[int] = Field(None, ge=0, le=10)
    angst: Optional[int] = Field(None, ge=0, le=10)
    konzentration: Optional[int] = Field(None, ge=0, le=10)
    husten: Optional[int] = Field(None, ge=0, le=10)
    atemnot: Optional[int] = Field(None, ge=0, le=10)
    mens: Optional[bool] = None
    notizen: Optional[str] = None
    other: Optional[str] = None
    model_config = {"populate_by_name": True}


class HealthDataOut(HealthDataBase):
    id: int


def _health_row_to_out(row: dict) -> HealthDataOut:
# row kann von psycopg dict_row kommen
    return HealthDataOut(
        id=row["id"],
        day=row["day"],
        borg=row["borg"],
        temperatur=row["temperatur"],
        erschoepfung=row.get("erschöpfung", 0),
        muskelschwaeche=row.get("muskelschwäche", 0),
        schmerzen=row.get("schmerzen", 0),
        angst=row.get("angst", 0),
        konzentration=row.get("konzentration", 0),
        husten=row.get("husten", 0),
        atemnot=row.get("atemnot", 0),
        mens=row.get("mens", False),
        notizen=row.get("notizen"),
        other=row.get("other"),
    )

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")


@app.get("/portfolio", include_in_schema=False)
def portfolio():
    return FileResponse("portfolio/index.html")


@app.get("/portfolio/projects", include_in_schema=False)
def portfolio_projects():
    return FileResponse("portfolio/projects.html")


@app.get("/portfolio/about", include_in_schema=False)
def portfolio_about():
    return FileResponse("portfolio/about.html")


@app.get("/portfolio/contact", include_in_schema=False)
def portfolio_contact():
    return FileResponse("portfolio/contact.html")


@app.get("/portfolio/skills", include_in_schema=False)
def portfolio_skills():
    return FileResponse("portfolio/skills.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/icon.png")


@app.get("/projects", include_in_schema=False)
def projects():
    return RedirectResponse(url="/portfolio/projects", status_code=308)


@app.get("/about", include_in_schema=False)
def about():
    return RedirectResponse(url="/portfolio/about", status_code=308)


@app.get("/contact", include_in_schema=False)
def contact():
    return RedirectResponse(url="/portfolio/contact", status_code=308)


async def notify_discord(p: Contact) -> bool:
    if not DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL fehlt – Nachricht wird nicht gesendet.")
        return False
    content = (
        f"**New contact**\n"
        f"**Name:** {p.name}\n**Email:** {p.email}\n"
        f"**Subject:** {p.subject}\n\n{p.message}"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.post(DISCORD_WEBHOOK_URL, json={"content": content})
            r.raise_for_status()
        logger.info("Discord-Webhook OK (%s)", r.status_code)
        return True
    except httpx.HTTPStatusError as e:
        logger.error(
            "Discord-Webhook HTTP-Fehler: %s - %s",
            e.response.status_code,
            e.response.text,
        )
    except Exception as e:
        logger.exception("Discord-Webhook Exception: %s", e)
    return False


@app.post("/api/contact")
async def create_contact(
    payload: Contact, x_auth_token: str | None = Header(default=None)
):
    if payload.website:
        raise HTTPException(status_code=400, detail="Bot detected")
    expected = os.getenv("FORM_TOKEN")
    if expected and x_auth_token != expected:
        raise HTTPException(status_code=401, detail="Invalid token")

    sent = await notify_discord(payload)
    if not sent:
        # Optional: in dev 200 zurückgeben, aber im Log steht’s dann.
        # Oder härter: Fehler rausgeben:
        # raise HTTPException(status_code=502, detail="Failed to deliver to Discord")
        pass

    return {"ok": True}


@app.get("/api/ping")
async def ping():
    """Simple health/latency endpoint."""
    return {
        "ok": True,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "service": "taoma.space",
    }


@app.post("/api/echo")
async def echo(payload: dict[str, Any]):
    data = payload  # <-- dict, kein model_dump()
    parts = []
    for key, value in data.items():
        key_safe = html.escape(str(key))
        value_safe = html.escape(json.dumps(value, ensure_ascii=False))
        parts.append(f"<p><strong>{key_safe}</strong> = <code>{value_safe}</code></p>")
    html_out = "\n".join(parts)
    return {"ok": True, "data": html_out}


@app.post("/api/job")
async def create_job(job: JobIn, background: BackgroundTasks):
    """Creates a fake background job and returns a job_id."""
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {"status": "queued", "progress": 0}

    async def worker():
        JOB_STORE[job_id]["status"] = "running"
        for i in range(job.seconds):
            await asyncio.sleep(1)
            JOB_STORE[job_id]["progress"] = int((i + 1) / job.seconds * 100)
        JOB_STORE[job_id]["status"] = "done"

    background.add_task(worker)
    return {"ok": True, "job_id": job_id}


@app.get("/api/job/{job_id}")
async def job_status(job_id: str):
    data = JOB_STORE.get(job_id)
    if not data:
        return {"ok": False, "error": "not_found"}
    return {"ok": True, **data}


@app.get("/skills", include_in_schema=False)
def skills():
    return RedirectResponse(url="/portfolio/skills", status_code=308)


# ------------ GIF API ------------ #


def _clean_str_list(values: Optional[List[str]]) -> List[str]:
    return [s for s in (str(v).strip() for v in (values or [])) if s]


def _normalize_gif_url(raw_url: str, *, error_on_invalid: bool = True) -> str | None:
    """
    Accepts URLs ending with .gif (case-insensitive).
    .gifv is allowed but converted to .gif.
    Returns normalized URL or None if invalid and error_on_invalid=False.
    """
    if not isinstance(raw_url, str):
        if error_on_invalid:
            raise HTTPException(status_code=400, detail="GIF URL must end with .gif")
        return None

    candidate = raw_url.strip()
    try:
        parsed = urlparse(candidate)
    except Exception:
        if error_on_invalid:
            raise HTTPException(status_code=400, detail="Invalid GIF URL")
        return None

    path = parsed.path or ""
    lower_path = path.lower()

    if lower_path.endswith(".gifv"):
        path = path[:-1]  # drop trailing "v"
    elif not lower_path.endswith(".gif"):
        if error_on_invalid:
            raise HTTPException(
                status_code=400,
                detail="GIF URL must end with .gif ('.gifv' is auto-converted).",
            )
        return None

    return urlunparse(parsed._replace(path=path))


def _cleanup_gif_urls_on_startup():
    """
    Drop GIF rows with unsupported URLs and normalize .gifv -> .gif.
    Runs once during application startup for both backends.
    """
    updated_ids: list[int] = []
    deleted_ids: list[int] = []
    try:
        if isinstance(db, PgGifDB):
            with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
                cur.execute("SELECT id, url FROM gifs ORDER BY id")
                rows = cur.fetchall()
                seen: set[str] = set()
                for row in rows:
                    normalized = _normalize_gif_url(row.get("url", ""), error_on_invalid=False)
                    if not normalized:
                        cur.execute("DELETE FROM gifs WHERE id=%s", (row["id"],))
                        deleted_ids.append(row["id"])
                        continue
                    key = normalized.lower()
                    if key in seen:
                        cur.execute("DELETE FROM gifs WHERE id=%s", (row["id"],))
                        deleted_ids.append(row["id"])
                        continue
                    seen.add(key)
                    if normalized != (row.get("url") or "").strip():
                        cur.execute("UPDATE gifs SET url=%s WHERE id=%s", (normalized, row["id"]))
                        updated_ids.append(row["id"])
                conn.commit()
        else:
            with db._connect() as conn:  # type: ignore[attr-defined]
                rows = conn.execute("SELECT id, url FROM gifs ORDER BY id").fetchall()
                seen: set[str] = set()
                for row in rows:
                    normalized = _normalize_gif_url(row["url"], error_on_invalid=False)
                    if not normalized:
                        conn.execute("DELETE FROM gifs WHERE id = ?", (row["id"],))
                        deleted_ids.append(row["id"])
                        continue
                    key = normalized.lower()
                    if key in seen:
                        conn.execute("DELETE FROM gifs WHERE id = ?", (row["id"],))
                        deleted_ids.append(row["id"])
                        continue
                    seen.add(key)
                    if normalized != (row["url"] or "").strip():
                        conn.execute("UPDATE gifs SET url = ? WHERE id = ?", (normalized, row["id"]))
                        updated_ids.append(row["id"])
                conn.commit()
    except Exception as exc:
        logger.warning("GIF URL cleanup failed: %s", exc)
        return

    if updated_ids or deleted_ids:
        logger.info(
            "GIF URL cleanup completed (updated=%d, deleted=%d)",
            len(updated_ids),
            len(deleted_ids),
        )

_cleanup_gif_urls_on_startup()


def _validate_gif_fields(
    title: str | None, anime: str | None, characters: List[str] | None, tags: List[str] | None
) -> tuple[str, str, List[str], List[str]]:
    clean_title = (title or "").strip()
    if not clean_title:
        raise HTTPException(status_code=400, detail="title must be provided")
    clean_anime = (anime or "").strip()
    if not clean_anime:
        raise HTTPException(status_code=400, detail="anime must be provided")
    clean_characters = _clean_str_list(characters)
    if not clean_characters:
        raise HTTPException(status_code=400, detail="characters must not be empty")
    clean_tags = _clean_str_list(tags)
    if not clean_tags:
        raise HTTPException(status_code=400, detail="tags must not be empty")
    return clean_title, clean_anime, clean_characters, clean_tags


def _ensure_gif_write_allowed(user: dict):
    if user.get("admin"):
        return
    if db.is_user_blacklisted(user["id"]):
        raise HTTPException(
            status_code=403,
            detail="You are blacklisted from creating, editing, or deleting GIFs.",
        )


def _ensure_gif_owner_or_admin(user: dict, gif: dict):
    _ensure_gif_write_allowed(user)
    if user.get("admin"):
        return
    if gif.get("created_by") != user["id"]:
        raise HTTPException(status_code=403, detail="You can only manage your own GIFs.")


def _maybe_award_gif_badges(user_id: int):
    if not GIF_BADGE_THRESHOLDS or not isinstance(db, PgGifDB):
        return
    thresholds = sorted(GIF_BADGE_THRESHOLDS, key=lambda t: t[0])
    try:
        gif_count = db.count_user_gifs(user_id)
    except Exception as e:
        logger.warning("Failed to count gifs for badge assignment: %s", e)
        return
    eligible = [(t, code) for t, code in thresholds if gif_count >= t]
    if not eligible:
        return
    top_threshold, top_code = eligible[-1]
    try:
        db.grant_icon(user_id, top_code, displayed=False)
    except KeyError:
        logger.warning("Configured GIF badge icon '%s' not found; skipping grant", top_code)
    except Exception as e:
        logger.warning("Failed to grant gif badge %s: %s", top_code, e)
    lower_codes = [code for threshold, code in thresholds if threshold < top_threshold]
    for code in lower_codes:
        try:
            db.revoke_icon(user_id, code)
        except Exception:
            continue


def _ensure_target_user_exists(user_id: int):
    if isinstance(db, PgGifDB):
        target = db.getUser(user_id)
        if not target:
            raise HTTPException(status_code=404, detail="User not found")
@app.get("/api/auth/verify")
def verify(token: str = Depends(require_token)):
    user = db.get_token_user(token) or {}

    # try to resolve slug if missing
    linktree_slug = user.get("linktree_slug")
    if not linktree_slug and user.get("linktree_id") and isinstance(db, PgGifDB):
        try:
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT slug FROM linktrees WHERE id=%s", (user["linktree_id"],)
                )
                row = cur.fetchone()
                if row:
                    linktree_slug = row[0]
        except Exception:
            linktree_slug = None

    user_out = {
        "id": user.get("id"),
        "username": user.get("username"),
        "admin": bool(user.get("admin", False)),
        "linktree_id": user.get("linktree_id"),
        "linktree_slug": linktree_slug,  # <-- add this
        "profile_picture": user.get("profile_picture"),
        "created_at": (
            user.get("created_at").isoformat()
            if isinstance(user.get("created_at"), datetime)
            else user.get("created_at")
        ),
        "updated_at": (
            user.get("updated_at").isoformat()
            if isinstance(user.get("updated_at"), datetime)
            else user.get("updated_at")
        ),
    }
    return {
        "ok": True,
        "expires_at": db.get_token_expiry(token),
        "user": user_out,
    }


@app.get("/api/linktrees/{linktree_id}/manage", response_model=LinktreeOut)
def get_linktree_manage(linktree_id: int, user: dict = Depends(require_user)):
    _ensure_pg()
    _require_tree_owner_or_admin(linktree_id, user)

    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        # Linktree-Stammdaten
        cur.execute(
            """
            SELECT id, user_id, slug, location, quote, song_url, song_icon_url, background_url,
                   COALESCE(show_audio_player, false) AS show_audio_player,
                   audio_player_bg_color,
                   COALESCE(audio_player_bg_alpha, 60) AS audio_player_bg_alpha,
                   audio_player_text_color,
                   audio_player_accent_color,
                   COALESCE(background_is_video, false) AS background_is_video,
                   COALESCE(transparency, 0)          AS transparency,
                   COALESCE(name_effect, 'none')       AS name_effect,
                   COALESCE(background_effect,'none')  AS background_effect,
                   device_type,
                   COALESCE(display_name_mode,'slug')  AS display_name_mode,
                    custom_display_name,
                    link_color,
                    link_bg_color,
                    COALESCE(link_bg_alpha, 100)        AS link_bg_alpha,
                    card_color,
                   text_color,
                   name_color,
                   location_color,
                   quote_color,
                   cursor_url,
                   COALESCE(discord_frame_enabled, false) AS discord_frame_enabled,
                    COALESCE(discord_presence_enabled, false) AS discord_presence_enabled,
                    COALESCE(discord_presence, 'online') AS discord_presence,
                    COALESCE(discord_status_enabled, false) AS discord_status_enabled,
                    discord_status_text,
                    COALESCE(discord_badges_enabled, false) AS discord_badges_enabled,
                    COALESCE(show_visit_counter, false) AS show_visit_counter,
                    visit_counter_color,
                    visit_counter_bg_color,
                    COALESCE(visit_counter_bg_alpha, 20) AS visit_counter_bg_alpha
               FROM linktrees
              WHERE id = %s
        """,
            (linktree_id,),
        )
        lt = cur.fetchone()
        if not lt:
            raise HTTPException(404, "Linktree not found")

        # Username + Avatar des Owners
        cur.execute(
            "SELECT username, profile_picture FROM users WHERE id=%s", (lt["user_id"],)
        )
        urow = cur.fetchone()
        user_username = urow["username"] if urow else None
        user_pfp = urow["profile_picture"] if urow else None

        # Links (UNGEFILTERT → auch is_active=false)
        cur.execute(
            """
            SELECT id, url, label, icon_url, position, is_active
              FROM linktree_links
             WHERE linktree_id = %s
             ORDER BY COALESCE(position, 0), id
        """,
            (linktree_id,),
        )
        links = cur.fetchall() or []

        # Icons (UNGEFILTERT → displayed kann true/false sein)
        cur.execute(
            """
            SELECT i.id, i.code, i.image_url, i.description,
                ui.displayed, ui.acquired_at
            FROM user_icons ui
            JOIN icons i ON i.id = ui.icon_id   -- <- HIER!
            WHERE ui.user_id = %s
            ORDER BY i.code
        """,
            (lt["user_id"],),
        )
        icons = cur.fetchall() or []

    visit_count = 0
    try:
        canonical_id = _get_canonical_linktree_id(lt.get("slug", "")) or lt["id"]
        visit_count = db.get_linktree_visit_count(canonical_id)
    except Exception:
        visit_count = 0

    discord_acct = _load_discord_account(lt["user_id"])
    discord_linked = bool(discord_acct)
    decoration_url = discord_acct.get("decoration_url") if discord_acct else None
    discord_badges = (
        _discord_badges_from_account(discord_acct) if discord_linked else []
    )
    presence_value, status_text = _resolve_discord_presence(lt, discord_acct)

    return {
        "id": lt["id"],
        "user_id": lt["user_id"],
        "slug": lt["slug"],
        "device_type": lt.get("device_type", "pc"),
        "location": lt.get("location"),
        "quote": lt.get("quote"),
        "song_url": lt.get("song_url"),
        "song_icon_url": lt.get("song_icon_url"),
        "background_url": lt.get("background_url"),
        "background_is_video": bool(lt.get("background_is_video")),
        "transparency": int(lt.get("transparency") or 0),
        "name_effect": lt.get("name_effect") or "none",
        "background_effect": lt.get("background_effect") or "none",
        "display_name_mode": lt.get("display_name_mode") or "slug",
        "custom_display_name": lt.get("custom_display_name"),
        "link_color": lt.get("link_color"),
        "link_bg_color": lt.get("link_bg_color"),
        "link_bg_alpha": int(lt.get("link_bg_alpha") or 100),
        "card_color": lt.get("card_color"),
        "text_color": lt.get("text_color"),
        "name_color": lt.get("name_color"),
        "location_color": lt.get("location_color"),
        "quote_color": lt.get("quote_color"),
        "cursor_url": lt.get("cursor_url"),
        "show_audio_player": bool(lt.get("show_audio_player", False)),
        "audio_player_bg_color": lt.get("audio_player_bg_color"),
        "audio_player_bg_alpha": int(lt.get("audio_player_bg_alpha", 60) or 60),
        "audio_player_text_color": lt.get("audio_player_text_color"),
        "audio_player_accent_color": lt.get("audio_player_accent_color"),
        "discord_frame_enabled": bool(lt.get("discord_frame_enabled", False)),
        "discord_decoration_url": decoration_url,
        "discord_presence_enabled": bool(lt.get("discord_presence_enabled", False)),
        "discord_presence": presence_value,
        "discord_status_enabled": bool(lt.get("discord_status_enabled", False)),
        "discord_status_text": status_text,
        "discord_badges_enabled": bool(lt.get("discord_badges_enabled", False)),
        "discord_badges": discord_badges if lt.get("discord_badges_enabled") else [],
        "discord_linked": discord_linked,
        "profile_picture": user_pfp,
        "user_username": user_username,
        "show_visit_counter": bool(lt.get("show_visit_counter", False)),
        "visit_count": int(visit_count or 0),
        "visit_counter_color": lt.get("visit_counter_color"),
        "visit_counter_bg_color": lt.get("visit_counter_bg_color"),
        "visit_counter_bg_alpha": int(lt.get("visit_counter_bg_alpha", 20) or 20),
        "links": [
            {
                "id": r["id"],
                "url": r["url"],
                "label": r.get("label"),
                "icon_url": r.get("icon_url"),
                "position": r.get("position", 0),
                "is_active": bool(r.get("is_active", True)),
            }
            for r in links
        ],
        "icons": [
            {
                "id": r["id"],
                "code": r["code"],
                "image_url": r["image_url"],
                "description": r.get("description"),
                "displayed": bool(r.get("displayed", False)),
                "acquired_at": (
                    r["acquired_at"].isoformat() if r.get("acquired_at") else None
                ),
            }
            for r in icons
        ],
    }


@app.get("/api/linktrees/by-slug/{slug}/manage", response_model=LinktreeOut)
def get_linktree_manage_by_slug(
    slug: str,
    device: DeviceType = Query("pc"),
    user: dict = Depends(require_user),
):
    _ensure_pg()
    lt = db.get_linktree_by_slug(slug, device)
    if not lt:
        raise HTTPException(404, "Linktree not found")
    if not (user.get("admin") or user["id"] == lt["user_id"]):
        raise HTTPException(403, "Forbidden (owner or admin only)")
    return get_linktree_manage(lt["id"], user)


@app.get("/api/gif/admin", response_class=HTMLResponse)
def admin_page():
    # Page is public, but actions on it are protected via the API endpoints.
    return FileResponse("gifApiAdmin.html")


@app.get("/admin", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
def admin_page():
    return FileResponse("admin.html")


@app.get("/api", response_class=HTMLResponse)
def root():
    return FileResponse("gifApiMain.html")


@app.get("/api/admin/gifs", dependencies=[Depends(require_admin)])
def admin_list_gifs(
    q: Optional[str] = Query("", description="Titel-Contains, leer = alle"),
    nsfw: str = Query("true", description="false|true|only"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    return db.search_by_title(q or "", nsfw_mode=nsfw, limit=limit, offset=offset)


@app.get("/api/my/gifs")
def list_my_gifs(
    q: Optional[str] = Query("", description="Title contains filter"),
    nsfw: str = Query("true", description="false|true|only"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user: dict = Depends(require_user),
):
    return db.search_user_gifs(
        user_id=user["id"],
        query=q or "",
        nsfw_mode=nsfw,
        limit=limit,
        offset=offset,
    )


@app.get("/api/admin/gif-blacklist", dependencies=[Depends(require_admin)])
def list_gif_blacklist():
    return db.list_gif_blacklist()


@app.get("/api/admin/gif-blacklist/{user_id}", dependencies=[Depends(require_admin)])
def get_gif_blacklist_entry(user_id: int):
    return {"user_id": user_id, "blacklisted": db.is_user_blacklisted(user_id)}


@app.post("/api/admin/gif-blacklist/{user_id}", dependencies=[Depends(require_admin)])
def add_gif_blacklist(user_id: int, payload: GifBlacklistIn | None = None):
    _ensure_target_user_exists(user_id)
    reason = payload.reason if payload else None
    db.add_to_gif_blacklist(user_id, reason)
    return {"user_id": user_id, "blacklisted": True, "reason": reason}


@app.delete("/api/admin/gif-blacklist/{user_id}", dependencies=[Depends(require_admin)])
def remove_gif_blacklist(user_id: int):
    _ensure_target_user_exists(user_id)
    db.remove_from_gif_blacklist(user_id)
    return {"user_id": user_id, "blacklisted": False}


@app.post("/api/auth/login", response_model=LoginOut)
def login(payload: LoginIn):
    if not ADMIN_PASSWORD:
        raise HTTPException(500, "Server not configured: GIFAPI_ADMIN_PASSWORD missing")
    if payload.password != ADMIN_PASSWORD:
        raise HTTPException(401, "Invalid password")

    admin = db.getUserByUsername(os.getenv("DEFAULT_ADMIN_USERNAME", "")) or None

    if not admin or not bool(admin.get("admin", False)):
        raise HTTPException(500, "No admin user to bind the session to")

    token = db.create_token(hours_valid=24, user_id=admin["id"])
    expires = db.get_token_expiry(token)
    return _session_response({"token": token, "expires_at": expires}, token)


@app.post("/api/auth/logout")
def logout(token: str = Depends(require_token)):
    db.revoke_token(token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("taoma_token")
    return resp


@app.get("/api/gifs")
def unified_get_gifs(
    q: Optional[str] = Query(None, description="Title contains (case-insensitive)"),
    tag: Optional[str] = Query(None, description="Pick random by tag"),
    anime: Optional[str] = Query(None, description="Pick random by anime"),
    character: Optional[str] = Query(None, description="Pick random by character"),
    list: Optional[str] = Query(
        None, alias="list", description="Use 'tags' to list all tags"
    ),
    nsfw: str = Query("false", description="false|true|only"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    nsfw_mode = (nsfw or "false").lower()
    if nsfw_mode not in {"false", "true", "only"}:
        raise HTTPException(
            status_code=400, detail="nsfw must be one of: false, true, only"
        )

    if list is not None:
        if list.lower() == "tags":
            return db.get_all_tags(nsfw_mode=nsfw_mode)
        raise HTTPException(
            status_code=400, detail="Unsupported list value. Use list=tags"
        )

    if q is not None:
        if any([tag, anime, character]):
            raise HTTPException(
                status_code=400, detail="Use either q OR one of tag/anime/character."
            )
        return db.search_by_title(q, nsfw_mode=nsfw_mode, limit=limit, offset=offset)

    filters_set = [x for x in [tag, anime, character] if x]
    if len(filters_set) > 1:
        raise HTTPException(
            status_code=400, detail="Use only one of tag, anime, or character."
        )

    try:
        if tag:
            return db.get_random_by_tag(tag, nsfw_mode=nsfw_mode)
        if anime:
            return db.get_random_by_anime(anime, nsfw_mode=nsfw_mode)
        if character:
            return db.get_random_by_character(character, nsfw_mode=nsfw_mode)
    except KeyError:
        if anime:
            suggestions = db.suggest_anime(anime, limit=5)
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"no gifs found for anime '{anime}'",
                    "suggestions": {"meintest_du": suggestions},
                },
            )
        if character:
            suggestions = db.suggest_character(character, limit=5)
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"no gifs found for character '{character}'",
                    "suggestions": {"meintest_du": suggestions},
                },
            )
        raise

    try:
        return db.get_random(nsfw_mode=nsfw_mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="no gifs in database")


@app.get(
    "/api/gifs/random/url",
    response_class=Response,
    summary="Get a random GIF URL only",
)
def random_gif_url(
    nsfw: str = Query("false", description="false|true|only"),
):
    """
    Returns only the URL (plain text) of a random GIF.
    """
    nsfw_mode = (nsfw or "false").lower()
    if nsfw_mode not in {"false", "true", "only"}:
        raise HTTPException(
            status_code=400, detail="nsfw must be one of: false, true, only"
        )
    try:
        gif = db.get_random(nsfw_mode=nsfw_mode)
    except KeyError:
        raise HTTPException(status_code=404, detail="no gifs in database")
    return Response(content=str(gif.get("url", "")), media_type="text/plain")


@app.post("/api/gifs", response_model=GifOut, status_code=201)
def create_or_update_gif(payload: GifIn, user: dict = Depends(require_user)):
    _ensure_gif_write_allowed(user)
    title, anime, characters, tags = _validate_gif_fields(
        payload.title, payload.anime, payload.characters, payload.tags
    )
    normalized_url = _normalize_gif_url(str(payload.url))
    try:
        existing = None
        try:
            existing = db.get_gif_by_url(normalized_url)
        except KeyError:
            pass

        if existing:
            _ensure_gif_owner_or_admin(user, existing)
            db.update_gif(
                existing["id"],
                title=title,
                url=normalized_url,
                nsfw=payload.nsfw,
                anime=anime,
                characters=characters,
                tags=tags,
            )
            return db.get_gif(existing["id"])

        gif_id = db.insert_gif(
            title=title,
            url=normalized_url,
            nsfw=payload.nsfw,
            anime=anime,
            characters=characters,
            tags=tags,
            created_by=user["id"],
        )
        _maybe_award_gif_badges(user["id"])
        return db.get_gif(gif_id)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/gifs/{gif_id}", response_model=GifOut)
def read_gif(gif_id: int):
    try:
        return db.get_gif(gif_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="GIF not found")


@app.patch("/api/gifs/{gif_id}", response_model=GifOut)
def update_gif(gif_id: int, payload: GifUpdate, user: dict = Depends(require_user)):
    try:
        gif = db.get_gif(gif_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="GIF not found")

    _ensure_gif_owner_or_admin(user, gif)
    title, anime, characters, tags = _validate_gif_fields(
        payload.title if payload.title is not None else gif.get("title"),
        payload.anime if payload.anime is not None else gif.get("anime"),
        payload.characters if payload.characters is not None else gif.get("characters", []),
        payload.tags if payload.tags is not None else gif.get("tags", []),
    )
    nsfw_value = payload.nsfw if payload.nsfw is not None else gif["nsfw"]
    url_value = (
        _normalize_gif_url(str(payload.url)) if payload.url is not None else None
    )

    db.update_gif(
        gif_id,
        title=title,
        url=url_value,
        nsfw=nsfw_value,
        anime=anime,
        characters=characters,
        tags=tags,
    )
    return db.get_gif(gif_id)


@app.delete("/api/gifs/{gif_id}", status_code=204)
def delete_gif(gif_id: int, user: dict = Depends(require_user)):
    try:
        gif = db.get_gif(gif_id)  # 404, falls nicht vorhanden
    except KeyError:
        raise HTTPException(status_code=404, detail="GIF not found")
    _ensure_gif_owner_or_admin(user, gif)
    db.delete_gif(gif_id)
    return Response(status_code=204)


@app.post("/api/visits/increment", status_code=204)
def incrementcount():
    db.incrementCounter()
    return Response(status_code=204)


@app.get("/api/visits/value", response_model=CounterOut)
def value():
    total = db.getCounterValue()
    return {"total": int(total or 0)}


@app.get("/user/{user_id}", response_model=UserOut)
def get_user(user_id: int, current: dict = Depends(require_user)):
    row = db.getUser(user_id)
    if not row:
        raise HTTPException(404, "User not found")
    if current["id"] != user_id and not bool(current.get("admin", False)):
        raise HTTPException(403, "Forbidden")
    row["admin"] = row.get("admin", False)
    for k in ("created_at", "updated_at"):
        if isinstance(row.get(k), datetime):
            row[k] = row[k].isoformat()
    return row


@app.post(
    "/user",
    response_model=UserOut,
    status_code=201,
    dependencies=[Depends(require_admin)],
)
def create_user(payload: UserCreateIn):
    try:
        new_id = db.createUser(
            username=payload.username.strip(),
            hashed_password=hashPassword(payload.password),
            linktree_id=payload.linktree_id,
            profile_picture=payload.profile_picture,
            admin=False,  # ← hart
        )
    except pg_errors.UniqueViolation:
        # Case-insensitive Unique-Index auf username → 409
        raise HTTPException(status_code=409, detail="username already exists")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to create user: {e}")

    row = db.getUser(new_id)
    # defensive postprocess
    if not row:
        raise HTTPException(status_code=500, detail="User created but not found")
    row["admin"] = row.get("admin", False)
    for k in ("created_at", "updated_at"):
        if isinstance(row.get(k), datetime):
            row[k] = row[k].isoformat()
    return row


@app.patch("/user/{user_id}", response_model=UserOut)
def update_user(
    user_id: int = PathParam(..., ge=1),
    payload: UserUpdateIn = ...,
    current: dict = Depends(require_user),  # liefert dict des eingeloggten Users
):
    target = db.getUser(user_id)
    if not target:
        raise HTTPException(404, "User not found")

    is_admin = bool(current.get("admin", False))
    is_self = current["id"] == user_id

    # Nicht-Admin darf nur sich selbst ändern und admin-Feld wird ignoriert
    if not is_admin and not is_self:
        raise HTTPException(403, "Forbidden")
    admin_value = payload.admin if is_admin else None

    try:
        db.updateUser(
            user_id,
            username=payload.username.strip() if payload.username is not None else None,
            password=(
                hashPassword(payload.password) if payload.password is not None else None
            ),
            linktree_id=payload.linktree_id,
            profile_picture=payload.profile_picture,
            admin=admin_value,  # nur Admin kann das setzen
        )
    except pg_errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="username already exists")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to update user: {e}")

    row = db.getUser(user_id)
    row["admin"] = row.get("admin", False)
    for k in ("created_at", "updated_at"):
        if isinstance(row.get(k), datetime):
            row[k] = row[k].isoformat()
    return row


@app.post("/api/auth/login_user", response_model=LoginOut)
def login_user(payload: LoginUserIn):
    user = db.getUserByUsername(payload.username.strip())
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not CheckPassword(user["password"], payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = db.create_token(hours_valid=24, user_id=user["id"])
    expires = db.get_token_expiry(token)
    return _session_response({"token": token, "expires_at": expires}, token)


@app.post("/api/auth/register", response_model=RegisterOut)
def register(payload: RegisterIn):
    uname = payload.username.strip()
    if db.getUserByUsername(uname):
        raise HTTPException(status_code=409, detail="username already exists")

    # hash
    hashed = hashPassword(payload.password)

    try:
        new_id = db.createUser(
            username=uname,
            hashed_password=hashed,
            linktree_id=payload.linktree_id,
            profile_picture=payload.profile_picture,
            admin=False,  # public signup → niemals admin
        )
    except pg_errors.UniqueViolation:
        raise HTTPException(status_code=409, detail="username already exists")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to create user: {e}")

    # auto-login
    token = db.create_token(hours_valid=24, user_id=new_id)
    exp = db.get_token_expiry(token)
    row = db.getUser(new_id)
    user_out = {
        "id": row["id"],
        "username": row["username"],
        "admin": bool(row.get("admin", False)),
        "linktree_id": row.get("linktree_id"),
        "profile_picture": row.get("profile_picture"),
        "created_at": (
            row.get("created_at").isoformat()
            if isinstance(row.get("created_at"), datetime)
            else row.get("created_at")
        ),
        "updated_at": (
            row.get("updated_at").isoformat()
            if isinstance(row.get("updated_at"), datetime)
            else row.get("updated_at")
        ),
    }
    return _session_response(
        {"token": token, "expires_at": exp, "user": user_out}, token
    )


@app.get("/register", include_in_schema=False)
def register_page():
    return FileResponse("register.html")


@app.get("/login", include_in_schema=False)
def login_page():
    return FileResponse("login.html")


@app.post("/admin/users/{user_id}/grant", dependencies=[Depends(require_admin)])
def grant_admin(user_id: int):
    db.updateUser(user_id, admin=True)
    return {"ok": True}


@app.get("/profile", include_in_schema=False)
def profile_page():
    return FileResponse("profile.html")


@app.patch("/api/users/me", response_model=UserOut)
def update_me(payload: MeUpdateIn, current: dict = Depends(require_user)):
    # Passwortwechsel: nur wenn new_password gesetzt ist → current_password nötig
    if payload.new_password is not None:
        if not payload.current_password:
            raise HTTPException(400, "current_password required to set new_password")
        u = db.getUser(current["id"])
        if not u or not CheckPassword(u["password"], payload.current_password):
            raise HTTPException(401, "Current password incorrect")
        # NEU: gleiches Passwort verhindern
        if CheckPassword(u["password"], payload.new_password):
            raise HTTPException(400, "New password must differ from current password")

    # Username-Kollisionsschutz (case-insensitiv) nur wenn er sich ändert
    if payload.username is not None:
        other = db.getUserByUsername(payload.username.strip())
        if other and other["id"] != current["id"]:
            raise HTTPException(409, "username already exists")

    try:
        db.updateUser(
            current["id"],
            username=payload.username.strip() if payload.username is not None else None,
            password=(
                hashPassword(payload.new_password)
                if payload.new_password is not None
                else None
            ),
            linktree_id=payload.linktree_id,
            profile_picture=payload.profile_picture,
            admin=None,  # niemals über /me
        )
    except Exception as e:
        raise HTTPException(502, f"Failed to update: {e}")

    row = db.getUser(current["id"])
    row["admin"] = bool(row.get("admin", False))
    for k in ("created_at", "updated_at"):
        if isinstance(row.get(k), datetime):
            row[k] = row[k].isoformat()
    return row


@app.get("/api/discord/status")
def discord_status(current: dict = Depends(require_user)):
    _ensure_pg()
    acct = _load_discord_account(current["id"])
    linked = bool(acct)
    badges = _discord_badges_from_account(acct) if linked else []
    presence_value, status_text = _resolve_discord_presence({}, acct if linked else None)

    return {
        "linked": linked,
        "configured": _discord_configured(),
        "discord_user_id": acct.get("discord_user_id") if linked else None,
        "username": acct.get("discord_username") if linked else None,
        "global_name": acct.get("discord_global_name") if linked else None,
        "decoration_url": acct.get("decoration_url") if linked else None,
        "badges": badges,
        "presence": presence_value if linked else "offline",
        "status_text": status_text if linked else None,
    }


@app.get("/api/discord/oauth-url")
def discord_oauth_url(current: dict = Depends(require_user)):
    _ensure_pg()
    if not _discord_configured():
        return {
            "configured": False,
            "url": None,
            "reason": "Discord linking not configured on server",
        }
    state = _new_discord_state(current["id"])
    scope_param = quote_plus(" ".join((DISCORD_OAUTH_SCOPES or "identify").split()))
    redirect = quote_plus(DISCORD_REDIRECT_URI)
    url = (
        "https://discord.com/oauth2/authorize"
        f"?client_id={quote_plus(DISCORD_CLIENT_ID)}"
        f"&response_type=code&redirect_uri={redirect}"
        f"&scope={scope_param}"
        f"&state={quote_plus(state)}"
        "&prompt=consent"
    )
    return {"url": url, "state": state, "configured": True}


@app.get("/api/discord/callback", include_in_schema=False)
async def discord_callback(code: str | None = None, state: str | None = None):
    if not code or not state:
        raise HTTPException(400, "Missing code or state")
    _ensure_pg()
    _ensure_discord_config()
    user_id = _pop_discord_state(state)
    if not user_id:
        raise HTTPException(400, "Invalid or expired state")

    token_payload = {
        "client_id": DISCORD_CLIENT_ID,
        "client_secret": DISCORD_CLIENT_SECRET,
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": DISCORD_REDIRECT_URI,
    }
    async with httpx.AsyncClient() as client:
        token_res = await client.post(
            "https://discord.com/api/oauth2/token",
            data=token_payload,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if token_res.status_code >= 400:
            raise HTTPException(token_res.status_code, token_res.text)
        token_json = token_res.json()
        access_token = token_json.get("access_token")
        refresh_token = token_json.get("refresh_token", "")
        expires_in = token_json.get("expires_in", 3600) or 3600
        scopes = token_json.get("scope") or DISCORD_OAUTH_SCOPES or "identify"

        user_res = await client.get(
            "https://discord.com/api/users/@me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if user_res.status_code >= 400:
            raise HTTPException(user_res.status_code, user_res.text)
        data = user_res.json()

    decoration = data.get("avatar_decoration_data") or data.get("avatar_decoration")
    public_flags = data.get("public_flags") or data.get("flags") or 0
    premium_type = data.get("premium_type") or 0
    token_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))
    decoration_json = json.dumps(decoration) if decoration is not None else None
    db.upsert_discord_account(
        user_id=user_id,
        discord_user_id=data.get("id"),
        discord_username=data.get("username"),
        discord_global_name=data.get("global_name"),
        avatar_hash=data.get("avatar"),
        avatar_decoration=decoration_json,
        public_flags=int(public_flags or 0),
        premium_type=int(premium_type or 0),
        access_token=access_token,
        refresh_token=refresh_token,
        token_expires_at=token_expires_at,
        scopes=scopes,
    )

    html_doc = """
<!doctype html>
<html><body style="font-family:system-ui;padding:16px;text-align:center;">
<p>Discord connected. You can close this window.</p>
<script>
  try {
    if (window.opener) {
      window.opener.postMessage({ source: 'taoma', type: 'discord-linked', ok: true }, '*');
    }
  } catch (e) {}
  window.close();
</script>
</body></html>
"""
    return HTMLResponse(content=html_doc)


@app.delete("/api/discord/account")
def discord_unlink(current: dict = Depends(require_user)):
    _ensure_pg()
    try:
        db.delete_discord_account(current["id"])
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE linktrees
                   SET discord_frame_enabled = FALSE,
                       discord_presence_enabled = FALSE,
                       discord_status_enabled = FALSE,
                       discord_badges_enabled = FALSE,
                       updated_at = NOW()
                 WHERE user_id=%s
                """,
                (current["id"],),
            )
            conn.commit()
    except Exception as e:
        raise HTTPException(502, f"Failed to unlink Discord: {e}")
    return {"ok": True}


HEIC_BRANDS = {
    b"ftypheic",
    b"ftypheix",
    b"ftyphevc",
    b"ftyphevx",
    b"ftypheif",
    b"ftypmif1",
    b"ftypmsf1",
}


def _detect_image_ext(b: bytes) -> str:
    """
    Verifiziert Image-Bytes (inkl. HEIC/HEIF Header) und gibt die Zielendung zurück.
    Keine Re-Encodierung, nur Header-Check.
    """
    fmt = ""
    try:
        with Image.open(io.BytesIO(b)) as img:
            img.verify()  # prüft Header & Konsistenz
            fmt = (img.format or "").upper()
    except UnidentifiedImageError:
        fmt = ""
    except Exception:
        fmt = ""

    if fmt == "JPEG":
        return "jpg"
    if fmt in {"PNG", "WEBP", "GIF"}:
        return fmt.lower()

    # Pillow ohne heif-Plugin -> selbst auf HEIC/HEIF Header prüfen
    if len(b) >= 12 and b[4:12] in HEIC_BRANDS:
        return "heic"
    return ""


def _detect_video_ext(b: bytes, content_type: str | None) -> str:
    """Sehr defensiver Container-Check für MP4/MOV/WEBM, keine vollständige Demux."""
    ct = (content_type or "").lower()
    if ct == "video/webm":
        if b.startswith(b"\x1a\x45\xdf\xa3"):
            return "webm"
        return ""

    if ct in {"video/mp4", "video/quicktime"}:
        if len(b) < 12 or b[4:8] != b"ftyp":
            return ""
        brand = b[8:12]
        if brand in {b"isom", b"iso2", b"avc1", b"mp41", b"mp42"}:
            return "mp4"
        if brand in {b"qt  ", b"MSNV"}:
            return "mov"  # QuickTime / MOV
        # Manche MOVs haben andere Brands; fallback: treat as mp4 if MP4 CT
        if ct == "video/mp4":
            return "mp4"
    return ""


def _looks_like_video_url(url: str | None) -> bool:
    if not url:
        return False
    lower = str(url).lower()
    return any(ext in lower for ext in VIDEO_EXTENSIONS)


@app.post("/api/users/me/avatar", response_model=UserOut)
async def upload_avatar(
    file: UploadFile = File(...), current: dict = Depends(require_user)
):
    if file.content_type not in ALLOWED_IMAGE_CT:
        raise HTTPException(415, "Unsupported media type")
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "File too large (max 5MB)")

    ext = _detect_image_ext(data)
    if not ext:
        raise HTTPException(400, "File is not a valid image")

    # Altes Bild merken (für Cleanup)
    old_url = None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT profile_picture FROM users WHERE id=%s", (current["id"],)
            )
            row = cur.fetchone()
            old_url = row[0] if row else None
    except Exception:
        old_url = None

    fname = f"user{current['id']}_{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"

    db.updateUser(current["id"], profile_picture=url)

    # Cleanup (nach erfolgreichem Update)
    if old_url and old_url != url:
        _delete_if_unreferenced(old_url)

    row = db.getUser(current["id"])
    row["admin"] = bool(row.get("admin", False))
    for k in ("created_at", "updated_at"):
        if isinstance(row.get(k), datetime):
            row[k] = row[k].isoformat()
    return row


@app.get("/api/avatars")
def list_avatars():
    base = pathlib.Path("static/avatars")
    if not base.exists():
        # Fallback: leere Liste
        return []
    # nur übliche Endungen
    items = []
    for p in sorted(base.glob("*")):
        if (
            p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}
            and p.is_file()
        ):
            items.append(f"/static/avatars/{p.name}")
    return items


@app.get("/linktree/config", include_in_schema=False)
def linktree_config_page():
    return FileResponse("linktree_config.html")


# ---------- Linktree: public read ----------


@app.get("/api/linktrees/{slug}", response_model=LinktreeOut)
def get_linktree(
    slug: str,
    device: DeviceType = Query("pc", description="pc or mobile"),
    request: Request = None,
    response: Response = None,
):
    _ensure_pg()
    lt = db.get_linktree_by_slug(slug, device)
    if not lt and device == "mobile":
        lt = db.get_linktree_by_slug(slug, "pc")  # fallback
        device = "pc"
    if not lt:
        raise HTTPException(404, "Linktree not found")

    # Username & Avatar des Owners holen
    user_username = None
    user_pfp = None
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT username, profile_picture FROM users WHERE id=%s", (lt["user_id"],)
        )
        row = cur.fetchone()
        if row:
            user_username, user_pfp = row[0], row[1]
    decoration_url = None
    discord_linked = False
    discord_badges = []
    acct = None
    try:
        acct = _load_discord_account(lt["user_id"])
        discord_linked = bool(acct)
        decoration_url = acct.get("decoration_url") if acct else None
        if discord_linked:
            discord_badges = _discord_badges_from_account(acct)
    except Exception:
        decoration_url = None
    presence_value, status_text = _resolve_discord_presence(lt, acct if discord_linked else None)

    icons = [
        {
            "id": i["id"],
            "code": i["code"],
            "image_url": i["image_url"],
            "description": i.get("description"),
            "displayed": i.get("displayed", False),
            "acquired_at": i.get("acquired_at"),
        }
        for i in lt["icons"]
        if i.get("displayed")
    ]

    links = [
        {
            "id": r["id"],
            "url": r["url"],
            "label": r.get("label"),
            "icon_url": r.get("icon_url"),
            "position": r.get("position", 0),
            "is_active": r.get("is_active", True),
        }
        for r in lt["links"]
        if r.get("is_active", True)
    ]
    frame_enabled = bool(lt.get("discord_frame_enabled", False))
    visit_count = 0
    if request is not None and response is not None:
        visit_count = _record_linktree_visit(lt, request, response)
    else:
        try:
            if isinstance(db, PgGifDB):
                canonical_id = _get_canonical_linktree_id(lt.get("slug", "")) or lt["id"]
                visit_count = db.get_linktree_visit_count(canonical_id)
            else:
                visit_count = 0
        except Exception:
            visit_count = 0

    return {
        "id": lt["id"],
        "user_id": lt["user_id"],
        "slug": lt["slug"],
        "device_type": lt.get("device_type", "pc"),
        "location": lt.get("location"),
        "quote": lt.get("quote"),
        "song_url": lt.get("song_url"),
        "song_icon_url": lt.get("song_icon_url"),
        "show_audio_player": bool(lt.get("show_audio_player", False)),
        "audio_player_bg_color": lt.get("audio_player_bg_color"),
        "audio_player_bg_alpha": int(lt.get("audio_player_bg_alpha", 60) or 60),
        "audio_player_text_color": lt.get("audio_player_text_color"),
        "audio_player_accent_color": lt.get("audio_player_accent_color"),
        "background_url": lt.get("background_url"),
        "background_is_video": lt.get("background_is_video", False),
        "transparency": lt.get("transparency", 0),
        "name_effect": lt.get("name_effect", "none"),
        "background_effect": lt.get("background_effect", "none"),
        "display_name_mode": lt.get("display_name_mode", "slug"),
        "custom_display_name": lt.get("custom_display_name"),
        "link_color": lt.get("link_color"),
        "link_bg_color": lt.get("link_bg_color"),
        "link_bg_alpha": lt.get("link_bg_alpha", 100),
        "card_color": lt.get("card_color"),
        "text_color": lt.get("text_color"),
        "name_color": lt.get("name_color"),
        "location_color": lt.get("location_color"),
        "quote_color": lt.get("quote_color"),
        "cursor_url": lt.get("cursor_url"),
        "discord_frame_enabled": frame_enabled,
        "discord_decoration_url": decoration_url if frame_enabled else None,
        "discord_presence_enabled": bool(lt.get("discord_presence_enabled", False)),
        "discord_presence": presence_value,
        "discord_status_enabled": bool(lt.get("discord_status_enabled", False)),
        "discord_status_text": status_text,
        "discord_badges_enabled": bool(lt.get("discord_badges_enabled", False)),
        "discord_badges": discord_badges if lt.get("discord_badges_enabled") else [],
        "discord_linked": discord_linked,
        "profile_picture": user_pfp,  # <= jetzt dabei
        "user_username": user_username,  # <= jetzt dabei
        "show_visit_counter": bool(lt.get("show_visit_counter", False)),
        "visit_count": int(visit_count or 0),
        "visit_counter_color": lt.get("visit_counter_color"),
        "visit_counter_bg_color": lt.get("visit_counter_bg_color"),
        "visit_counter_bg_alpha": int(lt.get("visit_counter_bg_alpha", 20) or 20),
        "links": links,
        "icons": icons,
    }


# Optional: HTML-Seite (statisches Template, das o.g. API nutzt)
@app.get("/tree/{slug}", include_in_schema=False)
def linktree_page(slug: str):
    return FileResponse("linktree.html")


# ---------- Linktree: create/update (owner/admin) ----------


@app.post(
    "/api/linktrees", response_model=LinktreeOut, dependencies=[Depends(require_user)]
)
def create_linktree_ep(payload: LinktreeCreateIn, user: dict = Depends(require_user)):
    _ensure_pg()
    # Ein Tree pro User + device_type
    existing = db.get_linktree_by_slug(payload.slug, payload.device_type)
    if existing and existing["user_id"] == user["id"]:
        raise HTTPException(409, "You already have a linktree with this slug and device")
    # globale Slug-Kollision prüfen (andere User)
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT user_id FROM linktrees WHERE lower(slug)=lower(%s) AND user_id<>%s LIMIT 1", (payload.slug, user["id"]))
        if cur.fetchone():
            raise HTTPException(409, "Slug already in use")
    try:
        bg_is_video = (
            _looks_like_video_url(payload.background_url)
            if payload.background_url
            else payload.background_is_video
        )
        linktree_id = db.create_linktree(
            user_id=user["id"],
            slug=payload.slug,
            device_type=payload.device_type,
            location=payload.location,
            quote=payload.quote,
            song_url=payload.song_url,
            song_icon_url=payload.song_icon_url,
            show_audio_player=payload.show_audio_player,
            audio_player_bg_color=payload.audio_player_bg_color,
            audio_player_bg_alpha=payload.audio_player_bg_alpha,
            audio_player_text_color=payload.audio_player_text_color,
            audio_player_accent_color=payload.audio_player_accent_color,
            background_url=payload.background_url,
            background_is_video=bg_is_video,
            transparency=payload.transparency,
            name_effect=payload.name_effect,
            background_effect=payload.background_effect,
            display_name_mode=payload.display_name_mode,  # <-- NEU
            custom_display_name=payload.custom_display_name,
            link_color=payload.link_color,
            link_bg_color=payload.link_bg_color,
            link_bg_alpha=payload.link_bg_alpha,
            card_color=payload.card_color,
            text_color=payload.text_color,
            name_color=payload.name_color,
            location_color=payload.location_color,
            quote_color=payload.quote_color,
            cursor_url=payload.cursor_url,
            discord_frame_enabled=payload.discord_frame_enabled,
            discord_presence_enabled=payload.discord_presence_enabled,
            discord_presence=payload.discord_presence,
            discord_status_enabled=payload.discord_status_enabled,
            discord_status_text=payload.discord_status_text,
            discord_badges_enabled=payload.discord_badges_enabled,
            show_visit_counter=payload.show_visit_counter,
            visit_counter_color=payload.visit_counter_color,
            visit_counter_bg_color=payload.visit_counter_bg_color,
            visit_counter_bg_alpha=payload.visit_counter_bg_alpha,
        )
    except pg_errors.UniqueViolation:
        raise HTTPException(409, "Slug already in use")
    lt = db.get_linktree_by_slug(payload.slug, payload.device_type)
    # public shaping wie oben:
    return get_linktree(payload.slug, device=payload.device_type)


@app.post("/api/linktrees/{slug}/clone", response_model=LinktreeOut)
def clone_linktree_variant(
    slug: str,
    target_device: DeviceType = Query("mobile"),
    user: dict = Depends(require_user),
):
    _ensure_pg()
    if target_device not in ("pc", "mobile"):
        raise HTTPException(400, "Invalid target device")
    source_device = "pc" if target_device == "mobile" else "mobile"
    src = db.get_linktree_by_slug(slug, source_device)
    if not src:
        raise HTTPException(404, "Source linktree not found")
    if not (user.get("admin") or user["id"] == src["user_id"]):
        raise HTTPException(403, "Forbidden (owner or admin only)")
    try:
        new_id = db.clone_linktree_variant(slug, source_device, target_device, src["user_id"])
    except ValueError as ve:
        raise HTTPException(409, str(ve))
    except KeyError:
        raise HTTPException(404, "Source linktree not found")
    return get_linktree_manage(new_id, user)


@app.delete("/api/linktrees/{linktree_id}")
def delete_linktree_ep(linktree_id: int, user: dict = Depends(require_user)):
    _ensure_pg()
    _require_tree_owner_or_admin(linktree_id, user)

    # Sammle Owner und evtl. Medien-URLs (für Cleanup)
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT user_id, song_url, background_url, cursor_url FROM linktrees WHERE id=%s",
            (linktree_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Linktree not found")

        owner_id = int(row["user_id"])
        song_url = row.get("song_url")
        bg_url = row.get("background_url")
        cursor_url = row.get("cursor_url")

        # Icon-URLs der Links sammeln (können lokale Medien sein)
        cur.execute(
            "SELECT icon_url FROM linktree_links WHERE linktree_id=%s AND icon_url IS NOT NULL",
            (linktree_id,),
        )
        icon_urls = [r["icon_url"] for r in cur.fetchall() or [] if r.get("icon_url")]

        # 1) Links löschen
        cur.execute("DELETE FROM linktree_links WHERE linktree_id=%s", (linktree_id,))
        # 2) Verknüpfung beim Owner lösen, falls gesetzt
        cur.execute(
            "UPDATE users SET linktree_id = NULL WHERE id=%s AND linktree_id=%s",
            (owner_id, linktree_id),
        )
        # 3) Linktree löschen
        cur.execute("DELETE FROM linktrees WHERE id=%s", (linktree_id,))
        conn.commit()

    # Medien aufräumen (nur wenn nirgendwo mehr referenziert)
    for url in [song_url, bg_url, cursor_url, *icon_urls]:
        if url:
            _delete_if_unreferenced(url)

    return {"ok": True}


# ---------- Links: add/update/delete (owner/admin) ----------


@app.post("/api/linktrees/{linktree_id}/links", response_model=LinkOut)
def add_link_ep(
    linktree_id: int, payload: LinkCreateIn, user: dict = Depends(require_user)
):
    _ensure_pg()
    _require_tree_owner_or_admin(linktree_id, user)
    link_id = db.add_link(
        linktree_id,
        url=str(payload.url),
        label=payload.label,
        icon_url=payload.icon_url,
        position=payload.position,
        is_active=payload.is_active,
    )
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, url, label, icon_url, position, is_active FROM linktree_links WHERE id=%s",
            (link_id,),
        )
        r = cur.fetchone()
    return {
        "id": r["id"],
        "url": r["url"],
        "label": r["label"],
        "icon_url": r["icon_url"],
        "position": r["position"],
        "is_active": r["is_active"],
    }


@app.patch("/api/links/{link_id}", response_model=dict)
def update_link_ep(
    link_id: int, payload: LinkUpdateIn, user: dict = Depends(require_user)
):
    _ensure_pg()
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT lt.user_id, l.linktree_id, l.icon_url
              FROM linktree_links l
              JOIN linktrees lt ON lt.id = l.linktree_id
             WHERE l.id = %s
        """,
            (link_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Link not found")
        owner_id, linktree_id, old_icon = (
            int(row["user_id"]),
            int(row["linktree_id"]),
            row["icon_url"],
        )

    if not (user.get("admin") or user["id"] == owner_id):
        raise HTTPException(403, "Forbidden (owner or admin only)")

    # Werde die neue icon_url gleich aus dem Payload holen
    new_icon = payload.icon_url if payload.icon_url is not None else old_icon

    fields = payload.model_dump(exclude_unset=True)
    if "background_url" in fields:
        fields["background_is_video"] = _looks_like_video_url(
            fields.get("background_url")
        )
    if "url" in fields and fields["url"] is not None:
        fields["url"] = str(fields["url"])

    db.update_link(link_id, **fields)

    # Cleanup, wenn sich icon_url geändert hat
    if old_icon and old_icon != new_icon:
        _delete_if_unreferenced(old_icon)

    return {"ok": True}


@app.delete("/api/links/{link_id}", status_code=204)
def delete_link_ep(link_id: int, user: dict = Depends(require_user)):
    _ensure_pg()
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT lt.user_id
              FROM linktree_links l
              JOIN linktrees lt ON lt.id = l.linktree_id
             WHERE l.id = %s
        """,
            (link_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Link not found")
        owner_id = int(row[0])
    if not (user.get("admin") or user["id"] == owner_id):
        raise HTTPException(403, "Forbidden (owner or admin only)")
    db.delete_link(link_id)
    return Response(status_code=204)


# ---------- Icon-Katalog & Besitz ----------


@app.get("/api/icons", response_model=List[IconOut])
def list_icons():
    _ensure_pg()
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute("SELECT id, code, image_url, description FROM icons ORDER BY code")
        rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "code": r["code"],
            "image_url": r["image_url"],
            "description": r.get("description"),
        }
        for r in rows
    ]


@app.post("/api/icons", dependencies=[Depends(require_admin)], response_model=IconOut)
def upsert_icon_ep(payload: IconUpsertIn):
    _ensure_pg()
    icon_id = db.upsert_icon(payload.code, payload.image_url, payload.description)
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, code, image_url, description FROM icons WHERE id=%s", (icon_id,)
        )
        r = cur.fetchone()
    return {
        "id": r["id"],
        "code": r["code"],
        "image_url": r["image_url"],
        "description": r.get("description"),
    }


@app.post("/api/users/{user_id}/icons/{code}", dependencies=[Depends(require_admin)])
def grant_icon_ep(user_id: int, code: str, body: GrantIconIn):
    _ensure_pg()
    db.grant_icon(user_id, code, displayed=body.displayed)
    return {"ok": True}


@app.patch("/api/users/me/icons/{code}")
def toggle_my_icon_displayed(
    code: str, body: ToggleDisplayedIn, me: dict = Depends(require_user)
):
    _ensure_pg()
    db.set_icon_displayed(me["id"], code, body.displayed)
    return {"ok": True}


@app.post("/api/users/me/song")
async def upload_song(
    file: UploadFile = File(...),
    current: dict = Depends(require_user),
    device: DeviceType | None = Query(None, description="pc or mobile"),
):
    _ensure_pg()
    if file.content_type not in ALLOWED_AUDIO:
        raise HTTPException(415, "Unsupported audio type")
    data = await file.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "File too large (max 15MB)")
    target_device = device or "pc"

    old_url = None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT lt.song_url
                  FROM linktrees lt
                 WHERE lt.user_id = %s AND lt.device_type = %s
            """,
                (current["id"], target_device),
            )
            row = cur.fetchone()
            old_url = row[0] if row else None
            if row is None:
                raise HTTPException(404, "Linktree for device not found")
    except HTTPException:
        raise
    except Exception:
        old_url = None

    audio_ext = {
        "audio/mpeg": "mp3",
        "audio/ogg": "ogg",
        "audio/wav": "wav",
    }.get(file.content_type)
    if not audio_ext:
        raise HTTPException(415, "Unsupported audio type")

    fname = f"user{current['id']}_song_{uuid.uuid4().hex}.{audio_ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"
    updated = db.update_linktree_by_user_and_device(
        current["id"], target_device, song_url=url
    )
    if not updated:
        raise HTTPException(404, "Linktree for device not found")

    # Cleanup
    if old_url and old_url != url:
        _delete_if_unreferenced(old_url)

    return {"url": url}


@app.post("/api/users/me/songicon")
async def upload_song_icon(
    file: UploadFile = File(...),
    current: dict = Depends(require_user),
    device: DeviceType | None = Query(None, description="pc or mobile"),
):
    _ensure_pg()
    if file.content_type not in ALLOWED_IMAGE_CT:
        raise HTTPException(415, "Unsupported image type")
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "File too large (max 5MB)")
    ext = _detect_image_ext(data)
    if not ext:
        raise HTTPException(400, "File is not a valid image")
    target_device = device or "pc"

    old_url = None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT lt.song_icon_url
                  FROM linktrees lt
                 WHERE lt.user_id = %s AND lt.device_type = %s
            """,
                (current["id"], target_device),
            )
            row = cur.fetchone()
            old_url = row[0] if row else None
            if row is None:
                raise HTTPException(404, "Linktree for device not found")
    except HTTPException:
        raise
    except Exception:
        old_url = None

    fname = f"user{current['id']}_songicon_{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"

    updated = db.update_linktree_by_user_and_device(
        current["id"], target_device, song_icon_url=url
    )
    if not updated:
        raise HTTPException(404, "Linktree for device not found")

    if old_url and old_url != url:
        _delete_if_unreferenced(old_url)

    return {"url": url}


@app.post("/api/users/me/background")
async def upload_background(
    file: UploadFile = File(...),
    current: dict = Depends(require_user),
    device: DeviceType | None = Query(None, description="pc or mobile"),
):
    _ensure_pg()
    ct = file.content_type or ""
    if ct not in ALLOWED_IMAGE_CT and ct not in ALLOWED_VIDEO_CT:
        raise HTTPException(415, "Unsupported background type")
    data = await file.read()
    if len(data) > MAX_BACKGROUND_BYTES:
        raise HTTPException(413, "File too large (max 50MB)")
    target_device = device or "pc"

    image_ext = _detect_image_ext(data)
    video_ext = "" if image_ext else _detect_video_ext(data, ct)
    if not image_ext and not video_ext:
        raise HTTPException(400, "File is not a supported image or video")

    # Altes BG merken
    old_url = None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT lt.background_url
                  FROM linktrees lt
                 WHERE lt.user_id = %s AND lt.device_type = %s
            """,
                (current["id"], target_device),
            )
            row = cur.fetchone()
            old_url = row[0] if row else None
            if row is None:
                raise HTTPException(404, "Linktree for device not found")
    except HTTPException:
        raise
    except Exception:
        old_url = None

    ext = image_ext or video_ext
    fname = f"user{current['id']}_bg_{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"
    is_video = bool(video_ext)

    updated = db.update_linktree_by_user_and_device(
        current["id"],
        target_device,
        background_url=url,
        background_is_video=is_video,
    )
    if not updated:
        raise HTTPException(404, "Linktree for device not found")

    # Cleanup
    if old_url and old_url != url:
        _delete_if_unreferenced(old_url)

    return {"url": url, "is_video": is_video}


@app.post("/api/users/me/cursor")
async def upload_cursor(
    file: UploadFile = File(...),
    current: dict = Depends(require_user),
    device: DeviceType | None = Query(None, description="pc or mobile"),
):
    _ensure_pg()
    target_device = device or "pc"
    if target_device != "pc":
        raise HTTPException(400, "Custom cursor is only available for desktop linktrees")
    if file.content_type not in ALLOWED_IMAGE_CT:
        raise HTTPException(415, "Unsupported image type for cursor")
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "File too large (max 5MB)")
    ext = _detect_image_ext(data)
    if not ext:
        raise HTTPException(400, "File is not a valid image")
    # Cursor-Kompatibilität: auf PNG normalisieren, falls anderes Format (z.B. WebP)
    if ext != "png":
        try:
            with Image.open(io.BytesIO(data)) as img:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                data = buf.getvalue()
                ext = "png"
                if len(data) > MAX_IMAGE_BYTES:
                    raise HTTPException(413, "Converted cursor is too large (>5MB)")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(400, "Cursor could not be converted to PNG")

    old_url = None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT cursor_url FROM linktrees WHERE user_id=%s AND device_type=%s",
                (current["id"], target_device),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(404, "Linktree for device not found")
            old_url = row[0]
    except HTTPException:
        raise
    except Exception:
        old_url = None

    fname = f"user{current['id']}_cursor_{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"

    updated = db.update_linktree_by_user_and_device(
        current["id"], target_device, cursor_url=url
    )
    if not updated:
        raise HTTPException(404, "Linktree for device not found")

    if old_url and old_url != url:
        _delete_if_unreferenced(old_url)

    return {"url": url}


@app.post("/api/users/me/linkicon")
async def upload_linkicon(
    file: UploadFile = File(...), current: dict = Depends(require_user)
):
    if file.content_type not in ALLOWED_IMAGE_CT:
        raise HTTPException(415, "Unsupported image type")
    data = await file.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "File too large (max 5MB)")
    ext = _detect_image_ext(data)
    if not ext:
        raise HTTPException(400, "File is not a valid image")
    fname = f"user{current['id']}_icon_{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"
    return {"url": url}


@app.api_route(
    "/api/admin/media/gc",
    methods=["POST", "GET"],
    dependencies=[Depends(require_admin)],
)
def media_garbage_collect(min_age_seconds: int = Query(60, ge=0)):
    """Löscht unreferenzierte Dateien im Upload-Verzeichnis, die älter als min_age_seconds sind."""
    return _run_media_gc(min_age_seconds, require_templates=True)


@app.get("/nanna/birthdays/2025", include_in_schema=False)
def home():
    return FileResponse("nannaBirthday2025.html")




@app.get("/alexandra/data/show", dependencies=[Depends(require_specific_user)])
def very_private_page():
    return FileResponse("healthData.html")


# ---------- HealthData CRUD ----------

@app.get("/alexandra/data/health", response_model=List[HealthDataOut], dependencies=[Depends(require_specific_user)])
def list_health_data_ep(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    _ensure_pg()
    rows = db.list_health_data(limit=limit, offset=offset)
    return [_health_row_to_out(r) for r in rows]


@app.get("/alexandra/data/health/{data_id}", response_model=HealthDataOut, dependencies=[Depends(require_specific_user)])
def get_health_data_ep(data_id: int):
    _ensure_pg()
    row = db.get_health_data(data_id)
    if not row:
        raise HTTPException(404, "Health data not found")
    return _health_row_to_out(row)


@app.get("/alexandra/data/health/by-day/{day}", response_model=HealthDataOut, dependencies=[Depends(require_specific_user)])
def get_health_data_by_day_ep(day: date):
    _ensure_pg()
    row = db.get_health_data_by_day(str(day))
    if not row:
        raise HTTPException(404, f"No health data for {day}")
    return _health_row_to_out(row)


@app.post("/alexandra/data/health", response_model=HealthDataOut, status_code=201, dependencies=[Depends(require_specific_user)])
def create_health_data_ep(payload: HealthDataIn):
    _ensure_pg()
    # Einfügen
    new_id = db.insert_health_data(
        day=str(payload.day),
        borg=payload.borg,
        temperatur=payload.temperatur,
        erschöpfung=payload.erschoepfung,
        muskelschwäche=payload.muskelschwaeche,
        schmerzen=payload.schmerzen,
        angst=payload.angst,
        konzentration=payload.konzentration,
        husten=payload.husten,
        atemnot=payload.atemnot,
        mens=payload.mens,
        notizen=payload.notizen,
        other=payload.other,
    )
    row = db.get_health_data(new_id)
    return _health_row_to_out(row)


@app.patch("/alexandra/data/health/{data_id}", response_model=HealthDataOut, dependencies=[Depends(require_specific_user)])
def update_health_data_ep(data_id: int, payload: HealthDataUpdate):
    _ensure_pg()
    if not db.get_health_data(data_id):
        raise HTTPException(404, "Health data not found")

    # Nur gesetzte Felder an DB weitergeben
    fields = payload.model_dump(exclude_unset=True, by_alias=True)
    # Mapping von ASCII → DB-Spalten (nur falls ASCII verwendet wurde)
    if "erschoepfung" in fields and "erschöpfung" not in fields:
        fields["erschöpfung"] = fields.pop("erschoepfung")
    if "muskelschwaeche" in fields and "muskelschwäche" not in fields:
        fields["muskelschwäche"] = fields.pop("muskelschwaeche")

    db.update_health_data(data_id, **fields)
    row = db.get_health_data(data_id)
    return _health_row_to_out(row)


@app.delete("/alexandra/data/health/{data_id}", status_code=204, dependencies=[Depends(require_specific_user)])
def delete_health_data_ep(data_id: int):
    _ensure_pg()
    if not db.get_health_data(data_id):
        raise HTTPException(404, "Health data not found")
    db.delete_health_data(data_id)
    return Response(status_code=204)

class GlobalChatManager:
    def __init__(self) -> None:
        self._connections: dict[WebSocket, str] = {}
        self._next_id = 1
        self._id_lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        async with self._id_lock:
            user_id = f"user{self._next_id}"
            self._next_id += 1
        self._connections[websocket] = user_id
        logger.info("Global chat client connected: %s", user_id)
        await websocket.send_text("Welcome to the WebSocket server!")
        await self.broadcast(f"New user connected: {user_id}", sender=websocket)
        return user_id

    async def disconnect(self, websocket: WebSocket) -> None:
        user_id = self._connections.pop(websocket, None)
        if user_id:
            logger.info("Global chat client disconnected: %s", user_id)
            await self.broadcast(f"User left: {user_id}", sender=websocket)

    async def broadcast(self, message: str, sender: WebSocket | None = None) -> None:
        for ws in list(self._connections):
            if ws is sender:
                continue
            try:
                await ws.send_text(message)
            except Exception as exc:
                logger.warning("Failed to send chat broadcast: %s", exc)
                self._connections.pop(ws, None)

    async def send_personal_message(self, message: str, websocket: WebSocket) -> None:
        try:
            await websocket.send_text(message)
        except Exception as exc:
            logger.warning("Failed to send chat message to client: %s", exc)
            self._connections.pop(websocket, None)


chat_manager = GlobalChatManager()


@app.websocket("/ws/global-chat")
async def global_chat_socket(websocket: WebSocket) -> None:
    user_id: str | None = None
    try:
        user_id = await chat_manager.connect(websocket)
        while True:
            text = await websocket.receive_text()
            text = text.strip()
            if not text:
                continue
            await chat_manager.broadcast(f"{user_id}: {text}", sender=websocket)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("Global chat websocket error: %s", exc)
        await websocket.close(code=1011)
    finally:
        await chat_manager.disconnect(websocket)

@app.get("/globalChat", include_in_schema=False)
def globalChat():
    return FileResponse("globalChat.html")

@app.get("/datenschutz.html", include_in_schema=False)
def privacyPolicy():
    return FileResponse("datenschutz.html")

@app.patch("/api/linktrees/{linktree_id}", response_model=dict)
def update_linktree_ep(
    linktree_id: int,
    payload: LinktreeUpdateIn,
    user: dict = Depends(require_user),
):
    _ensure_pg()
    # Owner/Admin prüfen
    _require_tree_owner_or_admin(linktree_id, user)

    fields = payload.model_dump(exclude_unset=True)

    # Vorherige Medien-URLs zum Aufräumen merken + device_type
    old_song = None
    old_song_icon = None
    old_bg = None
    current_device_type = None
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT song_url, song_icon_url, background_url, device_type FROM linktrees WHERE id=%s",
            (linktree_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Linktree not found")
        old_song = row.get("song_url")
        old_song_icon = row.get("song_icon_url")
        old_bg = row.get("background_url")
        current_device_type = row.get("device_type") or "pc"

    # Wenn slug geändert werden soll: einfache Kollision prüfen (gleicher device_type)
    if "slug" in fields and fields["slug"]:
        new_slug = fields["slug"]
        target_device = fields.get("device_type") or current_device_type
        with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM linktrees WHERE lower(slug)=lower(%s) AND id<>%s AND user_id<>%s",
                (new_slug, linktree_id, user["id"]),
            )
            if cur.fetchone():
                raise HTTPException(409, "Slug already in use")

    # Dynamisches UPDATE bauen
    cols = []
    vals = []
    for k, v in fields.items():
        cols.append(f"{k}=%s")
        vals.append(v)
    if cols:
        q = f"UPDATE linktrees SET {', '.join(cols)}, updated_at=NOW() WHERE id=%s"
        vals.append(linktree_id)
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(q, tuple(vals))
            # Wenn Slug geändert: auch andere Device-Variante des gleichen Users synchronisieren
            if "slug" in fields and fields["slug"]:
                cur.execute(
                    "UPDATE linktrees SET slug=%s, updated_at=NOW() WHERE user_id=%s AND device_type<>%s",
                    (fields["slug"], user["id"], current_device_type),
                )
            conn.commit()

    # Medien-Cleanup, falls URLs gewechselt wurden
    new_song = fields.get("song_url", old_song)
    new_song_icon = fields.get("song_icon_url", old_song_icon)
    new_bg = fields.get("background_url", old_bg)
    if old_song and old_song != new_song:
        _delete_if_unreferenced(old_song)
    if old_song_icon and old_song_icon != new_song_icon:
        _delete_if_unreferenced(old_song_icon)
    if old_bg and old_bg != new_bg:
        _delete_if_unreferenced(old_bg)

    return {"ok": True}

firestoreDB: Any | None = None
TEMPLATE_COLLECTION = "LinktreeTemplate"
TEMPLATE_SAVES_COLLECTION = "LinktreeTemplateSaves"
discord_gateway: DiscordPresenceGateway | None = None


def _fs():
    """Lazy Firestore client with clear error when credentials are missing."""
    global firestoreDB
    if firestoreDB is not None:
        return firestoreDB
    try:
        firestoreDB = get_firestore_client()
    except Exception as exc:
        logger.error("Firestore unavailable: %s", exc)
        raise HTTPException(503, "Firestore is not configured")
    return firestoreDB


@app.on_event("startup")
async def _startup_media_gc():
    try:
        res = _run_media_gc(60, require_templates=False)
        if not res.get("ok", False):
            logger.warning("Startup media GC skipped: %s", "; ".join(res.get("warnings") or []))
            return
        logger.info(
            "Startup media GC: deleted=%d skipped=%d warnings=%d",
            len(res.get("deleted", [])),
            len(res.get("skipped", [])),
            len(res.get("warnings", [])),
        )
    except Exception as exc:
        logger.warning("Startup media GC failed: %s", exc)
    global discord_gateway
    try:
        if discord_gateway is None:
            discord_gateway = DiscordPresenceGateway(DISCORD_BOT_TOKEN, db)
        discord_gateway.start()
    except Exception as exc:
        logger.warning("Discord presence gateway not started: %s", exc)


@app.on_event("shutdown")
async def _shutdown_presence_gateway():
    if discord_gateway:
        discord_gateway.stop()


class TemplateLinkIn(BaseModel):
    url: HttpUrl
    label: Optional[str] = None
    icon_url: Optional[str] = None
    position: Optional[int] = None
    is_active: bool = True


class TemplateVariantIn(BaseModel):
    device_type: DeviceType
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    song_icon_url: Optional[str] = None
    show_audio_player: bool = False
    audio_player_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    audio_player_bg_alpha: int = Field(60, ge=0, le=100)
    audio_player_text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    audio_player_accent_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    background_url: str = Field(..., min_length=1)
    background_is_video: bool = False
    transparency: int = Field(0, ge=0, le=100)
    name_effect: EffectName = "none"
    background_effect: BgEffectName = "none"
    display_name_mode: DisplayNameMode = "slug"
    custom_display_name: Optional[str] = Field(None, min_length=1, max_length=64)
    link_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_alpha: int = Field(100, ge=0, le=100)
    card_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    name_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    location_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    quote_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_url: Optional[str] = None
    discord_frame_enabled: bool = False
    discord_presence_enabled: bool = False
    discord_presence: DiscordPresence = "online"
    discord_status_enabled: bool = False
    discord_status_text: Optional[str] = Field(None, max_length=140)
    discord_badges_enabled: bool = False
    show_visit_counter: bool = False
    visit_counter_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    visit_counter_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    visit_counter_bg_alpha: int = Field(20, ge=0, le=100)


class TemplateCreateIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    description: Optional[str] = Field(None, max_length=500)
    creator: Optional[str] = Field(None, max_length=100)
    preview_image_url: Optional[str] = Field(None, max_length=500)
    is_public: bool = True
    variants: List[TemplateVariantIn] = Field(..., min_length=1, max_length=2)


class TemplateUpdateIn(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=64)
    description: Optional[str] = Field(None, max_length=500)
    creator: Optional[str] = Field(None, max_length=100)
    preview_image_url: Optional[str] = Field(None, max_length=500)
    is_public: Optional[bool] = None
    variants: Optional[List[TemplateVariantIn]] = None


class TemplateListOut(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    creator: Optional[str] = None
    preview_image_url: Optional[str] = None
    owner_id: int
    owner_username: Optional[str] = None
    owner_profile_picture: Optional[str] = None
    is_public: bool
    device_type: Optional[str] = None  # "pc", "mobile", or "both"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class TemplateDetailOut(TemplateListOut):
    variants: List[dict]
    data: Optional[dict] = None  # backwards compatibility


def _doc_time_iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return value if value else None


def _normalize_variant(payload: TemplateVariantIn) -> dict:
    data = payload.model_dump(exclude_none=True)
    data.setdefault("background_is_video", False)
    data.setdefault("transparency", 0)
    data.setdefault("name_effect", "none")
    data.setdefault("background_effect", "none")
    data.setdefault("display_name_mode", "slug")
    data.setdefault("link_bg_alpha", 100)
    data.setdefault("audio_player_bg_alpha", 60)
    data.setdefault("visit_counter_bg_alpha", 20)
    data.setdefault("show_visit_counter", False)
    data.setdefault("discord_frame_enabled", False)
    data.setdefault("show_audio_player", False)
    data.setdefault("discord_presence_enabled", False)
    data.setdefault("discord_presence", "online")
    data.setdefault("discord_status_enabled", False)
    data.setdefault("discord_badges_enabled", False)
    return data


def _template_doc_to_list(doc) -> TemplateListOut:
    data = doc.to_dict() or {}
    variants = data.get("variants") or []
    device_type = None
    if variants:
        if len(variants) >= 2:
            device_type = "both"
        else:
            device_type = variants[0].get("device_type")
    return TemplateListOut(
        id=doc.id,
        name=data.get("name") or "",
        description=data.get("description"),
        creator=data.get("creator"),
        preview_image_url=data.get("preview_image_url"),
        owner_id=int(data.get("owner_id") or 0),
        owner_username=data.get("owner_username"),
        owner_profile_picture=data.get("owner_profile_picture"),
        is_public=bool(data.get("is_public", False)),
        device_type=device_type,
        created_at=_doc_time_iso(data.get("created_at")),
        updated_at=_doc_time_iso(data.get("updated_at")),
    )


def _template_doc_to_detail(doc) -> TemplateDetailOut:
    data = doc.to_dict() or {}
    variants = data.get("variants") or []
    # backward-compat: also expose first variant as "data"
    first = variants[0] if variants else {}
    return TemplateDetailOut(
        id=doc.id,
        name=data.get("name") or "",
        description=data.get("description"),
        creator=data.get("creator"),
        preview_image_url=data.get("preview_image_url"),
        owner_id=int(data.get("owner_id") or 0),
        owner_username=data.get("owner_username"),
        owner_profile_picture=data.get("owner_profile_picture"),
        is_public=bool(data.get("is_public", False)),
        created_at=_doc_time_iso(data.get("created_at")),
        updated_at=_doc_time_iso(data.get("updated_at")),
        variants=variants,
        data=first,
    )


def _get_template_doc(template_id: str, *, user: dict) -> Any:
    doc = _fs().collection(TEMPLATE_COLLECTION).document(template_id).get()
    if not doc.exists:
        raise HTTPException(404, "Template not found")
    data = doc.to_dict() or {}
    owner_id = int(data.get("owner_id") or 0)
    if not data.get("is_public") and owner_id != int(user["id"]):
        raise HTTPException(403, "Template is private")
    return doc


def _resolve_user_slug(user: dict) -> str:
    slug = (user.get("linktree_slug") or "").strip()
    if slug:
        return slug
    if isinstance(db, PgGifDB):
        try:
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT slug FROM linktrees WHERE user_id=%s ORDER BY id LIMIT 1",
                    (user["id"],),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return row[0]
        except Exception:
            pass
    raw = (user.get("username") or "").lower()
    slug = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")
    if len(slug) < 2:
        slug = f"user{user['id']}"
    return slug[:48]


def _ensure_slug_unique(slug: str, user_id: int) -> str:
    if not isinstance(db, PgGifDB):
        return slug
    candidate = slug
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        for _ in range(5):
            cur.execute(
                "SELECT 1 FROM linktrees WHERE lower(slug)=lower(%s) AND user_id<>%s LIMIT 1",
                (candidate, user_id),
            )
            if not cur.fetchone():
                return candidate
            candidate = f"{slug}-{secrets.token_hex(2)}"
    return candidate


def _extract_linktree_fields(data: dict) -> dict:
    allowed = {
        "device_type",
        "location",
        "quote",
        "song_url",
        "song_icon_url",
        "show_audio_player",
        "audio_player_bg_color",
        "audio_player_bg_alpha",
        "audio_player_text_color",
        "audio_player_accent_color",
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
        "card_color",
        "text_color",
        "name_color",
        "location_color",
        "quote_color",
        "cursor_url",
        "discord_frame_enabled",
        "discord_presence_enabled",
        "discord_presence",
        "discord_status_enabled",
        "discord_status_text",
        "discord_badges_enabled",
        "show_visit_counter",
        "visit_counter_color",
        "visit_counter_bg_color",
        "visit_counter_bg_alpha",
    }
    return {k: v for k, v in data.items() if k in allowed}


@app.get("/marketplace", include_in_schema=False)
def marketplace_page():
    return FileResponse("marketplace.html")


@app.get("/marketplace/create", include_in_schema=False)
def marketplace_create_page():
    return FileResponse("marketplace_create.html")


@app.get("/marketplace/templates/{template_id}/demo", include_in_schema=False)
def marketplace_template_demo(template_id: str):
    return FileResponse("linktree.html")


@app.get("/api/marketplace/templates", response_model=List[TemplateListOut])
def list_public_templates(
    limit: int = Query(50, ge=1, le=100),
    user: dict = Depends(require_user),
):
    query = (
        _fs().collection(TEMPLATE_COLLECTION)
        .where("is_public", "==", True)
        .limit(limit)
    )
    docs = [_template_doc_to_list(doc) for doc in query.stream()]
    docs.sort(
        key=lambda d: d.created_at or "",
        reverse=True,
    )
    return docs


@app.get("/api/marketplace/templates/mine", response_model=List[TemplateListOut])
def list_my_templates(
    limit: int = Query(100, ge=1, le=200),
    user: dict = Depends(require_user),
):
    query = (
        _fs().collection(TEMPLATE_COLLECTION)
        .where("owner_id", "==", int(user["id"]))
        .limit(limit)
    )
    docs = [_template_doc_to_list(doc) for doc in query.stream()]
    docs.sort(
        key=lambda d: d.created_at or "",
        reverse=True,
    )
    return docs


@app.get("/api/marketplace/templates/saved", response_model=List[TemplateListOut])
def list_saved_templates(user: dict = Depends(require_user)):
    saves = (
        _fs().collection(TEMPLATE_SAVES_COLLECTION)
        .where("user_id", "==", int(user["id"]))
        .stream()
    )
    out = []
    for save in saves:
        data = save.to_dict() or {}
        template_id = data.get("template_id")
        if not template_id:
            continue
        doc = _fs().collection(TEMPLATE_COLLECTION).document(template_id).get()
        if not doc.exists:
            continue
        tdata = doc.to_dict() or {}
        if not tdata.get("is_public") and int(tdata.get("owner_id") or 0) != int(
            user["id"]
        ):
            continue
        out.append(_template_doc_to_list(doc))
    return out


@app.get("/api/marketplace/templates/{template_id}", response_model=TemplateDetailOut)
def get_template_detail(
    template_id: str, user: dict = Depends(require_user)
):
    doc = _get_template_doc(template_id, user=user)
    return _template_doc_to_detail(doc)


@app.post("/api/marketplace/templates", response_model=TemplateDetailOut, status_code=201)
def create_template(payload: TemplateCreateIn, user: dict = Depends(require_user)):
    variants = [_normalize_variant(v) for v in payload.variants]
    if len({v["device_type"] for v in variants}) != len(variants):
        raise HTTPException(400, "Duplicate device_type in variants")
    # Pick preview from pc variant background, fallback first variant
    preview = payload.preview_image_url
    if not preview:
        pc_var = next((v for v in variants if v.get("device_type") == "pc"), None)
        src = pc_var or (variants[0] if variants else None)
        preview = (src or {}).get("background_url") or "/static/icon.png"
    creator_name = (payload.creator or user.get("username") or "").strip() or None
    owner_pfp = user.get("profile_picture") if isinstance(user, dict) else None
    doc_ref = _fs().collection(TEMPLATE_COLLECTION).document()
    now = firestore.SERVER_TIMESTAMP
    doc_ref.set(
        {
            "name": payload.name.strip(),
            "description": payload.description.strip() if payload.description else None,
            "creator": creator_name,
            "preview_image_url": preview,
            "owner_id": int(user["id"]),
            "owner_username": user.get("username"),
            "owner_profile_picture": owner_pfp,
            "is_public": bool(payload.is_public),
            "created_at": now,
            "updated_at": now,
            "variants": variants,
        }
    )
    doc = doc_ref.get()
    return _template_doc_to_detail(doc)


@app.patch("/api/marketplace/templates/{template_id}", response_model=TemplateDetailOut)
def update_template(
    template_id: str, payload: TemplateUpdateIn, user: dict = Depends(require_user)
):
    doc_ref = _fs().collection(TEMPLATE_COLLECTION).document(template_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(404, "Template not found")
    existing = doc.to_dict() or {}
    if int(existing.get("owner_id") or 0) != int(user["id"]):
        raise HTTPException(403, "Forbidden")
    update: dict[str, Any] = {"updated_at": firestore.SERVER_TIMESTAMP}
    if payload.name is not None:
        update["name"] = payload.name.strip()
    if payload.description is not None:
        update["description"] = payload.description.strip() if payload.description else None
    if payload.creator is not None:
        update["creator"] = payload.creator.strip() if payload.creator else None
    if payload.preview_image_url is not None:
        update["preview_image_url"] = payload.preview_image_url or None
    if payload.is_public is not None:
        update["is_public"] = bool(payload.is_public)
    if payload.variants is not None:
        variants = [_normalize_variant(v) for v in payload.variants]
        if len({v["device_type"] for v in variants}) != len(variants):
            raise HTTPException(400, "Duplicate device_type in variants")
        update["variants"] = variants
        # update preview to background of pc variant if provided
        if not update.get("preview_image_url"):
            pc_var = next((v for v in variants if v.get("device_type") == "pc"), None)
            src = pc_var or (variants[0] if variants else None)
            update["preview_image_url"] = (src or {}).get("background_url") or None
    if payload.creator is not None or not existing.get("owner_profile_picture"):
        update["owner_profile_picture"] = user.get("profile_picture")
    doc_ref.set(update, merge=True)
    doc = doc_ref.get()
    return _template_doc_to_detail(doc)


@app.delete("/api/marketplace/templates/{template_id}", status_code=204)
def delete_template(template_id: str, user: dict = Depends(require_user)):
    doc_ref = _fs().collection(TEMPLATE_COLLECTION).document(template_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(404, "Template not found")
    existing = doc.to_dict() or {}
    if int(existing.get("owner_id") or 0) != int(user["id"]):
        raise HTTPException(403, "Forbidden")
    doc_ref.delete()
    return Response(status_code=204)


@app.post("/api/marketplace/templates/{template_id}/save")
def save_template(template_id: str, user: dict = Depends(require_user)):
    doc = _get_template_doc(template_id, user=user)
    save_id = f"{user['id']}_{doc.id}"
    _fs().collection(TEMPLATE_SAVES_COLLECTION).document(save_id).set(
        {
            "user_id": int(user["id"]),
            "template_id": doc.id,
            "created_at": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
    return {"ok": True}


@app.delete("/api/marketplace/templates/{template_id}/save", status_code=204)
def unsave_template(template_id: str, user: dict = Depends(require_user)):
    save_id = f"{user['id']}_{template_id}"
    _fs().collection(TEMPLATE_SAVES_COLLECTION).document(save_id).delete()
    return Response(status_code=204)


@app.post("/api/marketplace/templates/{template_id}/apply", response_model=LinktreeOut)
def apply_template_to_linktree(
    template_id: str,
    device: Optional[str] = Query(None, description="pc, mobile, or both"),
    user: dict = Depends(require_user),
):
    _ensure_pg()
    doc = _get_template_doc(template_id, user=user)
    template = doc.to_dict() or {}
    variants = template.get("variants") or []
    if not variants:
        data = template.get("data") or {}
        if data:
            variants = [data]
    var_map = {
        v.get("device_type"): v for v in variants if v.get("device_type") in {"pc", "mobile"}
    }
    target_choice = device or ("both" if len(var_map) == 2 else next(iter(var_map.keys()), "pc"))
    if target_choice not in {"pc", "mobile", "both"}:
        raise HTTPException(400, "Invalid device option")
    if target_choice == "both":
        if "pc" not in var_map or "mobile" not in var_map:
            raise HTTPException(400, "Template does not contain both variants")
        targets = ["pc", "mobile"]
    else:
        if target_choice not in var_map:
            raise HTTPException(404, f"Template has no {target_choice} variant")
        targets = [target_choice]

    results: list[tuple[int, str]] = []  # (id, device)
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        for target_device in targets:
            data = var_map[target_device] or {}
            link_fields = _extract_linktree_fields(data)
            link_fields.pop("device_type", None)

            cur.execute(
                "SELECT id, slug FROM linktrees WHERE user_id=%s AND device_type=%s",
                (user["id"], target_device),
            )
            row = cur.fetchone()
            if row:
                linktree_id = row["id"]
                if link_fields:
                    cols = [f"{k}=%s" for k in link_fields.keys()]
                    vals = list(link_fields.values()) + [linktree_id]
                    cur.execute(
                        f"UPDATE linktrees SET {', '.join(cols)}, updated_at=NOW() WHERE id=%s",
                        tuple(vals),
                    )
            else:
                slug = _ensure_slug_unique(_resolve_user_slug(user), int(user["id"]))
                linktree_id = db.create_linktree(
                    user_id=user["id"],
                    slug=slug,
                    device_type=target_device,
                    **link_fields,
                )
            results.append((linktree_id, target_device))
        conn.commit()

    # Return the manage view for the PC variant if available, otherwise the last applied
    preferred = next((lid for lid, dev in results if dev == "pc"), results[-1][0])
    return get_linktree_manage(preferred, user)
