from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
import os, httpx
import hashlib
import smtplib
from email.message import EmailMessage
import re
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
from typing import Callable, List, Optional
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
LayoutMode = Literal["center", "wide"]
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
MAX_AUDIO_BYTES = 30 * 1024 * 1024  # 30 MB
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB (Avatare/Icons)
MAX_BACKGROUND_BYTES = 100 * 1024 * 1024  # 100 MB (Hintergrund-Video/Bild)

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


FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._ ()-]+")


def _clean_upload_filename(name: str | None) -> str | None:
    if not name:
        return None
    base = os.path.basename(str(name)).strip().replace("\x00", "")
    if not base:
        return None
    cleaned = FILENAME_SAFE_RE.sub("", base)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "".join(ch for ch in base if ch.isprintable()).strip()
    if not cleaned:
        return None
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip()
    return cleaned


def _normalize_text_list(
    value: Any,
    *,
    max_items: int | None = None,
    max_len: int | None = None,
    dedupe: bool = True,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        if max_len is not None and len(text) > max_len:
            text = text[:max_len]
        if dedupe:
            if text in seen:
                continue
            seen.add(text)
        out.append(text)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _list_to_json(
    value: Any,
    *,
    max_items: int | None = None,
    max_len: int | None = None,
    allow_empty: bool = False,
) -> str | None:
    if value is None:
        return None
    items = _normalize_text_list(value, max_items=max_items, max_len=max_len)
    if not items and not allow_empty:
        return None
    return json.dumps(items)


def _json_to_list(
    raw: Any,
    *,
    max_items: int | None = None,
    max_len: int | None = None,
    none_if_missing: bool = False,
) -> list[str] | None:
    if raw is None:
        return None if none_if_missing else []
    if isinstance(raw, str):
        if not raw.strip():
            return None if none_if_missing else []
        try:
            data = json.loads(raw)
        except Exception:
            data = [raw]
    else:
        data = raw
    items = _normalize_text_list(data, max_items=max_items, max_len=max_len)
    if none_if_missing and raw is None:
        return None
    return items


def _normalize_section_order(value: Any) -> list[Any] | None:
    if value is None:
        return None
    data = value
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            data = json.loads(value)
        except Exception:
            data = [value]
    if not isinstance(data, (list, tuple, set)):
        data = [data]
    out: list[Any] = []
    seen: set[str] = set()
    for item in data:
        if item is None:
            continue
        if isinstance(item, (list, tuple, set)):
            row: list[str] = []
            for sub in item:
                raw = str(sub).strip().lower()
                key = SECTION_KEY_ALIASES.get(raw, raw)
                if not key or key not in SECTION_ORDER_ALLOWED:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                row.append(key)
                if len(row) >= 2:
                    break
            if len(row) == 1:
                out.append(row[0])
            elif len(row) > 1:
                out.append(row)
            continue
        raw = str(item).strip().lower()
        key = SECTION_KEY_ALIASES.get(raw, raw)
        if not key or key not in SECTION_ORDER_ALLOWED:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    if not out:
        return None
    for key in SECTION_ORDER_DEFAULT:
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _normalize_canvas_layout(value: Any) -> dict | None:
    if value is None:
        return None
    data = value
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            data = json.loads(value)
        except Exception:
            return None
    if not isinstance(data, dict):
        return None
    enabled = bool(data.get("enabled", False))
    plate_min_w = 1
    plate_min_h = 1
    plate_max_w = 20000
    plate_max_h = 20000
    size_min_w = 1
    size_min_h = 1
    size_max_w = 20000
    size_max_h = 20000
    try:
        grid = int(data.get("grid", 1))
    except Exception:
        grid = 1
    grid = max(1, min(1000, grid))
    plates_raw = data.get("plates") or {}
    groups_raw = data.get("groups") or {}
    size_raw = data.get("size")
    plates: dict[str, dict] = {}
    groups: dict[str, dict] = {}
    size: dict[str, int] | None = None
    if isinstance(size_raw, dict):
        try:
            sw = int(size_raw.get("w", 0))
            sh = int(size_raw.get("h", 0))
        except Exception:
            sw = 0
            sh = 0
        if sw > 0 and sh > 0:
            size = {
                "w": max(size_min_w, min(size_max_w, sw)),
                "h": max(size_min_h, min(size_max_h, sh)),
            }
    if isinstance(groups_raw, dict):
        for gid, raw in groups_raw.items():
            if not isinstance(raw, dict):
                continue
            try:
                gx = int(raw.get("x", 0))
                gy = int(raw.get("y", 0))
            except Exception:
                continue
            groups[str(gid)] = {"x": gx, "y": gy}
    if isinstance(plates_raw, dict):
        for key, raw in plates_raw.items():
            raw_key = str(key).strip().lower()
            k = SECTION_KEY_ALIASES.get(raw_key, raw_key)
            if not k or k not in SECTION_ORDER_ALLOWED:
                continue
            if not isinstance(raw, dict):
                continue
            try:
                x = int(raw.get("x", 0))
                y = int(raw.get("y", 0))
            except Exception:
                continue
            plate = {"x": x, "y": y}
            try:
                w = int(raw.get("w", 0))
            except Exception:
                w = 0
            try:
                h = int(raw.get("h", 0))
            except Exception:
                h = 0
            if w > 0:
                plate["w"] = max(plate_min_w, min(plate_max_w, w))
            if h > 0:
                plate["h"] = max(plate_min_h, min(plate_max_h, h))
            group = raw.get("group")
            if group is not None:
                gid = str(group)
                if gid in groups:
                    plate["group"] = gid
            plates[k] = plate
    if not plates and not groups and not enabled and not size:
        return None
    result = {
        "enabled": enabled,
        "grid": grid,
        "plates": plates,
        "groups": groups,
    }
    if size is not None:
        result["size"] = size
    return result


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
        cur.execute(
            "SELECT COUNT(*) FROM linktrees WHERE linktree_profile_picture = %s",
            (url,),
        )
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
            for k in (
                "song_url",
                "song_icon_url",
                "background_url",
                "cursor_url",
                "linktree_profile_picture",
            )
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
            for k in (
                "song_url",
                "song_icon_url",
                "background_url",
                "cursor_url",
                "linktree_profile_picture",
            )
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
            _add_local_media_url(variant.get("linktree_profile_picture"), urls)
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
            _add_local_media_url(legacy.get("linktree_profile_picture"), urls)
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
                "SELECT linktree_profile_picture FROM linktrees WHERE linktree_profile_picture IS NOT NULL",
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

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://taoma.space").rstrip("/")
RESET_TOKEN_TTL_MINUTES = int(os.getenv("RESET_TOKEN_TTL_MINUTES", "30"))
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587") or "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM")
SMTP_TLS = (os.getenv("SMTP_TLS", "true").lower() not in {"0", "false", "no"})


def _email_configured() -> bool:
    return bool(SMTP_HOST and SMTP_FROM)


def _send_email(to_email: str, subject: str, body: str) -> bool:
    if not _email_configured():
        return False
    msg = EmailMessage()
    msg["From"] = SMTP_FROM
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as smtp:
            if SMTP_TLS:
                smtp.starttls()
            if SMTP_USER and SMTP_PASS:
                smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
        return True
    except Exception as exc:
        logger.warning("Email send failed: %s", exc)
        return False


def _hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _create_password_reset(user_id: int) -> str:
    if not isinstance(db, PgGifDB):
        raise HTTPException(501, "Password reset requires PostgreSQL.")
    token = secrets.token_urlsafe(32)
    token_hash = _hash_reset_token(token)
    expires = datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_TTL_MINUTES)
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "DELETE FROM password_resets WHERE user_id=%s AND used_at IS NULL",
            (user_id,),
        )
        cur.execute(
            """
            INSERT INTO password_resets (user_id, token_hash, expires_at)
            VALUES (%s, %s, %s)
            """,
            (user_id, token_hash, expires),
        )
        conn.commit()
    return token


def _consume_password_reset(token: str) -> int:
    if not isinstance(db, PgGifDB):
        raise HTTPException(501, "Password reset requires PostgreSQL.")
    token_hash = _hash_reset_token(token)
    now = datetime.now(timezone.utc)
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, user_id, expires_at, used_at
            FROM password_resets
            WHERE token_hash=%s
            LIMIT 1
            """,
            (token_hash,),
        )
        row = cur.fetchone()
        if not row or row.get("used_at") or row.get("expires_at") < now:
            raise HTTPException(400, "Invalid or expired reset token")
        cur.execute(
            "UPDATE password_resets SET used_at=now() WHERE id=%s",
            (row["id"],),
        )
        conn.commit()
        return int(row["user_id"])


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

DISCORD_BADGE_ICON_MAP = {
    "staff": "/static/discord-badges/discordstaff.svg",
    "partner": "/static/discord-badges/discordpartner.svg",
    "hypesquad": "/static/discord-badges/hypesquadevents.svg",
    "bug_hunter_lvl1": "/static/discord-badges/discordbughunter1.svg",
    "bravery": "/static/discord-badges/hypesquadbravery.svg",
    "brilliance": "/static/discord-badges/hypesquadbrilliance.svg",
    "balance": "/static/discord-badges/hypesquadbalance.svg",
    "early_supporter": "/static/discord-badges/discordearlysupporter.svg",
    "bug_hunter_lvl2": "/static/discord-badges/discordbughunter2.svg",
    "verified_bot": "/static/discord-badges/premiumbot.png",
    "verified_dev": "/static/discord-badges/discordbotdev.svg",
    "certified_mod": "/static/discord-badges/discordmod.svg",
    "bot_http": "/static/discord-badges/supportscommands.svg",
    "active_dev": "/static/discord-badges/activedeveloper.svg",
    "nitro": "/static/discord-badges/discordnitro.svg",
}


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
                    "icon_url": DISCORD_BADGE_ICON_MAP.get(badge["code"]),
                }
            )
    premium = int(acct.get("premium_type") or 0)
    if premium > 0:
        badges.append(
            {
                "code": "nitro",
                "label": "Discord Nitro",
                "icon_url": DISCORD_BADGE_ICON_MAP.get("nitro"),
            }
        )
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
QuoteFontFamily = Literal["default", "serif", "mono", "script", "display"]
BgEffectName = Literal[
    "none",
    "night",
    "rain",
    "snow",
    "noise",
    "gradient",
    "parallax",
    "particles",
    "sweep",
    "mesh",
    "grid",
    "vignette",
    "scanlines",
    "glitch",
]
CursorEffectName = Literal[
    "none",
    "glow",
    "particles",
    "trail",
    "aura",
    "magnet",
    "morph",
    "snap",
    "velocity",
    "ripple",
    "blend",
    "sticky",
    "rotate",
]
DiscordPresence = Literal["online", "idle", "dnd", "offline"]
HEX_COLOR_RE = r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$"
SECTION_ORDER_DEFAULT = [
    "pfp",
    "name",
    "discord_status",
    "quote",
    "location",
    "badges",
    "audio",
    "links",
    "visit_counter",
]
SECTION_ORDER_ALLOWED = set(SECTION_ORDER_DEFAULT)
SECTION_KEY_ALIASES = {"discord_badges": "badges"}


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
    quote_typing_enabled: bool = False
    quote_typing_texts: Optional[List[str]] = None
    entry_text: Optional[str] = Field(None, max_length=120)
    quote_typing_speed: Optional[int] = Field(None, ge=20, le=200)
    quote_typing_pause: Optional[int] = Field(None, ge=200, le=10000)
    quote_font_size: Optional[int] = Field(None, ge=10, le=40)
    quote_font_family: QuoteFontFamily = "default"
    quote_effect: EffectName = "none"
    quote_effect_strength: int = Field(70, ge=0, le=100)
    entry_bg_alpha: int = Field(85, ge=0, le=100)
    entry_text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    entry_font_size: int = Field(16, ge=10, le=40)
    entry_font_family: QuoteFontFamily = "default"
    entry_effect: EffectName = "none"
    entry_overlay_alpha: int = Field(35, ge=0, le=100)
    entry_box_enabled: bool = True
    entry_border_enabled: bool = True
    entry_border_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    song_url: Optional[str] = None
    song_name: Optional[str] = Field(None, max_length=120)
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
    name_font_family: QuoteFontFamily = "default"
    background_effect: BgEffectName = "none"
    display_name_mode: DisplayNameMode = "slug"
    layout_mode: LayoutMode = "center"
    custom_display_name: Optional[str] = Field(None, min_length=1, max_length=64)
    linktree_profile_picture: Optional[str] = None
    section_order: Optional[List[Any]] = None
    canvas_layout: Optional[dict] = None
    link_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_alpha: int = Field(100, ge=0, le=100)
    link_columns: Optional[int] = Field(None, ge=1, le=8)
    link_icons_only: bool = False
    link_icons_only_size: int = Field(36, ge=16, le=128)
    link_icons_only_gap: int = Field(12, ge=0, le=64)
    link_icons_only_grouped: bool = False
    link_icons_only_direction: Literal["row", "column"] = "row"
    card_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    name_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    location_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    quote_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_url: Optional[str] = None
    cursor_effect: CursorEffectName = "none"
    cursor_effect_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_effect_alpha: int = Field(70, ge=0, le=100)
    discord_frame_enabled: bool = False
    discord_presence_enabled: bool = False
    discord_presence: DiscordPresence = "online"
    discord_status_enabled: bool = False
    discord_status_text: Optional[str] = Field(None, max_length=140)
    discord_badges_enabled: bool = False
    discord_badge_codes: Optional[List[str]] = None
    show_visit_counter: bool = False
    visit_counter_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    visit_counter_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    visit_counter_bg_alpha: int = Field(20, ge=0, le=100)
    demo_show_links: bool = False
    demo_link_label: Optional[str] = Field(None, max_length=80)
    demo_link_url: Optional[str] = Field(None, max_length=500)
    demo_link_icon_url: Optional[str] = Field(None, max_length=500)


class LinktreeUpdateIn(BaseModel):
    slug: Optional[str] = Field(
        None, min_length=2, max_length=48, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    device_type: Optional[DeviceType] = None
    location: Optional[str] = None
    quote: Optional[str] = None
    quote_typing_enabled: Optional[bool] = None
    quote_typing_texts: Optional[List[str]] = None
    entry_text: Optional[str] = Field(None, max_length=120)
    quote_typing_speed: Optional[int] = Field(None, ge=20, le=200)
    quote_typing_pause: Optional[int] = Field(None, ge=200, le=10000)
    quote_font_size: Optional[int] = Field(None, ge=10, le=40)
    quote_font_family: Optional[QuoteFontFamily] = None
    quote_effect: Optional[EffectName] = None
    quote_effect_strength: Optional[int] = Field(None, ge=0, le=100)
    entry_bg_alpha: Optional[int] = Field(None, ge=0, le=100)
    entry_text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    entry_font_size: Optional[int] = Field(None, ge=10, le=40)
    entry_font_family: Optional[QuoteFontFamily] = None
    entry_effect: Optional[EffectName] = None
    entry_overlay_alpha: Optional[int] = Field(None, ge=0, le=100)
    entry_box_enabled: Optional[bool] = None
    entry_border_enabled: Optional[bool] = None
    entry_border_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    song_url: Optional[str] = None
    song_name: Optional[str] = Field(None, max_length=120)
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
    name_font_family: Optional[QuoteFontFamily] = None
    background_effect: Optional[BgEffectName] = None
    display_name_mode: Optional[DisplayNameMode] = None
    layout_mode: Optional[LayoutMode] = None
    custom_display_name: Optional[str] = Field(None, min_length=1, max_length=64)
    linktree_profile_picture: Optional[str] = None
    section_order: Optional[List[Any]] = None
    canvas_layout: Optional[dict] = None
    link_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_alpha: Optional[int] = Field(None, ge=0, le=100)
    link_columns: Optional[int] = Field(None, ge=1, le=8)
    link_icons_only: Optional[bool] = None
    link_icons_only_size: Optional[int] = Field(None, ge=16, le=128)
    link_icons_only_gap: Optional[int] = Field(None, ge=0, le=64)
    link_icons_only_grouped: Optional[bool] = None
    link_icons_only_direction: Optional[Literal["row", "column"]] = None
    card_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    name_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    location_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    quote_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_url: Optional[str] = None
    cursor_effect: Optional[CursorEffectName] = None
    cursor_effect_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_effect_alpha: Optional[int] = Field(None, ge=0, le=100)
    discord_frame_enabled: Optional[bool] = None
    discord_presence_enabled: Optional[bool] = None
    discord_presence: Optional[DiscordPresence] = None
    discord_status_enabled: Optional[bool] = None
    discord_status_text: Optional[str] = Field(None, max_length=140)
    discord_badges_enabled: Optional[bool] = None
    discord_badge_codes: Optional[List[str]] = None
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
    quote_typing_enabled: bool = False
    quote_typing_texts: Optional[List[str]] = None
    entry_text: Optional[str] = None
    quote_typing_speed: Optional[int] = None
    quote_typing_pause: Optional[int] = None
    quote_font_size: Optional[int] = None
    quote_font_family: Optional[QuoteFontFamily] = None
    quote_effect: Optional[EffectName] = None
    quote_effect_strength: Optional[int] = None
    entry_bg_alpha: Optional[int] = None
    entry_text_color: Optional[str] = None
    entry_font_size: Optional[int] = None
    entry_font_family: Optional[QuoteFontFamily] = None
    entry_effect: Optional[EffectName] = None
    entry_overlay_alpha: Optional[int] = None
    entry_box_enabled: Optional[bool] = None
    entry_border_enabled: Optional[bool] = None
    entry_border_color: Optional[str] = None
    song_url: Optional[str] = None
    song_name: Optional[str] = None
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
    name_font_family: QuoteFontFamily = "default"
    background_effect: BgEffectName
    display_name_mode: DisplayNameMode  # NEU
    layout_mode: Optional[LayoutMode] = None
    custom_display_name: Optional[str] = None
    link_color: Optional[str] = None
    link_bg_color: Optional[str] = None
    link_bg_alpha: int = 100
    link_columns: Optional[int] = None
    link_icons_only: bool = False
    link_icons_only_size: int = 36
    link_icons_only_gap: int = 12
    link_icons_only_grouped: bool = False
    link_icons_only_direction: Literal["row", "column"] = "row"
    card_color: Optional[str] = None
    text_color: Optional[str] = None
    name_color: Optional[str] = None
    location_color: Optional[str] = None
    quote_color: Optional[str] = None
    cursor_url: Optional[str] = None
    cursor_effect: CursorEffectName = "none"
    cursor_effect_color: Optional[str] = None
    cursor_effect_alpha: int = 70
    discord_frame_enabled: bool = False
    discord_decoration_url: Optional[str] = None
    discord_presence_enabled: bool = False
    discord_presence: DiscordPresence = "online"
    discord_status_enabled: bool = False
    discord_status_text: Optional[str] = None
    discord_badges_enabled: bool = False
    discord_badges: Optional[List[DiscordBadgeOut]] = None
    discord_badge_codes: Optional[List[str]] = None
    discord_linked: bool = False
    linktree_profile_picture: Optional[str] = None
    section_order: Optional[List[Any]] = None
    canvas_layout: Optional[dict] = None
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
    email: Optional[EmailStr] = None
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
    email: Optional[str] = None
    linktree_id: Optional[int] = None
    linktree_slug: Optional[str] = None  # <- add this
    profile_picture: Optional[str] = None
    admin: bool
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class UserCreateIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)
    linktree_id: Optional[int] = None
    profile_picture: Optional[str] = None


class UserUpdateIn(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=32)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8)
    linktree_id: Optional[int] = None
    profile_picture: Optional[str] = None
    admin: Optional[bool] = None


JOB_STORE: dict[str, dict] = {}


class LoginUserIn(BaseModel):
    identifier: str = Field(
        ...,
        validation_alias=AliasChoices("identifier", "username"),
    )
    password: str
    email: Optional[EmailStr] = None


class RegisterIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    email: EmailStr
    password: str = Field(..., min_length=8)
    linktree_id: Optional[int] = None
    profile_picture: Optional[str] = None


class RegisterOut(BaseModel):
    token: str
    expires_at: Optional[str] = None
    user: UserOut


class PasswordResetRequestIn(BaseModel):
    email: EmailStr


class PasswordResetIn(BaseModel):
    token: str = Field(..., min_length=20, max_length=200)
    password: str = Field(..., min_length=8)


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


def _gif_url_is_live(url: str) -> tuple[bool, int | None]:
    if not isinstance(url, str) or not url.strip():
        return False, None
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True) as client:
            resp = client.head(url.strip())
            if not (200 <= resp.status_code < 300):
                resp = client.get(url.strip(), headers={"Range": "bytes=0-0"})
            return 200 <= resp.status_code < 300, resp.status_code
    except httpx.HTTPError:
        return False, None


def does_url_still_exist(fetch_gif: Callable[[], dict], *, max_attempts: int = 12):
    tried_ids: set[int] = set()
    for _ in range(max_attempts):
        gif = fetch_gif()
        if not isinstance(gif, dict):
            return gif

        gif_id = gif.get("id")
        if isinstance(gif_id, int):
            if gif_id in tried_ids:
                continue
            tried_ids.add(gif_id)

        url = str(gif.get("url") or "").strip()
        ok, status_code = _gif_url_is_live(url)
        if ok:
            return gif

        logger.warning(
            "Skipping unreachable GIF id=%s url=%s status=%s",
            gif_id,
            url,
            status_code,
        )
        # Hard-remove clear dead links so future requests can recover faster.
        if isinstance(gif_id, int) and status_code in {404, 410}:
            try:
                db.delete_gif(gif_id)
            except Exception as exc:
                logger.warning("Failed to delete dead GIF id=%s: %s", gif_id, exc)

    raise HTTPException(
        status_code=404, detail="no reachable gifs found for this request"
    )


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


# ---------- Linktree: public read ----------


# Optional: HTML-Seite (statisches Template, das o.g. API nutzt)


# ---------- Linktree: create/update (owner/admin) ----------


# ---------- Links: add/update/delete (owner/admin) ----------


# ---------- Icon-Katalog & Besitz ----------


@app.api_route(
    "/api/admin/media/gc",
    methods=["POST", "GET"],
    dependencies=[Depends(require_admin)],
)
def media_garbage_collect(min_age_seconds: int = Query(60, ge=0)):
    """Löscht unreferenzierte Dateien im Upload-Verzeichnis, die älter als min_age_seconds sind."""
    return _run_media_gc(min_age_seconds, require_templates=True)


# ---------- HealthData CRUD ----------


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
    quote_typing_enabled: bool = False
    quote_typing_texts: Optional[List[str]] = None
    quote_typing_speed: Optional[int] = Field(None, ge=20, le=200)
    quote_typing_pause: Optional[int] = Field(None, ge=200, le=10000)
    quote_font_size: Optional[int] = Field(None, ge=10, le=40)
    quote_font_family: Optional[QuoteFontFamily] = None
    quote_effect: Optional[EffectName] = None
    entry_text: Optional[str] = Field(None, max_length=120)
    entry_bg_alpha: Optional[int] = Field(None, ge=0, le=100)
    entry_text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    entry_font_size: Optional[int] = Field(None, ge=10, le=40)
    entry_font_family: Optional[QuoteFontFamily] = None
    entry_effect: Optional[EffectName] = None
    entry_overlay_alpha: Optional[int] = Field(None, ge=0, le=100)
    entry_box_enabled: Optional[bool] = None
    entry_border_enabled: Optional[bool] = None
    entry_border_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    song_url: Optional[str] = None
    song_name: Optional[str] = Field(None, max_length=120)
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
    name_font_family: Optional[QuoteFontFamily] = None
    background_effect: BgEffectName = "none"
    display_name_mode: DisplayNameMode = "slug"
    layout_mode: Optional[LayoutMode] = None
    custom_display_name: Optional[str] = Field(None, min_length=1, max_length=64)
    linktree_profile_picture: Optional[str] = None
    section_order: Optional[List[Any]] = None
    canvas_layout: Optional[dict] = None
    link_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    link_bg_alpha: int = Field(100, ge=0, le=100)
    link_columns: Optional[int] = Field(None, ge=1, le=8)
    link_icons_only: bool = False
    link_icons_only_size: int = Field(36, ge=16, le=128)
    link_icons_only_gap: int = Field(12, ge=0, le=64)
    link_icons_only_grouped: bool = False
    link_icons_only_direction: Literal["row", "column"] = "row"
    card_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    text_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    name_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    location_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    quote_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_url: Optional[str] = None
    cursor_effect: CursorEffectName = "none"
    cursor_effect_color: Optional[str] = Field(None, pattern=HEX_COLOR_RE)
    cursor_effect_alpha: int = Field(70, ge=0, le=100)
    discord_frame_enabled: bool = False
    discord_presence_enabled: bool = False
    discord_presence: DiscordPresence = "online"
    discord_status_enabled: bool = False
    discord_status_text: Optional[str] = Field(None, max_length=140)
    discord_badges_enabled: bool = False
    discord_badge_codes: Optional[List[str]] = None
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
    if "quote_typing_texts" in data:
        data["quote_typing_texts"] = _normalize_text_list(
            data.get("quote_typing_texts"),
            max_items=3,
            max_len=180,
            dedupe=False,
        )
    if "quote_typing_speed" in data:
        try:
            speed = int(data.get("quote_typing_speed"))
        except Exception:
            speed = None
        data["quote_typing_speed"] = (
            max(20, min(200, speed)) if speed is not None else None
        )
    if "quote_typing_pause" in data:
        try:
            pause = int(data.get("quote_typing_pause"))
        except Exception:
            pause = None
        data["quote_typing_pause"] = (
            max(200, min(10000, pause)) if pause is not None else None
        )
    if "quote_font_size" in data:
        try:
            size = int(data.get("quote_font_size"))
        except Exception:
            size = None
        data["quote_font_size"] = (
            max(10, min(40, size)) if size is not None else None
        )
    if "quote_font_family" in data:
        fam = str(data.get("quote_font_family") or "").lower()
        if fam in {"default", "serif", "mono", "script", "display"}:
            data["quote_font_family"] = fam
        else:
            data["quote_font_family"] = "default"
    if "quote_effect" in data:
        fx = str(data.get("quote_effect") or "").lower()
        if fx in {"none", "glow", "neon", "rainbow"}:
            data["quote_effect"] = fx
        else:
            data["quote_effect"] = "none"
    if "quote_effect_strength" in data:
        try:
            strength = int(data.get("quote_effect_strength"))
        except Exception:
            strength = None
        data["quote_effect_strength"] = (
            max(0, min(100, strength)) if strength is not None else 70
        )
    if "entry_bg_alpha" in data:
        try:
            alpha = int(data.get("entry_bg_alpha"))
        except Exception:
            alpha = None
        data["entry_bg_alpha"] = max(0, min(100, alpha)) if alpha is not None else 85
    if "entry_font_size" in data:
        try:
            size = int(data.get("entry_font_size"))
        except Exception:
            size = None
        data["entry_font_size"] = max(10, min(40, size)) if size is not None else 16
    if "entry_font_family" in data:
        fam = str(data.get("entry_font_family") or "").lower()
        if fam in {"default", "serif", "mono", "script", "display"}:
            data["entry_font_family"] = fam
        else:
            data["entry_font_family"] = "default"
    if "name_font_family" in data:
        fam = str(data.get("name_font_family") or "").lower()
        if fam in {"default", "serif", "mono", "script", "display"}:
            data["name_font_family"] = fam
        else:
            data["name_font_family"] = "default"
    if "entry_effect" in data:
        fx = str(data.get("entry_effect") or "").lower()
        if fx in {"none", "glow", "neon", "rainbow"}:
            data["entry_effect"] = fx
        else:
            data["entry_effect"] = "none"
    if "entry_overlay_alpha" in data:
        try:
            alpha = int(data.get("entry_overlay_alpha"))
        except Exception:
            alpha = None
        data["entry_overlay_alpha"] = (
            max(0, min(100, alpha)) if alpha is not None else 35
        )
    if "entry_box_enabled" in data:
        data["entry_box_enabled"] = bool(data.get("entry_box_enabled"))
    if "entry_border_enabled" in data:
        data["entry_border_enabled"] = bool(data.get("entry_border_enabled"))
    if "entry_border_color" in data and isinstance(data.get("entry_border_color"), str):
        val = data.get("entry_border_color").strip()
        data["entry_border_color"] = val if re.match(HEX_COLOR_RE, val) else None
    if "entry_text_color" in data and isinstance(data.get("entry_text_color"), str):
        val = data.get("entry_text_color").strip()
        data["entry_text_color"] = val if re.match(HEX_COLOR_RE, val) else None
    if "linktree_profile_picture" in data and isinstance(
        data.get("linktree_profile_picture"), str
    ):
        val = data.get("linktree_profile_picture").strip()
        data["linktree_profile_picture"] = val or None
    if "link_columns" in data:
        try:
            cols = int(data.get("link_columns"))
        except Exception:
            cols = None
        if cols is not None:
            cols = max(1, min(8, cols))
        data["link_columns"] = cols
    if "link_icons_only" in data:
        data["link_icons_only"] = bool(data.get("link_icons_only"))
    if "link_icons_only_size" in data:
        try:
            size = int(data.get("link_icons_only_size"))
        except Exception:
            size = None
        data["link_icons_only_size"] = (
            max(16, min(128, size)) if size is not None else 36
        )
    if "link_icons_only_gap" in data:
        try:
            gap = int(data.get("link_icons_only_gap"))
        except Exception:
            gap = None
        data["link_icons_only_gap"] = max(0, min(64, gap)) if gap is not None else 12
    if "link_icons_only_grouped" in data:
        data["link_icons_only_grouped"] = bool(data.get("link_icons_only_grouped"))
    if "link_icons_only_direction" in data:
        direction = str(data.get("link_icons_only_direction") or "row").lower()
        data["link_icons_only_direction"] = (
            direction if direction in {"row", "column"} else "row"
        )
    if "layout_mode" in data:
        mode = str(data.get("layout_mode") or "center").lower()
        data["layout_mode"] = mode if mode in {"center", "wide"} else "center"
    if "section_order" in data:
        data["section_order"] = _normalize_section_order(data.get("section_order"))
    if "canvas_layout" in data:
        data["canvas_layout"] = _normalize_canvas_layout(data.get("canvas_layout"))
    if "discord_badge_codes" in data:
        data["discord_badge_codes"] = _normalize_text_list(
            data.get("discord_badge_codes"),
            max_items=50,
            max_len=64,
        )
    data.setdefault("background_is_video", False)
    data.setdefault("transparency", 0)
    data.setdefault("name_effect", "none")
    data.setdefault("name_font_family", "default")
    data.setdefault("background_effect", "none")
    data.setdefault("cursor_effect", "none")
    data.setdefault("cursor_effect_alpha", 70)
    data.setdefault("display_name_mode", "slug")
    data.setdefault("layout_mode", "center")
    data.setdefault("link_bg_alpha", 100)
    data.setdefault("link_icons_only_size", 36)
    data.setdefault("link_icons_only_gap", 12)
    data.setdefault("link_icons_only_grouped", False)
    data.setdefault("link_icons_only_direction", "row")
    data.setdefault("audio_player_bg_alpha", 60)
    data.setdefault("visit_counter_bg_alpha", 20)
    data.setdefault("show_visit_counter", False)
    data.setdefault("discord_frame_enabled", False)
    data.setdefault("show_audio_player", False)
    data.setdefault("discord_presence_enabled", False)
    data.setdefault("discord_presence", "online")
    data.setdefault("discord_status_enabled", False)
    data.setdefault("discord_badges_enabled", False)
    data.setdefault("quote_typing_enabled", False)
    data.setdefault("quote_font_family", "default")
    data.setdefault("quote_effect", "none")
    data.setdefault("quote_effect_strength", 70)
    data.setdefault("entry_bg_alpha", 85)
    data.setdefault("entry_font_size", 16)
    data.setdefault("entry_font_family", "default")
    data.setdefault("entry_effect", "none")
    data.setdefault("entry_overlay_alpha", 35)
    data.setdefault("entry_box_enabled", True)
    data.setdefault("entry_border_enabled", True)
    data.setdefault("demo_show_links", False)
    for key in ("demo_link_label", "demo_link_url", "demo_link_icon_url"):
        if key in data and isinstance(data[key], str):
            data[key] = data[key].strip() or None
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
        "quote_typing_enabled",
        "quote_typing_texts",
        "entry_text",
        "quote_typing_speed",
        "quote_typing_pause",
        "quote_font_size",
        "quote_font_family",
        "quote_effect",
        "quote_effect_strength",
        "song_url",
        "song_name",
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
        "name_font_family",
        "background_effect",
        "display_name_mode",
        "layout_mode",
        "custom_display_name",
        "linktree_profile_picture",
        "section_order",
        "canvas_layout",
        "link_color",
        "link_bg_color",
        "link_bg_alpha",
        "link_columns",
        "link_icons_only",
        "link_icons_only_size",
        "link_icons_only_gap",
        "link_icons_only_grouped",
        "link_icons_only_direction",
        "card_color",
        "text_color",
        "name_color",
        "location_color",
        "quote_color",
        "entry_bg_alpha",
        "entry_text_color",
        "entry_font_size",
        "entry_font_family",
        "entry_effect",
        "entry_overlay_alpha",
        "entry_box_enabled",
        "entry_border_enabled",
        "entry_border_color",
        "cursor_url",
        "cursor_effect",
        "cursor_effect_color",
        "cursor_effect_alpha",
        "discord_frame_enabled",
        "discord_presence_enabled",
        "discord_presence",
        "discord_status_enabled",
        "discord_status_text",
        "discord_badges_enabled",
        "discord_badge_codes",
        "show_visit_counter",
        "visit_counter_color",
        "visit_counter_bg_color",
        "visit_counter_bg_alpha",
    }
    fields = {k: v for k, v in data.items() if k in allowed}
    if "quote_typing_texts" in fields:
        fields["quote_typing_texts"] = _list_to_json(
            fields.get("quote_typing_texts"),
            max_items=3,
            max_len=180,
        )
    if "section_order" in fields:
        order = _normalize_section_order(fields.get("section_order"))
        fields["section_order"] = json.dumps(order) if order else None
    if "canvas_layout" in fields:
        layout = _normalize_canvas_layout(fields.get("canvas_layout"))
        fields["canvas_layout"] = json.dumps(layout) if layout is not None else None
    if "quote_typing_speed" in fields:
        try:
            speed = int(fields.get("quote_typing_speed"))
        except Exception:
            speed = None
        fields["quote_typing_speed"] = (
            max(20, min(200, speed)) if speed is not None else None
        )
    if "quote_typing_pause" in fields:
        try:
            pause = int(fields.get("quote_typing_pause"))
        except Exception:
            pause = None
        fields["quote_typing_pause"] = (
            max(200, min(10000, pause)) if pause is not None else None
        )
    if "quote_font_size" in fields:
        try:
            size = int(fields.get("quote_font_size"))
        except Exception:
            size = None
        fields["quote_font_size"] = (
            max(10, min(40, size)) if size is not None else None
        )
    if "quote_font_family" in fields:
        fam = str(fields.get("quote_font_family") or "").lower()
        if fam in {"default", "serif", "mono", "script", "display"}:
            fields["quote_font_family"] = fam
        else:
            fields["quote_font_family"] = "default"
    if "quote_effect" in fields:
        fx = str(fields.get("quote_effect") or "").lower()
        if fx in {"none", "glow", "neon", "rainbow"}:
            fields["quote_effect"] = fx
        else:
            fields["quote_effect"] = "none"
    if "entry_bg_alpha" in fields:
        try:
            alpha = int(fields.get("entry_bg_alpha"))
        except Exception:
            alpha = None
        fields["entry_bg_alpha"] = (
            max(0, min(100, alpha)) if alpha is not None else None
        )
    if "entry_font_size" in fields:
        try:
            size = int(fields.get("entry_font_size"))
        except Exception:
            size = None
        fields["entry_font_size"] = (
            max(10, min(40, size)) if size is not None else None
        )
    if "entry_font_family" in fields:
        fam = str(fields.get("entry_font_family") or "default").lower()
        fields["entry_font_family"] = (
            fam if fam in {"default", "serif", "mono", "script", "display"} else "default"
        )
    if "entry_effect" in fields:
        fx = str(fields.get("entry_effect") or "none").lower()
        fields["entry_effect"] = (
            fx if fx in {"none", "glow", "neon", "rainbow"} else "none"
        )
    if "entry_text_color" in fields and isinstance(fields.get("entry_text_color"), str):
        val = fields.get("entry_text_color").strip()
        fields["entry_text_color"] = val if re.match(HEX_COLOR_RE, val) else None
    if "link_columns" in fields:
        try:
            cols = int(fields.get("link_columns"))
        except Exception:
            cols = None
        if cols is not None:
            cols = max(1, min(8, cols))
        fields["link_columns"] = cols
    if "link_icons_only" in fields:
        fields["link_icons_only"] = bool(fields.get("link_icons_only"))
    if "link_icons_only_size" in fields:
        try:
            size = int(fields.get("link_icons_only_size"))
        except Exception:
            size = None
        fields["link_icons_only_size"] = (
            max(16, min(128, size)) if size is not None else None
        )
    if "link_icons_only_gap" in fields:
        try:
            gap = int(fields.get("link_icons_only_gap"))
        except Exception:
            gap = None
        fields["link_icons_only_gap"] = max(0, min(64, gap)) if gap is not None else None
    if "link_icons_only_grouped" in fields:
        fields["link_icons_only_grouped"] = bool(fields.get("link_icons_only_grouped"))
    if "link_icons_only_direction" in fields:
        direction = str(fields.get("link_icons_only_direction") or "row").lower()
        fields["link_icons_only_direction"] = (
            direction if direction in {"row", "column"} else "row"
        )
    if "discord_badge_codes" in fields:
        fields["discord_badge_codes"] = _list_to_json(
            fields.get("discord_badge_codes"),
            max_items=50,
            max_len=64,
            allow_empty=True,
        )
    if "entry_text" in fields and isinstance(fields["entry_text"], str):
        fields["entry_text"] = fields["entry_text"].strip() or None
    return fields

from .routes.portfolio import register_portfolio_routes
from .routes.gif import register_gif_routes
from .routes.linktree import register_linktree_routes
from .routes.rest import register_rest_routes

register_portfolio_routes(app, globals())
register_gif_routes(app, globals())
register_linktree_routes(app, globals())
register_rest_routes(app, globals())
