from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
import os, httpx
from dotenv import load_dotenv
import logging
import json, html
from datetime import datetime, timezone
from fastapi import BackgroundTasks
import asyncio, uuid
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
from fastapi.responses import HTMLResponse
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
import threading
from psycopg import errors as pg_errors
import os, base64
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

DisplayNameMode = Literal['slug','username']
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
MAX_AUDIO_BYTES = 10 * 1024 * 1024  # 10 MB
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

def _count_db_references(url: str) -> int:
    if not url:
        return 0
    if not isinstance(db, PgGifDB):
        # SQLite-Pfad: implementiere hier ggf. analog, oder immer 0 zurückgeben
        with db_lock:
            pass
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        total = 0
        cur.execute("SELECT COUNT(*) FROM users WHERE profile_picture = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM linktrees WHERE song_url = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM linktrees WHERE background_url = %s", (url,))
        total += int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM linktree_links WHERE icon_url = %s", (url,))
        total += int(cur.fetchone()[0])
        return total

def _delete_if_unreferenced(url: str) -> bool:
    """Löscht die Datei, wenn lokales Media und DB-Referenzen == 0.
    Gibt True zurück, wenn gelöscht wurde."""
    if not _is_local_media_url(url):
        return False
    if _count_db_references(url) > 0:
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
    
def _session_response(payload: dict, token: str, max_age: int = 24*3600) -> JSONResponse:
    resp = JSONResponse(payload)
    resp.set_cookie(
        key="taoma_token",
        value=token,
        max_age=max_age,
        httponly=True,     # schützt vor JS-Zugriff
        secure=True,       # nur über HTTPS
        samesite="lax"     # Navigation-Links funktionieren
    )
    return resp
    
def _extract_token(
    x_auth_token: str | None,
    authorization: str | None,
    request: Request
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
    # 4) ?token=... als Queryparam
    tok = request.query_params.get("token")
    if tok:
        return tok
    return None


DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
logger = logging.getLogger("uvicorn.error")

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

MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # 2 MB
ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif"}


EffectName = Literal["none", "glow", "neon", "rainbow"]
BgEffectName = Literal["none", "night", "rain", "snow"]


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


class LinktreeCreateIn(BaseModel):
    slug: str = Field(..., min_length=2, max_length=48, pattern=r"^[a-zA-Z0-9_-]+$")
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    background_url: Optional[str] = None
    background_is_video: bool = False
    transparency: int = Field(0, ge=0, le=100)
    name_effect: EffectName = "none"
    background_effect: BgEffectName = "none"
    display_name_mode: DisplayNameMode = 'slug'


class LinktreeUpdateIn(BaseModel):
    slug: Optional[str] = Field(
        None, min_length=2, max_length=48, pattern=r"^[a-zA-Z0-9_-]+$"
    )
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    background_url: Optional[str] = None
    background_is_video: Optional[bool] = None
    transparency: Optional[int] = Field(None, ge=0, le=100)
    name_effect: Optional[EffectName] = None
    background_effect: Optional[BgEffectName] = None
    display_name_mode: Optional[DisplayNameMode] = None


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
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    background_url: Optional[str] = None
    background_is_video: bool
    transparency: int
    name_effect: EffectName
    background_effect: BgEffectName
    display_name_mode: DisplayNameMode                 # NEU
    profile_picture: Optional[str] = None              # NEU – fürs Avatar
    user_username: Optional[str] = None                # NEU – für „username“-Modus
    links: List[LinkOut]
    icons: List[IconOut]


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
    characters: List[str] = []
    tags: List[str] = []


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


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/icon.png")


@app.get("/projects", include_in_schema=False)
def projects():
    return FileResponse("projects.html")


@app.get("/about", include_in_schema=False)
def about():
    return FileResponse("about.html")


@app.get("/contact", include_in_schema=False)
def contact():
    return FileResponse("contact.html")


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
    return FileResponse("skills.html")


# ------------ GIF API ------------ #
@app.get("/api/auth/verify")
def verify(x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")):
    if not x_auth_token or not db.validate_token(x_auth_token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    user = db.get_token_user(x_auth_token) or {}

    # try to resolve slug if missing
    linktree_slug = user.get("linktree_slug")
    if not linktree_slug and user.get("linktree_id") and isinstance(db, PgGifDB):
        try:
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute("SELECT slug FROM linktrees WHERE id=%s", (user["linktree_id"],))
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
        "linktree_slug": linktree_slug,       # <-- add this
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
        "expires_at": db.get_token_expiry(x_auth_token),
        "user": user_out,
    }


@app.get(
    "/api/admin", response_class=HTMLResponse, dependencies=[Depends(require_admin)]
)
def admin_page():
    return FileResponse("gifApiAdmin.html")


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
def logout(x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")):
    if x_auth_token:
        db.revoke_token(x_auth_token)
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


@app.post(
    "/api/gifs",
    response_model=GifOut,
    status_code=201,
    dependencies=[Depends(require_user)],
)
def create_or_update_gif(payload: GifIn):
    try:
        existing = None
        try:
            existing = db.get_gif_by_url(str(payload.url))
        except KeyError:
            pass

        if existing:
            db.update_gif(
                existing["id"],
                title=payload.title,
                nsfw=payload.nsfw,
                anime=payload.anime,
                characters=payload.characters,
                tags=payload.tags,
            )
            return db.get_gif(existing["id"])

        gif_id = db.insert_gif(
            title=payload.title,
            url=str(payload.url),
            nsfw=payload.nsfw,
            anime=payload.anime,
            characters=payload.characters,
            tags=payload.tags,
        )
        return db.get_gif(gif_id)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/gifs/{gif_id}", response_model=GifOut)
def read_gif(gif_id: int):
    try:
        return db.get_gif(gif_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="GIF not found")


@app.patch(
    "/api/gifs/{gif_id}", response_model=GifOut, dependencies=[Depends(require_user)]
)
def update_gif(gif_id: int, payload: GifUpdate):
    try:
        _ = db.get_gif(gif_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="GIF not found")

    db.update_gif(
        gif_id,
        title=payload.title,
        url=str(payload.url) if payload.url is not None else None,
        nsfw=payload.nsfw,
        anime=payload.anime,
        characters=payload.characters,
        tags=payload.tags,
    )
    return db.get_gif(gif_id)


@app.delete("/api/gifs/{gif_id}", status_code=204, dependencies=[Depends(require_user)])
def delete_gif(gif_id: int):
    try:
        _ = db.get_gif(gif_id)  # 404, falls nicht vorhanden
    except KeyError:
        raise HTTPException(status_code=404, detail="GIF not found")
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
    return _session_response({"token": token, "expires_at": exp, "user": user_out}, token)


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


def _safe_image_bytes(b: bytes) -> str:
    """
    Validiert, dass b ein echtes Bild ist und gibt das ermittelte Format zurück
    (z.B. 'PNG', 'JPEG', 'WEBP', 'GIF'). Wir re-encoden NICHT, um Animationen
    (GIF/WEBP) zu erhalten – nur Validierung.
    """
    try:
        with Image.open(io.BytesIO(b)) as img:
            img.verify()  # prüft Header & Konsistenz
            fmt = (img.format or "").upper()
            return fmt
    except UnidentifiedImageError:
        return ""
    except Exception:
        return ""


@app.post("/api/users/me/avatar", response_model=UserOut)
async def upload_avatar(file: UploadFile = File(...), current: dict = Depends(require_user)):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, "Unsupported media type")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 2MB)")

    detected = _safe_image_bytes(data)
    if not detected:
        raise HTTPException(400, "File is not a valid image")

    # Altes Bild merken (für Cleanup)
    old_url = None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT profile_picture FROM users WHERE id=%s", (current["id"],))
            row = cur.fetchone()
            old_url = row[0] if row else None
    except Exception:
        old_url = None

    ext = {"image/png":"png","image/jpeg":"jpg","image/webp":"webp","image/gif":"gif"}[file.content_type]
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
def get_linktree(slug: str):
    _ensure_pg()
    lt = db.get_linktree_by_slug(slug)
    if not lt:
        raise HTTPException(404, "Linktree not found")

    # Username & Avatar des Owners holen
    user_username = None
    user_pfp = None
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT username, profile_picture FROM users WHERE id=%s", (lt["user_id"],))
        row = cur.fetchone()
        if row:
            user_username, user_pfp = row[0], row[1]

    icons = [
        {
            "id": i["id"], "code": i["code"], "image_url": i["image_url"],
            "description": i.get("description"),
            "displayed": i.get("displayed", False),
            "acquired_at": (i["acquired_at"].isoformat() if i.get("acquired_at") else None),
        }
        for i in lt["icons"] if i.get("displayed")
    ]

    links = [
        {
            "id": r["id"], "url": r["url"], "label": r.get("label"),
            "icon_url": r.get("icon_url"), "position": r.get("position", 0),
            "is_active": r.get("is_active", True),
        }
        for r in lt["links"] if r.get("is_active", True)
    ]

    return {
        "id": lt["id"],
        "user_id": lt["user_id"],
        "slug": lt["slug"],
        "location": lt.get("location"),
        "quote": lt.get("quote"),
        "song_url": lt.get("song_url"),
        "background_url": lt.get("background_url"),
        "background_is_video": lt.get("background_is_video", False),
        "transparency": lt.get("transparency", 0),
        "name_effect": lt.get("name_effect", "none"),
        "background_effect": lt.get("background_effect", "none"),
        "display_name_mode": lt.get("display_name_mode", "slug"),
        "profile_picture": user_pfp,              # <= jetzt dabei
        "user_username": user_username,           # <= jetzt dabei
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
    # Ein Tree pro User – DB hat UNIQUE(user_id), aber wir geben eine saubere 409
    existing = db.get_linktree_by_slug(payload.slug)
    if existing and existing["user_id"] == user["id"]:
        raise HTTPException(409, "You already have a linktree with this slug")
    try:
        linktree_id = db.create_linktree(
            user_id=user["id"],
            slug=payload.slug,
            location=payload.location,
            quote=payload.quote,
            song_url=payload.song_url,
            background_url=payload.background_url,
            background_is_video=payload.background_is_video,
            transparency=payload.transparency,
            name_effect=payload.name_effect,
            background_effect=payload.background_effect,
            display_name_mode=payload.display_name_mode,  # <-- NEU
        )
    except pg_errors.UniqueViolation:
        raise HTTPException(409, "Slug already in use")
    lt = db.get_linktree_by_slug(payload.slug)
    # public shaping wie oben:
    return get_linktree(payload.slug)


@app.patch("/api/linktrees/{linktree_id}", response_model=dict)
def update_linktree_ep(
    linktree_id: int, payload: LinktreeUpdateIn, user: dict = Depends(require_user)
):
    _ensure_pg()
    _require_tree_owner_or_admin(linktree_id, user)
    try:
        db.update_linktree(
            linktree_id,
            **{k: v for k, v in payload.model_dump(exclude_unset=True).items()},
        )
    except pg_errors.UniqueViolation:
        raise HTTPException(409, "Slug already in use")
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
def update_link_ep(link_id: int, payload: LinkUpdateIn, user: dict = Depends(require_user)):
    _ensure_pg()
    with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT lt.user_id, l.linktree_id, l.icon_url
              FROM linktree_links l
              JOIN linktrees lt ON lt.id = l.linktree_id
             WHERE l.id = %s
        """, (link_id,))
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Link not found")
        owner_id, linktree_id, old_icon = int(row["user_id"]), int(row["linktree_id"]), row["icon_url"]

    if not (user.get("admin") or user["id"] == owner_id):
        raise HTTPException(403, "Forbidden (owner or admin only)")

    # Werde die neue icon_url gleich aus dem Payload holen
    new_icon = payload.icon_url if payload.icon_url is not None else old_icon

    fields = payload.model_dump(exclude_unset=True)
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
async def upload_song(file: UploadFile = File(...), current: dict = Depends(require_user)):
    if file.content_type not in ALLOWED_AUDIO:
        raise HTTPException(415, "Unsupported audio type")
    data = await file.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "File too large (max 10MB)")

    old_url = None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT lt.song_url
                  FROM linktrees lt
                  JOIN users u ON u.linktree_id = lt.id
                 WHERE u.id = %s
            """, (current["id"],))
            row = cur.fetchone()
            old_url = row[0] if row else None
    except Exception:
        old_url = None

    ext = file.filename.split(".")[-1].lower()
    fname = f"user{current['id']}_song_{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"
    db.update_linktree_by_user(current["id"], song_url=url)

    # Cleanup
    if old_url and old_url != url:
        _delete_if_unreferenced(old_url)

    return {"url": url}


@app.post("/api/users/me/background")
async def upload_background(file: UploadFile = File(...), current: dict = Depends(require_user)):
    if file.content_type not in ALLOWED_MIME and not file.content_type.startswith("video/"):
        raise HTTPException(415, "Unsupported background type")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES * 5:
        raise HTTPException(413, "File too large (max 10MB)")

    # Altes BG merken
    old_url = None
    try:
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT lt.background_url
                  FROM linktrees lt
                  JOIN users u ON u.linktree_id = lt.id
                 WHERE u.id = %s
            """, (current["id"],))
            row = cur.fetchone()
            old_url = row[0] if row else None
    except Exception:
        old_url = None

    ext = file.filename.split(".")[-1].lower()
    fname = f"user{current['id']}_bg_{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"
    is_video = file.content_type.startswith("video/")

    db.update_linktree_by_user(current["id"], background_url=url, background_is_video=is_video)

    # Cleanup
    if old_url and old_url != url:
        _delete_if_unreferenced(old_url)

    return {"url": url, "is_video": is_video}
@app.post("/api/users/me/linkicon")
async def upload_linkicon(file: UploadFile = File(...), current: dict = Depends(require_user)):
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(415, "Unsupported image type")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 2MB)")
    ext = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
    }[file.content_type]
    fname = f"user{current['id']}_icon_{uuid.uuid4().hex}.{ext}"
    out_path = UPLOAD_DIR / fname
    out_path.write_bytes(data)
    url = f"/media/{UPLOAD_DIR.name}/{fname}"
    return {"url": url}

@app.post("/api/admin/media/gc", dependencies=[Depends(require_admin)])
def media_garbage_collect(min_age_seconds: int = 60):
    """Löscht unreferenzierte Dateien im Upload-Verzeichnis, die älter als min_age_seconds sind."""
    deleted = []
    skipped = []
    now = datetime.now().timestamp()

    for p in UPLOAD_DIR.iterdir():
        if not p.is_file():
            continue
        url = f"/media/{UPLOAD_DIR.name}/{p.name}"
        age = now - p.stat().st_mtime
        if age < min_age_seconds:
            skipped.append(p.name)  # zu frisch
            continue
        try:
            if _is_local_media_url(url) and _count_db_references(url) == 0:
                p.unlink(missing_ok=True)
                deleted.append(p.name)
            else:
                skipped.append(p.name)
        except Exception as e:
            logger.warning("GC failed for %s: %s", p, e)
            skipped.append(p.name)

    return {"ok": True, "deleted": deleted, "skipped": skipped}