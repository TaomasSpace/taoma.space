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
import os, hmac, base64, hashlib
from typing import Tuple
from fastapi import Path
from typing import Literal
import psycopg

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


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def require_token(x_auth_token: str | None = Header(None, alias="X-Auth-Token")):
    if not x_auth_token or not db.validate_token(x_auth_token):
        raise HTTPException(401, "Unauthorized")
    return x_auth_token


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


import bcrypt
from psycopg.rows import dict_row

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
    slug: str = Field(..., min_length=2, max_length=48, regex=r"^[a-zA-Z0-9_-]+$")
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    background_url: Optional[str] = None
    background_is_video: bool = False
    transparency: int = Field(0, ge=0, le=100)
    name_effect: EffectName = "none"
    background_effect: BgEffectName = "none"


class LinktreeUpdateIn(BaseModel):
    slug: Optional[str] = Field(
        None, min_length=2, max_length=48, regex=r"^[a-zA-Z0-9_-]+$"
    )
    location: Optional[str] = None
    quote: Optional[str] = None
    song_url: Optional[str] = None
    background_url: Optional[str] = None
    background_is_video: Optional[bool] = None
    transparency: Optional[int] = Field(None, ge=0, le=100)
    name_effect: Optional[EffectName] = None
    background_effect: Optional[BgEffectName] = None


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
    links: List[LinkOut]
    icons: List[IconOut]


class IconUpsertIn(BaseModel):
    code: str = Field(..., min_length=2, max_length=64, regex=r"^[a-z0-9_\-]+$")
    image_url: str
    description: Optional[str] = None


class GrantIconIn(BaseModel):
    displayed: bool = False


class ToggleDisplayedIn(BaseModel):
    displayed: bool


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
    user = None
    try:
        user = db.get_token_user(x_auth_token)
    except AttributeError:
        user = None
    # mask sensible Felder
    if user:
        user_out = {
            "id": user["id"],
            "username": user["username"],
            "admin": bool(user.get("admin", False)),
            "linktree_id": user.get("linktree_id"),
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
    else:
        user_out = None
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
    return {"token": token, "expires_at": expires}


@app.post("/api/auth/logout")
def logout(x_auth_token: str | None = Header(default=None, alias="X-Auth-Token")):
    if not x_auth_token:
        raise HTTPException(400, "Missing token")
    db.revoke_token(x_auth_token)
    return {"ok": True}


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
    user_id: int = Path(..., ge=1),
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
    return {"token": token, "expires_at": expires}


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
    return {"token": token, "expires_at": exp, "user": user_out}


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


# ---------- Linktree: public read ----------


@app.get("/api/linktrees/{slug}", response_model=LinktreeOut)
def get_linktree(slug: str):
    _ensure_pg()
    lt = db.get_linktree_by_slug(slug)
    if not lt:
        raise HTTPException(404, "Linktree not found")

    # Nur **angezeigte** Icons nach außen geben
    icons = [
        {
            "id": i["id"],
            "code": i["code"],
            "image_url": i["image_url"],
            "description": i.get("description"),
            "displayed": i.get("displayed", False),
            "acquired_at": (
                i["acquired_at"].isoformat() if i.get("acquired_at") else None
            ),
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
    # kleine Readback-Query:
    lt = db.get_linktree_by_slug(
        db.get_linktree_by_slug.__self__.slug if False else db.get_linktree_by_slug
    )  # no-op trick
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
    # Owner ermitteln via Join
    with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT lt.user_id, l.linktree_id
              FROM linktree_links l
              JOIN linktrees lt ON lt.id = l.linktree_id
             WHERE l.id = %s
        """,
            (link_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(404, "Link not found")
        owner_id, linktree_id = int(row[0]), int(row[1])
    if not (user.get("admin") or user["id"] == owner_id):
        raise HTTPException(403, "Forbidden (owner or admin only)")

    db.update_link(
        link_id, **{k: v for k, v in payload.model_dump(exclude_unset=True).items()}
    )
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


@app.get("/linktree/config", include_in_schema=False)
def linktree_config_page():
    return FileResponse("linktree_config.html")
