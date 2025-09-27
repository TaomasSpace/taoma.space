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

load_dotenv()
app = FastAPI(title="Anime GIF API", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
logger = logging.getLogger("uvicorn.error")  # nutzt Uvicorn-Logger

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

class Echo(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    meta: dict | None = None
class JobIn(BaseModel):
    seconds: int = Field(3, ge=1, le=20)
class Contact(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    subject: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=10, max_length=5000)
    website: str | None = None  # Honeypot

JOB_STORE: dict[str, dict] = {}

@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")


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
async def echo(payload: Echo):
    data = payload.model_dump()
    parts = []
    for key, value in data.items():
        key_safe = html.escape(str(key))
        value_safe = html.escape(json.dumps(value, ensure_ascii=False))
        parts.append(f'<p><strong>{key_safe}</strong> = <code>{value_safe}</code></p>')
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