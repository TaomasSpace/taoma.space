from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
from pydantic import BaseModel, EmailStr, Field
import os, httpx
from dotenv import load_dotenv
import logging

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


class Contact(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    email: EmailStr
    subject: str = Field(..., min_length=1, max_length=200)
    message: str = Field(..., min_length=10, max_length=5000)
    website: str | None = None  # Honeypot


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
            r.raise_for_status()  # wirft bei 4xx/5xx
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
