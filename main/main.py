from fastapi import FastAPI
import os
from fastapi import Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse


app = FastAPI(title="Anime GIF API", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

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
    allow_credentials=False,  # wir nutzen Header-Token, keine Cookies
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Auth-Token"],
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/icon.png")


@app.get("/projects", include_in_schema=False)
def projects():
    return FileResponse("projects.html")


@app.get("/about")
def about():
    return FileResponse("about.html")


@app.get("/contact")
def contact():
    return FileResponse(contact.html)
