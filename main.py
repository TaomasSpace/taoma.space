from fastapi import FastAPI, HTTPException, Query, Header, Depends
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel
import os
from fastapi import Depends, Header

from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi import Response


app = FastAPI(title="Anime GIF API", version="0.1.0")
DATABASE_URL = os.getenv("DATABASE_URL")  # von Render
from fastapi.middleware.cors import CORSMiddleware

ADMIN_PASSWORD = os.getenv("GIFAPI_ADMIN_PASSWORD", "")

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


@app.get("/", response_class=HTMLResponse)
def root():
    path = Path(__file__).resolve().parents[1] / "index.html"
    return path.read_text(encoding="utf-8")
