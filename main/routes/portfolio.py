"""Route registrations extracted from main.py."""

def register_portfolio_routes(app, ns):
    globals().update(ns)

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

    @app.get("/projects", include_in_schema=False)
    def projects():
        return RedirectResponse(url="/portfolio/projects", status_code=308)

    @app.get("/about", include_in_schema=False)
    def about():
        return RedirectResponse(url="/portfolio/about", status_code=308)

    @app.get("/contact", include_in_schema=False)
    def contact():
        return RedirectResponse(url="/portfolio/contact", status_code=308)

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

