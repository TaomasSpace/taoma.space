"""Route registrations extracted from main.py."""

def register_gif_routes(app, ns):
    globals().update(ns)

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
                return does_url_still_exist(
                    lambda: db.get_random_by_tag(tag, nsfw_mode=nsfw_mode)
                )
            if anime:
                return does_url_still_exist(
                    lambda: db.get_random_by_anime(anime, nsfw_mode=nsfw_mode)
                )
            if character:
                return does_url_still_exist(
                    lambda: db.get_random_by_character(character, nsfw_mode=nsfw_mode)
                )
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
            return does_url_still_exist(lambda: db.get_random(nsfw_mode=nsfw_mode))
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
            gif = does_url_still_exist(lambda: db.get_random(nsfw_mode=nsfw_mode))
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

