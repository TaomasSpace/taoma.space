"""Route registrations extracted from main.py."""

def register_rest_routes(app, ns):
    globals().update(ns)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse("static/icon.png")

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
            "email": user.get("email"),
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
        email = payload.email.strip().lower() if payload.email else None
        if email and db.getUserByEmail(email):
            raise HTTPException(status_code=409, detail="email already exists")
        password_raw = payload.password.strip() if payload.password else None
        if not password_raw:
            if not email:
                raise HTTPException(status_code=400, detail="email required")
            password_raw = secrets.token_urlsafe(12)
        try:
            new_id = db.createUser(
                username=payload.username.strip(),
                email=email,
                hashed_password=hashPassword(password_raw),
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
        if payload.email is not None:
            email = payload.email.strip().lower()
            if not email:
                raise HTTPException(400, "email required")
            other = db.getUserByEmail(email)
            if other and other["id"] != user_id:
                raise HTTPException(409, "email already exists")

        admin_value = payload.admin if is_admin else None

        try:
            db.updateUser(
                user_id,
                username=payload.username.strip() if payload.username is not None else None,
                email=payload.email.strip().lower() if payload.email is not None else None,
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
        identifier = payload.identifier.strip()
        email_input = payload.email.strip().lower() if payload.email else None
        user = db.getUserByEmail(identifier) or db.getUserByUsername(identifier)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not CheckPassword(user["password"], payload.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        existing_email = (user.get("email") or "").strip()
        if not existing_email:
            if not email_input:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "email_required",
                        "message": "Email required for this account.",
                    },
                )
            other = db.getUserByEmail(email_input)
            if other and other["id"] != user["id"]:
                raise HTTPException(status_code=409, detail="email already exists")
            db.updateUser(user["id"], email=email_input)
        elif email_input and existing_email.lower() != email_input.lower():
            raise HTTPException(status_code=400, detail="email does not match this account")
        token = db.create_token(hours_valid=24, user_id=user["id"])
        expires = db.get_token_expiry(token)
        return _session_response({"token": token, "expires_at": expires}, token)

    @app.post("/api/auth/register", response_model=RegisterOut)
    def register(payload: RegisterIn):
        uname = payload.username.strip()
        email = payload.email.strip().lower()
        if db.getUserByUsername(uname):
            raise HTTPException(status_code=409, detail="username already exists")
        if db.getUserByEmail(email):
            raise HTTPException(status_code=409, detail="email already exists")

        # hash
        hashed = hashPassword(payload.password)

        try:
            new_id = db.createUser(
                username=uname,
                email=email,
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
            "email": row.get("email"),
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

    @app.post("/api/auth/password/request")
    def request_password_reset(
        payload: PasswordResetRequestIn, background: BackgroundTasks
    ):
        email = payload.email.strip().lower()
        user = db.getUserByEmail(email)
        if not user:
            return {"ok": True}
        token = _create_password_reset(user["id"])
        reset_url = f"{PUBLIC_BASE_URL}/reset?token={token}"
        subject = "Reset your TAOMA password"
        body = (
            "We received a request to reset your password.\n\n"
            f"Reset link: {reset_url}\n\n"
            "If you did not request this, you can ignore this email."
        )
        if _email_configured():
            background.add_task(_send_email, email, subject, body)
        else:
            logger.warning("Password reset requested but SMTP is not configured")
        return {"ok": True}

    @app.post("/api/auth/password/reset")
    def reset_password(payload: PasswordResetIn):
        token = payload.token.strip()
        user_id = _consume_password_reset(token)
        db.updateUser(user_id, password=hashPassword(payload.password))
        if isinstance(db, PgGifDB):
            db.revoke_user_tokens(user_id)
        return {"ok": True}

    @app.get("/register", include_in_schema=False)
    def register_page():
        return FileResponse("register.html")

    @app.get("/login", include_in_schema=False)
    def login_page():
        return FileResponse("login.html")

    @app.get("/reset", include_in_schema=False)
    def reset_page():
        return FileResponse("reset.html")

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

        if payload.email is not None:
            email = payload.email.strip().lower()
            if not email:
                raise HTTPException(400, "email required")
            other = db.getUserByEmail(email)
            if other and other["id"] != current["id"]:
                raise HTTPException(409, "email already exists")

        try:
            db.updateUser(
                current["id"],
                username=payload.username.strip() if payload.username is not None else None,
                email=payload.email.strip().lower() if payload.email is not None else None,
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

    @app.get("/nanna/birthdays/2025", include_in_schema=False)
    def home():
        return FileResponse("nannaBirthday2025.html")

    @app.get("/alexandra/data/show", dependencies=[Depends(require_specific_user)])
    def very_private_page():
        return FileResponse("healthData.html")

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

    @app.get("/globalChat", include_in_schema=False)
    def globalChat():
        return FileResponse("globalChat.html")

    @app.get("/datenschutz.html", include_in_schema=False)
    def privacyPolicy():
        return FileResponse("datenschutz.html")

    @app.get("/impressum.html", include_in_schema=False)
    def impressum_page():
        return FileResponse("impressum.html")

