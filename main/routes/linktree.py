"""Route registrations extracted from main.py."""

from ..main import *  # noqa: F401,F403


def register_linktree_routes(app):

    @app.get("/api/linktrees/{linktree_id}/manage", response_model=LinktreeOut)
    def get_linktree_manage(linktree_id: int, user: dict = Depends(require_user)):
        _ensure_pg()
        _require_tree_owner_or_admin(linktree_id, user)

        with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            # Linktree-Stammdaten
            cur.execute(
                """
                SELECT id, user_id, slug, location, quote, song_url, song_name, song_icon_url, background_url,
                       COALESCE(quote_typing_enabled, false) AS quote_typing_enabled,
                       quote_typing_texts,
                       quote_typing_speed,
                       quote_typing_pause,
                       quote_font_size,
                       quote_font_family,
                       COALESCE(quote_effect, 'none') AS quote_effect,
                       COALESCE(quote_effect_strength, 70) AS quote_effect_strength,
                       entry_text,
                       COALESCE(entry_bg_alpha, 85) AS entry_bg_alpha,
                       entry_text_color,
                       COALESCE(entry_font_size, 16) AS entry_font_size,
                       COALESCE(entry_font_family, 'default') AS entry_font_family,
                       COALESCE(entry_effect, 'none') AS entry_effect,
                       COALESCE(entry_overlay_alpha, 35) AS entry_overlay_alpha,
                       COALESCE(entry_box_enabled, true) AS entry_box_enabled,
                       COALESCE(entry_border_enabled, true) AS entry_border_enabled,
                       entry_border_color,
                       COALESCE(show_audio_player, false) AS show_audio_player,
                       audio_player_bg_color,
                       COALESCE(audio_player_bg_alpha, 60) AS audio_player_bg_alpha,
                       audio_player_text_color,
                       audio_player_accent_color,
                       COALESCE(background_is_video, false) AS background_is_video,
                       COALESCE(transparency, 0)          AS transparency,
                       COALESCE(name_effect, 'none')       AS name_effect,
                       COALESCE(name_font_family, 'default') AS name_font_family,
                       COALESCE(background_effect,'none')  AS background_effect,
                       device_type,
                       COALESCE(display_name_mode,'slug')  AS display_name_mode,
                       COALESCE(layout_mode,'center')      AS layout_mode,
                        custom_display_name,
                        linktree_profile_picture,
                        section_order,
                        canvas_layout,
                        link_color,
                        link_bg_color,
                        COALESCE(link_bg_alpha, 100)        AS link_bg_alpha,
                        link_columns,
                        COALESCE(link_icons_only, false)    AS link_icons_only,
                        card_color,
                       text_color,
                       name_color,
                       location_color,
                       quote_color,
                       cursor_url,
                       COALESCE(cursor_effect, 'none') AS cursor_effect,
                       cursor_effect_color,
                       COALESCE(cursor_effect_alpha, 70) AS cursor_effect_alpha,
                       COALESCE(discord_frame_enabled, false) AS discord_frame_enabled,
                        COALESCE(discord_presence_enabled, false) AS discord_presence_enabled,
                        COALESCE(discord_presence, 'online') AS discord_presence,
                        COALESCE(discord_status_enabled, false) AS discord_status_enabled,
                        discord_status_text,
                        COALESCE(discord_badges_enabled, false) AS discord_badges_enabled,
                        discord_badge_codes,
                        COALESCE(show_visit_counter, false) AS show_visit_counter,
                        visit_counter_color,
                        visit_counter_bg_color,
                        COALESCE(visit_counter_bg_alpha, 20) AS visit_counter_bg_alpha
                   FROM linktrees
                  WHERE id = %s
            """,
                (linktree_id,),
            )
            lt = cur.fetchone()
            if not lt:
                raise HTTPException(404, "Linktree not found")

            # Username + Avatar des Owners
            cur.execute(
                "SELECT username, profile_picture FROM users WHERE id=%s", (lt["user_id"],)
            )
            urow = cur.fetchone()
            user_username = urow["username"] if urow else None
            user_pfp = urow["profile_picture"] if urow else None

            # Links (UNGEFILTERT → auch is_active=false)
            cur.execute(
                """
                SELECT id, url, label, icon_url, position, is_active
                  FROM linktree_links
                 WHERE linktree_id = %s
                 ORDER BY COALESCE(position, 0), id
            """,
                (linktree_id,),
            )
            links = cur.fetchall() or []

            # Icons (UNGEFILTERT → displayed kann true/false sein)
            cur.execute(
                """
                SELECT i.id, i.code, i.image_url, i.description,
                    ui.displayed, ui.acquired_at
                FROM user_icons ui
                JOIN icons i ON i.id = ui.icon_id   -- <- HIER!
                WHERE ui.user_id = %s
                ORDER BY i.code
            """,
                (lt["user_id"],),
            )
            icons = cur.fetchall() or []

        visit_count = 0
        try:
            canonical_id = _get_canonical_linktree_id(lt.get("slug", "")) or lt["id"]
            visit_count = db.get_linktree_visit_count(canonical_id)
        except Exception:
            visit_count = 0

        discord_acct = _load_discord_account(lt["user_id"])
        discord_linked = bool(discord_acct)
        decoration_url = discord_acct.get("decoration_url") if discord_acct else None
        discord_badges = (
            _discord_badges_from_account(discord_acct) if discord_linked else []
        )
        presence_value, status_text = _resolve_discord_presence(lt, discord_acct)

        return {
            "id": lt["id"],
            "user_id": lt["user_id"],
            "slug": lt["slug"],
            "device_type": lt.get("device_type", "pc"),
            "location": lt.get("location"),
            "quote": lt.get("quote"),
            "quote_typing_enabled": bool(lt.get("quote_typing_enabled", False)),
            "quote_typing_texts": _json_to_list(
                lt.get("quote_typing_texts"),
                max_items=3,
                max_len=180,
            ),
            "quote_typing_speed": lt.get("quote_typing_speed"),
            "quote_typing_pause": lt.get("quote_typing_pause"),
            "quote_font_size": lt.get("quote_font_size"),
            "quote_font_family": lt.get("quote_font_family") or "default",
            "quote_effect": lt.get("quote_effect") or "none",
            "quote_effect_strength": int(lt.get("quote_effect_strength", 70) or 70),
            "entry_text": lt.get("entry_text"),
            "entry_bg_alpha": int(lt.get("entry_bg_alpha", 85) or 85),
            "entry_text_color": lt.get("entry_text_color"),
            "entry_font_size": int(lt.get("entry_font_size", 16) or 16),
            "entry_font_family": lt.get("entry_font_family") or "default",
            "entry_effect": lt.get("entry_effect") or "none",
            "entry_overlay_alpha": int(lt.get("entry_overlay_alpha", 35) or 35),
            "entry_box_enabled": bool(lt.get("entry_box_enabled", True)),
            "entry_border_enabled": bool(lt.get("entry_border_enabled", True)),
            "entry_border_color": lt.get("entry_border_color"),
            "song_url": lt.get("song_url"),
            "song_name": lt.get("song_name"),
            "song_icon_url": lt.get("song_icon_url"),
            "background_url": lt.get("background_url"),
            "background_is_video": bool(lt.get("background_is_video")),
            "transparency": int(lt.get("transparency") or 0),
            "name_effect": lt.get("name_effect") or "none",
            "name_font_family": lt.get("name_font_family") or "default",
            "background_effect": lt.get("background_effect") or "none",
            "display_name_mode": lt.get("display_name_mode") or "slug",
            "layout_mode": lt.get("layout_mode") or "center",
            "custom_display_name": lt.get("custom_display_name"),
            "linktree_profile_picture": lt.get("linktree_profile_picture"),
            "section_order": _normalize_section_order(lt.get("section_order")),
            "canvas_layout": _normalize_canvas_layout(lt.get("canvas_layout")),
            "link_color": lt.get("link_color"),
            "link_bg_color": lt.get("link_bg_color"),
            "link_bg_alpha": int(lt.get("link_bg_alpha") or 100),
            "link_columns": lt.get("link_columns"),
            "link_icons_only": bool(lt.get("link_icons_only", False)),
            "card_color": lt.get("card_color"),
            "text_color": lt.get("text_color"),
            "name_color": lt.get("name_color"),
            "location_color": lt.get("location_color"),
            "quote_color": lt.get("quote_color"),
            "cursor_url": lt.get("cursor_url"),
            "cursor_effect": lt.get("cursor_effect") or "none",
            "cursor_effect_color": lt.get("cursor_effect_color"),
            "cursor_effect_alpha": int(lt.get("cursor_effect_alpha") or 70),
            "show_audio_player": bool(lt.get("show_audio_player", False)),
            "audio_player_bg_color": lt.get("audio_player_bg_color"),
            "audio_player_bg_alpha": int(lt.get("audio_player_bg_alpha", 60) or 60),
            "audio_player_text_color": lt.get("audio_player_text_color"),
            "audio_player_accent_color": lt.get("audio_player_accent_color"),
            "discord_frame_enabled": bool(lt.get("discord_frame_enabled", False)),
            "discord_decoration_url": decoration_url,
            "discord_presence_enabled": bool(lt.get("discord_presence_enabled", False)),
            "discord_presence": presence_value,
            "discord_status_enabled": bool(lt.get("discord_status_enabled", False)),
            "discord_status_text": status_text,
            "discord_badges_enabled": bool(lt.get("discord_badges_enabled", False)),
            "discord_badges": discord_badges if lt.get("discord_badges_enabled") else [],
            "discord_badge_codes": _json_to_list(
                lt.get("discord_badge_codes"),
                max_items=50,
                max_len=64,
                none_if_missing=True,
            ),
            "discord_linked": discord_linked,
            "profile_picture": user_pfp,
            "user_username": user_username,
            "show_visit_counter": bool(lt.get("show_visit_counter", False)),
            "visit_count": int(visit_count or 0),
            "visit_counter_color": lt.get("visit_counter_color"),
            "visit_counter_bg_color": lt.get("visit_counter_bg_color"),
            "visit_counter_bg_alpha": int(lt.get("visit_counter_bg_alpha", 20) or 20),
            "links": [
                {
                    "id": r["id"],
                    "url": r["url"],
                    "label": r.get("label"),
                    "icon_url": r.get("icon_url"),
                    "position": r.get("position", 0),
                    "is_active": bool(r.get("is_active", True)),
                }
                for r in links
            ],
            "icons": [
                {
                    "id": r["id"],
                    "code": r["code"],
                    "image_url": r["image_url"],
                    "description": r.get("description"),
                    "displayed": bool(r.get("displayed", False)),
                    "acquired_at": (
                        r["acquired_at"].isoformat() if r.get("acquired_at") else None
                    ),
                }
                for r in icons
            ],
        }

    @app.get("/api/linktrees/by-slug/{slug}/manage", response_model=LinktreeOut)
    def get_linktree_manage_by_slug(
        slug: str,
        device: DeviceType = Query("pc"),
        user: dict = Depends(require_user),
    ):
        _ensure_pg()
        lt = db.get_linktree_by_slug(slug, device)
        if not lt:
            raise HTTPException(404, "Linktree not found")
        if not (user.get("admin") or user["id"] == lt["user_id"]):
            raise HTTPException(403, "Forbidden (owner or admin only)")
        return get_linktree_manage(lt["id"], user)

    @app.post("/api/users/me/linktree-pfp")
    async def upload_linktree_pfp(
        file: UploadFile = File(...),
        current: dict = Depends(require_user),
        device: DeviceType | None = Query(None, description="pc or mobile"),
    ):
        _ensure_pg()
        if file.content_type not in ALLOWED_IMAGE_CT:
            raise HTTPException(415, "Unsupported image type")
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "File too large (max 5MB)")
        ext = _detect_image_ext(data)
        if not ext:
            raise HTTPException(400, "File is not a valid image")
        target_device = device or "pc"

        old_url = None
        try:
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT lt.linktree_profile_picture
                      FROM linktrees lt
                     WHERE lt.user_id = %s AND lt.device_type = %s
                """,
                    (current["id"], target_device),
                )
                row = cur.fetchone()
                old_url = row[0] if row else None
                if row is None:
                    raise HTTPException(404, "Linktree for device not found")
        except HTTPException:
            raise
        except Exception:
            old_url = None

        fname = f"user{current['id']}_ltpfp_{uuid.uuid4().hex}.{ext}"
        out_path = UPLOAD_DIR / fname
        out_path.write_bytes(data)
        url = f"/media/{UPLOAD_DIR.name}/{fname}"

        updated = db.update_linktree_by_user_and_device(
            current["id"], target_device, linktree_profile_picture=url
        )
        if not updated:
            raise HTTPException(404, "Linktree for device not found")

        if old_url and old_url != url:
            _delete_if_unreferenced(old_url)

        return {"url": url}

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

    @app.get("/linktree_canvas.html", include_in_schema=False)
    def linktree_canvas_page():
        return FileResponse("linktree_canvas.html")

    @app.get("/api/linktrees/{slug}", response_model=LinktreeOut)
    def get_linktree(
        slug: str,
        device: DeviceType = Query("pc", description="pc or mobile"),
        request: Request = None,
        response: Response = None,
    ):
        _ensure_pg()
        lt = db.get_linktree_by_slug(slug, device)
        if not lt and device == "mobile":
            lt = db.get_linktree_by_slug(slug, "pc")  # fallback
            device = "pc"
        if not lt:
            raise HTTPException(404, "Linktree not found")

        # Username & Avatar des Owners holen
        user_username = None
        user_pfp = None
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT username, profile_picture FROM users WHERE id=%s", (lt["user_id"],)
            )
            row = cur.fetchone()
            if row:
                user_username, user_pfp = row[0], row[1]
        decoration_url = None
        discord_linked = False
        discord_badges = []
        acct = None
        try:
            acct = _load_discord_account(lt["user_id"])
            discord_linked = bool(acct)
            decoration_url = acct.get("decoration_url") if acct else None
            if discord_linked:
                discord_badges = _discord_badges_from_account(acct)
        except Exception:
            decoration_url = None
        presence_value, status_text = _resolve_discord_presence(lt, acct if discord_linked else None)
        quote_typing_texts = _json_to_list(
            lt.get("quote_typing_texts"),
            max_items=3,
            max_len=180,
        )
        quote_typing_speed = lt.get("quote_typing_speed")
        quote_typing_pause = lt.get("quote_typing_pause")
        quote_font_size = lt.get("quote_font_size")
        quote_font_family = lt.get("quote_font_family") or "default"
        quote_effect = lt.get("quote_effect") or "none"
        quote_effect_strength = lt.get("quote_effect_strength", 70)
        entry_bg_alpha = lt.get("entry_bg_alpha", 85)
        entry_text_color = lt.get("entry_text_color")
        entry_font_size = lt.get("entry_font_size", 16)
        entry_font_family = lt.get("entry_font_family") or "default"
        entry_effect = lt.get("entry_effect") or "none"
        entry_overlay_alpha = lt.get("entry_overlay_alpha", 35)
        discord_badge_codes = _json_to_list(
            lt.get("discord_badge_codes"),
            max_items=50,
            max_len=64,
            none_if_missing=True,
        )

        icons = [
            {
                "id": i["id"],
                "code": i["code"],
                "image_url": i["image_url"],
                "description": i.get("description"),
                "displayed": i.get("displayed", False),
                "acquired_at": i.get("acquired_at"),
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
        frame_enabled = bool(lt.get("discord_frame_enabled", False))
        visit_count = 0
        if request is not None and response is not None:
            visit_count = _record_linktree_visit(lt, request, response)
        else:
            try:
                if isinstance(db, PgGifDB):
                    canonical_id = _get_canonical_linktree_id(lt.get("slug", "")) or lt["id"]
                    visit_count = db.get_linktree_visit_count(canonical_id)
                else:
                    visit_count = 0
            except Exception:
                visit_count = 0

        return {
            "id": lt["id"],
            "user_id": lt["user_id"],
            "slug": lt["slug"],
            "device_type": lt.get("device_type", "pc"),
            "location": lt.get("location"),
            "quote": lt.get("quote"),
            "quote_typing_enabled": bool(lt.get("quote_typing_enabled", False)),
            "quote_typing_texts": quote_typing_texts,
            "quote_typing_speed": quote_typing_speed,
            "quote_typing_pause": quote_typing_pause,
            "quote_font_size": quote_font_size,
            "quote_font_family": quote_font_family,
            "quote_effect": quote_effect,
            "quote_effect_strength": int(quote_effect_strength or 70),
            "entry_text": lt.get("entry_text"),
            "entry_bg_alpha": int(entry_bg_alpha or 85),
            "entry_text_color": entry_text_color,
            "entry_font_size": int(entry_font_size or 16),
            "entry_font_family": entry_font_family,
            "entry_effect": entry_effect,
            "entry_overlay_alpha": int(entry_overlay_alpha or 35),
            "entry_box_enabled": bool(lt.get("entry_box_enabled", True)),
            "entry_border_enabled": bool(lt.get("entry_border_enabled", True)),
            "entry_border_color": lt.get("entry_border_color"),
            "song_url": lt.get("song_url"),
            "song_name": lt.get("song_name"),
            "song_icon_url": lt.get("song_icon_url"),
            "show_audio_player": bool(lt.get("show_audio_player", False)),
            "audio_player_bg_color": lt.get("audio_player_bg_color"),
            "audio_player_bg_alpha": int(lt.get("audio_player_bg_alpha", 60) or 60),
            "audio_player_text_color": lt.get("audio_player_text_color"),
            "audio_player_accent_color": lt.get("audio_player_accent_color"),
            "background_url": lt.get("background_url"),
            "background_is_video": lt.get("background_is_video", False),
            "transparency": lt.get("transparency", 0),
            "name_effect": lt.get("name_effect", "none"),
            "name_font_family": lt.get("name_font_family") or "default",
            "background_effect": lt.get("background_effect", "none"),
            "display_name_mode": lt.get("display_name_mode", "slug"),
            "layout_mode": lt.get("layout_mode") or "center",
            "custom_display_name": lt.get("custom_display_name"),
            "linktree_profile_picture": lt.get("linktree_profile_picture"),
            "section_order": _normalize_section_order(lt.get("section_order")),
            "canvas_layout": _normalize_canvas_layout(lt.get("canvas_layout")),
            "link_color": lt.get("link_color"),
            "link_bg_color": lt.get("link_bg_color"),
            "link_bg_alpha": lt.get("link_bg_alpha", 100),
            "link_columns": lt.get("link_columns"),
            "link_icons_only": bool(lt.get("link_icons_only", False)),
            "card_color": lt.get("card_color"),
            "text_color": lt.get("text_color"),
            "name_color": lt.get("name_color"),
            "location_color": lt.get("location_color"),
            "quote_color": lt.get("quote_color"),
            "cursor_url": lt.get("cursor_url"),
            "cursor_effect": lt.get("cursor_effect") or "none",
            "cursor_effect_color": lt.get("cursor_effect_color"),
            "cursor_effect_alpha": int(lt.get("cursor_effect_alpha") or 70),
            "discord_frame_enabled": frame_enabled,
            "discord_decoration_url": decoration_url if frame_enabled else None,
            "discord_presence_enabled": bool(lt.get("discord_presence_enabled", False)),
            "discord_presence": presence_value,
            "discord_status_enabled": bool(lt.get("discord_status_enabled", False)),
            "discord_status_text": status_text,
            "discord_badges_enabled": bool(lt.get("discord_badges_enabled", False)),
            "discord_badges": discord_badges if lt.get("discord_badges_enabled") else [],
            "discord_badge_codes": discord_badge_codes,
            "discord_linked": discord_linked,
            "profile_picture": user_pfp,  # <= jetzt dabei
            "user_username": user_username,  # <= jetzt dabei
            "show_visit_counter": bool(lt.get("show_visit_counter", False)),
            "visit_count": int(visit_count or 0),
            "visit_counter_color": lt.get("visit_counter_color"),
            "visit_counter_bg_color": lt.get("visit_counter_bg_color"),
            "visit_counter_bg_alpha": int(lt.get("visit_counter_bg_alpha", 20) or 20),
            "links": links,
            "icons": icons,
        }

    @app.get("/tree/{slug}", include_in_schema=False)
    def linktree_page(slug: str):
        return FileResponse("linktree.html")

    @app.post(
        "/api/linktrees", response_model=LinktreeOut, dependencies=[Depends(require_user)]
    )
    def create_linktree_ep(payload: LinktreeCreateIn, user: dict = Depends(require_user)):
        _ensure_pg()
        # Ein Tree pro User + device_type
        existing = db.get_linktree_by_slug(payload.slug, payload.device_type)
        if existing and existing["user_id"] == user["id"]:
            raise HTTPException(409, "You already have a linktree with this slug and device")
        # globale Slug-Kollision prüfen (andere User)
        with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT user_id FROM linktrees WHERE lower(slug)=lower(%s) AND user_id<>%s LIMIT 1", (payload.slug, user["id"]))
            if cur.fetchone():
                raise HTTPException(409, "Slug already in use")
        try:
            bg_is_video = (
                _looks_like_video_url(payload.background_url)
                if payload.background_url
                else payload.background_is_video
            )
            song_name = _clean_upload_filename(payload.song_name)
            quote_texts = _normalize_text_list(
                payload.quote_typing_texts,
                max_items=3,
                max_len=180,
                dedupe=False,
            )
            if payload.quote and not quote_texts:
                quote_texts = [payload.quote]
            quote_texts_json = json.dumps(quote_texts) if quote_texts else None
            badge_codes_json = _list_to_json(
                payload.discord_badge_codes,
                max_items=50,
                max_len=64,
                allow_empty=True,
            )
            section_order = _normalize_section_order(payload.section_order)
            section_order_json = json.dumps(section_order) if section_order else None
            canvas_layout = _normalize_canvas_layout(payload.canvas_layout)
            canvas_layout_json = json.dumps(canvas_layout) if canvas_layout is not None else None
            entry_text = payload.entry_text.strip() if isinstance(payload.entry_text, str) else None
            entry_bg_alpha = (
                int(payload.entry_bg_alpha)
                if payload.entry_bg_alpha is not None
                else 85
            )
            entry_bg_alpha = max(0, min(100, entry_bg_alpha))
            entry_overlay_alpha = (
                int(payload.entry_overlay_alpha)
                if payload.entry_overlay_alpha is not None
                else 35
            )
            entry_overlay_alpha = max(0, min(100, entry_overlay_alpha))
            entry_box_enabled = (
                bool(payload.entry_box_enabled)
                if payload.entry_box_enabled is not None
                else True
            )
            entry_border_enabled = (
                bool(payload.entry_border_enabled)
                if payload.entry_border_enabled is not None
                else True
            )
            entry_border_color = (
                payload.entry_border_color.strip()
                if isinstance(payload.entry_border_color, str)
                else None
            )
            if entry_border_color and not re.match(HEX_COLOR_RE, entry_border_color):
                entry_border_color = None
            entry_font_size = (
                int(payload.entry_font_size)
                if payload.entry_font_size is not None
                else 16
            )
            entry_font_size = max(10, min(40, entry_font_size))
            entry_font_family = payload.entry_font_family or "default"
            entry_effect = payload.entry_effect or "none"
            entry_text_color = (
                payload.entry_text_color.strip()
                if isinstance(payload.entry_text_color, str)
                else None
            )
            if entry_text_color and not re.match(HEX_COLOR_RE, entry_text_color):
                entry_text_color = None
            quote_speed = (
                int(payload.quote_typing_speed)
                if payload.quote_typing_speed is not None
                else None
            )
            if quote_speed is not None:
                quote_speed = max(20, min(200, quote_speed))
            quote_pause = (
                int(payload.quote_typing_pause)
                if payload.quote_typing_pause is not None
                else None
            )
            if quote_pause is not None:
                quote_pause = max(200, min(10000, quote_pause))
            quote_size = int(payload.quote_font_size) if payload.quote_font_size is not None else None
            if quote_size is not None:
                quote_size = max(10, min(40, quote_size))
            quote_font_family = payload.quote_font_family or "default"
            quote_effect = payload.quote_effect or "none"
            quote_effect_strength = (
                int(payload.quote_effect_strength)
                if payload.quote_effect_strength is not None
                else 70
            )
            quote_effect_strength = max(0, min(100, quote_effect_strength))
            name_font_family = payload.name_font_family or "default"
            link_columns = None
            if payload.link_columns is not None:
                try:
                    link_columns = int(payload.link_columns)
                except Exception:
                    link_columns = None
                if link_columns is not None:
                    link_columns = max(1, min(8, link_columns))
            link_icons_only = bool(payload.link_icons_only)
            linktree_id = db.create_linktree(
                user_id=user["id"],
                slug=payload.slug,
                device_type=payload.device_type,
                location=payload.location,
                quote=payload.quote,
                quote_typing_enabled=payload.quote_typing_enabled,
                quote_typing_texts=quote_texts_json,
                quote_typing_speed=quote_speed,
                quote_typing_pause=quote_pause,
                quote_font_size=quote_size,
                quote_font_family=quote_font_family,
                quote_effect=quote_effect,
                quote_effect_strength=quote_effect_strength,
                entry_text=entry_text,
                entry_bg_alpha=entry_bg_alpha,
                entry_text_color=entry_text_color,
                entry_font_size=entry_font_size,
                entry_font_family=entry_font_family,
                entry_effect=entry_effect,
                entry_overlay_alpha=entry_overlay_alpha,
                entry_box_enabled=entry_box_enabled,
                entry_border_enabled=entry_border_enabled,
                entry_border_color=entry_border_color,
                song_url=payload.song_url,
                song_name=song_name,
                song_icon_url=payload.song_icon_url,
                show_audio_player=payload.show_audio_player,
                audio_player_bg_color=payload.audio_player_bg_color,
                audio_player_bg_alpha=payload.audio_player_bg_alpha,
                audio_player_text_color=payload.audio_player_text_color,
                audio_player_accent_color=payload.audio_player_accent_color,
                background_url=payload.background_url,
                background_is_video=bg_is_video,
                transparency=payload.transparency,
                name_effect=payload.name_effect,
                name_font_family=name_font_family,
                background_effect=payload.background_effect,
                display_name_mode=payload.display_name_mode,  # <-- NEU
                layout_mode=payload.layout_mode,
                custom_display_name=payload.custom_display_name,
                linktree_profile_picture=payload.linktree_profile_picture,
                section_order=section_order_json,
                canvas_layout=canvas_layout_json,
                link_color=payload.link_color,
                link_bg_color=payload.link_bg_color,
                link_bg_alpha=payload.link_bg_alpha,
                link_columns=link_columns,
                link_icons_only=link_icons_only,
                card_color=payload.card_color,
                text_color=payload.text_color,
                name_color=payload.name_color,
                location_color=payload.location_color,
                quote_color=payload.quote_color,
                cursor_url=payload.cursor_url,
                cursor_effect=payload.cursor_effect,
                cursor_effect_color=payload.cursor_effect_color,
                cursor_effect_alpha=payload.cursor_effect_alpha,
                discord_frame_enabled=payload.discord_frame_enabled,
                discord_presence_enabled=payload.discord_presence_enabled,
                discord_presence=payload.discord_presence,
                discord_status_enabled=payload.discord_status_enabled,
                discord_status_text=payload.discord_status_text,
                discord_badges_enabled=payload.discord_badges_enabled,
                discord_badge_codes=badge_codes_json,
                show_visit_counter=payload.show_visit_counter,
                visit_counter_color=payload.visit_counter_color,
                visit_counter_bg_color=payload.visit_counter_bg_color,
                visit_counter_bg_alpha=payload.visit_counter_bg_alpha,
            )
        except pg_errors.UniqueViolation:
            raise HTTPException(409, "Slug already in use")
        lt = db.get_linktree_by_slug(payload.slug, payload.device_type)
        # public shaping wie oben:
        return get_linktree(payload.slug, device=payload.device_type)

    @app.post("/api/linktrees/{slug}/clone", response_model=LinktreeOut)
    def clone_linktree_variant(
        slug: str,
        target_device: DeviceType = Query("mobile"),
        user: dict = Depends(require_user),
    ):
        _ensure_pg()
        if target_device not in ("pc", "mobile"):
            raise HTTPException(400, "Invalid target device")
        source_device = "pc" if target_device == "mobile" else "mobile"
        src = db.get_linktree_by_slug(slug, source_device)
        if not src:
            raise HTTPException(404, "Source linktree not found")
        if not (user.get("admin") or user["id"] == src["user_id"]):
            raise HTTPException(403, "Forbidden (owner or admin only)")
        try:
            new_id = db.clone_linktree_variant(slug, source_device, target_device, src["user_id"])
        except ValueError as ve:
            raise HTTPException(409, str(ve))
        except KeyError:
            raise HTTPException(404, "Source linktree not found")
        return get_linktree_manage(new_id, user)

    @app.delete("/api/linktrees/{linktree_id}")
    def delete_linktree_ep(linktree_id: int, user: dict = Depends(require_user)):
        _ensure_pg()
        _require_tree_owner_or_admin(linktree_id, user)

        # Sammle Owner und evtl. Medien-URLs (für Cleanup)
        with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, song_url, background_url, cursor_url, linktree_profile_picture FROM linktrees WHERE id=%s",
                (linktree_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Linktree not found")

            owner_id = int(row["user_id"])
            song_url = row.get("song_url")
            bg_url = row.get("background_url")
            cursor_url = row.get("cursor_url")
            pfp_url = row.get("linktree_profile_picture")

            # Icon-URLs der Links sammeln (können lokale Medien sein)
            cur.execute(
                "SELECT icon_url FROM linktree_links WHERE linktree_id=%s AND icon_url IS NOT NULL",
                (linktree_id,),
            )
            icon_urls = [r["icon_url"] for r in cur.fetchall() or [] if r.get("icon_url")]

            # 1) Links löschen
            cur.execute("DELETE FROM linktree_links WHERE linktree_id=%s", (linktree_id,))
            # 2) Verknüpfung beim Owner lösen, falls gesetzt
            cur.execute(
                "UPDATE users SET linktree_id = NULL WHERE id=%s AND linktree_id=%s",
                (owner_id, linktree_id),
            )
            # 3) Linktree löschen
            cur.execute("DELETE FROM linktrees WHERE id=%s", (linktree_id,))
            conn.commit()

        # Medien aufräumen (nur wenn nirgendwo mehr referenziert)
        for url in [song_url, bg_url, cursor_url, pfp_url, *icon_urls]:
            if url:
                _delete_if_unreferenced(url)

        return {"ok": True}

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
    def update_link_ep(
        link_id: int, payload: LinkUpdateIn, user: dict = Depends(require_user)
    ):
        _ensure_pg()
        with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT lt.user_id, l.linktree_id, l.icon_url
                  FROM linktree_links l
                  JOIN linktrees lt ON lt.id = l.linktree_id
                 WHERE l.id = %s
            """,
                (link_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Link not found")
            owner_id, linktree_id, old_icon = (
                int(row["user_id"]),
                int(row["linktree_id"]),
                row["icon_url"],
            )

        if not (user.get("admin") or user["id"] == owner_id):
            raise HTTPException(403, "Forbidden (owner or admin only)")

        # Werde die neue icon_url gleich aus dem Payload holen
        new_icon = payload.icon_url if payload.icon_url is not None else old_icon

        fields = payload.model_dump(exclude_unset=True)
        if "song_name" in fields:
            fields["song_name"] = _clean_upload_filename(fields.get("song_name"))
        if "song_url" in fields and not fields.get("song_url"):
            fields["song_url"] = None
            fields["song_name"] = None
        if "background_url" in fields:
            fields["background_is_video"] = _looks_like_video_url(
                fields.get("background_url")
            )
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

    @app.post("/api/users/me/song")
    async def upload_song(
        file: UploadFile = File(...),
        current: dict = Depends(require_user),
        device: DeviceType | None = Query(None, description="pc or mobile"),
    ):
        _ensure_pg()
        if file.content_type not in ALLOWED_AUDIO:
            raise HTTPException(415, "Unsupported audio type")
        data = await file.read()
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "File too large (max 15MB)")
        target_device = device or "pc"
        original_name = _clean_upload_filename(file.filename)

        old_url = None
        try:
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT lt.song_url
                      FROM linktrees lt
                     WHERE lt.user_id = %s AND lt.device_type = %s
                """,
                    (current["id"], target_device),
                )
                row = cur.fetchone()
                old_url = row[0] if row else None
                if row is None:
                    raise HTTPException(404, "Linktree for device not found")
        except HTTPException:
            raise
        except Exception:
            old_url = None

        audio_ext = {
            "audio/mpeg": "mp3",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
        }.get(file.content_type)
        if not audio_ext:
            raise HTTPException(415, "Unsupported audio type")

        fname = f"user{current['id']}_song_{uuid.uuid4().hex}.{audio_ext}"
        out_path = UPLOAD_DIR / fname
        out_path.write_bytes(data)
        url = f"/media/{UPLOAD_DIR.name}/{fname}"
        updated = db.update_linktree_by_user_and_device(
            current["id"], target_device, song_url=url, song_name=original_name
        )
        if not updated:
            raise HTTPException(404, "Linktree for device not found")

        # Cleanup
        if old_url and old_url != url:
            _delete_if_unreferenced(old_url)

        return {"url": url, "name": original_name}

    @app.post("/api/users/me/songicon")
    async def upload_song_icon(
        file: UploadFile = File(...),
        current: dict = Depends(require_user),
        device: DeviceType | None = Query(None, description="pc or mobile"),
    ):
        _ensure_pg()
        if file.content_type not in ALLOWED_IMAGE_CT:
            raise HTTPException(415, "Unsupported image type")
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "File too large (max 5MB)")
        ext = _detect_image_ext(data)
        if not ext:
            raise HTTPException(400, "File is not a valid image")
        target_device = device or "pc"

        old_url = None
        try:
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT lt.song_icon_url
                      FROM linktrees lt
                     WHERE lt.user_id = %s AND lt.device_type = %s
                """,
                    (current["id"], target_device),
                )
                row = cur.fetchone()
                old_url = row[0] if row else None
                if row is None:
                    raise HTTPException(404, "Linktree for device not found")
        except HTTPException:
            raise
        except Exception:
            old_url = None

        fname = f"user{current['id']}_songicon_{uuid.uuid4().hex}.{ext}"
        out_path = UPLOAD_DIR / fname
        out_path.write_bytes(data)
        url = f"/media/{UPLOAD_DIR.name}/{fname}"

        updated = db.update_linktree_by_user_and_device(
            current["id"], target_device, song_icon_url=url
        )
        if not updated:
            raise HTTPException(404, "Linktree for device not found")

        if old_url and old_url != url:
            _delete_if_unreferenced(old_url)

        return {"url": url}

    @app.post("/api/users/me/background")
    async def upload_background(
        file: UploadFile = File(...),
        current: dict = Depends(require_user),
        device: DeviceType | None = Query(None, description="pc or mobile"),
    ):
        _ensure_pg()
        ct = file.content_type or ""
        if ct not in ALLOWED_IMAGE_CT and ct not in ALLOWED_VIDEO_CT:
            raise HTTPException(415, "Unsupported background type")
        data = await file.read()
        if len(data) > MAX_BACKGROUND_BYTES:
            raise HTTPException(413, "File too large (max 50MB)")
        target_device = device or "pc"

        image_ext = _detect_image_ext(data)
        video_ext = "" if image_ext else _detect_video_ext(data, ct)
        if not image_ext and not video_ext:
            raise HTTPException(400, "File is not a supported image or video")

        # Altes BG merken
        old_url = None
        try:
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT lt.background_url
                      FROM linktrees lt
                     WHERE lt.user_id = %s AND lt.device_type = %s
                """,
                    (current["id"], target_device),
                )
                row = cur.fetchone()
                old_url = row[0] if row else None
                if row is None:
                    raise HTTPException(404, "Linktree for device not found")
        except HTTPException:
            raise
        except Exception:
            old_url = None

        ext = image_ext or video_ext
        fname = f"user{current['id']}_bg_{uuid.uuid4().hex}.{ext}"
        out_path = UPLOAD_DIR / fname
        out_path.write_bytes(data)
        url = f"/media/{UPLOAD_DIR.name}/{fname}"
        is_video = bool(video_ext)

        updated = db.update_linktree_by_user_and_device(
            current["id"],
            target_device,
            background_url=url,
            background_is_video=is_video,
        )
        if not updated:
            raise HTTPException(404, "Linktree for device not found")

        # Cleanup
        if old_url and old_url != url:
            _delete_if_unreferenced(old_url)

        return {"url": url, "is_video": is_video}

    @app.post("/api/users/me/cursor")
    async def upload_cursor(
        file: UploadFile = File(...),
        current: dict = Depends(require_user),
        device: DeviceType | None = Query(None, description="pc or mobile"),
    ):
        _ensure_pg()
        target_device = device or "pc"
        if target_device != "pc":
            raise HTTPException(400, "Custom cursor is only available for desktop linktrees")
        if file.content_type not in ALLOWED_IMAGE_CT:
            raise HTTPException(415, "Unsupported image type for cursor")
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "File too large (max 5MB)")
        ext = _detect_image_ext(data)
        if not ext:
            raise HTTPException(400, "File is not a valid image")
        # Cursor-Kompatibilität: auf PNG normalisieren, falls anderes Format (z.B. WebP)
        if ext != "png":
            try:
                with Image.open(io.BytesIO(data)) as img:
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    data = buf.getvalue()
                    ext = "png"
                    if len(data) > MAX_IMAGE_BYTES:
                        raise HTTPException(413, "Converted cursor is too large (>5MB)")
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(400, "Cursor could not be converted to PNG")

        old_url = None
        try:
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT cursor_url FROM linktrees WHERE user_id=%s AND device_type=%s",
                    (current["id"], target_device),
                )
                row = cur.fetchone()
                if row is None:
                    raise HTTPException(404, "Linktree for device not found")
                old_url = row[0]
        except HTTPException:
            raise
        except Exception:
            old_url = None

        fname = f"user{current['id']}_cursor_{uuid.uuid4().hex}.{ext}"
        out_path = UPLOAD_DIR / fname
        out_path.write_bytes(data)
        url = f"/media/{UPLOAD_DIR.name}/{fname}"

        updated = db.update_linktree_by_user_and_device(
            current["id"], target_device, cursor_url=url
        )
        if not updated:
            raise HTTPException(404, "Linktree for device not found")

        if old_url and old_url != url:
            _delete_if_unreferenced(old_url)

        return {"url": url}

    @app.post("/api/users/me/linkicon")
    async def upload_linkicon(
        file: UploadFile = File(...), current: dict = Depends(require_user)
    ):
        if file.content_type not in ALLOWED_IMAGE_CT:
            raise HTTPException(415, "Unsupported image type")
        data = await file.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "File too large (max 5MB)")
        ext = _detect_image_ext(data)
        if not ext:
            raise HTTPException(400, "File is not a valid image")
        fname = f"user{current['id']}_icon_{uuid.uuid4().hex}.{ext}"
        out_path = UPLOAD_DIR / fname
        out_path.write_bytes(data)
        url = f"/media/{UPLOAD_DIR.name}/{fname}"
        return {"url": url}

    @app.patch("/api/linktrees/{linktree_id}", response_model=dict)
    def update_linktree_ep(
        linktree_id: int,
        payload: LinktreeUpdateIn,
        user: dict = Depends(require_user),
    ):
        _ensure_pg()
        # Owner/Admin prüfen
        _require_tree_owner_or_admin(linktree_id, user)

        fields = payload.model_dump(exclude_unset=True)
        if "quote_typing_texts" in fields:
            qt = _normalize_text_list(
                fields.get("quote_typing_texts"),
                max_items=3,
                max_len=180,
                dedupe=False,
            )
            if fields.get("quote") and not qt:
                qt = [str(fields.get("quote")).strip()]
            fields["quote_typing_texts"] = json.dumps(qt) if qt else None
        if "discord_badge_codes" in fields:
            fields["discord_badge_codes"] = _list_to_json(
                fields.get("discord_badge_codes"),
                max_items=50,
                max_len=64,
                allow_empty=True,
            )
        if "section_order" in fields:
            order = _normalize_section_order(fields.get("section_order"))
            fields["section_order"] = json.dumps(order) if order else None
        if "canvas_layout" in fields:
            layout = _normalize_canvas_layout(fields.get("canvas_layout"))
            fields["canvas_layout"] = json.dumps(layout) if layout is not None else None
        if "entry_text" in fields and isinstance(fields["entry_text"], str):
            fields["entry_text"] = fields["entry_text"].strip() or None
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
        if "name_font_family" in fields:
            fam = str(fields.get("name_font_family") or "default").lower()
            fields["name_font_family"] = (
                fam if fam in {"default", "serif", "mono", "script", "display"} else "default"
            )
        if "entry_effect" in fields:
            fx = str(fields.get("entry_effect") or "none").lower()
            fields["entry_effect"] = (
                fx if fx in {"none", "glow", "neon", "rainbow"} else "none"
            )
        if "entry_overlay_alpha" in fields:
            try:
                alpha = int(fields.get("entry_overlay_alpha"))
            except Exception:
                alpha = None
            fields["entry_overlay_alpha"] = (
                max(0, min(100, alpha)) if alpha is not None else None
            )
        if "entry_box_enabled" in fields:
            fields["entry_box_enabled"] = bool(fields.get("entry_box_enabled"))
        if "entry_border_enabled" in fields:
            fields["entry_border_enabled"] = bool(fields.get("entry_border_enabled"))
        if "entry_border_color" in fields and isinstance(fields.get("entry_border_color"), str):
            val = fields.get("entry_border_color").strip()
            fields["entry_border_color"] = val if re.match(HEX_COLOR_RE, val) else None
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
        if "quote_font_family" in fields:
            fam = str(fields.get("quote_font_family") or "default").lower()
            fields["quote_font_family"] = (
                fam if fam in {"default", "serif", "mono", "script", "display"} else "default"
            )
        if "quote_effect" in fields:
            fx = str(fields.get("quote_effect") or "none").lower()
            fields["quote_effect"] = (
                fx if fx in {"none", "glow", "neon", "rainbow"} else "none"
            )
        if "layout_mode" in fields:
            mode = str(fields.get("layout_mode") or "center").lower()
            fields["layout_mode"] = mode if mode in {"center", "wide"} else "center"
        if "quote_effect_strength" in fields:
            try:
                strength = int(fields.get("quote_effect_strength"))
            except Exception:
                strength = None
            fields["quote_effect_strength"] = (
                max(0, min(100, strength)) if strength is not None else 70
            )

        # Vorherige Medien-URLs zum Aufräumen merken + device_type
        old_song = None
        old_song_icon = None
        old_bg = None
        old_pfp = None
        current_device_type = None
        with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT song_url, song_icon_url, background_url, linktree_profile_picture, device_type FROM linktrees WHERE id=%s",
                (linktree_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Linktree not found")
            old_song = row.get("song_url")
            old_song_icon = row.get("song_icon_url")
            old_bg = row.get("background_url")
            old_pfp = row.get("linktree_profile_picture")
            current_device_type = row.get("device_type") or "pc"

        # Wenn slug geändert werden soll: einfache Kollision prüfen (gleicher device_type)
        if "slug" in fields and fields["slug"]:
            new_slug = fields["slug"]
            target_device = fields.get("device_type") or current_device_type
            with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM linktrees WHERE lower(slug)=lower(%s) AND id<>%s AND user_id<>%s",
                    (new_slug, linktree_id, user["id"]),
                )
                if cur.fetchone():
                    raise HTTPException(409, "Slug already in use")

        # Dynamisches UPDATE bauen
        cols = []
        vals = []
        for k, v in fields.items():
            cols.append(f"{k}=%s")
            vals.append(v)
        if cols:
            q = f"UPDATE linktrees SET {', '.join(cols)}, updated_at=NOW() WHERE id=%s"
            vals.append(linktree_id)
            with psycopg.connect(db.dsn) as conn, conn.cursor() as cur:
                cur.execute(q, tuple(vals))
                # Wenn Slug geändert: auch andere Device-Variante des gleichen Users synchronisieren
                if "slug" in fields and fields["slug"]:
                    cur.execute(
                        "UPDATE linktrees SET slug=%s, updated_at=NOW() WHERE user_id=%s AND device_type<>%s",
                        (fields["slug"], user["id"], current_device_type),
                    )
                conn.commit()

        # Medien-Cleanup, falls URLs gewechselt wurden
        new_song = fields.get("song_url", old_song)
        new_song_icon = fields.get("song_icon_url", old_song_icon)
        new_bg = fields.get("background_url", old_bg)
        new_pfp = fields.get("linktree_profile_picture", old_pfp)
        if old_song and old_song != new_song:
            _delete_if_unreferenced(old_song)
        if old_song_icon and old_song_icon != new_song_icon:
            _delete_if_unreferenced(old_song_icon)
        if old_bg and old_bg != new_bg:
            _delete_if_unreferenced(old_bg)
        if old_pfp and old_pfp != new_pfp:
            _delete_if_unreferenced(old_pfp)

        return {"ok": True}

    @app.get("/marketplace", include_in_schema=False)
    def marketplace_page():
        return FileResponse("marketplace.html")

    @app.get("/marketplace/create", include_in_schema=False)
    def marketplace_create_page():
        return FileResponse("marketplace_create.html")

    @app.get("/marketplace/templates/{template_id}/demo", include_in_schema=False)
    def marketplace_template_demo(template_id: str):
        return FileResponse("linktree.html")

    @app.get("/api/marketplace/templates", response_model=List[TemplateListOut])
    def list_public_templates(
        limit: int = Query(50, ge=1, le=100),
        user: dict = Depends(require_user),
    ):
        query = (
            _fs().collection(TEMPLATE_COLLECTION)
            .where("is_public", "==", True)
            .limit(limit)
        )
        docs = [_template_doc_to_list(doc) for doc in query.stream()]
        docs.sort(
            key=lambda d: d.created_at or "",
            reverse=True,
        )
        return docs

    @app.get("/api/marketplace/templates/mine", response_model=List[TemplateListOut])
    def list_my_templates(
        limit: int = Query(100, ge=1, le=200),
        user: dict = Depends(require_user),
    ):
        query = (
            _fs().collection(TEMPLATE_COLLECTION)
            .where("owner_id", "==", int(user["id"]))
            .limit(limit)
        )
        docs = [_template_doc_to_list(doc) for doc in query.stream()]
        docs.sort(
            key=lambda d: d.created_at or "",
            reverse=True,
        )
        return docs

    @app.get("/api/marketplace/templates/saved", response_model=List[TemplateListOut])
    def list_saved_templates(user: dict = Depends(require_user)):
        saves = (
            _fs().collection(TEMPLATE_SAVES_COLLECTION)
            .where("user_id", "==", int(user["id"]))
            .stream()
        )
        out = []
        for save in saves:
            data = save.to_dict() or {}
            template_id = data.get("template_id")
            if not template_id:
                continue
            doc = _fs().collection(TEMPLATE_COLLECTION).document(template_id).get()
            if not doc.exists:
                continue
            tdata = doc.to_dict() or {}
            if not tdata.get("is_public") and int(tdata.get("owner_id") or 0) != int(
                user["id"]
            ):
                continue
            out.append(_template_doc_to_list(doc))
        return out

    @app.get("/api/marketplace/templates/{template_id}", response_model=TemplateDetailOut)
    def get_template_detail(
        template_id: str, user: dict = Depends(require_user)
    ):
        doc = _get_template_doc(template_id, user=user)
        return _template_doc_to_detail(doc)

    @app.post("/api/marketplace/templates", response_model=TemplateDetailOut, status_code=201)
    def create_template(payload: TemplateCreateIn, user: dict = Depends(require_user)):
        variants = [_normalize_variant(v) for v in payload.variants]
        if len({v["device_type"] for v in variants}) != len(variants):
            raise HTTPException(400, "Duplicate device_type in variants")
        # Pick preview from pc variant background, fallback first variant
        preview = payload.preview_image_url
        if not preview:
            pc_var = next((v for v in variants if v.get("device_type") == "pc"), None)
            src = pc_var or (variants[0] if variants else None)
            preview = (src or {}).get("background_url") or "/static/icon.png"
        creator_name = (payload.creator or user.get("username") or "").strip() or None
        owner_pfp = user.get("profile_picture") if isinstance(user, dict) else None
        doc_ref = _fs().collection(TEMPLATE_COLLECTION).document()
        now = firestore.SERVER_TIMESTAMP
        doc_ref.set(
            {
                "name": payload.name.strip(),
                "description": payload.description.strip() if payload.description else None,
                "creator": creator_name,
                "preview_image_url": preview,
                "owner_id": int(user["id"]),
                "owner_username": user.get("username"),
                "owner_profile_picture": owner_pfp,
                "is_public": bool(payload.is_public),
                "created_at": now,
                "updated_at": now,
                "variants": variants,
            }
        )
        doc = doc_ref.get()
        return _template_doc_to_detail(doc)

    @app.patch("/api/marketplace/templates/{template_id}", response_model=TemplateDetailOut)
    def update_template(
        template_id: str, payload: TemplateUpdateIn, user: dict = Depends(require_user)
    ):
        doc_ref = _fs().collection(TEMPLATE_COLLECTION).document(template_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(404, "Template not found")
        existing = doc.to_dict() or {}
        if int(existing.get("owner_id") or 0) != int(user["id"]):
            raise HTTPException(403, "Forbidden")
        update: dict[str, Any] = {"updated_at": firestore.SERVER_TIMESTAMP}
        if payload.name is not None:
            update["name"] = payload.name.strip()
        if payload.description is not None:
            update["description"] = payload.description.strip() if payload.description else None
        if payload.creator is not None:
            update["creator"] = payload.creator.strip() if payload.creator else None
        if payload.preview_image_url is not None:
            update["preview_image_url"] = payload.preview_image_url or None
        if payload.is_public is not None:
            update["is_public"] = bool(payload.is_public)
        if payload.variants is not None:
            variants = [_normalize_variant(v) for v in payload.variants]
            if len({v["device_type"] for v in variants}) != len(variants):
                raise HTTPException(400, "Duplicate device_type in variants")
            update["variants"] = variants
            # update preview to background of pc variant if provided
            if not update.get("preview_image_url"):
                pc_var = next((v for v in variants if v.get("device_type") == "pc"), None)
                src = pc_var or (variants[0] if variants else None)
                update["preview_image_url"] = (src or {}).get("background_url") or None
        if payload.creator is not None or not existing.get("owner_profile_picture"):
            update["owner_profile_picture"] = user.get("profile_picture")
        doc_ref.set(update, merge=True)
        doc = doc_ref.get()
        return _template_doc_to_detail(doc)

    @app.delete("/api/marketplace/templates/{template_id}", status_code=204)
    def delete_template(template_id: str, user: dict = Depends(require_user)):
        doc_ref = _fs().collection(TEMPLATE_COLLECTION).document(template_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(404, "Template not found")
        existing = doc.to_dict() or {}
        if int(existing.get("owner_id") or 0) != int(user["id"]):
            raise HTTPException(403, "Forbidden")
        doc_ref.delete()
        return Response(status_code=204)

    @app.post("/api/marketplace/templates/{template_id}/save")
    def save_template(template_id: str, user: dict = Depends(require_user)):
        doc = _get_template_doc(template_id, user=user)
        save_id = f"{user['id']}_{doc.id}"
        _fs().collection(TEMPLATE_SAVES_COLLECTION).document(save_id).set(
            {
                "user_id": int(user["id"]),
                "template_id": doc.id,
                "created_at": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        return {"ok": True}

    @app.delete("/api/marketplace/templates/{template_id}/save", status_code=204)
    def unsave_template(template_id: str, user: dict = Depends(require_user)):
        save_id = f"{user['id']}_{template_id}"
        _fs().collection(TEMPLATE_SAVES_COLLECTION).document(save_id).delete()
        return Response(status_code=204)

    @app.post("/api/marketplace/templates/{template_id}/apply", response_model=LinktreeOut)
    def apply_template_to_linktree(
        template_id: str,
        device: Optional[str] = Query(None, description="pc, mobile, or both"),
        user: dict = Depends(require_user),
    ):
        _ensure_pg()
        doc = _get_template_doc(template_id, user=user)
        template = doc.to_dict() or {}
        variants = template.get("variants") or []
        if not variants:
            data = template.get("data") or {}
            if data:
                variants = [data]
        var_map = {
            v.get("device_type"): v for v in variants if v.get("device_type") in {"pc", "mobile"}
        }
        target_choice = device or ("both" if len(var_map) == 2 else next(iter(var_map.keys()), "pc"))
        if target_choice not in {"pc", "mobile", "both"}:
            raise HTTPException(400, "Invalid device option")
        if target_choice == "both":
            if "pc" not in var_map or "mobile" not in var_map:
                raise HTTPException(400, "Template does not contain both variants")
            targets = ["pc", "mobile"]
        else:
            if target_choice not in var_map:
                raise HTTPException(404, f"Template has no {target_choice} variant")
            targets = [target_choice]

        results: list[tuple[int, str]] = []  # (id, device)
        with psycopg.connect(db.dsn, row_factory=dict_row) as conn, conn.cursor() as cur:
            for target_device in targets:
                data = var_map[target_device] or {}
                link_fields = _extract_linktree_fields(data)
                link_fields.pop("device_type", None)

                cur.execute(
                    "SELECT id, slug FROM linktrees WHERE user_id=%s AND device_type=%s",
                    (user["id"], target_device),
                )
                row = cur.fetchone()
                if row:
                    linktree_id = row["id"]
                    if link_fields:
                        cols = [f"{k}=%s" for k in link_fields.keys()]
                        vals = list(link_fields.values()) + [linktree_id]
                        cur.execute(
                            f"UPDATE linktrees SET {', '.join(cols)}, updated_at=NOW() WHERE id=%s",
                            tuple(vals),
                        )
                else:
                    slug = _ensure_slug_unique(_resolve_user_slug(user), int(user["id"]))
                    linktree_id = db.create_linktree(
                        user_id=user["id"],
                        slug=slug,
                        device_type=target_device,
                        **link_fields,
                    )
                results.append((linktree_id, target_device))
            conn.commit()

        # Return the manage view for the PC variant if available, otherwise the last applied
        preferred = next((lid for lid, dev in results if dev == "pc"), results[-1][0])
        return get_linktree_manage(preferred, user)

