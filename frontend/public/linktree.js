async function resizeCursorImage(url, size = 32) {
  return new Promise((resolve, reject) => {
    const img = new Image();

    img.crossOrigin = "anonymous"; // Wichtig, falls das PNG von deiner Domain kommt
    img.onload = () => {
      // Canvas erstellen
      const canvas = document.createElement("canvas");
      canvas.width = size;
      canvas.height = size;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, size, size);

      // Bild proportional skalieren
      const scale = Math.min(size / img.width, size / img.height);
      const w = img.width * scale;
      const h = img.height * scale;

      const x = (size - w) / 2;
      const y = (size - h) / 2;

      ctx.drawImage(img, x, y, w, h);

      // Base64 URL erzeugen
      const resizedUrl = canvas.toDataURL("image/png");
      resolve(resizedUrl);
    };

    img.onerror = reject;
    img.src = url;
  });
}

(() => {
  "use strict";

  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  const soundBtn = document.getElementById("soundBtn");
  const soundLabel = document.getElementById("soundLabel");
  let bgVideo = null;
  let bgAudio = null;
  let soundOn = false;
  let gateActive = false;

  function safeUrl(
    raw,
    { allowRelative = true, allowedProtocols = ["http:", "https:"] } = {}
  ) {
    if (typeof raw !== "string") return "";
    const trimmed = raw.trim();
    if (!trimmed) return "";
    if (/^javascript:/i.test(trimmed)) return "";
    try {
      const parsed = new URL(trimmed, window.location.origin);
      if (allowedProtocols.includes(parsed.protocol)) {
        if (
          (parsed.protocol === "http:" || parsed.protocol === "https:") &&
          !trimmed.startsWith("http") &&
          allowRelative
        ) {
          return trimmed;
        }
        return parsed.href;
      }
    } catch (_) {
      if (
        allowRelative &&
        trimmed.startsWith("/") &&
        !trimmed.startsWith("//")
      ) {
        return trimmed;
      }
    }
    if (
      allowRelative &&
      trimmed.startsWith("/") &&
      !trimmed.startsWith("//")
    ) {
      return trimmed;
    }
    return "";
  }

  const safeMediaUrl = (raw) =>
    safeUrl(raw, {
      allowRelative: true,
      allowedProtocols: ["http:", "https:"],
    });

  function hexToRgba(hex, alpha = 1) {
    const m = /^#?([a-fA-F0-9]{3}|[a-fA-F0-9]{6})$/.exec((hex || "").trim());
    if (!m) return null;
    let h = m[1];
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const int = parseInt(h, 16);
    const r = (int >> 16) & 255;
    const g = (int >> 8) & 255;
    const b = int & 255;
    const a = Math.min(1, Math.max(0, alpha));
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
  const safeLinkUrl = (raw) =>
    safeUrl(raw, {
      allowRelative: false,
      allowedProtocols: ["http:", "https:", "mailto:", "tel:"],
    });

  const VIDEO_EXTENSIONS = [".mp4", ".webm", ".m4v", ".mov"];
  const looksLikeVideo = (url) =>
    VIDEO_EXTENSIONS.some((ext) => (url || "").toLowerCase().includes(ext));

  function pickBackgroundSource(data, preferLite) {
    const primary = safeMediaUrl(data.background_url);
    const low =
      preferLite &&
      (safeMediaUrl(data.background_low_url) ||
        safeMediaUrl(data.background_preview_url) ||
        null);
    const chosen = (low || primary || "").trim();
    const useVideo =
      data.background_is_video &&
      looksLikeVideo(chosen) &&
      !(preferLite && low && !looksLikeVideo(low));
    return { url: chosen, isVideo: useVideo };
  }

  const DEFAULT_LINK_COLOR = "#e9ebff";
  const DEFAULT_LINK_BG = "#171b3b";
  const DEFAULT_TEXT_COLOR = "#e9ebff";
  const DEFAULT_NAME_COLOR = "#e9ebff";
  const DEFAULT_LOCATION_COLOR = "#a2a9c8";
  const DEFAULT_QUOTE_COLOR = "#e9ebff";

  function sanitizeHexColor(raw) {
    if (typeof raw !== "string") return "";
    const trimmed = raw.trim();
    return /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(trimmed)
      ? trimmed
      : "";
  }

  function hexToRgb(hex) {
    const clean = sanitizeHexColor(hex);
    if (!clean) return null;
    const raw = clean.slice(1);
    const parts =
      raw.length === 3
        ? raw.split("").map((ch) => ch + ch)
        : [raw.slice(0, 2), raw.slice(2, 4), raw.slice(4, 6)];
    const [r, g, b] = parts.map((p) => parseInt(p, 16));
    return { r, g, b };
  }

  function rgbToRgba(rgb, alpha = 1) {
    if (!rgb) return "";
    const a = Math.min(1, Math.max(0, alpha));
    return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${a})`;
  }

  function lightenRgb(rgb, amount = 0.08) {
    const t = Math.min(1, Math.max(0, amount));
    return {
      r: Math.round(rgb.r + (255 - rgb.r) * t),
      g: Math.round(rgb.g + (255 - rgb.g) * t),
      b: Math.round(rgb.b + (255 - rgb.b) * t),
    };
  }

  function applyLinkStyles(linkColorRaw, linkBgRaw, linkBgAlphaRaw) {
    const rootStyle = document.documentElement.style;
    const linkColor = sanitizeHexColor(linkColorRaw) || DEFAULT_LINK_COLOR;
    const bgHex = sanitizeHexColor(linkBgRaw) || DEFAULT_LINK_BG;
    const bgRgb = hexToRgb(bgHex) || hexToRgb(DEFAULT_LINK_BG);
    const alpha =
      Math.min(100, Math.max(0, Number(linkBgAlphaRaw ?? 100))) / 100;
    const base = rgbToRgba(bgRgb, alpha);
    const hover = rgbToRgba(lightenRgb(bgRgb, 0.07), alpha);
    rootStyle.setProperty("--link-color", linkColor);
    rootStyle.setProperty("--link-bg", base || DEFAULT_LINK_BG);
    rootStyle.setProperty("--link-bg-hover", hover || base || DEFAULT_LINK_BG);
  }

  function applyTextColors(data) {
    const rootStyle = document.documentElement.style;
    const base = sanitizeHexColor(data?.text_color) || DEFAULT_TEXT_COLOR;
    const location =
      sanitizeHexColor(data?.location_color) ||
      sanitizeHexColor(data?.text_color) ||
      DEFAULT_LOCATION_COLOR;
    const name =
      sanitizeHexColor(data?.name_color) ||
      sanitizeHexColor(data?.link_color) ||
      DEFAULT_NAME_COLOR;
    const quote =
      sanitizeHexColor(data?.quote_color) || base || DEFAULT_QUOTE_COLOR;

    rootStyle.setProperty("--ink", base);
    rootStyle.setProperty("--muted", location);
    rootStyle.setProperty("--name-color", name);
    rootStyle.setProperty("--location-color", location);
    rootStyle.setProperty("--quote-color", quote);
  }

  function createLocationIcon() {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", "16");
    svg.setAttribute("height", "16");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("fill", "currentColor");
    svg.setAttribute("aria-hidden", "true");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute(
      "d",
      "M12 2C7.6 2 4 5.6 4 10c0 5.3 8 12 8 12s8-6.7 8-12c0-4.4-3.6-8-8-8zm0 11a3 3 0 1 1 0-6 3 3 0 0 1 0 6z"
    );
    svg.appendChild(path);
    return svg;
  }

  function updateSoundUI() {
    if (!soundBtn) return;
    soundBtn.classList.toggle("on", soundOn);
    soundLabel.textContent = soundOn ? "Sound on" : "Sound off";
  }

  function showGate(show) {
    const gate = document.getElementById("enterOverlay");
    if (!gate) return;
    if (show) {
      gate.classList.remove("hidden");
      gateActive = true;
      gate.focus({ preventScroll: true });
    } else {
      gate.classList.add("hidden");
      gateActive = false;
    }
  }

  async function turnSoundOn() {
    try {
      if (bgAudio) {
        bgAudio.muted = false;
        await bgAudio.play().catch(() => {});
      } else if (bgVideo) {
        bgVideo.muted = false;
        await bgVideo.play().catch(() => {});
      } else {
        return;
      }
      soundOn = true;
      updateSoundUI();
      soundBtn.classList.remove("hidden");
    } catch (_) {
      // bleibt off
    }
  }

  function turnSoundOff() {
    try {
      if (bgAudio) {
        bgAudio.muted = true;
      }
      if (bgVideo) {
        bgVideo.muted = true;
      }
      soundOn = false;
      updateSoundUI();
    } catch (_) {}
  }

  if (soundBtn) {
    soundBtn.addEventListener("click", () => {
      if (soundOn) turnSoundOff();
      else turnSoundOn();
    });
  }

  function enterExperience() {
    showGate(false);
    if (bgVideo) {
      bgVideo.muted = false;
      bgVideo.play().catch(() => {});
    }
    if (bgAudio) {
      bgAudio.muted = false;
      bgAudio
        .play()
        .then(() => {
          soundOn = true;
          updateSoundUI();
        })
        .catch(() => {});
    }
  }

  const gate = document.getElementById("enterOverlay");
  if (gate) {
    gate.addEventListener("click", enterExperience);
    gate.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        enterExperience();
      }
    });
  }

  // Ein “erster User-Input” schaltet optional sofort Ton an
  const firstGesture = () => {
    turnSoundOn();
    window.removeEventListener("pointerdown", firstGesture);
  };
  window.addEventListener("pointerdown", firstGesture, { once: true });

  const CACHE_PREFIX = "taoma:linktree:v1:";
  const CACHE_MS = 90 * 1000; // short-lived cache to speed up repeat visits
  let lastData = null;
  let lastMediaMode = null;
  let lastDevice = null;
  let discordStatusCache = null;
  let discordStatusPromise = null;

  function getConnectionProfile() {
    const conn =
      navigator.connection || navigator.mozConnection || navigator.webkitConnection;
    const downlink =
      conn && typeof conn.downlink === "number" ? conn.downlink : null;
    return {
      saveData:
        !!(conn && conn.saveData) ||
        (typeof window.matchMedia === "function" &&
          window.matchMedia("(prefers-reduced-data: reduce)").matches),
      effectiveType: (conn && conn.effectiveType) || "",
      downlink,
    };
  }

  function shouldUseLiteMedia(profile = getConnectionProfile()) {
    const dl = typeof profile.downlink === "number" ? profile.downlink : null;
    const slow =
      profile.saveData ||
      ["slow-2g", "2g", "3g"].includes(profile.effectiveType || "") ||
      (dl !== null && dl < 1.5);
    return !!slow;
  }

  function makeCacheKey({ slug, templateId, device }) {
    if (templateId) return `${CACHE_PREFIX}tpl:${templateId}:${device}`;
    return `${CACHE_PREFIX}slug:${slug || "anon"}:${device}`;
  }

  function readCachedTree(key) {
    try {
      const raw = sessionStorage.getItem(key);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || typeof parsed.savedAt !== "number" || !parsed.data) return null;
      if (Date.now() - parsed.savedAt > CACHE_MS) {
        sessionStorage.removeItem(key);
        return null;
      }
      return parsed.data;
    } catch (_) {
      return null;
    }
  }

  function writeCachedTree(key, data) {
    try {
      sessionStorage.setItem(
        key,
        JSON.stringify({ savedAt: Date.now(), data: data || null })
      );
    } catch (_) {}
  }

  async function fetchDiscordStatus() {
    if (discordStatusCache !== null) return discordStatusCache;
    if (discordStatusPromise) return discordStatusPromise;
    discordStatusPromise = (async () => {
      try {
        const r = await fetch("/api/discord/status", {
          credentials: "include",
        });
        if (!r.ok) throw new Error("discord status failed");
        const json = await r.json();
        discordStatusCache = json || { linked: false };
      } catch (_) {
        discordStatusCache = { linked: false };
      } finally {
        discordStatusPromise = null;
      }
      return discordStatusCache;
    })();
    return discordStatusPromise;
  }

  function updateProfileFromUser(data) {
    if (!data || !data.is_template_demo) return;
    if (!currentUser || !currentUser.profile_picture) return;
    const pfp = document.getElementById("pfp");
    const url = safeMediaUrl(currentUser.profile_picture);
    if (!pfp || !url) return;
    try {
      const absolute = new URL(url, window.location.origin).href;
      if (pfp.src === absolute) return;
    } catch (_) {
      // ignore comparison errors
    }
    pfp.src = url;
  }

  async function applyUserDiscordFrame(data) {
    if (!data || !data.is_template_demo || !data.discord_frame_enabled) return;
    const status = await fetchDiscordStatus();
    if (!status || !status.linked || !status.decoration_url) return;
    const frame = document.getElementById("pfpFrame");
    const url = safeMediaUrl(status.decoration_url);
    if (!frame || !url) return;
    frame.src = url;
    frame.style.display = "block";
  }

  async function hydrateUserEnhancements(data, device, userPromise) {
    try {
      const user = await userPromise.catch(() => null);
      currentUser = currentUser || user;
      if (!data) return;
      if (data.is_template_demo) {
        updateProfileFromUser(data);
        await applyUserDiscordFrame(data);
      }
    } catch (_) {
      // ignore hydration errors to avoid blocking initial render
    }
  }

  let currentUser = null;

  async function fetchSessionUser() {
    try {
      const r = await fetch("/api/auth/verify", { credentials: "include" });
      if (!r.ok) return null;
      const json = await r.json();
      currentUser = json && json.user ? json.user : null;
      return currentUser;
    } catch (_) {
      return null;
    }
  }

  function detectDevice(requestedDevice) {
    return requestedDevice === "mobile"
      ? "mobile"
      : requestedDevice === "pc"
      ? "pc"
      : window.matchMedia("(max-width: 720px)").matches
      ? "mobile"
      : "pc";
  }

  let visibilityHandlerAdded = false;

  async function fetchTreeData({ templateId, slug, device }) {
    if (templateId) {
      const r = await fetch(
        `/api/marketplace/templates/${encodeURIComponent(templateId)}`,
        { credentials: "include" }
      );
      if (!r.ok) return null;
      const tpl = await r.json();
      const variants = tpl.variants || [];
      const chosen =
        variants.find((v) => v.device_type === device) ||
        variants[0] ||
        tpl.data ||
        {};
      const data = { ...(chosen || {}) };
      data.slug = data.slug || tpl.name || "Template";
      data.device_type = data.device_type || device;
      data.user_username = data.user_username || tpl.owner_username || "Template";
      data.profile_picture = data.profile_picture || tpl.preview_image_url || null;
      data.is_template_demo = true;
      return data;
    }

    const r = await fetch(
      `/api/linktrees/${encodeURIComponent(slug)}?device=${device}`
    );
    if (!r.ok) return null;
    return await r.json();
  }

  function renderTree(
    data,
    { preferLite = shouldUseLiteMedia(), fromCache = false, device = detectDevice() } = {}
  ) {
    if (!data) return;
    lastData = data;
    lastMediaMode = preferLite ? "lite" : "full";
    lastDevice = device;
    const isMobile = device === "mobile";
    let needsGate = false;

    let bgIsVideo = false;
    let bgUrl = "";
    const bgEl = document.getElementById("bg");
    bgVideo = null;
    if (bgEl) {
      bgEl.textContent = "";
      const bgSource = pickBackgroundSource(data, preferLite);
      bgUrl = bgSource.url;
      bgIsVideo = bgSource.isVideo;
      if (bgUrl) {
        if (bgIsVideo) {
          const video = document.createElement("video");
          video.id = "bgvid";
          video.src = bgUrl;
          video.autoplay = false;
          video.muted = true;
          video.loop = true;
          video.setAttribute("muted", "");
          video.setAttribute("playsinline", "");
          video.setAttribute("webkit-playsinline", "");
          video.preload = preferLite ? "metadata" : "auto";
          if (preferLite) {
            video.setAttribute("data-lite", "true");
          }
          bgEl.appendChild(video);
          bgVideo = video;
          needsGate = true;
          if (!visibilityHandlerAdded) {
            document.addEventListener("visibilitychange", () => {
              if (!document.hidden && !gateActive && bgVideo)
                bgVideo.play().catch(() => {});
            });
            visibilityHandlerAdded = true;
          }
        } else {
          const img = document.createElement("img");
          img.src = bgUrl;
          img.alt = "Background";
          img.loading = "lazy";
          img.decoding = "async";
          bgEl.appendChild(img);
        }
      }
    }
    if (bgEl) {
      bgEl.classList.remove("night", "rain", "snow");
      if (data.background_effect && data.background_effect !== "none") {
        bgEl.classList.add(data.background_effect);
      }
    }

    // Audio (Song hat Vorrang vor Video-Ton)
    if (bgAudio) {
      try {
        bgAudio.pause();
      } catch {}
      bgAudio.remove();
      bgAudio = null;
    }
    const songUrl = safeMediaUrl(data.song_url);
    if (songUrl) {
      bgAudio = document.createElement("audio");
      bgAudio.src = songUrl;
      bgAudio.loop = true;
      bgAudio.preload = preferLite ? "metadata" : "auto";
      bgAudio.muted = true; // Autoplay-friendly start, gate will unmute
      document.body.appendChild(bgAudio);
      bgAudio.play().catch(() => {});
      needsGate = true;
      if (soundBtn) soundBtn.classList.remove("hidden"); // Button zeigen - Nutzer kann unmute'n
    } else if (bgIsVideo && bgVideo) {
      // Kein Song -> Videoton anbieten
      if (soundBtn) soundBtn.classList.remove("hidden");
    } else {
      if (soundBtn) soundBtn.classList.add("hidden"); // nichts abzuspielen
    }
    if (needsGate) {
      showGate(true);
    } else {
      showGate(false);
    }
    soundOn = false;
    updateSoundUI();

    const cardEl = document.getElementById("card");
    if (cardEl) {
      const raw = Number(data.transparency);
      const clamped = Number.isFinite(raw) ? Math.min(100, Math.max(0, raw)) : 0;
      const cardAlpha = 1 - clamped / 100;

      if (cardAlpha <= 0) {
        cardEl.classList.add("transparent");
      } else {
        cardEl.classList.remove("transparent");
        cardEl.style.setProperty("--card-alpha", cardAlpha.toString());
      }
    }
    applyTextColors(data);
    applyLinkStyles(data.link_color, data.link_bg_color, data.link_bg_alpha);

    const cursorUrl = safeMediaUrl(data.cursor_url);
    if (cursorUrl && !isMobile) {
      resizeCursorImage(cursorUrl, 32)
        .then((resized) => {
          const cursorDecl = `url("${resized}") 0 0, auto`;
          document.documentElement.style.setProperty("--cursor", cursorDecl);
        })
        .catch(() => {
          document.documentElement.style.setProperty("--cursor", "auto");
        });
    } else {
      document.documentElement.style.setProperty("--cursor", "auto");
    }

    // Profil/Badges/Links wie gehabt
    const pfp = document.getElementById("pfp");
    const pfpFrame = document.getElementById("pfpFrame");
    if (pfp) {
      const prefersUserPfp =
        data.is_template_demo && currentUser && currentUser.profile_picture;
      const profileUrl =
        safeMediaUrl(prefersUserPfp ? currentUser.profile_picture : data.profile_picture) ||
        safeMediaUrl(currentUser && currentUser.profile_picture) ||
        "/static/icon.png";
      pfp.src = profileUrl;
    }
    if (pfpFrame) {
      const frameUrl = safeMediaUrl(data.discord_decoration_url);
      if (data.discord_frame_enabled && frameUrl) {
        pfpFrame.src = frameUrl;
        pfpFrame.style.display = "block";
      } else {
        pfpFrame.style.display = "none";
      }
    }
    const nameEl = document.getElementById("name");
    if (nameEl) {
      const mode = data.display_name_mode || "slug";
      if (mode === "username") {
        nameEl.textContent = data.user_username || data.slug || "User";
      } else if (mode === "custom" && data.custom_display_name) {
        nameEl.textContent = data.custom_display_name;
      } else {
        nameEl.textContent = data.slug || "User";
      }

      // Effekt-Klassen setzen
      nameEl.classList.remove("glow", "neon", "rainbow");
      if (data.name_effect && data.name_effect !== "none") {
        nameEl.classList.add(data.name_effect);
      }
    }
    const quoteEl = document.getElementById("quote");
    if (quoteEl) quoteEl.textContent = data.quote || "";
    const locEl = document.getElementById("loc");
    if (locEl) {
      locEl.textContent = "";
      const locText = typeof data.location === "string" ? data.location.trim() : "";
      if (locText) {
        const icon = createLocationIcon();
        const text = document.createElement("span");
        text.textContent = ` ${locText}`;
        locEl.appendChild(icon);
        locEl.appendChild(text);
      }
    }

    const badgesEl = document.getElementById("badges");
    if (badgesEl) {
      badgesEl.innerHTML = "";
      const frag = document.createDocumentFragment();
      (data.icons || []).forEach((ic) => {
        if (!ic.displayed) return;

        const desc = ic.description || ic.code || "badge";
        const wrap = document.createElement("span");
        wrap.className = "badge";
        wrap.title = desc;
        wrap.setAttribute("aria-label", desc);

        const img = document.createElement("img");
        const iconUrl = safeMediaUrl(ic.image_url);
        if (!iconUrl) return;
        img.src = iconUrl;
        img.alt = desc;
        img.width = 20;
        img.height = 20;
        img.loading = "lazy";
        img.decoding = "async";

        wrap.appendChild(img);
        frag.appendChild(wrap);
      });
      badgesEl.appendChild(frag);
    }
    const linksEl = document.getElementById("links");
    if (linksEl) {
      linksEl.textContent = "";
      const frag = document.createDocumentFragment();
      (data.links || []).forEach((l) => {
        if (!l.is_active) return;
        const href = safeLinkUrl(l.url);
        if (!href) return;
        const a = document.createElement("a");
        a.className = "link";
        a.href = href;
        a.target = "_blank";
        a.rel = "noreferrer noopener";
        const iconUrl = safeMediaUrl(l.icon_url);
        if (iconUrl) {
          const img = document.createElement("img");
          img.src = iconUrl;
          img.alt = "";
          img.loading = "lazy";
          img.decoding = "async";
          a.appendChild(img);
        }
        const span = document.createElement("span");
        span.textContent = l.label || l.url;
        a.appendChild(span);
        frag.appendChild(a);
      });
      linksEl.appendChild(frag);
    }

    const visitBox = document.getElementById("visitCounter");
    const visitValue = document.getElementById("visitCounterValue");
    if (visitBox && visitValue) {
      if (data.show_visit_counter) {
        visitValue.textContent = Number(data.visit_count || 0).toLocaleString();
        const textColor =
          hexToRgba(data.visit_counter_color, 1) ||
          hexToRgba(data.text_color, 1) ||
          null;
        if (textColor) visitBox.style.color = textColor;
        const bgAlpha = Number.isFinite(Number(data.visit_counter_bg_alpha))
          ? Number(data.visit_counter_bg_alpha)
          : 20;
        const bgColor =
          hexToRgba(
            data.visit_counter_bg_color || "#ffffff",
            Math.min(1, Math.max(0, bgAlpha / 100))
          ) || "rgba(255,255,255,0.14)";
        visitBox.style.backgroundColor = bgColor;
        visitBox.style.display = "inline-flex";
      } else {
        visitBox.style.display = "none";
      }
    }
  }

  async function loadTree() {
    try {
      const params = new URLSearchParams(location.search);
      let templateId = params.get("template_id");
      const parts = location.pathname.split("/").filter(Boolean);
      if (!templateId && parts[0] === "marketplace" && parts[1] === "templates") {
        templateId = parts[2] || null;
      }
      const requestedDevice = params.get("device");
      const device = detectDevice(requestedDevice);
      const slug = decodeURIComponent(location.pathname.split("/").pop() || "");
      const cacheKey = makeCacheKey({ slug, templateId, device });
      const preferLite = shouldUseLiteMedia();

      const userPromise = fetchSessionUser();
      const cached = readCachedTree(cacheKey);
      if (cached) {
        renderTree(cached, { preferLite, fromCache: true, device });
        hydrateUserEnhancements(cached, device, userPromise).catch(() => {});
      }

      const dataPromise = fetchTreeData({ templateId, slug, device });
      const liveData = await dataPromise;
      let data = liveData || cached;
      if (!data) return;
      writeCachedTree(cacheKey, data);
      renderTree(data, { preferLite, fromCache: false, device });
      hydrateUserEnhancements(data, device, userPromise).catch(() => {});
    } catch (e) {
      console.error("loadTree failed:", e);
    }
    soundOn = false;
    updateSoundUI();
  }

  const connection =
    navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  const prefersReducedData =
    typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-reduced-data: reduce)")
      : null;

  function handleMediaPreferenceChange() {
    if (!lastData) return;
    const preferLite = shouldUseLiteMedia();
    if ((preferLite ? "lite" : "full") === lastMediaMode) return;
    renderTree(lastData, {
      preferLite,
      fromCache: true,
      device: lastDevice || detectDevice(),
    });
  }

  if (connection && connection.addEventListener) {
    connection.addEventListener("change", handleMediaPreferenceChange);
  }
  if (prefersReducedData && prefersReducedData.addEventListener) {
    prefersReducedData.addEventListener("change", handleMediaPreferenceChange);
  } else if (prefersReducedData && prefersReducedData.addListener) {
    prefersReducedData.addListener(handleMediaPreferenceChange);
  }

  loadTree();
  document.addEventListener("DOMContentLoaded", () => {
    console.log("Page loaded (HTML ready)");
  });
})();
