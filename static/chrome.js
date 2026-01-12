(function () {
  const navConfig = {
    primary: [
      { key: "linktree", href: "/linktree", label: "Linktree" },
      { key: "marketplace", href: "/marketplace", label: "Marketplace" },
    ],
    secondary: [
      { key: "builder", href: "/linktree/config", label: "Linktree Builder" },
    ],
    more: [
      { key: "api", href: "/api", label: "GIF API" },
      { key: "api-admin", href: "/api/gif/admin", label: "GIF Admin" },
      { key: "portfolio", href: "/portfolio", label: "Portfolio" },
    ],
  };

  const footerLinks = [
    { href: "/linktree", label: "Linktree" },
    { href: "/marketplace", label: "Marketplace" },
    { href: "/api", label: "GIF API" },
    { href: "/portfolio", label: "Portfolio" },
    { href: "/datenschutz.html", label: "Privacy" },
  ];

  const resolveActiveKey = (path) => {
    const normalized = (path || "").toLowerCase();
    if (!normalized || normalized === "/") return "home";
    if (normalized.includes("marketplace")) return "marketplace";
    if (normalized.includes("/linktree/config") || normalized.includes("linktree_config")) return "builder";
    if (normalized.includes("linktree")) return "linktree";
    if (normalized.includes("/api/gif/admin") || normalized.includes("gifapiadmin")) return "api-admin";
    if (normalized.startsWith("/api") || normalized.includes("gifapimain")) return "api";
    if (normalized.includes("portfolio")) return "portfolio";
    return "";
  };

  const createLink = (item, className) => {
    const a = document.createElement("a");
    a.href = item.href;
    a.textContent = item.label;
    if (item.key) a.dataset.key = item.key;
    if (className) a.className = className;
    return a;
  };

  const removeExisting = (selector, className) => {
    const nodes = Array.from(document.querySelectorAll(selector)).filter(
      (node) => !node.classList.contains(className)
    );
    nodes.forEach((node) => node.remove());
  };

  const renderHeader = () => {
    removeExisting("body > header", "shell-header");

    const header = document.createElement("header");
    header.className = "shell-header";
    header.innerHTML = `
      <div class="shell-container shell-nav">
        <a class="shell-brand" href="/">
          <img src="/static/icon.png" alt="TAOMA logo" loading="lazy" />
          <span>TAOMA</span>
        </a>
        <button class="shell-mobile-toggle" type="button" aria-label="Toggle menu" aria-expanded="false">
          Menu
        </button>
        <nav class="shell-links" aria-label="Hauptnavigation"></nav>
        <div class="shell-auth auth" id="authSlot"></div>
      </div>
    `;

    const navEl = header.querySelector(".shell-links");
    const mobileToggle = header.querySelector(".shell-mobile-toggle");

    navConfig.primary.forEach((item) => {
      navEl.appendChild(createLink(item, "primary"));
    });

    navConfig.secondary.forEach((item) => {
      navEl.appendChild(createLink(item, "secondary"));
    });

    const moreWrap = document.createElement("div");
    moreWrap.className = "shell-more";
    moreWrap.innerHTML = `
      <button type="button" aria-expanded="false">More v</button>
      <div class="shell-more-menu" role="menu"></div>
    `;
    const moreMenu = moreWrap.querySelector(".shell-more-menu");
    navConfig.more.forEach((item) => {
      moreMenu.appendChild(createLink(item, "minor"));
    });
    navEl.appendChild(moreWrap);

    const activeKey = resolveActiveKey(window.location.pathname);
    header.querySelectorAll("[data-key]").forEach((link) => {
      if (link.dataset.key === activeKey) {
        link.dataset.current = "true";
        link.setAttribute("aria-current", "page");
      }
    });

    mobileToggle?.addEventListener("click", () => {
      const open = navEl.classList.toggle("open");
      mobileToggle.setAttribute("aria-expanded", String(open));
      moreWrap.classList.remove("open");
    });

    const closeMenus = () => {
      navEl.classList.remove("open");
      mobileToggle?.setAttribute("aria-expanded", "false");
      moreWrap.classList.remove("open");
      moreWrap.querySelector("button")?.setAttribute("aria-expanded", "false");
    };

    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!header.contains(target)) {
        closeMenus();
      }
    });

    navEl.addEventListener("click", (event) => {
      const target = event.target;
      if (target.tagName === "A") {
        closeMenus();
      }
    });

    const moreBtn = moreWrap.querySelector("button");
    moreBtn?.addEventListener("click", (event) => {
      event.stopPropagation();
      const open = moreWrap.classList.toggle("open");
      moreBtn.setAttribute("aria-expanded", String(open));
    });

    document.body.prepend(header);
  };

  const renderFooter = () => {
    removeExisting("body > footer", "shell-footer");

    const footer = document.createElement("footer");
    footer.className = "shell-footer";
    const year = new Date().getFullYear();

    const linksHtml = footerLinks
      .map(
        (link) => `<a href="${link.href}">${link.label}</a>`
      )
      .join("");

    footer.innerHTML = `
      <div class="shell-container">
        <div class="footer-meta">© <span id="year">${year}</span> TAOMA™</div>
        <div class="footer-links">${linksHtml}</div>
        <div class="footer-visit" title="Visitors (daily de-duplicated)">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
            <path d="M12 5c-4.8 0-9 3.2-10.5 7 1.5 3.8 5.7 7 10.5 7s9-3.2 10.5-7C21 8.2 16.8 5 12 5Zm0 2c3.1 0 6.1 1.9 7.6 5C18.1 15.1 15.1 17 12 17s-6.1-1.9-7.6-5C5.9 8.9 8.9 7 12 7Zm0 2a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm0 2a1 1 0 1 1 0 2 1 1 0 0 1 0-2Z" fill="currentColor"/>
          </svg>
          <span id="visitor-count">—</span>
        </div>
      </div>
    `;

    document.body.appendChild(footer);
  };

  const init = () => {
    renderHeader();
    renderFooter();
    if (typeof window.renderAuthHeader === "function") {
      try {
        window.renderAuthHeader();
      } catch {
        /* noop */
      }
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
