import Link from "next/link";

export default function Home() {
  return (
    <main
      style={{
        minHeight: "100vh",
        display: "grid",
        placeItems: "center",
        padding: "48px 20px",
        textAlign: "center",
      }}
    >
      <div
        style={{
          maxWidth: 640,
          background: "rgba(13, 16, 35, 0.82)",
          borderRadius: 16,
          padding: "32px 28px",
          border: "1px solid #ffffff22",
          boxShadow: "0 16px 40px rgba(0,0,0,0.4)",
        }}
      >
        <p style={{ letterSpacing: "0.12em", fontSize: 12, opacity: 0.9 }}>
          TAOMA SPACE
        </p>
        <h1 style={{ margin: "8px 0 12px", fontSize: 32, lineHeight: 1.2 }}>
          Next.js Frontend Sandbox
        </h1>
        <p style={{ margin: "0 0 20px", opacity: 0.82 }}>
          Dies ist der neue Next.js-Entrypoint. Die bestehende FastAPI-Backends
          und Assets bleiben unverändert. Starte mit der Linktree-Seite oder
          ergänze weitere Seiten im <code>app/</code>-Verzeichnis.
        </p>
        <div style={{ display: "flex", gap: 12, justifyContent: "center" }}>
          <Link
            href="/linktree"
            style={{
              padding: "12px 16px",
              borderRadius: 12,
              border: "1px solid #ffffff33",
              background: "#7f8bff",
              color: "#0a0c18",
              fontWeight: 700,
            }}
          >
            Linktree öffnen
          </Link>
          <a
            href="https://nextjs.org/docs"
            target="_blank"
            rel="noreferrer"
            style={{
              padding: "12px 16px",
              borderRadius: 12,
              border: "1px solid #ffffff33",
              color: "inherit",
              fontWeight: 700,
              background: "#ffffff0d",
            }}
          >
            Next.js Docs
          </a>
        </div>
      </div>
    </main>
  );
}
