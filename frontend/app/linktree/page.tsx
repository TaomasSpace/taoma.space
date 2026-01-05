import type { Metadata } from "next";
import Script from "next/script";
import styles from "./linktree.module.css";

export const metadata: Metadata = {
  title: "TAOMA™ — Linktree",
  description:
    "TAOMA™ Linktree: zentrale Anlaufstelle für alle TAOMA Links, Socials und Projekte.",
  alternates: { canonical: "https://taoma.space/linktree" },
  openGraph: {
    title: "TAOMA™ — Linktree",
    description: "Alle TAOMA Links, Socials und Projekte auf einen Blick.",
    type: "profile",
    url: "https://taoma.space/linktree",
    images: [{ url: "https://taoma.space/static/icon.png" }],
  },
  twitter: {
    card: "summary",
    title: "TAOMA™ — Linktree",
    description: "Alle TAOMA Links, Socials und Projekte auf einen Blick.",
    images: ["https://taoma.space/static/icon.png"],
  },
  robots: "index,follow",
  themeColor: "#0f1223",
};

export default function LinktreePage() {
  return (
    <div className={styles.page}>
      <div className="bg" id="bg" />
      <div
        id="enterOverlay"
        className="enter-gate hidden"
        role="button"
        tabIndex={0}
        aria-label="Enter site to enable media"
      >
        <div className="enter-box">
          <p>Click to enter</p>
          <span className="muted">Enable video &amp; sound</span>
        </div>
      </div>
      <button id="soundBtn" className="sound-toggle hidden" aria-label="Toggle sound">
        <span className="dot" aria-hidden="true" />
        <span id="soundLabel">Sound off</span>
      </button>
      <main>
        <div className="card" id="card">
          <div className="pfp-wrap">
            <img
              id="pfp"
              className="pfp"
              src="/static/icon.png"
              loading="lazy"
              decoding="async"
              alt="Profile picture"
            />
            <img
              id="pfpFrame"
              className="pfp-frame"
              src=""
              alt=""
              loading="lazy"
              decoding="async"
              style={{ display: "none" }}
            />
          </div>
          <div id="name" className="name">
            User
          </div>
          <div id="quote" className="quote">
            -
          </div>
          <div className="location" id="loc" />
          <div className="badges" id="badges" />
          <div className="links" id="links" />
          <div id="visitCounter" className="visit-counter" style={{ display: "none" }}>
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M2.458 12C3.732 7.943 7.523 5 12 5c4.477 0 8.268 2.943 9.542 7-1.274 4.057-5.065 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
              />
            </svg>
            <span id="visitCounterValue">0</span>
          </div>
        </div>
      </main>
      <footer>
        © <span id="year"></span> TAOMA™
      </footer>
      <Script src="/linktree.js" strategy="afterInteractive" />
    </div>
  );
}
