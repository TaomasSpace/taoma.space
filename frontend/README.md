# TAOMA Next.js Frontend

Lokaler Dev-Start:

1. Node 18+ installieren (z. B. über `nvm install --lts`).
2. Ab ins Verzeichnis `frontend`.
3. `npm install`
4. Optional: `BACKEND_ORIGIN=http://localhost:8000` setzen, damit Next die `/api/*`-Calls an dein FastAPI-Backend proxyt (siehe `next.config.mjs`).
5. `npm run dev` starten und `http://localhost:3000/linktree` öffnen.

Assets liegen gespiegelt in `public/static`, die Linktree-Logik steckt in `public/linktree.js` und wird clientseitig per `<Script>` geladen. Weitere Seiten kannst du im `app/`-Verzeichnis ergänzen.

Legacy-Seiten: Alle bisherigen HTML-Dateien liegen unter `public/legacy`. Der Catch-all-Route `app/(legacy)/[...slug]/page.tsx` rendert sie serverseitig, damit alle alten Pfade (z. B. `/admin`, `/datenschutz`, `/marketplace`, `/portfolio/about`) weiter funktionieren, während du nach und nach echte React-Seiten bauen kannst. Bereits neu umgesetzt: `/` (Landing) und `/linktree`.
