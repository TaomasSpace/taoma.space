# Sicherheitsbewertung von taoma.space

## Überblick
Die Anwendung kombiniert statische Seiten mit einem FastAPI-Backend für Authentifizierung, Linktree-Verwaltung und Medien-Uploads. Viele Schutzmaßnahmen sind bereits vorhanden, etwa rollenbasierte Zugriffskontrollen und sichere Passwortspeicherung. Dennoch existieren einige Schwachstellen mit hohem Risiko, vor allem im Bereich der Client-seitigen Darstellung und des Token-Handlings.

## Positive Sicherheitsaspekte
- **Rollen- und Sitzungskontrollen:** Administrative Endpunkte setzen `require_admin` bzw. `require_user` ein und verlassen sich auf valide Tokens, sodass nur angemeldete Nutzer oder Administratoren die sensiblen APIs aufrufen können.【F:main/main.py†L93-L114】
- **Sichere Passwortverarbeitung:** Passwörter werden vor dem Speichern mit bcrypt gehasht und bei der Anmeldung korrekt geprüft; ein direktes Zurückrechnen wird explizit verhindert.【F:main/main.py†L117-L139】
- **Härtung bei Cookies und Uploads:** Login-Antworten setzen ein `HttpOnly`-/`Secure`-Cookie, wodurch das Session-Token vor einfachem JavaScript-Zugriff geschützt wird. Upload-Endpunkte validieren Typ, Größe und Bildintegrität bevor Dateien persistiert werden.【F:main/main.py†L197-L206】【F:main/main.py†L1162-L1201】
- **Mixed-Content-Schutz:** Die Startseite sendet jetzt eine `Content-Security-Policy` mit `upgrade-insecure-requests`, wodurch Browser gemischte HTTP-Ressourcen automatisch auf HTTPS anheben und keine "Nicht sicher"-Warnung mehr zeigen.【F:index.html†L13-L14】

## Kritische Risiken
1. **Persistente XSS in Linktree-Profilen:** Mehrere Felder (`location`, Link-Label) werden ohne Escaping per `innerHTML` in die Linktree-Seite eingefügt. Ein Angreifer kann dadurch JavaScript einschleusen, das beim Öffnen der öffentlichen Profilseite ausgeführt wird – inklusive potenziellem Diebstahl von Tokens oder Session-Daten anderer Besucher.【F:linktree.html†L860-L901】
2. **Token-Leakage über URL-Parameter:** `_extract_token` akzeptiert Tokens auch aus `?token=`-Query-Parametern. Solche URLs landen häufig in Browser-Historien, Server-Logs oder Referrern und können von Drittparteien abgefangen werden, was eine Session-Übernahme ermöglicht.【F:main/main.py†L209-L230】
3. **Token-Speicherung in `localStorage`:** Die Login-Seite speichert das erhaltene Token zusätzlich zu den sicheren Cookies in `localStorage`. Jeder XSS-Angriff (siehe Punkt 1) kann damit das Token auslesen und an Angreifer exfiltrieren. Es gibt keine Notwendigkeit für diesen Speicherort, weil der Server bereits ein `HttpOnly`-Cookie setzt.【F:login.html†L495-L508】

## Empfehlungen
- **Output-Encoding ergänzen:** Statt `innerHTML` sollten vertrauenswürdige Alternativen genutzt werden (z. B. `textContent`, dedizierte DOM-Knoten oder serverseitige Escaping-Funktionen), um Linktree-Daten sicher zu rendern.
- **URL-basierte Tokens deaktivieren:** Entfernen Sie die Query-Param-Unterstützung oder machen Sie sie nur für einmalige Einladungs-Workflows nutzbar. Tokens sollten ausschließlich über Header oder HttpOnly-Cookies transportiert werden.
- **LocalStorage-Verwendung eliminieren:** Entfernen Sie das Speichern des Tokens in `localStorage` und verlassen Sie sich stattdessen auf das bereits gesetzte Cookie. Falls ein JS-Zugriff zwingend nötig ist, sollten kurzfristige CSRF-geschützte Token oder Memory-Storage genutzt werden.

Durch die Priorisierung dieser Maßnahmen lassen sich die größten Risiken deutlich reduzieren, ohne die vorhandenen Sicherheitsstärken zu verlieren.
