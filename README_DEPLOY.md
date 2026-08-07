# Seeking Alpha Legal — Vercel deployment

This package makes your existing build deploy to Vercel unchanged. Static frontends
serve from `/public`; the FastAPI intake backend runs as a Python serverless
function under `/api`.

## What's inside
- `public/` — dashboard (index.html), intake console, tech framework, product brief,
  the Charlie teaser + SG self-filing pages, and the 43 SICC justice portraits.
- `api/index.py` — Vercel entry; mounts your `intake_service.py` FastAPI app at `/api`.
- `api/intake_service.py` — your backend, unchanged. Endpoints become `/api/health`
  and `/api/extract`.
- `vercel.json`, `requirements.txt`.

## Deploy (2 minutes)
1. Install the CLI: `npm i -g vercel`
2. From this folder: `vercel` (first run links/creates the project), then `vercel --prod`.
   - Or push this folder to a GitHub repo and "Import Project" in the Vercel dashboard.
3. In Vercel → Project → Settings → Environment Variables, add:
   - `ANTHROPIC_API_KEY` (required for live model extraction)
   - `OPENAI_API_KEY` (optional — only if you use audio/ASR)
4. The intake console already points at `/api/extract` (same origin), so live mode
   works as soon as the key is set. No CORS config needed.

## Known limits — read before you rely on it
- **The `/extract` endpoint is stateless** (it redacts, calls Claude, returns a record,
  and uses `/tmp` for transient files). That's a clean fit for Vercel serverless. Good.
- **SQLite write-state does NOT persist on Vercel.** Vercel functions have an ephemeral,
  read-only-ish filesystem, so anything that WRITES to `intake.db`/`sicc.db` will not
  survive between requests. The bundled `sicc.db` (read-only queries) can ship as a
  static asset, but for anything that writes, use a hosted DB. Lowest-friction option:
  **Turso / libSQL** (SQLite-compatible, minimal code change) or **Vercel Postgres**.
- **The heavier agents** (`judicial_agent.py`, `framing.py`, `intake_agent.py`) are CLI/
  batch tools, not part of the web service. If you want them web-exposed with their
  databases, host the backend on a persistent platform (**Render / Railway / Fly.io**)
  instead of Vercel, and keep only the static frontends on Vercel.

## Recommended split (cleanest)
- **Frontends → Vercel** (this package, works today).
- **Stateful backend + DBs → Render or Railway** (native FastAPI + SQLite, persistent disk).
- Point the console's endpoint field at that backend URL instead of `/api/extract`.

This package gets the frontends + stateless intake live on Vercel immediately; the
persistent-DB decision is the only thing standing between this and full production.
