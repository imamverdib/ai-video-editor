"""UI wiring: serves the built React single-page app (``frontend/dist``).

During development the SPA runs on the Vite dev server (port 5173) and talks to
this API through the CORS allowlist. In production ``npm run build`` emits
``frontend/dist`` which is mounted here, so the whole app ships as one process.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import settings

log = logging.getLogger(__name__)

_SETUP_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Text-Based Video Editor - build the frontend</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; min-height:100vh; display:grid; place-items:center;
         background:#0b0d12; color:#e8ecf4;
         font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  main { max-width:44rem; padding:2.5rem; }
  h1 { font-size:1.6rem; margin:0 0 .5rem; }
  p  { color:#9aa6bd; }
  pre { background:#151924; border:1px solid #222836; border-radius:.6rem;
         padding:1rem 1.1rem; overflow-x:auto; }
  code { color:#ffe94d; }
  a { color:#5eead4; }
</style></head>
<body><main>
  <h1>Backend is running ✅</h1>
  <p>The React UI has not been built yet. Either start the dev server:</p>
  <pre><code>cd frontend
npm install
npm run dev</code></pre>
  <p>…then open <a href="http://localhost:5173">http://localhost:5173</a>, or build
  it once for single-process serving:</p>
  <pre><code>cd frontend && npm install && npm run build</code></pre>
  <p>API docs: <a href="/docs">/docs</a> &middot;
     Capabilities: <a href="/api/capabilities">/api/capabilities</a></p>
</main></body></html>
"""


def mount_frontend(app: FastAPI) -> None:
    """Mount the SPA at ``/`` (must be called after all API routes)."""
    dist = settings.frontend_dist
    index = dist / "index.html"
    if index.is_file():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="spa")
        log.info("Serving built frontend from %s", dist)
        return

    log.warning("frontend/dist not found - serving the setup page at /")

    @app.get("/", include_in_schema=False, response_class=HTMLResponse)
    async def setup_page() -> HTMLResponse:  # pragma: no cover - trivial
        return HTMLResponse(_SETUP_PAGE)
