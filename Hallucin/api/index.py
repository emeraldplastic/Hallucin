from __future__ import annotations

from hallucination_detector.webapp import create_app

# Vercel expects a module-level ASGI/WSGI app object in the /api directory.
app = create_app()
