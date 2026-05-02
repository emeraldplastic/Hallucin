# Changelog

All notable changes to Hallucin Studio are documented here.

## [0.1.0] — 2026-05-02

### Added
- Offline-first hallucination detection with deterministic local hash encoder.
- Optional transformer embedding support via `sentence-transformers`.
- Hybrid semantic + lexical claim scoring with numeric and negation penalties.
- Flask web app with text input and file upload modes.
- Sliding window rate limiter with per-client tracking.
- Security headers (no-store, no-cache, X-Frame-Options, CSP controls).
- File upload validation (extension allowlist, binary rejection, size limits).
- In-memory upload processing — no files are written to disk.
- Vercel deployment configuration with `/health` endpoint.
- 29 unit tests covering detector, scorer, webapp, uploads, and deployment.
- CLI entrypoint via `hallucin` command.
- Python API: `from hallucination_detector import detect`.
- Configurable via environment variables (upload limits, rate limits, extensions).
- Responsive UI with animated score gauge, claim cards, and drag-and-drop uploads.
- `prefers-reduced-motion` media query support for accessibility.
