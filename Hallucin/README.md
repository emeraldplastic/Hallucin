<p align="center">
  <strong>Hallucin Studio</strong><br>
  <em>Offline-first grounding analysis for LLM responses</em>
</p>

<p align="center">
  <a href="#install">Install</a> •
  <a href="#run-the-web-app">Web App</a> •
  <a href="#python-api">Python API</a> •
  <a href="#rest-api">REST API</a> •
  <a href="#deploy-on-vercel">Deploy</a> •
  <a href="#test">Test</a> •
  <a href="#license">License</a>
</p>

---

## What Is Hallucin Studio?

Hallucin Studio is an **offline-first hallucination detection toolkit and web app**. It compares model responses against trusted source context, splits the response into atomic claims, scores each claim against the context using hybrid semantic and lexical matching, and returns a document-level grounding score with per-claim labels.

### Key Features

| Feature | Details |
|---|---|
| **Offline-first scoring** | Deterministic local hash encoder — no network, no GPU required |
| **Optional transformer embeddings** | Upgrade to `sentence-transformers` when local model weights are available |
| **Hybrid matching** | Blended semantic similarity + lexical coverage ratio |
| **Numeric & negation penalties** | Catches common factual number swaps and negation mismatches |
| **Web app** | Flask UI with text input and file upload modes |
| **Privacy by default** | In-memory processing — no uploads stored on disk |
| **Security hardened** | Cache-control, CSRF-safe headers, rate limiting, upload validation |
| **Vercel-ready** | One-command cloud deployment with health check endpoint |
| **Tested** | 29 unit tests covering detector, web API, uploads, and deployment |

---

## Install

**Minimal (offline scoring only):**

```bash
pip install -e .
```

**Full (with transformer embeddings and spaCy sentence splitting):**

```bash
pip install -e ".[full]"
python -m spacy download en_core_web_sm
```

### Requirements

- Python ≥ 3.9
- `numpy ≥ 1.24`
- `flask ≥ 2.3`

---

## Run The Web App

```bash
hallucin
```

Open **http://127.0.0.1:8000** in your browser.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HALLUCIN_HOST` | `127.0.0.1` | Bind address |
| `HALLUCIN_PORT` | `8000` | Bind port |
| `HALLUCIN_DEBUG` | `0` | Enable Flask debug mode (`1` to enable) |

---

## Python API

```python
from hallucination_detector import detect

result = detect(
    context="The Eiffel Tower is 330 metres tall and completed in 1889.",
    response="The Eiffel Tower is in Paris and 330 metres tall.",
)

print(result.score)           # 0.0 – 1.0
print(result.flagged_claims)  # List of unsupported claim strings
result.report()               # Print a formatted grounding report
```

### DetectionResult Properties

| Property | Type | Description |
|---|---|---|
| `score` | `float` | Document-level grounding score (0.0–1.0) |
| `claims` | `list[ClaimResult]` | All scored claims |
| `supported_claims` | `list[ClaimResult]` | Claims matching the context |
| `partial_claims` | `list[ClaimResult]` | Partially supported claims |
| `flagged_claims` | `list[str]` | Unsupported claim texts |
| `elapsed_ms` | `float` | Processing time in milliseconds |

---

## REST API

### `POST /api/analyze`

**JSON payload:**

```json
{
  "context": "trusted source text",
  "response": "model output to evaluate",
  "model_name": "local"
}
```

**Multipart file upload** is also supported with `context_file` and `response_file` fields.

**Response:**

```json
{
  "score": 0.85,
  "elapsed_ms": 12.34,
  "counts": { "supported": 3, "partial": 1, "unsupported": 0 },
  "claims": [
    {
      "claim": "The Eiffel Tower is 330 metres tall.",
      "label": "supported",
      "score": 0.91,
      "best_match": "It stands 330 metres tall..."
    }
  ]
}
```

### `POST /api/analyze/batch`

Scores multiple context/response pairs in a single request. Each item is scored independently and returned in the same order.

**JSON payload:**

```json
{
  "items": [
    { "context": "trusted source text", "response": "model output to evaluate" },
    { "context": "second source", "response": "second model output" }
  ]
}
```

**Response:**

```json
{
  "count": 2,
  "results": [
    { "score": 0.85, "elapsed_ms": 12.34, "counts": { "supported": 3, "partial": 1, "unsupported": 0 }, "claims": [] },
    { "score": 0.12, "elapsed_ms": 9.87, "counts": { "supported": 0, "partial": 1, "unsupported": 2 }, "claims": [] }
  ]
}
```

Invalid items are reported inline with an `"error"` field instead of failing the whole request. The batch size is limited by `HALLUCIN_MAX_BATCH_ITEMS` (default 50).

### `GET /health`

Returns `{"status": "ok"}` — use for uptime monitoring and load balancer checks.

---

## Configuration

All settings are optional and configured via environment variables.

| Variable | Default | Description |
|---|---|---|
| `HALLUCIN_MAX_UPLOAD_MB` | `128` | Maximum upload size in megabytes |
| `HALLUCIN_MAX_TEXT_CHARS` | `2000000` | Maximum text field length |
| `HALLUCIN_ALLOWED_UPLOAD_EXTENSIONS` | `.txt,.md,.json,.csv,.log,.html,.xml` | Comma-separated allowed file types |
| `HALLUCIN_RATE_LIMIT_REQUESTS` | `60` | Max requests per client per window |
| `HALLUCIN_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window duration |
| `HALLUCIN_ENABLE_PRIVACY_HEADERS` | `1` | Enable no-store/no-cache headers |
| `HALLUCIN_MAX_CONTEXT_CHUNKS` | `240` | Max context chunks for scoring |
| `HALLUCIN_MAX_BATCH_ITEMS` | `50` | Max context/response items per batch analysis request |
| `HALLUCIN_TOP_MATCH_CANDIDATES` | `3` | Top-N semantic candidates per claim |
| `HALLUCIN_USE_SPACY` | `0` | Use spaCy for sentence splitting |
| `HALLUCIN_API_KEYS` | (None) | Comma-separated allowed API keys |
| `HALLUCIN_REQUEST_TIMEOUT` | `30` | Request timeout in seconds |

---

## Deploy On Vercel

This project includes `vercel.json` and `api/index.py` for zero-config Vercel deployment.

```bash
vercel --prod
```

Health check: `GET /health`

> **Note:** Deploy from the `Hallucin` directory, or set the project root to `Hallucin` in the Vercel dashboard.

---

## Test

```bash
pytest -q
```

```
29 passed in 0.82s
```

Tests cover:
- Claim splitting (regex and spaCy fallback)
- Context chunking and deduplication
- Overall scoring aggregation
- End-to-end detection with grounded, hallucinated, and partial responses
- Web API (JSON payload, file uploads, validation, rate limiting)
- Security headers (enable/disable)
- Vercel entrypoint health check

---

## Project Structure

```
Hallucin/
├── hallucination_detector/
│   ├── __init__.py          # Public API exports
│   ├── __main__.py          # CLI entrypoint
│   ├── detector.py          # detect() orchestration
│   ├── scorer.py            # Claim scoring engine
│   ├── splitter.py          # Claim extraction
│   ├── webapp.py            # Flask application
│   ├── static/
│   │   ├── style.css        # UI styles
│   │   └── app.js           # Frontend logic
│   └── templates/
│       └── index.html       # Web interface
├── api/
│   └── index.py             # Vercel entrypoint
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── pyproject.toml           # Package metadata
├── requirements.txt         # Runtime dependencies
├── CHANGELOG.md             # Release history
├── CONTRIBUTING.md          # Contribution guide
├── LICENSE                  # MIT License
└── README.md
```

---

## Security

Hallucin Studio is designed with a strong security posture suitable for production deployments:

- **Input Sanitization** — All text inputs are sanitized using `bleach` to prevent XSS and injection attacks.
- **CSRF Protection** — API endpoints enforcing state-changing operations require a synchronized `X-CSRF-Token` header.
- **Authentication** — Optional API key enforcement via the `X-API-Key` or `Authorization` headers.
- **Security Headers** — Enforces strict `Content-Security-Policy`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, and `Cache-Control: no-store`.
- **DoS Protection** — Process execution timeouts, in-memory rate limiting (sliding window), and algorithmic limits on input dimensions and claims generation prevent resource exhaustion.
- **No File Persistence** — Uploaded files are processed strictly in memory and never written to disk.
- **Upload Validation** — Strict extension allowlists, binary content rejection, and payload size limits are enforced.
- **Structured Logging** — Comprehensive JSON-formatted audit trails for authentication failures, CSRF rejections, and rate limits.

---

## Credits

Special thanks to **Rume AI** for inspiring the production-grade security architecture, including the robust API key validation, CSRF implementations, configuration enforcement, and structured JSON observability patterns used in this project.

---

## License

[MIT](LICENSE) — see [LICENSE](LICENSE) for details.
