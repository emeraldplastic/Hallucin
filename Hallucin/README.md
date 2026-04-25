# Hallucin Studio

Hallucin Studio is an offline-first hallucination detection toolkit plus a web app.

It checks whether model responses are grounded in a trusted context, labels each claim,
and gives a document-level grounding score.

## Highlights

- Offline-first scoring with automatic fallback when transformer weights are unavailable
- Fast claim scoring via vectorized similarity matrix operations
- Hybrid semantic + lexical matching for stronger grounding accuracy
- Numeric and negation mismatch penalties to catch common fact swaps faster
- Modern Flask web app with drag-and-drop style uploads
- Privacy defaults (no-cache responses + in-memory file processing only)
- Context chunk capping for faster analysis on very large documents
- Large upload support (128 MB by default)
- API + UI + unit tests

## Install

```bash
pip install -e .
```

Optional full ML dependencies (transformer + spaCy):

```bash
pip install -e ".[full]"
```

Optional: if you want richer sentence splitting via spaCy model locally:

```bash
python -m spacy download en_core_web_sm
```

## Run The Web App

```bash
hallucin
```

Open `http://127.0.0.1:8000`.

Environment knobs:

- `HALLUCIN_MAX_UPLOAD_MB` (default: `128`)
- `HALLUCIN_MAX_TEXT_CHARS` (default: `2000000`)
- `HALLUCIN_ALLOWED_UPLOAD_EXTENSIONS` (default: `.txt,.md,.json,.csv,.log,.html,.xml`)
- `HALLUCIN_RATE_LIMIT_REQUESTS` (default: `60`, set `0` to disable)
- `HALLUCIN_RATE_LIMIT_WINDOW_SECONDS` (default: `60`)
- `HALLUCIN_ENABLE_PRIVACY_HEADERS` (default: `1`)
- `HALLUCIN_MAX_CONTEXT_CHUNKS` (default: `240`, lower is faster)
- `HALLUCIN_TOP_MATCH_CANDIDATES` (default: `3`)
- `HALLUCIN_FULL_CONTEXT_APPEND_LIMIT` (default: `24000`, chars)
- `HALLUCIN_HOST` (default: `127.0.0.1`)
- `HALLUCIN_PORT` (default: `8000`)
- `HALLUCIN_DEBUG` (`1` enables debug)

Uploads are validated server-side and must be UTF-8 text files with an allowed extension. Uploaded files are processed in-memory and not written to disk by the app.

## Python Usage

```python
from hallucination_detector import detect

result = detect(
    context="The Eiffel Tower is 330 metres tall and completed in 1889.",
    response="The Eiffel Tower is in Paris and 330 metres tall.",
)

print(result.score)
print(result.flagged_claims)
result.report()
```

To force the local offline model:

```python
result = detect(context=my_context, response=my_response, model_name="local")
```

## API Usage

### JSON

`POST /api/analyze`

```json
{
  "context": "trusted source text",
  "response": "model output to evaluate",
  "model_name": "local"
}
```

### Multipart Uploads

`POST /api/analyze`

- `context_file`: text file for source context
- `response_file`: text file for model response

## Deploy On Vercel

This repository is now configured for Vercel deployment (`vercel.json` + `api/index.py`).

```bash
vercel
```

Optional production deploy:

```bash
vercel --prod
```

If you previously deployed this app on Render, disable or delete that Render service in the Render dashboard so only the Vercel deployment stays active.

## Run Tests

```bash
pytest -q
```

## Project Layout

```text
hallucination_detector/
  __init__.py
  __main__.py
  detector.py
  scorer.py
  splitter.py
  webapp.py
  templates/
    index.html
  static/
    style.css
    app.js
api/
  index.py
tests/
  test_detector.py
  test_webapp.py
vercel.json
requirements.txt
```
