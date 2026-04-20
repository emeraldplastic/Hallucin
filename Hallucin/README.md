# Hallucin Studio

Hallucin Studio is an offline-first hallucination detection toolkit plus a web app.

It checks whether model responses are grounded in a trusted context, labels each claim,
and gives a document-level grounding score.

## Highlights

- Offline-first scoring with automatic fallback when transformer weights are unavailable
- Fast claim scoring via vectorized similarity matrix operations
- Numeric mismatch penalty to catch common fact swaps faster
- Modern Flask web app with drag-and-drop style uploads
- Large upload support (128 MB by default)
- API + UI + unit tests

## Install

```bash
pip install -e .
```

Optional: if you want richer sentence splitting via spaCy model:

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
- `HALLUCIN_HOST` (default: `127.0.0.1`)
- `HALLUCIN_PORT` (default: `8000`)
- `HALLUCIN_DEBUG` (`1` enables debug)

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
tests/
  test_detector.py
  test_webapp.py
```
