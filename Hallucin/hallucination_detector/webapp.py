from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import RequestEntityTooLarge

from .detector import detect

DEFAULT_MAX_UPLOAD_MB = int(os.getenv("HALLUCIN_MAX_UPLOAD_MB", "128"))
DEFAULT_MAX_TEXT_CHARS = int(os.getenv("HALLUCIN_MAX_TEXT_CHARS", "2000000"))


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        MAX_CONTENT_LENGTH=DEFAULT_MAX_UPLOAD_MB * 1024 * 1024,
        MAX_TEXT_CHARS=DEFAULT_MAX_TEXT_CHARS,
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            max_text_chars=app.config["MAX_TEXT_CHARS"],
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/api/analyze")
    def analyze():
        payload = request.get_json(silent=True) or request.form

        context = (payload.get("context", "") if payload else "").strip()
        response = (payload.get("response", "") if payload else "").strip()
        model_name = (payload.get("model_name") if payload else None) or "local"

        context_file = request.files.get("context_file")
        response_file = request.files.get("response_file")
        if context_file and context_file.filename:
            context = _read_upload_text(context_file, app.config["MAX_TEXT_CHARS"])
        if response_file and response_file.filename:
            response = _read_upload_text(response_file, app.config["MAX_TEXT_CHARS"])

        if not context or not response:
            return (
                jsonify(
                    {
                        "error": "Both context and response are required, either as text fields or uploaded files."
                    }
                ),
                400,
            )

        result = detect(context=context, response=response, model_name=model_name)
        return jsonify(
            {
                "score": result.score,
                "elapsed_ms": round(result.elapsed_ms, 2),
                "counts": {
                    "supported": len(result.supported_claims),
                    "partial": len(result.partial_claims),
                    "unsupported": len(result.flagged_claims),
                },
                "claims": [
                    {
                        "claim": claim.claim,
                        "label": claim.label,
                        "score": claim.score,
                        "best_match": claim.best_match,
                    }
                    for claim in result.claims
                ],
            }
        )

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge):
        return (
            jsonify(
                {
                    "error": (
                        "Upload too large. Current limit is "
                        f"{app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."
                    )
                }
            ),
            413,
        )

    @app.errorhandler(ValueError)
    def bad_upload(exc: ValueError):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):
        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.description}), exc.code
        app.logger.exception("Unhandled error while processing request")
        return jsonify({"error": "Internal server error"}), 500

    return app


def _read_upload_text(file_storage, max_text_chars: int) -> str:
    stream = file_storage.stream
    stream.seek(0)
    chunks: list[str] = []
    total_chars = 0

    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        text = chunk.decode("utf-8", errors="replace")
        total_chars += len(text)
        if total_chars > max_text_chars:
            raise ValueError(
                f"Uploaded text exceeds max length ({max_text_chars} chars)."
            )
        chunks.append(text)

    return "".join(chunks).strip()
