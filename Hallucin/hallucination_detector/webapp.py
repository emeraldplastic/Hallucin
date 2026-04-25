from __future__ import annotations

import codecs
import math
import os
import time
from collections import deque
from threading import Lock
from typing import Any, Callable

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import RequestEntityTooLarge

from .detector import detect

def _parse_allowed_extensions(raw_extensions: str) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in raw_extensions.split(","):
        extension = value.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.append(extension)

    unique = tuple(sorted(set(normalized)))
    if unique:
        return unique
    return (".txt",)


def _normalize_allowed_extensions_config(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return _parse_allowed_extensions(value)
    if isinstance(value, (list, tuple, set)):
        return _parse_allowed_extensions(",".join(str(item) for item in value))
    return _parse_allowed_extensions(str(value))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


DEFAULT_MAX_UPLOAD_MB = int(os.getenv("HALLUCIN_MAX_UPLOAD_MB", "128"))
DEFAULT_MAX_TEXT_CHARS = int(os.getenv("HALLUCIN_MAX_TEXT_CHARS", "2000000"))
DEFAULT_RATE_LIMIT_REQUESTS = int(os.getenv("HALLUCIN_RATE_LIMIT_REQUESTS", "60"))
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = int(
    os.getenv("HALLUCIN_RATE_LIMIT_WINDOW_SECONDS", "60")
)
DEFAULT_UPLOAD_ALLOWED_EXTENSIONS = _parse_allowed_extensions(
    os.getenv(
        "HALLUCIN_ALLOWED_UPLOAD_EXTENSIONS",
        ".txt,.md,.json,.csv,.log,.html,.xml",
    )
)
DEFAULT_ENABLE_PRIVACY_HEADERS = os.getenv("HALLUCIN_ENABLE_PRIVACY_HEADERS", "1") == "1"


class SlidingWindowRateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.max_requests = max(0, int(max_requests))
        self.window_seconds = max(0, int(window_seconds))
        self._time_fn = time_fn or time.time
        self._requests: dict[str, deque[float]] = {}
        self._lock = Lock()

    def check(self, client_key: str) -> tuple[bool, int]:
        if self.max_requests == 0 or self.window_seconds == 0:
            return True, 0

        now = self._time_fn()
        cutoff = now - self.window_seconds

        with self._lock:
            client_window = self._requests.get(client_key)
            if client_window is None:
                client_window = deque()
                self._requests[client_key] = client_window

            while client_window and client_window[0] <= cutoff:
                client_window.popleft()

            if len(client_window) >= self.max_requests:
                retry_after = max(
                    1, int(math.ceil(client_window[0] + self.window_seconds - now))
                )
                return False, retry_after

            client_window.append(now)

        return True, 0


def create_app(config: dict[str, Any] | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(
        MAX_CONTENT_LENGTH=DEFAULT_MAX_UPLOAD_MB * 1024 * 1024,
        MAX_TEXT_CHARS=DEFAULT_MAX_TEXT_CHARS,
        RATE_LIMIT_REQUESTS=DEFAULT_RATE_LIMIT_REQUESTS,
        RATE_LIMIT_WINDOW_SECONDS=DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
        UPLOAD_ALLOWED_EXTENSIONS=DEFAULT_UPLOAD_ALLOWED_EXTENSIONS,
        ENABLE_PRIVACY_HEADERS=DEFAULT_ENABLE_PRIVACY_HEADERS,
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)

    app.config["RATE_LIMIT_REQUESTS"] = max(0, int(app.config["RATE_LIMIT_REQUESTS"]))
    app.config["RATE_LIMIT_WINDOW_SECONDS"] = max(
        0, int(app.config["RATE_LIMIT_WINDOW_SECONDS"])
    )
    app.config["UPLOAD_ALLOWED_EXTENSIONS"] = _normalize_allowed_extensions_config(
        app.config["UPLOAD_ALLOWED_EXTENSIONS"]
    )
    app.config["ENABLE_PRIVACY_HEADERS"] = _as_bool(
        app.config["ENABLE_PRIVACY_HEADERS"]
    )

    rate_limiter = SlidingWindowRateLimiter(
        max_requests=app.config["RATE_LIMIT_REQUESTS"],
        window_seconds=app.config["RATE_LIMIT_WINDOW_SECONDS"],
    )
    app.extensions["rate_limiter"] = rate_limiter

    @app.before_request
    def enforce_rate_limit():
        if request.method != "POST" or request.endpoint != "analyze":
            return None

        allowed, retry_after = rate_limiter.check(_get_client_identifier())
        if allowed:
            return None

        return (
            jsonify(
                {
                    "error": "Rate limit exceeded. Try again later.",
                    "retry_after_seconds": retry_after,
                    "limit": app.config["RATE_LIMIT_REQUESTS"],
                    "window_seconds": app.config["RATE_LIMIT_WINDOW_SECONDS"],
                }
            ),
            429,
            {"Retry-After": str(retry_after)},
        )

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            max_text_chars=app.config["MAX_TEXT_CHARS"],
            allowed_upload_accept=",".join(app.config["UPLOAD_ALLOWED_EXTENSIONS"]),
            allowed_upload_types=", ".join(app.config["UPLOAD_ALLOWED_EXTENSIONS"]),
        )

    @app.after_request
    def set_privacy_headers(response):
        if not app.config["ENABLE_PRIVACY_HEADERS"]:
            return response

        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response

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
            _validate_upload_file(context_file, app.config["UPLOAD_ALLOWED_EXTENSIONS"])
            context = _read_upload_text(context_file, app.config["MAX_TEXT_CHARS"])
        if response_file and response_file.filename:
            _validate_upload_file(response_file, app.config["UPLOAD_ALLOWED_EXTENSIONS"])
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


def _validate_upload_file(file_storage, allowed_extensions: tuple[str, ...]) -> None:
    filename = (file_storage.filename or "").strip()
    if not filename:
        raise ValueError("Uploaded file is missing a filename.")

    extension = os.path.splitext(filename)[1].lower()
    if extension not in allowed_extensions:
        allowed = ", ".join(allowed_extensions)
        shown_extension = extension or "<none>"
        raise ValueError(
            f"Unsupported file type '{shown_extension}'. Allowed types: {allowed}."
        )


def _read_upload_text(file_storage, max_text_chars: int) -> str:
    stream = file_storage.stream
    stream.seek(0)
    chunks: list[str] = []
    total_chars = 0
    decoder = codecs.getincrementaldecoder("utf-8")()

    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        if b"\x00" in chunk:
            raise ValueError(
                "Uploaded files must be UTF-8 text and cannot contain binary content."
            )
        try:
            text = decoder.decode(chunk)
        except UnicodeDecodeError as exc:
            raise ValueError("Uploaded files must be UTF-8 text.") from exc
        total_chars += len(text)
        if total_chars > max_text_chars:
            raise ValueError(
                f"Uploaded text exceeds max length ({max_text_chars} chars)."
            )
        chunks.append(text)

    try:
        final_text = decoder.decode(b"", final=True)
    except UnicodeDecodeError as exc:
        raise ValueError("Uploaded files must be UTF-8 text.") from exc

    if final_text:
        total_chars += len(final_text)
        if total_chars > max_text_chars:
            raise ValueError(
                f"Uploaded text exceeds max length ({max_text_chars} chars)."
            )
        chunks.append(final_text)

    return "".join(chunks).strip()


def _get_client_identifier() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        first_ip = forwarded_for.split(",")[0].strip()
        if first_ip:
            return first_ip
    return request.remote_addr or "unknown"
