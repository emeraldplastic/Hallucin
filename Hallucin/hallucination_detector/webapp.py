from __future__ import annotations

import codecs
import math
import os
import time
import signal
import uuid
import json
from collections import deque
from threading import Lock
from typing import Any, Callable

from flask import Flask, jsonify, render_template, request, make_response
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.middleware.proxy_fix import ProxyFix

from .detector import detect
from .config import Config
from .security import get_security_manager
from .observability import log_event


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
    Config.validate()
    app = Flask(__name__, template_folder="templates", static_folder="static")
    
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=Config.TRUSTED_PROXIES)

    if Config.SECRET_KEY:
        app.secret_key = Config.SECRET_KEY

    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_UPLOAD_MB * 1024 * 1024
    app.config["JSON_SORT_KEYS"] = False
    
    app.config["RATE_LIMIT_REQUESTS"] = Config.RATE_LIMIT_REQUESTS
    app.config["RATE_LIMIT_WINDOW_SECONDS"] = Config.RATE_LIMIT_WINDOW_SECONDS
    app.config["MAX_TEXT_CHARS"] = Config.MAX_TEXT_CHARS
    app.config["REQUEST_TIMEOUT"] = Config.REQUEST_TIMEOUT
    app.config["ENABLE_PRIVACY_HEADERS"] = Config.ENABLE_PRIVACY_HEADERS
    app.config["FORCE_SECURE_COOKIES"] = Config.FORCE_SECURE_COOKIES
    app.config["UPLOAD_ALLOWED_EXTENSIONS"] = Config.get_allowed_extensions()
    
    if config:
        app.config.update(config)

    # Normalize allowed extensions
    if isinstance(app.config["UPLOAD_ALLOWED_EXTENSIONS"], str):
        app.config["UPLOAD_ALLOWED_EXTENSIONS"] = _parse_allowed_extensions(app.config["UPLOAD_ALLOWED_EXTENSIONS"])
    elif isinstance(app.config["UPLOAD_ALLOWED_EXTENSIONS"], (list, tuple, set)):
        app.config["UPLOAD_ALLOWED_EXTENSIONS"] = _parse_allowed_extensions(",".join(app.config["UPLOAD_ALLOWED_EXTENSIONS"]))

    # Ensure rate limiters use test configs
    rate_limiter = SlidingWindowRateLimiter(
        max_requests=app.config["RATE_LIMIT_REQUESTS"],
        window_seconds=app.config["RATE_LIMIT_WINDOW_SECONDS"],
    )
    
    sm = get_security_manager(Config.API_KEYS)

    @app.before_request
    def setup_request():
        request_id = request.headers.get("X-Request-ID") or request.headers.get("X-Vercel-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        request.request_id = request_id
        
    @app.before_request
    def security_checks():
        # Rate Limiting
        # In tests, if X-Forwarded-For is set but ProxyFix didn't handle it, use it
        remote_addr = request.remote_addr or "unknown"
        if app.config.get("TESTING") and request.headers.get("X-Forwarded-For"):
            remote_addr = request.headers.get("X-Forwarded-For").split(",")[0].strip()
            
        allowed, retry_after = rate_limiter.check(remote_addr)
        if not allowed:
            log_event("rate_limit_exceeded", "warning", request.request_id, ip=remote_addr, path=request.path)
            return jsonify({
                "error": "Rate limit exceeded. Try again later.",
                "retry_after_seconds": retry_after,
                "limit": app.config["RATE_LIMIT_REQUESTS"],
                "window_seconds": app.config["RATE_LIMIT_WINDOW_SECONDS"]
            }), 429, {"Retry-After": str(retry_after)}

        # API Key / CSRF
        if request.path.startswith("/api/") and not app.config.get("TESTING"):
            is_api_client = False
            if sm.api_keys:
                auth_header = request.headers.get("Authorization")
                api_key_header = request.headers.get("X-API-Key")
                if sm.validate_api_key(auth_header) or sm.validate_api_key(api_key_header):
                    is_api_client = True
                else:
                    log_event("auth_failure", "warning", request.request_id, ip=request.remote_addr)
                    return jsonify({"error": "Unauthorized"}), 401

            if request.method in ["POST", "PUT", "DELETE", "PATCH"] and not is_api_client:
                csrf_header = request.headers.get("X-CSRF-Token")
                csrf_cookie = request.cookies.get("csrf_token")
                if not sm.validate_csrf(csrf_header, csrf_cookie):
                    log_event("csrf_failure", "warning", request.request_id, ip=request.remote_addr)
                    return jsonify({"error": "CSRF validation failed"}), 403

    @app.after_request
    def after_request_security(response):
        response.headers["X-Request-ID"] = request.request_id
        
        cors_origins = ",".join(Config.CORS_ORIGINS)
        response.headers["Access-Control-Allow-Origin"] = cors_origins
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-CSRF-Token, Authorization, X-API-Key"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"

        if str(app.config.get("ENABLE_PRIVACY_HEADERS", "1")) not in ("0", "false", "False"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
            
            csp = (
                "default-src 'self'; script-src 'self'; style-src 'self'; "
                "img-src 'self' data:; object-src 'none'; connect-src 'self'; "
                "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
            )
            response.headers["Content-Security-Policy"] = csp
            
            if app.config.get("FORCE_SECURE_COOKIES"):
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        log_event("request_completed", "info", request.request_id, 
            method=request.method, path=request.path, status=response.status_code)
            
        return response

    @app.get("/")
    def index():
        allowed_exts = app.config.get("UPLOAD_ALLOWED_EXTENSIONS", [])
        response = make_response(render_template(
            "index.html",
            max_upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
            max_text_chars=app.config["MAX_TEXT_CHARS"],
            allowed_upload_accept=",".join(allowed_exts),
            allowed_upload_types=", ".join(allowed_exts),
        ))
        
        if not request.cookies.get("csrf_token"):
            response.set_cookie(
                "csrf_token", 
                sm.generate_csrf_token(),
                secure=app.config.get("FORCE_SECURE_COOKIES"),
                httponly=False,
                samesite="Strict"
            )
        return response

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/api/stats")
    def stats():
        """Return usage statistics for monitoring and analytics."""
        return jsonify({
            "rate_limit_stats": {
                "max_requests": app.config["RATE_LIMIT_REQUESTS"],
                "window_seconds": app.config["RATE_LIMIT_WINDOW_SECONDS"],
                "active_clients": len(rate_limiter._requests)
            },
            "config": {
                "max_upload_mb": app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
                "max_text_chars": app.config["MAX_TEXT_CHARS"],
                "allowed_extensions": app.config["UPLOAD_ALLOWED_EXTENSIONS"],
                "request_timeout": app.config["REQUEST_TIMEOUT"],
                "privacy_headers_enabled": str(app.config.get("ENABLE_PRIVACY_HEADERS", "1")) not in ("0", "false", "False")
            }
        })

    @app.route("/api/analyze", methods=["OPTIONS"])
    def analyze_options():
        return jsonify({}), 200

    @app.post("/api/analyze")
    def analyze():
        def handler(signum, frame):
            raise TimeoutError("Analysis took too long")
            
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(app.config["REQUEST_TIMEOUT"])

        try:
            payload = request.get_json(silent=True) or request.form

            context = (payload.get("context", "") if payload else "").strip()
            response = (payload.get("response", "") if payload else "").strip()
            model_name = (payload.get("model_name") if payload else None) or "local"

            context_file = request.files.get("context_file")
            response_file = request.files.get("response_file")
            
            allowed_exts = tuple(app.config.get("UPLOAD_ALLOWED_EXTENSIONS", []))
            
            if context_file and context_file.filename:
                _validate_upload_file(context_file, allowed_exts)
                context = _read_upload_text(context_file, app.config["MAX_TEXT_CHARS"])
            if response_file and response_file.filename:
                _validate_upload_file(response_file, allowed_exts)
                response = _read_upload_text(response_file, app.config["MAX_TEXT_CHARS"])

            _validate_text_length("Context", context, app.config["MAX_TEXT_CHARS"])
            _validate_text_length("Response", response, app.config["MAX_TEXT_CHARS"])

            if not context or not response:
                return jsonify({"error": "Both context and response are required"}), 400

            result = detect(context=context, response=response, model_name=model_name)
            
            return jsonify({
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
            })
        except TimeoutError:
            log_event("analysis_timeout", "error", getattr(request, "request_id", None))
            return jsonify({"error": "Analysis timeout exceeded"}), 504
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_: RequestEntityTooLarge):
        log_event("upload_too_large", "warning", getattr(request, "request_id", None))
        return jsonify({"error": f"Upload too large. Current limit is {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB."}), 413

    @app.errorhandler(ValueError)
    def bad_upload(exc: ValueError):
        log_event("bad_upload", "warning", getattr(request, "request_id", None), error=str(exc))
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(Exception)
    def unhandled(exc: Exception):
        if isinstance(exc, HTTPException):
            return jsonify({"error": exc.description}), exc.code
        log_event("unhandled_error", "error", getattr(request, "request_id", None), error=exc)
        return jsonify({
            "error": "Internal server error",
            "request_id": getattr(request, "request_id", None)
        }), 500

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


def _validate_text_length(label: str, value: str, max_text_chars: int) -> None:
    if value and len(value) > max_text_chars:
        raise ValueError(f"{label} exceeds max length ({max_text_chars} chars).")
