import os
from typing import Dict, Any

class Config:
    """Central configuration with production validation."""

    # Environment
    ENV = os.environ.get("HALLUCIN_ENV", "development").lower()
    DEBUG = os.environ.get("HALLUCIN_DEBUG", "0") == "1"

    # Server binding
    HOST = os.environ.get("HALLUCIN_HOST", "127.0.0.1")
    PORT = int(os.environ.get("HALLUCIN_PORT", "8000"))

    # Security secrets
    SECRET_KEY = os.environ.get("HALLUCIN_SECRET_KEY", "")
    API_KEYS = [k.strip() for k in os.environ.get("HALLUCIN_API_KEYS", "").split(",") if k.strip()]

    # Security settings
    CORS_ORIGINS = [o.strip() for o in os.environ.get("HALLUCIN_CORS_ORIGINS", "*").split(",") if o.strip()]
    TRUSTED_PROXIES = int(os.environ.get("HALLUCIN_TRUSTED_PROXIES", "1"))
    FORCE_SECURE_COOKIES = os.environ.get("HALLUCIN_FORCE_SECURE_COOKIES", "0") == "1"
    ENABLE_PRIVACY_HEADERS = os.environ.get("HALLUCIN_ENABLE_PRIVACY_HEADERS", "1") == "1"

    # Rate limiting
    RATE_LIMIT_REQUESTS = int(os.environ.get("HALLUCIN_RATE_LIMIT_REQUESTS", "60"))
    RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("HALLUCIN_RATE_LIMIT_WINDOW_SECONDS", "60"))
    RATE_LIMIT_STORAGE = os.environ.get("HALLUCIN_RATE_LIMIT_STORAGE", "memory://")

    # Application limits
    MAX_UPLOAD_MB = int(os.environ.get("HALLUCIN_MAX_UPLOAD_MB", "128"))
    MAX_TEXT_CHARS = int(os.environ.get("HALLUCIN_MAX_TEXT_CHARS", "2000000"))
    MAX_CONTEXT_CHUNKS = int(os.environ.get("HALLUCIN_MAX_CONTEXT_CHUNKS", "240"))
    MAX_CLAIMS = int(os.environ.get("HALLUCIN_MAX_CLAIMS", "200"))
    TOP_MATCH_CANDIDATES = int(os.environ.get("HALLUCIN_TOP_MATCH_CANDIDATES", "3"))
    REQUEST_TIMEOUT = int(os.environ.get("HALLUCIN_REQUEST_TIMEOUT", "30"))
    FULL_CONTEXT_APPEND_LIMIT = int(os.environ.get("HALLUCIN_FULL_CONTEXT_APPEND_LIMIT", "20000"))
    ALLOWED_UPLOAD_EXTENSIONS = os.environ.get("HALLUCIN_ALLOWED_UPLOAD_EXTENSIONS", ".txt,.md,.json,.csv,.log,.html,.xml")
    
    # Observability
    LOG_LEVEL = os.environ.get("HALLUCIN_LOG_LEVEL", "INFO").upper()
    LOG_FORMAT = os.environ.get("HALLUCIN_LOG_FORMAT", "json").lower()
    
    # Internal options
    USE_SPACY = os.environ.get("HALLUCIN_USE_SPACY", "0") == "1"

    @classmethod
    def validate(cls):
        """Validates configuration for production readiness."""
        if cls.ENV != "production":
            return

        missing = []
        weak = []

        if not cls.SECRET_KEY:
            missing.append("HALLUCIN_SECRET_KEY")
        elif len(cls.SECRET_KEY) < 32:
            weak.append("HALLUCIN_SECRET_KEY (must be at least 32 characters)")
        else:
            weak_placeholders = ["dev-key", "change-me", "secret", "placeholder", "your-secret-key"]
            for w in weak_placeholders:
                if w in cls.SECRET_KEY.lower():
                    weak.append("HALLUCIN_SECRET_KEY (contains weak placeholder)")
                    break

        if cls.DEBUG:
            weak.append("HALLUCIN_DEBUG (must be 0 in production)")

        if missing or weak:
            msg = "Production configuration validation failed.\n"
            if missing:
                msg += f"Missing required secrets: {', '.join(missing)}\n"
            if weak:
                msg += f"Weak or invalid config: {', '.join(weak)}\n"
            raise RuntimeError(msg)

    @classmethod
    def get_allowed_extensions(cls):
        return [ext.strip() for ext in cls.ALLOWED_UPLOAD_EXTENSIONS.split(",") if ext.strip()]
