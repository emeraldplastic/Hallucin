import bleach
import secrets
import hmac
from typing import Optional, List
from werkzeug.utils import secure_filename

class SecurityManager:
    """Central security manager for input sanitization and CSRF/API key validation."""

    ALLOWED_MODELS = {
        "local",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2"
    }

    def __init__(self, api_keys: List[str] = None):
        self.api_keys = set(api_keys) if api_keys else set()

    def sanitize(self, text: str, max_length: int = 2000000) -> str:
        """Strips HTML tags and truncates."""
        if not text:
            return ""
        # Strip all HTML tags
        clean_text = bleach.clean(text, tags=[], attributes={}, strip=True)
        # Truncate
        return clean_text[:max_length]

    def safe_model_name(self, name: str) -> str:
        """Validates model name against whitelist."""
        if not name or name not in self.ALLOWED_MODELS:
            return "local"
        return name

    def generate_csrf_token(self) -> str:
        """Generates a secure CSRF token."""
        return secrets.token_urlsafe(32)

    def validate_csrf(self, header_token: Optional[str], cookie_token: Optional[str]) -> bool:
        """Validates CSRF tokens using double-submit pattern."""
        if not header_token or not cookie_token:
            return False
        return hmac.compare_digest(header_token, cookie_token)

    def validate_api_key(self, key: Optional[str]) -> bool:
        """Validates an API key."""
        if not self.api_keys:
            return True # Open mode
        if not key:
            return False
        # If header format is "Bearer <key>"
        if key.startswith("Bearer "):
            key = key[7:]
        return key in self.api_keys

    def safe_filename(self, filename: str) -> str:
        """Sanitizes a filename for logging or download."""
        if not filename:
            return "unknown"
        safe_name = secure_filename(filename)
        return safe_name[:180] or "unknown"

_security_manager: Optional[SecurityManager] = None

def get_security_manager(api_keys: List[str] = None) -> SecurityManager:
    """Returns the singleton SecurityManager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager(api_keys)
    return _security_manager
