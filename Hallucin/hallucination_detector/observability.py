import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

class StructuredLogger:
    """Handles structured JSON logging."""
    
    def __init__(self, log_level: str = "INFO", log_format: str = "json"):
        self.logger = logging.getLogger("hallucin")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.propagate = False
        
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            self.logger.addHandler(handler)
            
        self.log_format = log_format.lower()

    def _truncate(self, value: Any, max_len: int = 1000) -> Any:
        """Truncates strings for safe logging."""
        if isinstance(value, str) and len(value) > max_len:
            return value[:max_len] + "... [truncated]"
        return value

    def log_event(self, event: str, level: str = "info", request_id: str = None, **kwargs):
        """Logs a structured event."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.upper(),
            "event": event,
        }
        
        if request_id:
            log_data["request_id"] = request_id
            
        for k, v in kwargs.items():
            if k == "error" and isinstance(v, Exception):
                log_data[k] = self._truncate(str(v))
            else:
                log_data[k] = self._truncate(v)
                
        if self.log_format == "json":
            self.logger.log(
                getattr(logging, level.upper(), logging.INFO),
                json.dumps(log_data)
            )
        else:
            # Text format
            parts = [f"[{log_data['timestamp']}]", f"[{log_data['level']}]"]
            if request_id:
                parts.append(f"[{request_id}]")
            parts.append(f"{event}:")
            
            for k, v in kwargs.items():
                parts.append(f"{k}={v}")
                
            self.logger.log(
                getattr(logging, level.upper(), logging.INFO),
                " ".join(parts)
            )

_logger: Optional[StructuredLogger] = None

def get_logger(log_level: str = "INFO", log_format: str = "json") -> StructuredLogger:
    global _logger
    if _logger is None:
        _logger = StructuredLogger(log_level, log_format)
    return _logger

def log_event(event: str, level: str = "info", request_id: str = None, **kwargs):
    get_logger().log_event(event, level, request_id, **kwargs)
