"""Security Audit Logging & Redaction Engine."""

from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional

from trading_ai.schemas.security import SecurityAuditEntry
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)

# Patterns for redacting sensitive secrets from audit logs and errors
SENSITIVE_PATTERNS = [
    (re.compile(r"(?i)(api[_-]?key|secret|token|password|bearer|auth|private[_-]?key)\s*[:=]\s*['\"]?([a-zA-Z0-9_\-\.]{6,})['\"]?"), r"\1=[REDACTED]"),
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)(4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})"), "[REDACTED_CARD]"),
]


def redact_sensitive_text(text: str) -> str:
    """Deterministic redaction of sensitive credentials, keys, and tokens."""
    if not text:
        return text
    result = text
    for pattern, repl in SENSITIVE_PATTERNS:
        result = pattern.sub(repl, result)
    return result


def redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redact dictionary values containing sensitive keys."""
    redacted = {}
    sensitive_keys = {"token", "secret", "key", "password", "auth", "private_key", "bearer", "api_key", "shared_secret"}
    for k, v in data.items():
        if any(s in k.lower() for s in sensitive_keys):
            redacted[k] = "[REDACTED]"
        elif isinstance(v, dict):
            redacted[k] = redact_dict(v)
        elif isinstance(v, str):
            redacted[k] = redact_sensitive_text(v)
        elif isinstance(v, list):
            redacted[k] = [redact_dict(item) if isinstance(item, dict) else (redact_sensitive_text(item) if isinstance(item, str) else item) for item in v]
        else:
            redacted[k] = v
    return redacted


class SecurityAuditLogger:
    """
    Immutable, privacy-compliant audit log manager for all security-relevant events.
    """

    def __init__(self, repository: Optional[Repository] = None):
        self._repo = repository
        self._in_memory_logs: List[SecurityAuditEntry] = []

    def log(
        self,
        action: str,
        decision: str,
        requesting_component: str,
        capability: Optional[str] = None,
        is_user_authorized: bool = False,
        authorized_by: Optional[str] = None,
        device_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityAuditEntry:
        """Record an auditable security entry with deterministic secret redaction."""
        safe_details = redact_dict(details or {})
        entry = SecurityAuditEntry(
            action=action,
            decision=decision,
            requesting_component=requesting_component,
            capability=capability,
            is_user_authorized=is_user_authorized,
            authorized_by=authorized_by,
            device_id=device_id,
            details=safe_details,
            timestamp=datetime.now(timezone.utc),
        )

        self._in_memory_logs.append(entry)
        if len(self._in_memory_logs) > 1000:
            self._in_memory_logs.pop(0)

        if self._repo and hasattr(self._repo, "save_security_audit_entry"):
            try:
                self._repo.save_security_audit_entry(entry)
            except Exception as e:
                logger.error(f"Failed to persist security audit entry: {e}")

        logger.info(
            f"SECURITY_AUDIT: action={action} decision={decision} "
            f"component={requesting_component} capability={capability} auth={is_user_authorized}"
        )
        return entry

    def get_recent_entries(self, limit: int = 50) -> List[SecurityAuditEntry]:
        """Retrieve latest audit entries."""
        if self._repo and hasattr(self._repo, "get_recent_security_audit_entries"):
            try:
                persisted = self._repo.get_recent_security_audit_entries(limit=limit)
                if persisted:
                    return persisted
            except Exception as e:
                logger.debug(f"Could not load persisted security audit entries: {e}")
        return self._in_memory_logs[-limit:]
