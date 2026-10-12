"""Private-Only Authentication, Session Management & Access Control.

Ensures strict access control for personal-only deployment:
- No public self-registration.
- Constant-time token verification (secrets.compare_digest).
- Signed HMAC session tokens with expiration and replay prevention.
- Rate limiting on failed authentication attempts.
- Secure session revocation and token rotation.
- Audit logging via SecurityAuditLogger with credentials redacted.
"""

from datetime import datetime, timezone, timedelta
import hashlib
import hmac
import logging
import secrets
import time
from typing import Dict, Optional, Set, Tuple

from trading_ai.security.audit import SecurityAuditLogger

logger = logging.getLogger(__name__)

# Security constraints
MAX_FAILED_ATTEMPTS = 5
RATE_LIMIT_WINDOW_SECONDS = 60.0
SESSION_TTL_HOURS = 168.0  # 7 days


class PrivateAuthManager:
    """Manages personal authentication, rate-limiting, and signed sessions."""

    def __init__(
        self,
        auth_token: Optional[str] = None,
        session_secret: Optional[str] = None,
        auth_enabled: bool = False,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ):
        self._auth_enabled = auth_enabled
        self._auth_token = auth_token or ""
        self._session_secret = session_secret or secrets.token_hex(32)
        self._audit = audit_logger or SecurityAuditLogger()

        # In-memory session tracking & revocation: session_id -> expires_at_timestamp
        self._active_sessions: Dict[str, float] = {}
        self._revoked_sessions: Set[str] = set()

        # Rate limiting: ip_address -> list of timestamp failures
        self._failed_attempts: Dict[str, list[float]] = {}

    @property
    def is_auth_enabled(self) -> bool:
        return self._auth_enabled

    def set_auth_enabled(self, enabled: bool) -> None:
        self._auth_enabled = enabled

    def set_auth_token(self, token: str) -> None:
        self._auth_token = token

    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if an IP has exceeded failed login thresholds."""
        now = time.time()
        attempts = [t for t in self._failed_attempts.get(client_ip, []) if now - t < RATE_LIMIT_WINDOW_SECONDS]
        self._failed_attempts[client_ip] = attempts
        return len(attempts) >= MAX_FAILED_ATTEMPTS

    def record_failed_attempt(self, client_ip: str) -> None:
        """Record a failed login attempt for rate-limiting."""
        now = time.time()
        if client_ip not in self._failed_attempts:
            self._failed_attempts[client_ip] = []
        self._failed_attempts[client_ip].append(now)
        self._audit.log(
            action="LOGIN_FAILED",
            decision="DENIED",
            requesting_component="PrivateAuthManager",
            details={"client_ip": client_ip, "recent_failures": len(self._failed_attempts[client_ip])},
            is_user_authorized=False,
        )

    def verify_token(self, token: str, client_ip: str = "127.0.0.1") -> bool:
        """Constant-time verification of personal access token."""
        if not self._auth_enabled:
            return True

        if not self._auth_token:
            logger.warning("AUTH_ENABLED is True but AUTH_TOKEN is not configured.")
            return False

        if self.is_rate_limited(client_ip):
            logger.warning("Rate limit exceeded for client IP %s", client_ip)
            return False

        is_valid = secrets.compare_digest(token.strip(), self._auth_token.strip())
        if is_valid:
            # Clear rate limit history on successful login
            self._failed_attempts.pop(client_ip, None)
            self._audit.log(
                action="LOGIN_SUCCESS",
                decision="ALLOWED",
                requesting_component="PrivateAuthManager",
                details={"client_ip": client_ip},
                is_user_authorized=True,
                authorized_by="owner",
            )
        else:
            self.record_failed_attempt(client_ip)
        return is_valid

    def create_session(self, client_ip: str = "127.0.0.1") -> str:
        """Create a cryptographically signed session token."""
        expires_at = time.time() + (SESSION_TTL_HOURS * 3600)
        nonce = secrets.token_hex(16)
        payload = f"{nonce}:{expires_at}"
        signature = hmac.new(
            self._session_secret.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        session_id = f"{payload}:{signature}"
        self._active_sessions[session_id] = expires_at
        return session_id

    def verify_session(self, session_id: Optional[str]) -> bool:
        """Validate HMAC signature, expiration, and revocation status."""
        if not self._auth_enabled:
            return True

        if not session_id or session_id in self._revoked_sessions:
            return False

        try:
            parts = session_id.split(":")
            if len(parts) != 3:
                return False
            nonce, exp_str, sig = parts
            expires_at = float(exp_str)
            if time.time() > expires_at:
                self._active_sessions.pop(session_id, None)
                return False

            payload = f"{nonce}:{exp_str}"
            expected_sig = hmac.new(
                self._session_secret.encode("utf-8"),
                payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return secrets.compare_digest(sig, expected_sig)
        except Exception as e:
            logger.warning("Session validation error: %s", e)
            return False

    def revoke_session(self, session_id: Optional[str]) -> bool:
        """Revoke a session immediately."""
        if not session_id:
            return False
        self._revoked_sessions.add(session_id)
        self._active_sessions.pop(session_id, None)
        self._audit.log(
            action="SESSION_REVOKED",
            decision="REVOKED",
            requesting_component="PrivateAuthManager",
            details={"session_id_prefix": session_id[:8]},
            is_user_authorized=True,
        )
        return True

    def rotate_secret(self, new_token: str) -> None:
        """Rotate the master authentication token and revoke all active sessions."""
        self._auth_token = new_token
        self._session_secret = secrets.token_hex(32)
        self._active_sessions.clear()
        self._revoked_sessions.clear()
        self._audit.log(
            action="TOKEN_ROTATED",
            decision="ROTATED",
            requesting_component="PrivateAuthManager",
            details={"message": "All prior sessions invalidated upon token rotation"},
            is_user_authorized=True,
            authorized_by="owner",
        )
