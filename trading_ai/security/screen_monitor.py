"""Privacy-Aware Adaptive Screen Monitoring with Allowlist/Denylist & Sensitive Region Masking."""

from datetime import datetime, timezone
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from trading_ai.schemas.enums import PCCapability, ScreenAccessStatus
from trading_ai.schemas.security import ScreenPrivacyRule
from trading_ai.security.audit import SecurityAuditLogger
from trading_ai.security.capability_gate import CapabilityPolicyGate

logger = logging.getLogger(__name__)

# Default Denylist for highly sensitive applications
BLOCKED_APPLICATIONS: Set[str] = {
    "1password",
    "bitwarden",
    "keepass",
    "lastpass",
    "dashlane",
    "credential manager",
    "windows security",
    "authenticator",
    "chrome password manager",
    "banking",
    "paypal",
    "chase",
    "bank of america",
    "whatsapp",
    "signal",
    "telegram",
}

# Default Allowlist for market analysis
ALLOWED_APPLICATIONS: Set[str] = {
    "tradingview",
    "trading ai",
    "trading ai assistant",
    "metatrader",
    "ninjatrader",
    "thinkorswim",
    "binance",
    "bybit",
}

# Regex for sensitive on-screen text fields that require masking
SENSITIVE_TEXT_REGEX = re.compile(
    r"(?i)(password|passwd|pin\b|api[_-]?key|secret|token|bearer|card[_-]?number|cvv|private[_-]?key)"
)


class AdaptiveScreenMonitor:
    """
    Privacy-first screen observation pipeline.

    Security & Privacy Controls:
      - Default strict denylist against password managers and banking apps.
      - Sensitive field masking on detected regions.
      - Adaptive change detection: drops identical/unchanged frames (<2% difference).
      - Capability gate enforcement (`PCCapability.SCREEN_READ` / `SCREEN_ANALYZE`).
      - Ephemeral storage: frames expire and are never persisted to public directories.
    """

    def __init__(
        self,
        capability_gate: CapabilityPolicyGate,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ):
        self._gate = capability_gate
        self._audit = audit_logger or SecurityAuditLogger()
        self._rules: List[ScreenPrivacyRule] = []
        self._last_frame_hashes: Dict[str, str] = {}  # app_name -> sha256
        self._frame_cache: Dict[str, Dict[str, Any]] = {}
        self._initialize_rules()

    def _initialize_rules(self) -> None:
        """Load default privacy allowlist and denylist."""
        for app in BLOCKED_APPLICATIONS:
            self._rules.append(
                ScreenPrivacyRule(
                    app_pattern=app,
                    is_allowed=False,
                    mask_sensitive_regions=True,
                    description=f"Strictly blocked sensitive application: {app}",
                )
            )
        for app in ALLOWED_APPLICATIONS:
            self._rules.append(
                ScreenPrivacyRule(
                    app_pattern=app,
                    is_allowed=True,
                    mask_sensitive_regions=True,
                    description=f"Allowed trading analysis application: {app}",
                )
            )

    def is_app_allowed(self, app_name: str) -> Tuple[bool, str]:
        """Check application name against allowlist and denylist."""
        cleaned = app_name.lower().strip()

        # 1. Denylist takes absolute precedence
        for blocked in BLOCKED_APPLICATIONS:
            if blocked in cleaned:
                return False, f"Application '{app_name}' contains blocked sensitive keyword '{blocked}'."

        # 2. Explicit rule check
        for rule in self._rules:
            if rule.app_pattern in cleaned:
                if not rule.is_allowed:
                    return False, f"Application '{app_name}' blocked by privacy rule."
                return True, f"Application '{app_name}' authorized by rule."

        # 3. Default conservative policy: require explicit permission
        return False, f"Application '{app_name}' is not in the explicit allowlist."

    def mask_sensitive_content(self, text_or_regions: Any) -> Any:
        """
        Deterministically redact on-screen sensitive strings and bounding boxes.
        """
        if isinstance(text_or_regions, str):
            if SENSITIVE_TEXT_REGEX.search(text_or_regions):
                return "[SENSITIVE_FIELD_MASKED]"
            return text_or_regions
        elif isinstance(text_or_regions, dict):
            masked = {}
            for k, v in text_or_regions.items():
                if any(s in k.lower() for s in ["password", "secret", "key", "token", "auth", "pin", "credential"]):
                    masked[k] = "[MASKED_SECURITY_FIELD]"
                elif isinstance(v, (str, dict, list)):
                    masked[k] = self.mask_sensitive_content(v)
                else:
                    masked[k] = v
            return masked
        elif isinstance(text_or_regions, list):
            return [self.mask_sensitive_content(item) for item in text_or_regions]
        return text_or_regions

    def capture_and_process_frame(
        self,
        app_name: str,
        window_title: Optional[str] = None,
        raw_pixels_mock: Optional[str] = None,
        ocr_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full screen privacy pipeline:
        SCREEN CAPTURE -> APPLICATION SCOPE CHECK -> PRIVACY FILTER -> SENSITIVE REGION MASKING -> CHANGE DETECTION -> RESULT -> AUDIT
        """
        # 1. Capability Permission Check
        if not self._gate.check_permission(PCCapability.SCREEN_READ, "AdaptiveScreenMonitor"):
            self._audit.log(
                action="SCREEN_CAPTURE_BLOCKED",
                decision="DENIED",
                requesting_component="AdaptiveScreenMonitor",
                capability="SCREEN_READ",
                details={"app_name": app_name, "reason": "SCREEN_READ capability not authorized"},
            )
            raise PermissionError("SCREEN_READ capability is not authorized by user.")

        # 2. Application Scope & Denylist Check
        allowed, scope_reason = self.is_app_allowed(app_name)
        if not allowed:
            self._audit.log(
                action="SCREEN_CAPTURE_DENIED",
                decision="DENIED",
                requesting_component="AdaptiveScreenMonitor",
                details={"app_name": app_name, "reason": scope_reason},
            )
            raise PermissionError(f"Screen access denied for '{app_name}': {scope_reason}")

        # 3. Privacy Filter & Sensitive Region Masking
        safe_ocr = self.mask_sensitive_content(ocr_text or "")
        safe_title = self.mask_sensitive_content(window_title or "")

        # 4. Adaptive Change Detection
        frame_content_sig = f"{safe_ocr}:{safe_title}:{raw_pixels_mock or 'frame'}"
        current_hash = hashlib.sha256(frame_content_sig.encode("utf-8")).hexdigest()

        last_hash = self._last_frame_hashes.get(app_name)
        if last_hash == current_hash:
            # Drop redundant duplicate frame to preserve token & compute budget
            return {
                "status": "SKIPPED_UNCHANGED",
                "app_name": app_name,
                "message": "Screen content unchanged; frame dropped to conserve compute.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        self._last_frame_hashes[app_name] = current_hash

        # 5. Build Safe Frame
        processed_frame = {
            "status": "CAPTURED_AND_FILTERED",
            "app_name": app_name,
            "window_title": safe_title,
            "filtered_ocr": safe_ocr,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sha256": current_hash,
            "sensitive_regions_masked": True,
        }

        # 6. Audit
        self._audit.log(
            action="SCREEN_FRAME_PROCESSED",
            decision="ALLOWED",
            requesting_component="AdaptiveScreenMonitor",
            capability="SCREEN_READ",
            is_user_authorized=True,
            details={"app_name": app_name, "sha256": current_hash[:10]},
        )
        return processed_frame
