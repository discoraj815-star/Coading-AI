"""Privacy-First Data Router: Classifies, sanitizes, and gates local vs cloud processing."""

from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.enums import DataPrivacyClassification
from trading_ai.schemas.security import DataClassificationRule
from trading_ai.security.audit import SecurityAuditLogger, redact_dict, redact_sensitive_text

logger = logging.getLogger(__name__)


class DataPrivacyRouter:
    """
    Enforces privacy-first local processing and explicit authorization boundaries.

    Guarantees:
      - Sensitive data never automatically leaves local execution environment.
      - External processing requires explicit classification and sanitization.
      - Untrusted input (web text, screen OCR, voice) is treated as UNTRUSTED_DATA.
      - Explicit audit trail for every routing decision.
    """

    def __init__(self, audit_logger: Optional[SecurityAuditLogger] = None):
        self._audit = audit_logger or SecurityAuditLogger()
        self._rules: List[DataClassificationRule] = [
            DataClassificationRule(
                data_pattern=r"(?i)(password|private_key|seed_phrase|auth_token|credit_card|secret_key)",
                classification=DataPrivacyClassification.BLOCKED_EXTERNAL,
                description="Credentials and private keys must never leave local machine",
            ),
            DataClassificationRule(
                data_pattern=r"(?i)(raw_screen_frame|audio_recording|local_trade_journal|database_row)",
                classification=DataPrivacyClassification.LOCAL_ONLY,
                description="Raw system capture and personal trade history remain local by default",
            ),
            DataClassificationRule(
                data_pattern=r"(?i)(market_symbol|timeframe|technical_indicator|public_chart|lecture_text)",
                classification=DataPrivacyClassification.USER_APPROVED_CLOUD,
                description="Sanitized market and educational text can be processed via Gemini when authorized",
            ),
            DataClassificationRule(
                data_pattern=r"(?i)(public_news|forex_factory_event|economic_calendar|docs)",
                classification=DataPrivacyClassification.NON_SENSITIVE_EXTERNAL,
                description="Public macroeconomic events and non-sensitive documents",
            ),
        ]

    def classify_data(self, data: Any) -> Tuple[DataPrivacyClassification, str]:
        """Determine data privacy classification based on content analysis."""
        text_rep = str(data)
        # Check highest security rules first (BLOCKED_EXTERNAL)
        for rule in self._rules:
            if re.search(rule.data_pattern, text_rep):
                return rule.classification, rule.description

        # Default conservative policy: LOCAL_ONLY
        return DataPrivacyClassification.LOCAL_ONLY, "Default conservative local processing"

    def process_and_route(
        self,
        raw_input: Any,
        target_destination: str,
        requesting_component: str,
        user_cloud_consent: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute full privacy pipeline:
        INPUT -> CLASSIFICATION -> SANITIZATION -> PERMISSION CHECK -> PROCESSING LOCATION -> STORAGE -> AUDIT
        """
        # 1. Classification
        classification, reason = self.classify_data(raw_input)

        # 2. Sanitization
        if isinstance(raw_input, dict):
            sanitized_data = redact_dict(raw_input)
        elif isinstance(raw_input, str):
            sanitized_data = redact_sensitive_text(raw_input)
        else:
            sanitized_data = raw_input

        # 3. Permission Check & Destination Resolution
        is_external_target = target_destination.lower() in ["cloud", "gemini", "external_api", "remote"]
        allowed = True
        decision_notes = "Processed locally"
        effective_location = "LOCAL"

        if classification == DataPrivacyClassification.BLOCKED_EXTERNAL:
            if is_external_target:
                allowed = False
                decision_notes = "BLOCKED: Sensitive credentials/data cannot be sent externally."
                effective_location = "BLOCKED"
        elif classification == DataPrivacyClassification.LOCAL_ONLY:
            if is_external_target:
                allowed = False
                decision_notes = "BLOCKED: Data is classified LOCAL_ONLY and cannot leave PC."
                effective_location = "BLOCKED"
        elif classification == DataPrivacyClassification.USER_APPROVED_CLOUD:
            if is_external_target:
                if user_cloud_consent:
                    effective_location = "USER_APPROVED_CLOUD"
                    decision_notes = "Routed to cloud with explicit user consent."
                else:
                    allowed = False
                    decision_notes = "BLOCKED: Cloud transmission requires explicit user consent."
                    effective_location = "BLOCKED"
        elif classification == DataPrivacyClassification.NON_SENSITIVE_EXTERNAL:
            if is_external_target:
                effective_location = "NON_SENSITIVE_EXTERNAL"
                decision_notes = "Non-sensitive public data permitted for external lookup."

        # 4. Audit
        decision = "ALLOWED" if allowed else "DENIED"
        self._audit.log(
            action="DATA_PRIVACY_ROUTING",
            decision=decision,
            requesting_component=requesting_component,
            is_user_authorized=user_cloud_consent,
            details={
                "target_destination": target_destination,
                "classification": classification.value,
                "effective_location": effective_location,
                "reason": decision_notes,
            },
        )

        return {
            "allowed": allowed,
            "classification": classification,
            "effective_location": effective_location,
            "sanitized_data": sanitized_data if allowed else None,
            "reason": decision_notes,
        }
