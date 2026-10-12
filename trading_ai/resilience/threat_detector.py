"""Cyber Wall Threat Detection & Scanner Adapter."""
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.schemas.enums import EventPriority, SystemEventType, ThreatLevel
from trading_ai.schemas.recovery import ThreatIncident
from trading_ai.schemas.performance import SystemEvent
from trading_ai.storage.repository import Repository


class ThreatScannerAdapter:
    """Interface and default heuristic scanner adapter."""

    def __init__(self, is_external_available: bool = False):
        self.is_external_available = is_external_available

    def scan_content(self, text_or_bytes: str) -> ThreatLevel:
        """Scan string content for known malware signatures, shell injections, or prompt exploits."""
        text = str(text_or_bytes).lower()

        # Critical exploits / dangerous system commands
        critical_patterns = [
            r"powershell\s+-enc",
            r"format\s+[a-z]:",
            r"rmdir\s+/s\s+/q",
            r"drop\s+database",
            r"drop\s+table\s+users",
            r"chmod\s+777",
            r"curl\s+.*\s*\|\s*sh",
            r"wget\s+.*\s*\|\s*sh",
        ]
        for p in critical_patterns:
            if re.search(p, text):
                return ThreatLevel.CRITICAL

        # Suspicious prompt injection commands
        suspicious_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|security)\s+(instructions|rules)",
            r"you\s+are\s+now\s+in\s+(developer|unrestricted|god)\s+mode",
            r"bypass\s+all\s+guardrails",
            r"system:\s*override",
        ]
        for p in suspicious_patterns:
            if re.search(p, text):
                return ThreatLevel.SUSPICIOUS

        return ThreatLevel.NORMAL


class ThreatDetector:
    """Detects, classifies, and reports project-level security threats and anomalies."""

    def __init__(
        self,
        scanner_adapter: Optional[ThreatScannerAdapter] = None,
        event_bus: Optional[ReactiveEventBus] = None,
        repository: Optional[Repository] = None,
    ):
        self.scanner = scanner_adapter or ThreatScannerAdapter()
        self.event_bus = event_bus
        self.repository = repository
        self._safe_restricted_mode: bool = False
        self._incidents: List[ThreatIncident] = []

    @property
    def in_safe_restricted_mode(self) -> bool:
        return self._safe_restricted_mode

    def evaluate_payload(self, content: str, source: str = "EXTERNAL_INPUT") -> ThreatIncident:
        """Scan and evaluate an untrusted payload."""
        try:
            level = self.scanner.scan_content(content)
        except Exception:
            # FAIL-CLOSED: If scanner fails or is unavailable, enter SAFE_RESTRICTED_MODE
            self._safe_restricted_mode = True
            level = ThreatLevel.BLOCKED

        incident = ThreatIncident(
            incident_id=f"threat_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            threat_level=level,
            source=source,
            description=f"Evaluated input with threat level {level}",
            action_taken="BLOCKED" if level in (ThreatLevel.BLOCKED, ThreatLevel.CRITICAL) else "LOGGED",
            is_mitigated=True,
        )

        self._incidents.append(incident)

        if self.repository and hasattr(self.repository, "save_threat_incident"):
            self.repository.save_threat_incident(incident)

        # Publish event to Step 11 EventBus if severe
        if self.event_bus and level in (ThreatLevel.SUSPICIOUS, ThreatLevel.BLOCKED, ThreatLevel.CRITICAL):
            prio = EventPriority.CRITICAL if level == ThreatLevel.CRITICAL else EventPriority.HIGH
            evt = SystemEvent(
                event_type=SystemEventType.SECURITY_AUDIT_LOGGED,
                payload={
                    "threat_level": level.value if hasattr(level, "value") else str(level),
                    "source": source,
                    "incident_id": incident.incident_id,
                },
                priority=prio,
                provenance="CyberWallThreatDetector",
            )
            self.event_bus.publish(evt)

        return incident

    def record_custom_threat(
        self,
        source: str,
        description: str,
        level: ThreatLevel = ThreatLevel.BLOCKED,
        details: Optional[Dict[str, Any]] = None,
    ) -> ThreatIncident:
        """Explicitly record a threat detected by another subsystem (e.g. capability violation)."""
        incident = ThreatIncident(
            incident_id=f"threat_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            threat_level=level,
            source=source,
            description=description,
            details=details or {},
            action_taken="BLOCKED",
            is_mitigated=True,
        )
        self._incidents.append(incident)

        if self.repository and hasattr(self.repository, "save_threat_incident"):
            self.repository.save_threat_incident(incident)

        return incident

    def list_active_threats(self) -> List[ThreatIncident]:
        """Return all recorded threat incidents."""
        if self.repository and hasattr(self.repository, "list_threat_incidents"):
            return self.repository.list_threat_incidents()
        return list(reversed(self._incidents))
