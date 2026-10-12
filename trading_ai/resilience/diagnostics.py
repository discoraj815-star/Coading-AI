"""Self-Diagnostics & Composite Health Monitoring."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.schemas.enums import AudioHardwareMode, HealthState, KnowledgeProtectionState, ThreatLevel
from trading_ai.schemas.recovery import ServiceHealthReport, SystemDiagnosticsReport
from trading_ai.storage.repository import Repository


class SystemDiagnostics:
    """Internal self-diagnostics and system health reporter."""

    def __init__(self, repository: Optional[Repository] = None):
        self._repo = repository
        self._services: Dict[str, ServiceHealthReport] = {}

    def update_service_heartbeat(
        self,
        service_name: str,
        state: HealthState = HealthState.HEALTHY,
        message: str = "Operating nominally",
        error_count: Optional[int] = None,
        recovery_count: Optional[int] = None,
    ) -> ServiceHealthReport:
        """Record or update heartbeat and health state for a service."""
        now = datetime.now(timezone.utc)
        curr = self._services.get(service_name)
        if not curr:
            curr = ServiceHealthReport(
                service_name=service_name,
                state=state,
                last_heartbeat=now,
                last_successful_op=now if state == HealthState.HEALTHY else None,
                error_count=error_count or 0,
                recovery_count=recovery_count or 0,
                message=message,
            )
        else:
            curr.state = state
            curr.last_heartbeat = now
            if state == HealthState.HEALTHY:
                curr.last_successful_op = now
            if error_count is not None:
                curr.error_count = error_count
            if recovery_count is not None:
                curr.recovery_count = recovery_count
            curr.message = message

        self._services[service_name] = curr
        return curr

    def generate_diagnostics_report(
        self,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        active_threats: int = 0,
        watchdog_tasks_count: int = 0,
        threat_level: ThreatLevel = ThreatLevel.NORMAL,
        audio_mode: AudioHardwareMode = AudioHardwareMode.REAL_HARDWARE,
        knowledge_state: KnowledgeProtectionState = KnowledgeProtectionState.PROTECTED,
    ) -> SystemDiagnosticsReport:
        """Produce a comprehensive self-diagnostic evaluation report."""
        # Calculate overall system state
        if isinstance(threat_level, str):
            clean_str = threat_level.split(".")[-1].upper()
            try:
                threat_level = ThreatLevel(clean_str)
            except Exception:
                threat_level = ThreatLevel.NORMAL

        overall = HealthState.HEALTHY
        recommendations: List[str] = []

        if threat_level in (ThreatLevel.BLOCKED, ThreatLevel.CRITICAL):
            overall = HealthState.WARNING if threat_level == ThreatLevel.BLOCKED else HealthState.CRITICAL
            recommendations.append("Review Cyber Wall threat log for unauthorized access attempts.")

        if memory_percent > 85.0 or cpu_percent > 85.0:
            overall = HealthState.DEGRADED
            recommendations.append("Resource threshold high: backpressure enabled.")

        for name, s in self._services.items():
            if s.state in (HealthState.ERROR, HealthState.CRITICAL):
                overall = HealthState.ERROR
                recommendations.append(f"Service '{name}' reported {s.state}: {s.message}")
            elif s.state == HealthState.DEGRADED and overall == HealthState.HEALTHY:
                overall = HealthState.DEGRADED

        if not recommendations:
            recommendations.append("All core systems operational and fully protected.")

        return SystemDiagnosticsReport(
            overall_state=overall,
            timestamp=datetime.now(timezone.utc),
            cpu_percent=round(cpu_percent, 1),
            memory_percent=round(memory_percent, 1),
            threat_level=threat_level,
            active_threats_count=active_threats,
            active_watchdog_tasks=watchdog_tasks_count,
            knowledge_integrity_status=knowledge_state,
            services=dict(self._services),
            audio_mode=audio_mode,
            recommendations=recommendations,
        )
