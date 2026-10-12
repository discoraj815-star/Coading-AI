"""Step 13 Resilience, Cyber Wall & Auto-Recovery Facade Core."""
from typing import Optional

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.resilience.backup_manager import BackupManager
from trading_ai.resilience.cyber_wall import CyberWallSandbox
from trading_ai.resilience.diagnostics import SystemDiagnostics
from trading_ai.resilience.knowledge_guard import KnowledgeIntegrityGuard
from trading_ai.resilience.recovery_coordinator import AutoRecoveryCoordinator
from trading_ai.resilience.resource_guard import ResourceGuard
from trading_ai.resilience.threat_detector import ThreatDetector, ThreatScannerAdapter
from trading_ai.resilience.watchdog import AntiHangWatchdog
from trading_ai.schemas.enums import ThreatLevel
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier


class ResilienceCore:
    """Unified facade coordinating Data Security, Cyber Wall Sandbox, Anti-Hang Watchdog and Auto-Recovery."""

    def __init__(
        self,
        repository: Optional[Repository] = None,
        event_bus: Optional[ReactiveEventBus] = None,
        workspace_root: Optional[str] = None,
        backup_dir: Optional[str] = None,
    ):
        self.repository = repository
        self.event_bus = event_bus

        # Initialize core sub-services
        self.knowledge_guard = KnowledgeIntegrityGuard(repository=repository)
        self.backup_manager = BackupManager(backup_dir=backup_dir, repository=repository)
        self.cyber_wall = CyberWallSandbox(workspace_root=workspace_root)
        self.threat_detector = ThreatDetector(
            scanner_adapter=ThreatScannerAdapter(),
            event_bus=event_bus,
            repository=repository,
        )
        self.watchdog = AntiHangWatchdog(event_bus=event_bus)
        self.resource_guard = ResourceGuard()
        self.diagnostics = SystemDiagnostics(repository=repository)

        if repository:
            barrier = CommitBarrier(repository)
            self.recovery_coordinator = AutoRecoveryCoordinator(
                repository=repository,
                commit_barrier=barrier,
            )
        else:
            self.recovery_coordinator = None

    def get_composite_status(self):
        """Produce real-time aggregated diagnostics status."""
        usage = self.resource_guard.get_resource_usage()
        active_threats = len(self.threat_detector.list_active_threats())
        active_watchdog = len(self.watchdog.list_active_tasks())
        max_threat = ThreatLevel.NORMAL
        for t in self.threat_detector.list_active_threats()[:5]:
            if t.threat_level in (ThreatLevel.CRITICAL, ThreatLevel.BLOCKED):
                max_threat = t.threat_level
                break

        return self.diagnostics.generate_diagnostics_report(
            cpu_percent=usage["cpu_percent"],
            memory_percent=usage["memory_percent"],
            active_threats=active_threats,
            watchdog_tasks_count=active_watchdog,
            threat_level=max_threat,
        )
