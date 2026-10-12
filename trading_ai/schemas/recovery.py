"""Step 13 Data Security, Cyber Wall, Watchdog & Auto-Recovery Schemas."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    AudioHardwareMode,
    HealthState,
    KnowledgeProtectionState,
    RecoveryClassification,
    TaskWatchdogState,
    ThreatLevel,
)


class ThreatIncident(BaseModel):
    """Record of a detected cyber wall threat or suspicious activity."""

    incident_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    threat_level: ThreatLevel
    source: str
    target_path_or_resource: Optional[str] = None
    description: str
    details: Dict[str, Any] = Field(default_factory=dict)
    action_taken: str = "BLOCKED"
    is_mitigated: bool = True


class KnowledgeIntegrityRecord(BaseModel):
    """Manifest entry storing cryptographic checksum for knowledge item/version."""

    item_id: str
    version_id: Optional[str] = None
    sha256_hash: str
    content_length: int
    epistemic_state: str
    provenance_hash: Optional[str] = None
    last_verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: KnowledgeProtectionState = KnowledgeProtectionState.PROTECTED


class BackupMetadata(BaseModel):
    """Isolated knowledge backup manifest record."""

    backup_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    backup_path: str
    sha256_checksum: str
    total_items: int
    file_size_bytes: int
    is_verified: bool = True
    retention_tag: str = "AUTOMATED_DAILY"
    notes: Optional[str] = None


class RestoreResult(BaseModel):
    """Outcome of a disaster recovery restore execution."""

    success: bool
    backup_id: str
    restored_items_count: int
    verified_checksum: bool
    restored_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None


class TaskWatchdogEntry(BaseModel):
    """Execution state tracked by Anti-Hang Watchdog."""

    task_id: str
    task_name: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_progress_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: float = 10.0
    state: TaskWatchdogState = TaskWatchdogState.RUNNING
    is_critical_operation: bool = False  # Exempts DB commit or live broker order from force stop
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ServiceHealthReport(BaseModel):
    """Health snapshot for an individual system service."""

    service_name: str
    state: HealthState
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_successful_op: Optional[datetime] = None
    error_count: int = 0
    recovery_count: int = 0
    message: str = "Operating nominally"


class SystemDiagnosticsReport(BaseModel):
    """Composite self-diagnostics report across all subsystem layers."""

    overall_state: HealthState
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    threat_level: ThreatLevel = ThreatLevel.NORMAL
    active_threats_count: int = 0
    active_watchdog_tasks: int = 0
    knowledge_integrity_status: KnowledgeProtectionState = KnowledgeProtectionState.PROTECTED
    services: Dict[str, ServiceHealthReport] = Field(default_factory=dict)
    audio_mode: AudioHardwareMode = AudioHardwareMode.REAL_HARDWARE
    recommendations: List[str] = Field(default_factory=list)


class RecoveryCheckpoint(BaseModel):
    """Resilient checkpoint record for long-running pipelines and tasks."""

    checkpoint_id: str
    job_type: str  # lecture, market, backtest, journal, screen
    job_id: str
    serial_number: Optional[int] = None
    state_payload: Dict[str, Any] = Field(default_factory=dict)
    sha256_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    classification: RecoveryClassification = RecoveryClassification.AUTO_RECOVERABLE
    requires_user_approval: bool = False


class CyberWallValidationRequest(BaseModel):
    """Path or action validation request against Cyber Wall sandbox."""

    target_path: str
    action: str = "READ"  # READ, WRITE, EXECUTE, DELETE
    file_type: Optional[str] = None


class CyberWallValidationResponse(BaseModel):
    """Evaluation response from Cyber Wall sandbox."""

    allowed: bool
    canonical_path: str
    threat_level: ThreatLevel
    reason: str
