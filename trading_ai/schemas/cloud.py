"""Step 16: 24×7 Cloud Runtime, Advanced Event Stream & Multi-Channel Notification Schemas."""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    CloudRuntimeState,
    EventPriorityLevel,
    NotificationChannel,
    NotificationDeliveryStatus,
    NotificationPrivacyLevel,
    SetupNotificationState,
)


class CloudComponentHealth(BaseModel):
    """Health status and telemetry for a single cloud component."""

    name: str
    status: str = "HEALTHY"  # HEALTHY, DEGRADED, OFFLINE, RECOVERING
    latency_ms: float = 0.0
    details: Dict[str, Any] = Field(default_factory=dict)
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CloudRuntimeHealth(BaseModel):
    """Comprehensive health matrix for the 24×7 cloud runtime."""

    runtime_state: CloudRuntimeState = CloudRuntimeState.ONLINE
    uptime_seconds: float = 0.0
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active_workers: int = 4
    components: Dict[str, CloudComponentHealth] = Field(default_factory=dict)
    stats: Dict[str, Any] = Field(default_factory=dict)


class CloudStreamEvent(BaseModel):
    """Authoritative, immutable event broadcast across WebSocket and notification layers."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: Optional[str] = None
    priority: EventPriorityLevel = EventPriorityLevel.P5_INFORMATIONAL
    source: str = "CLOUD_AI_CORE"
    confidence: float = 1.0
    evidence_reference: Dict[str, Any] = Field(default_factory=dict)
    rule_version: Optional[str] = None
    correlation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    deduplication_key: str = ""
    payload: Dict[str, Any] = Field(default_factory=dict)
    delivery_status: NotificationDeliveryStatus = NotificationDeliveryStatus.QUEUED


class NotificationRecord(BaseModel):
    """Persisted delivery record of an alert dispatched via any notification channel."""

    notification_id: str = Field(default_factory=lambda: f"notif_{uuid.uuid4().hex[:10]}")
    event_id: str
    priority: EventPriorityLevel = EventPriorityLevel.P5_INFORMATIONAL
    title: str
    message_hindi: str
    message_english: str
    symbol: Optional[str] = None
    channel: NotificationChannel = NotificationChannel.WEBSOCKET
    delivery_status: NotificationDeliveryStatus = NotificationDeliveryStatus.QUEUED
    privacy_level: NotificationPrivacyLevel = NotificationPrivacyLevel.NORMAL
    deduplication_key: str = ""
    target_device_id: Optional[str] = None
    retry_count: int = 0
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: Optional[datetime] = None


class UserNotificationPreferences(BaseModel):
    """Configurable user preferences, privacy constraints, and channel rate limits."""

    notification_enabled: bool = True
    push_enabled: bool = True
    sms_enabled: bool = False
    voice_enabled: bool = True
    email_enabled: bool = False
    in_app_enabled: bool = True

    # Quiet hours & overrides
    quiet_hours_enabled: bool = True
    quiet_hours_start_utc: int = 22  # 22:00 UTC (approx 3:30 AM IST)
    quiet_hours_end_utc: int = 4    # 04:00 UTC (approx 9:30 AM IST)
    critical_alert_override: bool = True  # P0, P1, P2 break through quiet hours

    # Rate limiting & deduplication
    max_sms_per_day: int = 5
    max_voice_alerts_per_day: int = 25
    sms_sent_today: int = 0
    voice_alerts_sent_today: int = 0
    last_reset_date: str = ""

    # Thresholds
    minimum_alert_confidence: float = 0.70
    minimum_confluence: int = 2
    why_not_trade_enabled: bool = True

    # Localization & Voice
    greeting_name: str = "Alam"
    notification_language: str = "hi"  # "hi" for Hindi, "en" for English
    notification_verbosity: str = "detailed"  # "short", "detailed"
    voice_speed: float = 1.0
    voice_type: str = "hindi_natural"
    privacy_level: NotificationPrivacyLevel = NotificationPrivacyLevel.NORMAL


class DevicePushTokenRegistration(BaseModel):
    """Registered push token for mobile/web push delivery."""

    device_id: str
    push_token: str
    platform: str = "web"  # web, android, ios
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VectorStoreStats(BaseModel):
    """Health and metrics for the C2 Vector Store Adapter."""

    backend: str = "SQLiteVectorStore"
    total_embeddings: int = 0
    dimension: int = 1536
    health: str = "HEALTHY"
    status_details: Dict[str, Any] = Field(default_factory=dict)


class TaskQueueJob(BaseModel):
    """Task item handled by the C3 Task Queue Adapter."""

    job_id: str = Field(default_factory=lambda: f"job_{uuid.uuid4().hex[:10]}")
    task_type: str
    priority: int = 2
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, FAILED, RETRYING, CANCELLED
    retry_count: int = 0
    max_retries: int = 3
    serial_number: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
