"""Security, Privacy, and Human-Computer Interaction Schemas for Step 12."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    CapabilityState,
    DataPrivacyClassification,
    DeviceStatus,
    NotificationPriority,
    PCCapability,
    VoiceProcessingState,
)


class DataClassificationRule(BaseModel):
    """Rule determining privacy classification of input data."""

    rule_id: str = Field(default_factory=lambda: f"priv_{uuid.uuid4().hex[:8]}")
    data_pattern: str
    classification: DataPrivacyClassification
    description: str
    redaction_regex: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PCCapabilityRule(BaseModel):
    """Explicit capability permission policy."""

    capability: PCCapability
    state: CapabilityState = CapabilityState.DENIED
    requires_user_approval: bool = True
    is_dangerous: bool = False
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: Optional[str] = None
    reason: Optional[str] = None


class CapabilityRequest(BaseModel):
    """Request submitted to invoke a specific PC capability."""

    request_id: str = Field(default_factory=lambda: f"capreq_{uuid.uuid4().hex[:8]}")
    capability: PCCapability
    requesting_component: str
    context: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CapabilityApproval(BaseModel):
    """Explicit user authorization for a capability request."""

    request_id: str
    capability: PCCapability
    approved: bool
    is_user: bool = False
    approved_by: str = ""
    reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DeviceRecord(BaseModel):
    """Trusted device metadata and cryptographic status."""

    device_id: str
    device_name: str
    device_type: str = "phone"  # phone, tablet, secondary_pc
    status: DeviceStatus = DeviceStatus.UNPAIRED
    public_key_or_token_hash: str
    paired_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    session_token_hash: Optional[str] = None
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None


class DevicePairingRequest(BaseModel):
    """Inbound request from a new client device seeking pairing."""

    device_id: str
    device_name: str
    device_type: str = "phone"
    pairing_code: str
    nonce: str
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class DevicePairingApproval(BaseModel):
    """User decision to authorize or reject device pairing."""

    device_id: str
    approved: bool
    is_user: bool = False
    approved_by: str = ""
    shared_secret: Optional[str] = None


class EncryptedPayload(BaseModel):
    """Secure transit envelope for cross-device communication."""

    device_id: str
    ciphertext: str
    nonce: str
    signature: str
    timestamp: float
    key_id: str = "v1"


class SecurityAuditEntry(BaseModel):
    """Auditable log entry for security and capability invocations."""

    audit_id: str = Field(default_factory=lambda: f"audit_{uuid.uuid4().hex[:10]}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    capability: Optional[str] = None
    action: str
    requesting_component: str
    decision: str  # ALLOWED, DENIED, REVOKED, ERROR
    is_user_authorized: bool = False
    authorized_by: Optional[str] = None
    device_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class NotificationItem(BaseModel):
    """Privacy-minimized notification item dispatched across devices."""

    notification_id: str = Field(default_factory=lambda: f"notif_{uuid.uuid4().hex[:8]}")
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.MEDIUM
    category: str = "system"  # system, market, security, task
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    read: bool = False
    delivered_devices: List[str] = Field(default_factory=list)


class ScreenPrivacyRule(BaseModel):
    """Allowlist/Denylist rule for screen monitoring."""

    rule_id: str = Field(default_factory=lambda: f"spriv_{uuid.uuid4().hex[:8]}")
    app_pattern: str
    is_allowed: bool
    mask_sensitive_regions: bool = True
    description: str = ""


class VoiceCommandRequest(BaseModel):
    """Audio or transcribed voice command submitted for execution."""

    command_text: str
    audio_format: Optional[str] = None
    audio_base64: Optional[str] = None
    device_id: Optional[str] = None
    session_id: Optional[str] = None
    is_user_voice: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class VoiceCommandResult(BaseModel):
    """Structured response to a voice command."""

    intent: str
    raw_query: str
    response_text: str
    audio_output_url: Optional[str] = None
    state: VoiceProcessingState = VoiceProcessingState.RESPONDING
    authorized: bool = True
    denial_reason: Optional[str] = None
    latency_ms: float = 0.0
