"""Step 12: Security, Privacy, and Human-Computer Interaction Package."""

from trading_ai.security.audit import SecurityAuditLogger, redact_dict, redact_sensitive_text
from trading_ai.security.capability_gate import CapabilityPolicyGate
from trading_ai.security.core import SecurityCore
from trading_ai.security.device_manager import DeviceManager
from trading_ai.security.notifications import CrossDeviceNotificationService
from trading_ai.security.privacy_router import DataPrivacyRouter
from trading_ai.security.screen_monitor import AdaptiveScreenMonitor
from trading_ai.security.voice_gateway import SecureVoiceGateway

__all__ = [
    "SecurityAuditLogger",
    "CapabilityPolicyGate",
    "DataPrivacyRouter",
    "DeviceManager",
    "AdaptiveScreenMonitor",
    "SecureVoiceGateway",
    "CrossDeviceNotificationService",
    "SecurityCore",
    "redact_sensitive_text",
    "redact_dict",
]
