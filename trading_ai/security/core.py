"""Security Core: Unified orchestration facade for all Step 12 security, privacy, and HCI services."""

from typing import Optional

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.security.audit import SecurityAuditLogger
from trading_ai.security.auth import PrivateAuthManager
from trading_ai.security.capability_gate import CapabilityPolicyGate
from trading_ai.security.device_manager import DeviceManager
from trading_ai.security.notifications import CrossDeviceNotificationService
from trading_ai.security.privacy_router import DataPrivacyRouter
from trading_ai.security.screen_monitor import AdaptiveScreenMonitor
from trading_ai.security.voice_gateway import SecureVoiceGateway
from trading_ai.storage.repository import Repository


class SecurityCore:
    """
    Centralized controller for Step 12 security, device pairing, screen monitoring, voice, and notifications.
    """

    def __init__(
        self,
        repository: Optional[Repository] = None,
        event_bus: Optional[ReactiveEventBus] = None,
        auth_token: Optional[str] = None,
        session_secret: Optional[str] = None,
        auth_enabled: bool = False,
    ):
        self.repository = repository
        self.event_bus = event_bus

        # 1. Audit logger
        self.audit_logger = SecurityAuditLogger(repository=repository)

        # 1b. Private Personal Authentication & Session Manager
        self.auth_manager = PrivateAuthManager(
            auth_token=auth_token,
            session_secret=session_secret,
            auth_enabled=auth_enabled,
            audit_logger=self.audit_logger,
        )

        # 2. Capability Policy Gate (Least-privilege default DENY)
        self.capability_gate = CapabilityPolicyGate(repository=repository, audit_logger=self.audit_logger)

        # 3. Privacy-First Data Router
        self.privacy_router = DataPrivacyRouter(audit_logger=self.audit_logger)

        # 4. Device Identity & Encrypted Transport
        self.device_manager = DeviceManager(repository=repository, audit_logger=self.audit_logger)

        # 5. Screen Observation with Privacy Masking
        self.screen_monitor = AdaptiveScreenMonitor(capability_gate=self.capability_gate, audit_logger=self.audit_logger)

        # 6. Secure Two-Way Voice Gateway
        self.voice_gateway = SecureVoiceGateway(capability_gate=self.capability_gate, audit_logger=self.audit_logger)

        # 7. Cross-Device Notification Service
        self.notifications = CrossDeviceNotificationService(
            device_manager=self.device_manager,
            capability_gate=self.capability_gate,
            event_bus=event_bus,
            audit_logger=self.audit_logger,
        )
