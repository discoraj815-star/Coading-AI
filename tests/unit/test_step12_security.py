"""Unit tests for Step 12 Security, Privacy, and Human-Computer Interaction."""

import time
import pytest
from datetime import datetime, timezone

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.schemas.enums import (
    CapabilityState,
    DataPrivacyClassification,
    DeviceStatus,
    NotificationPriority,
    PCCapability,
    VoiceProcessingState,
)
from trading_ai.schemas.security import (
    CapabilityApproval,
    DevicePairingApproval,
    DevicePairingRequest,
    VoiceCommandRequest,
)
from trading_ai.security.audit import SecurityAuditLogger, redact_dict, redact_sensitive_text
from trading_ai.security.capability_gate import CapabilityPolicyGate
from trading_ai.security.device_manager import DeviceManager
from trading_ai.security.notifications import CrossDeviceNotificationService
from trading_ai.security.privacy_router import DataPrivacyRouter
from trading_ai.security.screen_monitor import AdaptiveScreenMonitor
from trading_ai.security.voice_gateway import SecureVoiceGateway


# --- Capability Gate Tests ---
def test_capability_default_deny():
    gate = CapabilityPolicyGate()
    for cap in PCCapability:
        rule = gate.get_rule(cap)
        assert rule.state == CapabilityState.DENIED
        assert not gate.check_permission(cap, "TestComponent")


def test_capability_user_authorization_lifecycle():
    gate = CapabilityPolicyGate()
    req = gate.request_capability(PCCapability.ALERT_TRIGGER, "CompanionEngine")
    assert req.request_id.startswith("capreq_")

    # Authorize as user
    approval = CapabilityApproval(
        request_id=req.request_id,
        capability=PCCapability.ALERT_TRIGGER,
        approved=True,
        is_user=True,
        approved_by="Trader",
    )
    rule = gate.authorize_capability(approval)
    assert rule.state == CapabilityState.APPROVED_ALWAYS
    assert gate.check_permission(PCCapability.ALERT_TRIGGER, "CompanionEngine") is True

    # Revoke capability
    revoked = gate.revoke_capability(PCCapability.ALERT_TRIGGER, "Trader")
    assert revoked.state == CapabilityState.REVOKED
    assert gate.check_permission(PCCapability.ALERT_TRIGGER, "CompanionEngine") is False


def test_dangerous_capability_denies_automated_elevation():
    gate = CapabilityPolicyGate()
    req = gate.request_capability(PCCapability.TERMINAL_EXECUTE, "AutonomousAgent")

    # Automated agent tries to approve itself (is_user = False)
    approval = CapabilityApproval(
        request_id=req.request_id,
        capability=PCCapability.TERMINAL_EXECUTE,
        approved=True,
        is_user=False,
        approved_by="AutomatedAgent",
    )
    with pytest.raises(PermissionError, match="strictly requires human user authorization"):
        gate.authorize_capability(approval)

    # Permission must remain DENIED
    assert gate.check_permission(PCCapability.TERMINAL_EXECUTE, "AutonomousAgent") is False


# --- Privacy Router & Sanitization Tests ---
def test_privacy_classification_and_routing():
    router = DataPrivacyRouter()

    # 1. Blocked external (passwords / seed phrases)
    res_blocked = router.process_and_route(
        raw_input="my secret seed_phrase is orange apple tree",
        target_destination="cloud",
        requesting_component="Test",
    )
    assert res_blocked["allowed"] is False
    assert res_blocked["classification"] == DataPrivacyClassification.BLOCKED_EXTERNAL

    # 2. Local only without consent
    res_cloud_no_consent = router.process_and_route(
        raw_input="EUR/USD market_symbol breakout",
        target_destination="cloud",
        requesting_component="Test",
        user_cloud_consent=False,
    )
    assert res_cloud_no_consent["allowed"] is False

    # 3. Cloud with user consent
    res_cloud_approved = router.process_and_route(
        raw_input="EUR/USD market_symbol breakout",
        target_destination="cloud",
        requesting_component="Test",
        user_cloud_consent=True,
    )
    assert res_cloud_approved["allowed"] is True
    assert res_cloud_approved["effective_location"] == "USER_APPROVED_CLOUD"


def test_sensitive_text_and_dict_redaction():
    text = "Authorization: Bearer sk-1234567890abcdef1234567890 with api_key=secret_xyz_123456"
    redacted = redact_sensitive_text(text)
    assert "sk-1234567890abcdef1234567890" not in redacted
    assert "secret_xyz_123456" not in redacted

    data = {
        "user": "Alice",
        "api_key": "supersecretkey123",
        "nested": {"token": "mytoken"},
    }
    redacted_data = redact_dict(data)
    assert redacted_data["api_key"] == "[REDACTED]"
    assert redacted_data["nested"]["token"] == "[REDACTED]"
    assert redacted_data["user"] == "Alice"


# --- Device Pairing & Encrypted Transport Tests ---
def test_device_pairing_and_revocation_lifecycle():
    dev_mgr = DeviceManager()
    dev_id = "phone_iphone15_001"

    # Step 1: Generate PIN
    pin = dev_mgr.generate_pairing_code(dev_id, "Alice's iPhone")
    assert len(pin) == 6

    # Step 2: Device submits PIN
    req = DevicePairingRequest(
        device_id=dev_id,
        device_name="Alice's iPhone",
        pairing_code=pin,
        nonce="nonce_test_123",
    )
    assert dev_mgr.submit_pairing_request(req) is True

    # Step 3: Human User approves pairing
    approval = DevicePairingApproval(
        device_id=dev_id,
        approved=True,
        is_user=True,
        approved_by="Alice",
    )
    ok, secret = dev_mgr.approve_pairing(approval)
    assert ok is True
    assert secret is not None

    device = dev_mgr.get_device(dev_id)
    assert device.status == DeviceStatus.ACTIVE

    # Step 4: Encrypt from PC to Phone
    cipher_payload = dev_mgr.encrypt_for_device(dev_id, "Market update: BOS confirmed")
    assert cipher_payload.ciphertext != "Market update: BOS confirmed"

    # Step 5: Decrypt from Phone
    decrypted = dev_mgr.decrypt_from_device(cipher_payload)
    assert decrypted == "Market update: BOS confirmed"

    # Step 6: Revoke Device
    assert dev_mgr.revoke_device(dev_id, "Phone was lost") is True
    assert dev_mgr.get_device(dev_id).status == DeviceStatus.REVOKED

    # Post-revocation encryption must fail
    with pytest.raises(PermissionError):
        dev_mgr.encrypt_for_device(dev_id, "Secret message")


def test_device_replay_protection():
    dev_mgr = DeviceManager()
    dev_id = "phone_android_002"
    pin = dev_mgr.generate_pairing_code(dev_id, "Pixel 8")
    dev_mgr.submit_pairing_request(DevicePairingRequest(device_id=dev_id, device_name="Pixel 8", pairing_code=pin, nonce="n1"))
    dev_mgr.approve_pairing(DevicePairingApproval(device_id=dev_id, approved=True, is_user=True))

    payload = dev_mgr.encrypt_for_device(dev_id, "Trade alert")
    # First decryption succeeds
    assert dev_mgr.decrypt_from_device(payload) == "Trade alert"

    # Replay with same nonce must raise ValueError
    with pytest.raises(ValueError, match="Duplicate message nonce detected"):
        dev_mgr.decrypt_from_device(payload)


# --- Screen Monitoring Tests ---
def test_screen_monitor_scope_and_masking():
    gate = CapabilityPolicyGate()
    gate.authorize_capability(CapabilityApproval(request_id="r1", capability=PCCapability.SCREEN_READ, approved=True, is_user=True))
    monitor = AdaptiveScreenMonitor(capability_gate=gate)

    # 1. Allowed App: TradingView
    frame = monitor.capture_and_process_frame(
        app_name="TradingView",
        window_title="BTCUSD Chart",
        ocr_text="Open: 64000 High: 65000 | Token: sk-test-key",
    )
    assert frame["status"] == "CAPTURED_AND_FILTERED"
    assert "sk-test-key" not in frame["filtered_ocr"]

    # 2. Blocked App: 1Password
    with pytest.raises(PermissionError, match="blocked sensitive keyword"):
        monitor.capture_and_process_frame(app_name="1Password", window_title="Vault")

    # 3. Adaptive Change Detection (Duplicate frame dropped)
    frame_dup = monitor.capture_and_process_frame(
        app_name="TradingView",
        window_title="BTCUSD Chart",
        ocr_text="Open: 64000 High: 65000 | Token: sk-test-key",
    )
    assert frame_dup["status"] == "SKIPPED_UNCHANGED"


# --- Voice Gateway Tests ---
def test_voice_gateway_capability_gating():
    gate = CapabilityPolicyGate()
    gateway = SecureVoiceGateway(capability_gate=gate)

    # Non-privileged voice query (market status) -> Allowed
    res_status = gateway.process_voice_command(VoiceCommandRequest(command_text="What is the market status?"))
    assert res_status.authorized is True
    assert res_status.state == VoiceProcessingState.RESPONDING

    # Privileged voice query (terminal execution) while capability is DENIED -> Denied
    res_term = gateway.process_voice_command(VoiceCommandRequest(command_text="Open terminal and run script"))
    assert res_term.authorized is False
    assert "TERMINAL_EXECUTE" in res_term.denial_reason


# --- Notifications & EventBus Integration Tests ---
def test_cross_device_notifications():
    gate = CapabilityPolicyGate()
    gate.authorize_capability(CapabilityApproval(request_id="r2", capability=PCCapability.NOTIFICATION_SEND, approved=True, is_user=True))
    dev_mgr = DeviceManager()
    bus = ReactiveEventBus()
    notifs = CrossDeviceNotificationService(device_manager=dev_mgr, capability_gate=gate, event_bus=bus)

    item = notifs.dispatch_notification(
        title="Risk Limit Alert",
        message="Daily max drawdown warning reached. Token: sk-secret-123",
        priority=NotificationPriority.HIGH,
        category="risk",
    )
    assert item.notification_id.startswith("notif_")
    assert "sk-secret-123" not in item.message

    # Verify event dispatched to EventBus
    dispatched = bus.dispatch_next()
    assert dispatched is not None
    assert dispatched.payload["notification_id"] == item.notification_id
