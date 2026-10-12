"""Adversarial and Edge-Case Reliability Tests for Step 16."""
from datetime import datetime, timezone
import pytest
from trading_ai.events.gateway import WebSocketGateway
from trading_ai.notifications.orchestrator import NotificationOrchestrator
from trading_ai.notifications.preferences import PreferencesManager
from trading_ai.notifications.providers import SMSNotificationProvider
from trading_ai.schemas.cloud import CloudStreamEvent, NotificationRecord, UserNotificationPreferences
from trading_ai.schemas.enums import (
    DeviceStatus,
    EventPriorityLevel,
    NotificationChannel,
    NotificationDeliveryStatus,
)
from trading_ai.schemas.security import DeviceRecord
from trading_ai.security.device_manager import DeviceManager
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


def test_revoked_device_websocket_denial(repo):
    dev_mgr = DeviceManager(repo)
    # Register device then revoke it
    dev_mgr._devices["dev_stolen"] = DeviceRecord(
        device_id="dev_stolen",
        device_name="Stolen Phone",
        device_type="phone",
        public_key_or_token_hash="stolen_token_hash",
        status=DeviceStatus.REVOKED,
    )

    gateway = WebSocketGateway(repo, device_manager=dev_mgr)

    # Mock websocket object with close method
    class MockWebSocket:
        def __init__(self):
            self.closed = False
            self.close_code = None

        async def accept(self):
            pass

        async def close(self, code=1000, reason=""):
            self.closed = True
            self.close_code = code

    ws = MockWebSocket()
    import asyncio
    loop = asyncio.new_event_loop()
    connected = loop.run_until_complete(gateway.connect(ws, client_id="test", device_id="dev_stolen"))
    loop.close()

    assert connected is False
    assert ws.closed is True
    assert ws.close_code == 4003


def test_high_rate_spam_deduplication_and_daily_limits(repo):
    gateway = WebSocketGateway(repo)
    prefs_mgr = PreferencesManager(repo)
    prefs = UserNotificationPreferences(
        max_sms_per_day=2,
        sms_enabled=True,
    )
    prefs_mgr.save_preferences(prefs)

    orchestrator = NotificationOrchestrator(repo, gateway, preferences_manager=prefs_mgr)

    # 1. Dispatch 10 events with identical setup_id
    delivered_count = 0
    first_event_count = 0
    for i in range(10):
        evt = CloudStreamEvent(
            event_id=f"evt_{i}",
            event_type="SETUP_CONFIRMED",
            symbol="BTCUSD",
            priority=EventPriorityLevel.P3_CONFIRMED_SETUP,
            confidence=0.95,
            payload={"setup_id": "btc_fvg_001", "pattern_name": "FVG"},
        )
        recs = orchestrator.process_event(evt)
        delivered_count += len(recs)
        if i == 0:
            first_event_count = len(recs)

    # Strict deduplication invariant: only the first event dispatches notifications; all 9 subsequent are suppressed
    assert first_event_count > 0
    assert delivered_count == first_event_count
    assert orchestrator.health()["suppressed_duplicates"] >= 9


def test_provider_outage_resilience(repo):
    sms_provider = SMSNotificationProvider(repo)
    sms_provider.set_available(False)

    rec = NotificationRecord(
        notification_id="notif_test_err",
        event_id="evt_01",
        title="Test Alert",
        message_hindi="परीक्षण",
        message_english="Test",
        channel=NotificationChannel.SMS,
        delivery_status=NotificationDeliveryStatus.QUEUED,
    )

    status = sms_provider.send(rec)
    # Fail-closed provider invariant: marked UNAVAILABLE without crashing
    assert status == NotificationDeliveryStatus.UNAVAILABLE
    assert rec.last_error == "SMS Gateway Provider Unavailable"


def test_offline_client_recovery_smallest_sufficient_context(repo):
    gateway = WebSocketGateway(repo)

    # Persist mix of P5 (informational tick) and P1/P3 (critical) events
    p5_evt = CloudStreamEvent(
        event_id="evt_tick",
        event_type="PRICE_UPDATE",
        priority=EventPriorityLevel.P5_INFORMATIONAL,
        payload={"price": 1.0850},
    )
    p1_evt = CloudStreamEvent(
        event_id="evt_sec",
        event_type="SECURITY_ALERT",
        priority=EventPriorityLevel.P1_SECURITY,
        payload={"action": "UNAUTHORIZED_LOGIN"},
    )
    p3_evt = CloudStreamEvent(
        event_id="evt_setup",
        event_type="SETUP_CONFIRMED",
        priority=EventPriorityLevel.P3_CONFIRMED_SETUP,
        payload={"symbol": "EURUSD"},
    )

    repo.save_stream_event(p5_evt)
    repo.save_stream_event(p1_evt)
    repo.save_stream_event(p3_evt)

    # Replay for client
    replayed = gateway.replay_missed_events("client_phone", limit=10)
    replayed_ids = [e["event_id"] for e in replayed]

    # Invariant: P1 and P3 must be replayed; low-priority ticks (P5) must NOT pollute context!
    assert "evt_sec" in replayed_ids
    assert "evt_setup" in replayed_ids
    assert "evt_tick" not in replayed_ids
