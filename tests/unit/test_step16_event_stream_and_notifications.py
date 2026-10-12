"""Unit tests for Step 16 Event Stream, Deduplication, and Multi-Channel Notifications."""
from datetime import datetime, timezone
import pytest
from trading_ai.events.gateway import WebSocketGateway
from trading_ai.events.router import EventRouter
from trading_ai.notifications.orchestrator import NotificationOrchestrator
from trading_ai.notifications.preferences import PreferencesManager
from trading_ai.notifications.templates import NotificationTemplateFormatter
from trading_ai.notifications.voice import CloudVoiceSynthesizer
from trading_ai.schemas.cloud import CloudStreamEvent, UserNotificationPreferences
from trading_ai.schemas.enums import (
    EventPriorityLevel,
    NotificationChannel,
    NotificationDeliveryStatus,
    NotificationPrivacyLevel,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


@pytest.fixture
def orchestrator(repo):
    gateway = WebSocketGateway(repo)
    synth = CloudVoiceSynthesizer()
    return NotificationOrchestrator(repo, gateway, voice_synthesizer=synth)


def test_notification_template_formatter_hindi_and_epistemic_layers():
    formatter = NotificationTemplateFormatter(greeting_name="Alam")
    title, hindi, english = formatter.format_alert(
        event_type="SETUP_CONFIRMED",
        priority=EventPriorityLevel.P3_CONFIRMED_SETUP,
        symbol="EURUSD",
        details={
            "pattern_name": "Bullish FVG",
            "direction": "BUY",
            "timeframe": "15m",
            "poi_level": 1.0850,
            "invalidation_level": 1.0820,
        },
        privacy_level=NotificationPrivacyLevel.NORMAL,
        verbosity="detailed",
    )

    assert "Hello Alam" in title
    assert "EURUSD" in title
    # Check all four epistemic layers in Hindi text
    assert "OBSERVED (देखा गया)" in hindi
    assert "LEARNED (सीखा गया)" in hindi
    assert "INFERRED (निष्कर्ष)" in hindi
    assert "DECIDED (निर्णय)" in hindi
    assert "1.082" in hindi
    # Check English fallback
    assert "OBSERVED" in english
    assert "DECIDED" in english


def test_notification_secure_privacy_mode():
    formatter = NotificationTemplateFormatter(greeting_name="Alam")
    title, hindi, english = formatter.format_alert(
        event_type="SETUP_CONFIRMED",
        priority=EventPriorityLevel.P3_CONFIRMED_SETUP,
        symbol="EURUSD",
        details={"poi_level": 1.0850},
        privacy_level=NotificationPrivacyLevel.SECURE,
    )

    assert "सुरक्षित विवरण" in hindi or "App खोलकर" in hindi
    # Must NOT leak specific price levels or setup in secure mode
    assert "1.0850" not in hindi
    assert "secure app" in english.lower()


def test_notification_deduplication(orchestrator):
    event = CloudStreamEvent(
        event_type="SETUP_CONFIRMED",
        symbol="EURUSD",
        priority=EventPriorityLevel.P3_CONFIRMED_SETUP,
        confidence=0.95,
        payload={"setup_id": "setup_123", "pattern_name": "FVG", "timeframe": "15m"},
    )

    # First dispatch succeeds
    records_1 = orchestrator.process_event(event)
    assert len(records_1) >= 1

    # Second dispatch with identical setup_id and event_type is deduplicated
    records_2 = orchestrator.process_event(event)
    assert len(records_2) == 0
    assert orchestrator.health()["suppressed_duplicates"] >= 1


def test_preferences_quiet_hours_and_emergency_override(repo):
    prefs_mgr = PreferencesManager(repo)
    prefs = UserNotificationPreferences(
        quiet_hours_enabled=True,
        quiet_hours_start_utc=22,
        quiet_hours_end_utc=4,
        critical_alert_override=True,
    )
    prefs_mgr.save_preferences(prefs)

    # Simulated time at 23:30 UTC (inside quiet hours)
    quiet_time = datetime(2026, 10, 6, 23, 30, tzinfo=timezone.utc)

    # Normal setup (P3) should NOT deliver during quiet hours
    can_deliver_p3 = prefs_mgr.should_deliver(
        EventPriorityLevel.P3_CONFIRMED_SETUP,
        NotificationChannel.PUSH,
        now=quiet_time,
    )
    assert can_deliver_p3 is False

    # Emergency Risk Warning (P2) SHOULD override and deliver
    can_deliver_p2 = prefs_mgr.should_deliver(
        EventPriorityLevel.P2_ACTIVE_POSITION_RISK,
        NotificationChannel.PUSH,
        now=quiet_time,
    )
    assert can_deliver_p2 is True


def test_cloud_voice_synthesis():
    synth = CloudVoiceSynthesizer(voice_type="hindi_natural", voice_speed=1.0)
    res = synth.synthesize_speech("Hello Alam, EURUSD पर confirmation मिला है।")
    assert res["format"] == "audio/mp3"
    assert res["duration_estimated_seconds"] > 0.0
    assert len(res["audio_base64"]) > 10
    assert synth.health()["synthesis_count"] >= 1
