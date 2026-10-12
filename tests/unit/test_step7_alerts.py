"""Unit tests for Step 7 ProactiveAlertEngine: Lifecycle, Noise Filtering, and Alert Qualification."""

import pytest
from datetime import datetime, timezone
from trading_ai.companion.alert_engine import ProactiveAlertEngine
from trading_ai.schemas.companion import AlertEvent, DisciplineConfig, DisciplineState
from trading_ai.schemas.enums import (
    AlertDecision,
    AlertLifecycleState,
    AlertPriority,
    AlertQuality,
    AlertType,
    EventImpact,
    EventStatus,
    LevelType,
    MarketBias,
    MarketDirection,
    SetupStatus,
)
from trading_ai.schemas.fundamental import FundamentalEvent
from trading_ai.schemas.market import CurrentSituationSnapshot, MarketLevel, MarketSetup
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


@pytest.fixture
def alert_engine(repo):
    return ProactiveAlertEngine(repo)


def test_setup_confirmed_alert_quality_and_emission(alert_engine, repo):
    setup = MarketSetup(
        setup_id="stp_100",
        symbol="EUR/USD",
        setup_type="Silver Bullet FVG",
        timeframe="15m",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.CONFIRMED,
        confirmation_evidence=["15m FVG fill", "LTF MSS confirmed"],
    )
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Strong bullish expansion",
        active_confirmed_setup=setup,
        setup_confirmation_state="CONFIRMED",
        invalidation_level=1.0820,
    )
    repo.save_current_situation(sit)

    alerts = alert_engine.evaluate_and_generate_alerts("EUR/USD", current_situation=sit)
    assert len(alerts) >= 1
    setup_alert = next(a for a in alerts if "Setup Confirmed" in a.title)

    assert setup_alert.alert_type == AlertType.SETUP_HIGH_PROBABILITY
    assert setup_alert.quality == AlertQuality.HIGH_QUALITY
    assert setup_alert.decision == AlertDecision.ALERT
    assert setup_alert.lifecycle_state == AlertLifecycleState.ALERTED
    assert setup_alert.priority == AlertPriority.HIGH
    assert "Silver Bullet FVG" in setup_alert.message


def test_zero_noise_duplicate_suppression(alert_engine, repo):
    setup = MarketSetup(
        setup_id="stp_101",
        symbol="GBP/USD",
        setup_type="London Breakout",
        timeframe="15m",
        direction=MarketDirection.BEARISH,
        status=SetupStatus.CONFIRMED,
        confirmation_evidence=["Liquidity sweep"],
    )
    sit = CurrentSituationSnapshot(
        symbol="GBP/USD",
        current_price=1.2650,
        overall_bias=MarketBias.BEARISH_BIAS,
        bias_summary="Bearish displacement",
        active_confirmed_setup=setup,
        setup_confirmation_state="CONFIRMED",
    )

    # First evaluation: Emits alert
    alerts1 = alert_engine.evaluate_and_generate_alerts("GBP/USD", current_situation=sit)
    assert len(alerts1) == 1
    assert alerts1[0].decision == AlertDecision.ALERT

    # Second evaluation immediately after with identical setup: Zero-Noise filter SUPPRESSES duplicate
    alerts2 = alert_engine.evaluate_and_generate_alerts("GBP/USD", current_situation=sit)
    assert len(alerts2) == 0  # Emitted list is empty because decision was SUPPRESS


def test_bias_change_detection(alert_engine, repo):
    prev_sit = CurrentSituationSnapshot(
        symbol="BTC/USD",
        current_price=64000.0,
        overall_bias=MarketBias.BEARISH_BIAS,
        bias_summary="Bearish market structure",
    )
    curr_sit = CurrentSituationSnapshot(
        symbol="BTC/USD",
        current_price=66500.0,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Break of structure and change of character upward",
    )

    alerts = alert_engine.evaluate_and_generate_alerts(
        "BTC/USD", current_situation=curr_sit, previous_situation=prev_sit
    )
    bias_alert = next((a for a in alerts if a.alert_type == AlertType.MARKET_BIAS_CHANGE), None)
    assert bias_alert is not None
    assert bias_alert.priority == AlertPriority.HIGH
    assert "from BEARISH_BIAS to BULLISH_BIAS" in bias_alert.message


def test_news_imminent_alert_timing(alert_engine, repo):
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Trending upward",
        next_high_impact_event=FundamentalEvent(
            event_id="evt_cpi_1",
            source="ForexFactory",
            title="CPI m/m",
            currency="USD",
            event_time=datetime.now(timezone.utc),
            impact=EventImpact.HIGH,
            status=EventStatus.SCHEDULED,
        ),
        news_time_distance_minutes=12.0,
        news_risk="HIGH_RISK",
    )

    alerts = alert_engine.evaluate_and_generate_alerts("EUR/USD", current_situation=sit)
    news_alert = next((a for a in alerts if a.alert_type == AlertType.NEWS_IMMINENT), None)
    assert news_alert is not None
    assert news_alert.priority == AlertPriority.CRITICAL  # <= 15 min is CRITICAL
    assert "CPI m/m" in news_alert.message
    assert "12 minutes" in news_alert.message


def test_poi_reaction_alert(alert_engine, repo):
    poi = MarketLevel(
        level_id="poi_1",
        symbol="EUR/USD",
        timeframe="15m",
        price=1.0851,  # within 0.1% of current 1.0850
        level_type=LevelType.OB_HIGH,
        evidence_json={"name": "Key 15m Bullish OB"},
    )
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Rebound expected",
        current_poi=poi,
    )

    alerts = alert_engine.evaluate_and_generate_alerts("EUR/USD", current_situation=sit)
    poi_alert = next((a for a in alerts if a.alert_type == AlertType.POI_REACTION), None)
    assert poi_alert is not None
    assert poi_alert.quality == AlertQuality.MEDIUM_QUALITY
    assert "Key 15m Bullish OB" in poi_alert.message


def test_alert_acknowledgment_lifecycle(alert_engine, repo):
    setup = MarketSetup(
        setup_id="stp_ack",
        symbol="SOL/USD",
        setup_type="Asia Sweep",
        timeframe="15m",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.CONFIRMED,
        confirmation_evidence=["Asian range swept"],
    )
    sit = CurrentSituationSnapshot(
        symbol="SOL/USD",
        current_price=145.0,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Bullish trend",
        active_confirmed_setup=setup,
        setup_confirmation_state="CONFIRMED",
    )

    emitted = alert_engine.evaluate_and_generate_alerts("SOL/USD", current_situation=sit)
    assert len(emitted) >= 1
    alert_id = emitted[0].alert_id

    # Initially unacknowledged
    unacked = repo.get_alerts(symbol="SOL/USD", unacknowledged_only=True)
    assert any(a.alert_id == alert_id for a in unacked)

    # Acknowledge alert
    success = repo.acknowledge_alert(alert_id)
    assert success is True

    # No longer in unacknowledged list
    unacked_after = repo.get_alerts(symbol="SOL/USD", unacknowledged_only=True)
    assert not any(a.alert_id == alert_id for a in unacked_after)

    # Still present in full history
    all_alerts = repo.get_alerts(symbol="SOL/USD", unacknowledged_only=False)
    acked_item = next(a for a in all_alerts if a.alert_id == alert_id)
    assert acked_item.is_acknowledged is True
    assert acked_item.acknowledged_at is not None
