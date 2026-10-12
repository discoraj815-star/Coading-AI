"""Unit tests for Step 7 TradingDisciplineEngine and Trade Confirmation Gate."""

import pytest
from datetime import datetime, timezone, timedelta
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.schemas.companion import (
    DisciplineConfig,
    DisciplineState,
    TradeIntentRequest,
)
from trading_ai.schemas.enums import MarketBias, MarketDirection, SetupStatus
from trading_ai.schemas.market import CurrentSituationSnapshot, MarketSetup
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


@pytest.fixture
def discipline_engine(repo):
    return TradingDisciplineEngine(repo)


def test_default_discipline_config_unconfigured(discipline_engine):
    state, violations = discipline_engine.evaluate_discipline_status("user_1")
    assert state.config.daily_loss_limit is None
    assert state.config.max_daily_trades is None
    assert state.config.stop_loss_required is True
    assert len(violations) == 0


def test_trade_confirmation_gate_stop_loss_missing(discipline_engine, repo):
    intent = TradeIntentRequest(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=None,  # Missing mandatory stop loss
    )
    res = discipline_engine.check_trade_confirmation_gate(intent, "user_1")
    assert res.status == "USER_CONFIRMATION_REQUIRED"
    assert res.can_proceed is False
    assert res.checklist["stop_loss_defined"] is False
    assert any("STOP_LOSS_REQUIRED" in w for w in res.warnings)


def test_trade_confirmation_gate_risk_limit_exceeded(discipline_engine, repo):
    # Configure 2% max risk per trade
    cfg = DisciplineConfig(
        user_id="user_2",
        stop_loss_required=True,
        max_risk_per_trade_pct=2.0,
    )
    discipline_engine.update_config("user_2", cfg)

    # Intent with 5% risk ($500 on $10,000 balance)
    intent = TradeIntentRequest(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0800,
        risk_amount=500.0,
        account_balance=10000.0,
    )
    res = discipline_engine.check_trade_confirmation_gate(intent, "user_2")
    assert res.status == "USER_CONFIRMATION_REQUIRED"
    assert res.can_proceed is False
    assert res.is_risk_acceptable is False
    assert any("RISK_LIMIT_EXCEEDED" in w for w in res.warnings)


def test_trade_confirmation_gate_daily_loss_exceeded(discipline_engine, repo):
    cfg = DisciplineConfig(
        user_id="user_3",
        stop_loss_required=True,
        daily_loss_limit=300.0,
    )
    discipline_engine.update_config("user_3", cfg)

    # Record trade with $350 loss
    discipline_engine.record_trade_execution("EUR/USD", user_id="user_3", realized_loss=350.0)

    intent = TradeIntentRequest(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0820,
        risk_amount=50.0,
        account_balance=10000.0,
    )
    res = discipline_engine.check_trade_confirmation_gate(intent, "user_3")
    assert res.can_proceed is False
    assert any("DAILY_LOSS_EXCEEDED" in w for w in res.warnings)


def test_overtrading_rapid_window_trigger(discipline_engine):
    cfg = DisciplineConfig(
        user_id="user_4",
        overtrading_trades_threshold=3,
        trades_window_minutes=10,
    )
    discipline_engine.update_config("user_4", cfg)

    # Execute 3 trades quickly
    discipline_engine.record_trade_execution("EUR/USD", user_id="user_4")
    discipline_engine.record_trade_execution("EUR/USD", user_id="user_4")
    discipline_engine.record_trade_execution("EUR/USD", user_id="user_4")

    state, violations = discipline_engine.evaluate_discipline_status("user_4")
    assert any("OVERTRADING_TRIGGERED" in v for v in violations)


def test_trade_confirmation_gate_valid_intent(discipline_engine, repo):
    # Set up active confirmed setup in repository
    setup = MarketSetup(
        setup_id="stp_valid",
        symbol="EUR/USD",
        setup_type="ICT Order Block Long",
        timeframe="15m",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.CONFIRMED,
    )
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Bullish",
        active_confirmed_setup=setup,
    )
    repo.save_current_situation(sit)

    cfg = DisciplineConfig(
        user_id="user_5",
        stop_loss_required=True,
        max_risk_per_trade_pct=2.0,
        daily_loss_limit=500.0,
    )
    discipline_engine.update_config("user_5", cfg)

    intent = TradeIntentRequest(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0830,
        risk_amount=100.0,  # 1% risk on 10k
        account_balance=10000.0,
    )
    res = discipline_engine.check_trade_confirmation_gate(intent, "user_5")
    assert res.status == "USER_CONFIRMATION_REQUIRED"
    assert res.can_proceed is True
    assert res.is_discipline_passed is True
    assert res.is_setup_aligned is True
    assert res.is_risk_acceptable is True
    assert "TRADE INTENT VALIDATED" in res.recommended_action
