"""Unit tests for Step 8 Risk Engine and Engagement Modes."""

import pytest
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.execution.risk_engine import RiskEngine
from trading_ai.schemas.enums import EngagementMode, PositionStatus
from trading_ai.schemas.execution import ExecutionPolicy, Position
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


@pytest.fixture
def discipline_engine(repo):
    return TradingDisciplineEngine(repo)


@pytest.fixture
def risk_engine(repo, discipline_engine):
    return RiskEngine(repo, discipline_engine)


def test_risk_engine_no_stop_loss(risk_engine):
    policy = ExecutionPolicy()
    risk = risk_engine.assess_risk_and_size(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=None,
        target_price=1.0950,
        account_equity=10000.0,
        policy=policy,
    )
    assert risk.is_risk_acceptable is False
    assert any("NO_VALID_STOP_LOSS" in w for w in risk.warnings)


def test_risk_engine_position_sizing_and_rr(risk_engine):
    policy = ExecutionPolicy(min_rr_ratio=2.0)
    # Entry 1.0850, SL 1.0800 (stop distance 0.0050), Target 1.0950 (reward 0.0100) -> RR 1:2.0
    risk = risk_engine.assess_risk_and_size(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0800,
        target_price=1.0950,
        account_equity=10000.0,
        policy=policy,
        engagement_mode=EngagementMode.CALCULATED_AGGRESSION,  # 100% base
    )
    assert risk.is_risk_acceptable is True
    assert risk.is_rr_acceptable is True
    assert risk.risk_reward_ratio == 2.0
    # 1.0% of $10,000 = $100 risk amount
    # $100 / 0.0050 stop distance = 20,000 units
    assert pytest.approx(risk.calculated_quantity, rel=1e-4) == 20000.0
    assert risk.rounded_quantity == 20000.0


def test_risk_engine_defensive_mode_scaling(risk_engine):
    policy = ExecutionPolicy(defensive_size_pct=0.25)
    risk = risk_engine.assess_risk_and_size(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0800,
        target_price=1.0950,
        account_equity=10000.0,
        policy=policy,
        engagement_mode=EngagementMode.DEFENSIVE,
    )
    assert risk.sizing_factor == 0.25
    # Base risk 1.0% scaled to 0.25% ($25 risk)
    # $25 / 0.0050 = 5,000 units
    assert risk.rounded_quantity == 5000.0


def test_risk_engine_risk_off_mode_blocks_trade(risk_engine):
    policy = ExecutionPolicy()
    risk = risk_engine.assess_risk_and_size(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        stop_loss=1.0800,
        target_price=1.0950,
        account_equity=10000.0,
        policy=policy,
        engagement_mode=EngagementMode.RISK_OFF,
    )
    assert risk.is_risk_acceptable is False
    assert risk.sizing_factor == 0.0
    assert any("ENGAGEMENT_RISK_OFF" in w for w in risk.warnings)


def test_risk_engine_pyramiding_validation_blocks_losing_position(risk_engine):
    policy = ExecutionPolicy()
    losing_pos = Position(
        proposal_id="prop_lose",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        current_quantity=10000.0,
        initial_quantity=10000.0,
        initial_sl=1.0800,
        current_sl=1.0800,
        tp1=1.0925,
        final_target=1.0950,
        status=PositionStatus.OPEN,
        unrealized_pnl=-50.0,  # Negative PnL
        sl_moved_to_entry=False,
    )

    risk = risk_engine.assess_risk_and_size(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0820,
        stop_loss=1.0780,
        target_price=1.0950,
        account_equity=10000.0,
        policy=policy,
        engagement_mode=EngagementMode.PYRAMIDING,
        existing_positions=[losing_pos],
    )
    assert risk.is_risk_acceptable is False
    assert any("NO_PYRAMIDING_ON_LOSING_POSITION" in w for w in risk.warnings)


def test_risk_engine_pyramiding_allowed_on_winner(risk_engine):
    policy = ExecutionPolicy()
    winning_pos = Position(
        proposal_id="prop_win",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0800,
        current_quantity=10000.0,
        initial_quantity=10000.0,
        initial_sl=1.0750,
        current_sl=1.0800,  # Protected SL at entry
        tp1=1.0875,
        final_target=1.0950,
        status=PositionStatus.OPEN,
        unrealized_pnl=100.0,  # Positive PnL
        sl_moved_to_entry=True,
    )

    risk = risk_engine.assess_risk_and_size(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0890,
        stop_loss=1.0840,
        target_price=1.0990,
        account_equity=10000.0,
        policy=policy,
        engagement_mode=EngagementMode.PYRAMIDING,
        existing_positions=[winning_pos],
    )
    assert risk.is_risk_acceptable is True
    assert risk.sizing_factor == 0.5
