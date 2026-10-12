"""
Phase C: Step 8 Final Hardening Targeted Test Suite.

Covers all 22 mandatory requirements:
- Environment isolation (1-4)
- Live protection (5-10)
- Paper isolation (11-14)
- Restart safety (15-16)
- Network safety (17-19)
- Audit integrity (20-22)
"""

import os
import pytest
from datetime import datetime, timezone

from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.execution.broker_adapter import LiveBrokerAdapter, PaperBrokerAdapter
from trading_ai.execution.core import HedgeFundExecutionCore
from trading_ai.execution.execution_gate import ExecutionGate
from trading_ai.execution.position_manager import PositionManager
from trading_ai.execution.risk_engine import RiskEngine
from trading_ai.schemas.enums import (
    AlertQuality,
    ConfluenceCategory,
    EngagementMode,
    ExecutionEnvironment,
    ExecutionMode,
    OrderStatus,
    OrderType,
    PositionStatus,
    ProposalLifecycleState,
)
from trading_ai.schemas.execution import (
    ExecutionIntent,
    ExecutionPolicy,
    Order,
    Position,
    RiskAssessment,
    TradeConfluence,
    TradeProposal,
    TradeThesis,
)
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
def execution_core(repo, discipline_engine):
    return HedgeFundExecutionCore(repository=repo, discipline_engine=discipline_engine)


def make_valid_proposal(
    proposal_id: str = "prop_test_1",
    symbol: str = "EUR/USD",
    direction: str = "BUY",
    entry_price: float = 1.0850,
    sl: float = 1.0800,
    target: float = 1.0950,
    env: ExecutionEnvironment = ExecutionEnvironment.PAPER,
) -> TradeProposal:
    confluences = [
        TradeConfluence(
            category=ConfluenceCategory.MARKET_STRUCTURE,
            source="MTF",
            independence_group="HTF_BIAS",
            timeframe="1h",
            evidence_reference="Bullish structure",
        ),
        TradeConfluence(
            category=ConfluenceCategory.LECTURE_RULE,
            source="LectureValidator",
            independence_group="RULE_OB",
            timeframe="15m",
            evidence_reference="OB rejection confirmed",
        ),
    ]
    risk = RiskAssessment(
        account_equity=10000.0,
        risk_percentage=1.0,
        risk_amount=100.0,
        entry_price=entry_price,
        stop_loss=sl,
        stop_distance=abs(entry_price - sl) if sl else 0.0,
        stop_distance_pct=0.46,
        target_price=target,
        reward_distance=abs(target - entry_price),
        risk_reward_ratio=2.0,
        calculated_quantity=20000.0,
        rounded_quantity=20000.0,
        engagement_mode=EngagementMode.CALCULATED_AGGRESSION,
        is_rr_acceptable=True,
        is_risk_acceptable=True,
    )
    thesis = TradeThesis(
        summary="EUR/USD long thesis",
        invalidation_premise="Invalid if 1.0800 breaks",
    )
    return TradeProposal(
        proposal_id=proposal_id,
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        hard_stop_loss=sl,
        tp1=1.0925,
        final_target=target,
        invalidation_condition="Below 1.0800",
        proposed_quantity=20000.0,
        confluences=confluences,
        risk_assessment=risk,
        trade_thesis=thesis,
        quality=AlertQuality.HIGH_QUALITY,
        lifecycle_state=ProposalLifecycleState.PROPOSED,
        execution_environment=env,
    )


# ==============================================================================
# 1. Environment Isolation (Tests 1 - 4)
# ==============================================================================

def test_1_paper_cannot_call_live_broker_adapter():
    live_broker = LiveBrokerAdapter(api_key="mock_key", api_secret="mock_sec", account_id="mock_acc")
    paper_order = Order(
        intent_id="int_p1",
        proposal_id="prop_p1",
        symbol="EUR/USD",
        order_type=OrderType.MARKET,
        direction="BUY",
        quantity=100.0,
        execution_environment=ExecutionEnvironment.PAPER,
    )
    status, broker_id, fill_px, meta, err = live_broker.submit_order(paper_order)
    assert status == OrderStatus.REJECTED
    assert "LIVE_ADAPTER_CANNOT_EXECUTE_PAPER" in err
    assert fill_px is None


def test_2_live_cannot_silently_call_paper_broker_adapter():
    paper_broker = PaperBrokerAdapter(simulate_latency_ms=0.0)
    live_order = Order(
        intent_id="int_l1",
        proposal_id="prop_l1",
        symbol="EUR/USD",
        order_type=OrderType.MARKET,
        direction="BUY",
        quantity=100.0,
        execution_environment=ExecutionEnvironment.LIVE,
    )
    status, broker_id, fill_px, meta, err = paper_broker.submit_order(live_order)
    assert status == OrderStatus.REJECTED
    assert "PAPER_BROKER_ERROR: PAPER_ADAPTER_CANNOT_EXECUTE_LIVE" in err
    assert fill_px is None


def test_3_paper_failure_cannot_fall_back_to_live(execution_core, repo):
    prop = make_valid_proposal(proposal_id="prop_fail_p", env=ExecutionEnvironment.PAPER)
    repo.save_trade_proposal(prop)

    # Force error on paper broker
    execution_core.broker_adapter.set_force_error_for_testing("PAPER_SIMULATION_FAILURE")

    success, order, pos, errs = execution_core.execute_proposal("prop_fail_p", user_confirmed=True)
    assert success is False
    assert pos is None
    assert order is not None
    assert order.status == OrderStatus.BROKER_ERROR
    assert order.execution_environment == ExecutionEnvironment.PAPER
    assert order.is_simulated is True
    # Verify Live adapter was NOT used to fill
    assert execution_core.live_broker_adapter._is_connected is False


def test_4_live_failure_cannot_fall_back_to_paper(execution_core, repo):
    # Configure policy for LIVE
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=True,
    )
    execution_core.update_policy(policy)

    prop = make_valid_proposal(proposal_id="prop_fail_l", env=ExecutionEnvironment.LIVE)
    repo.save_trade_proposal(prop)

    # Live adapter is disconnected
    success, order, pos, errs = execution_core.execute_proposal("prop_fail_l", user_confirmed=True)
    assert success is False
    assert pos is None
    assert order is not None
    assert order.status == OrderStatus.BROKER_ERROR
    assert order.execution_environment == ExecutionEnvironment.LIVE
    assert "LIVE_BROKER_DISCONNECTED" in (order.error_message or "")


# ==============================================================================
# 2. Live Protection (Tests 5 - 10)
# ==============================================================================

def test_5_live_disabled_blocks_live_execution(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=False,  # Explicitly disabled
    )
    execution_core.update_policy(policy)

    prop = make_valid_proposal(proposal_id="prop_dis_l", env=ExecutionEnvironment.LIVE)
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_dis_l", user_confirmed=True)
    assert success is False
    assert any("LIVE_EXECUTION_DISABLED" in e for e in errs)
    assert pos is None


def test_6_auto_alone_cannot_activate_live(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_mode=ExecutionMode.AUTO,
        auto_execution_enabled=True,
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=False,  # auto enabled but live disabled
    )
    execution_core.update_policy(policy)

    prop = make_valid_proposal(proposal_id="prop_auto_l", env=ExecutionEnvironment.LIVE)
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_auto_l", user_confirmed=False)
    assert success is False
    assert any("LIVE_EXECUTION_DISABLED" in e for e in errs)


def test_7_risk_failure_blocks_live(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=True,
    )
    execution_core.update_policy(policy)

    # Proposal with unacceptable risk
    prop = make_valid_proposal(proposal_id="prop_risk_fail", env=ExecutionEnvironment.LIVE)
    prop.risk_assessment.is_risk_acceptable = False
    prop.risk_assessment.warnings = ["RISK_EXCEEDS_MAX_CAP"]
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_risk_fail", user_confirmed=True)
    assert success is False
    assert any("RISK_EXCEEDS_MAX_CAP" in e for e in errs)


def test_8_missing_sl_blocks_live(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=True,
    )
    execution_core.update_policy(policy)

    prop = make_valid_proposal(proposal_id="prop_no_sl", env=ExecutionEnvironment.LIVE)
    prop.hard_stop_loss = 0.0  # Missing / zero SL
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_no_sl", user_confirmed=True)
    assert success is False
    assert any("NO_VALID_STOP_LOSS" in e for e in errs)


def test_9_invalid_rr_blocks_live(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=True,
    )
    execution_core.update_policy(policy)

    prop = make_valid_proposal(proposal_id="prop_bad_rr", env=ExecutionEnvironment.LIVE)
    prop.risk_assessment.is_rr_acceptable = False
    prop.risk_assessment.risk_reward_ratio = 1.0  # Below 1:2.0
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_bad_rr", user_confirmed=True)
    assert success is False
    assert any("RR_UNACCEPTABLE" in e for e in errs)


def test_10_invalid_proposal_state_blocks_live(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=True,
    )
    execution_core.update_policy(policy)

    prop = make_valid_proposal(proposal_id="prop_rejected", env=ExecutionEnvironment.LIVE)
    prop.lifecycle_state = ProposalLifecycleState.REJECTED
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_rejected", user_confirmed=True)
    assert success is False
    assert any("INVALID_STATE" in e for e in errs)


# ==============================================================================
# 3. Paper Isolation (Tests 11 - 14)
# ==============================================================================

def test_11_auto_plus_paper_remains_paper(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_mode=ExecutionMode.AUTO,
        auto_execution_enabled=True,
        execution_environment=ExecutionEnvironment.PAPER,
    )
    execution_core.update_policy(policy)

    prop = make_valid_proposal(proposal_id="prop_auto_paper", env=ExecutionEnvironment.PAPER)
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_auto_paper", user_confirmed=False)
    assert success is True
    assert order.execution_environment == ExecutionEnvironment.PAPER
    assert order.is_simulated is True
    assert pos.execution_environment == ExecutionEnvironment.PAPER


def test_12_manual_plus_paper_remains_paper(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_mode=ExecutionMode.MANUAL,
        auto_execution_enabled=False,
        execution_environment=ExecutionEnvironment.PAPER,
    )
    execution_core.update_policy(policy)

    prop = make_valid_proposal(proposal_id="prop_man_paper", env=ExecutionEnvironment.PAPER)
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_man_paper", user_confirmed=True)
    assert success is True
    assert order.execution_environment == ExecutionEnvironment.PAPER
    assert order.is_simulated is True
    assert pos.execution_environment == ExecutionEnvironment.PAPER


def test_13_paper_pnl_cannot_modify_live_state(execution_core, repo, discipline_engine):
    # Verify discipline state is initially clean
    initial_state = repo.get_discipline_state("default_user")
    initial_loss = initial_state.daily_realized_loss if initial_state else 0.0

    # Create parent proposal and close a losing paper position
    prop = make_valid_proposal(proposal_id="prop_paper_iso", env=ExecutionEnvironment.PAPER)
    repo.save_trade_proposal(prop)

    pos = Position(
        position_id="pos_paper_iso",
        proposal_id="prop_paper_iso",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        current_quantity=10000.0,
        initial_quantity=10000.0,
        initial_sl=1.0800,
        current_sl=1.0800,
        tp1=1.0925,
        final_target=1.0950,
        execution_environment=ExecutionEnvironment.PAPER,
    )
    repo.save_position(pos)

    # Hit stop loss on paper position
    execution_core.position_manager.update_position_on_price(pos, current_price=1.0790)

    # Live discipline state must remain untouched
    after_state = repo.get_discipline_state("default_user")
    after_loss = after_state.daily_realized_loss if after_state else 0.0
    assert after_loss == initial_loss


def test_14_paper_positions_cannot_modify_live_positions(repo):
    prop = make_valid_proposal(proposal_id="prop_p", env=ExecutionEnvironment.PAPER)
    repo.save_trade_proposal(prop)

    paper_pos = Position(
        position_id="pos_paper_only",
        proposal_id="prop_p",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        current_quantity=1000.0,
        initial_quantity=1000.0,
        initial_sl=1.0800,
        current_sl=1.0800,
        tp1=1.0925,
        final_target=1.0950,
        execution_environment=ExecutionEnvironment.PAPER,
    )
    repo.save_position(paper_pos)

    live_positions = repo.get_positions(environment=ExecutionEnvironment.LIVE)
    assert not any(p.position_id == "pos_paper_only" for p in live_positions)

    live_open = repo.get_open_positions(environment=ExecutionEnvironment.LIVE)
    assert not any(p.position_id == "pos_paper_only" for p in live_open)


# ==============================================================================
# 4. Restart Safety (Tests 15 - 16)
# ==============================================================================

def test_15_paper_remains_paper_after_restart(tmp_path):
    db_file = tmp_path / "restart_test.db"
    db_manager1 = DatabaseManager(str(db_file))
    repo1 = Repository(db_manager1)
    core1 = HedgeFundExecutionCore(repository=repo1, discipline_engine=TradingDisciplineEngine(repo1))

    policy1 = core1.get_or_create_policy("trader_1")
    assert policy1.execution_environment == ExecutionEnvironment.PAPER
    assert policy1.live_execution_enabled is False

    # Simulate restart by instantiating new manager/repo on the same DB file
    db_manager2 = DatabaseManager(str(db_file))
    repo2 = Repository(db_manager2)
    core2 = HedgeFundExecutionCore(repository=repo2, discipline_engine=TradingDisciplineEngine(repo2))

    policy2 = core2.get_or_create_policy("trader_1")
    assert policy2.execution_environment == ExecutionEnvironment.PAPER
    assert policy2.live_execution_enabled is False


def test_16_execution_state_survives_restart(tmp_path):
    db_file = tmp_path / "persistence_test.db"
    db_mgr1 = DatabaseManager(str(db_file))
    repo1 = Repository(db_mgr1)

    prop = make_valid_proposal(proposal_id="prop_persist", env=ExecutionEnvironment.PAPER)
    repo1.save_trade_proposal(prop)

    order = Order(
        order_id="ord_persist",
        intent_id="int_persist",
        proposal_id="prop_persist",
        symbol="EUR/USD",
        order_type=OrderType.MARKET,
        direction="BUY",
        quantity=1000.0,
        status=OrderStatus.FILLED,
        average_fill_price=1.0852,
        execution_environment=ExecutionEnvironment.PAPER,
        is_simulated=True,
    )
    repo1.save_order(order)

    # Reconnect
    db_mgr2 = DatabaseManager(str(db_file))
    repo2 = Repository(db_mgr2)

    loaded_prop = repo2.get_trade_proposal("prop_persist")
    assert loaded_prop is not None
    assert loaded_prop.execution_environment == ExecutionEnvironment.PAPER

    loaded_order = repo2.get_order("ord_persist")
    assert loaded_order is not None
    assert loaded_order.execution_environment == ExecutionEnvironment.PAPER
    assert loaded_order.is_simulated is True
    assert loaded_order.average_fill_price == 1.0852


# ==============================================================================
# 5. Network Safety (Tests 17 - 19)
# ==============================================================================

def test_17_network_failure_produces_unknown(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=True,
    )
    execution_core.update_policy(policy)

    # Force timeout on live broker
    execution_core.live_broker_adapter.set_force_timeout_for_testing(True)

    prop = make_valid_proposal(proposal_id="prop_timeout", env=ExecutionEnvironment.LIVE)
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_timeout", user_confirmed=True)
    assert success is False
    assert order is not None
    assert order.status == OrderStatus.UNKNOWN
    assert "NETWORK_ERROR_UNCERTAIN_STATE" in (order.error_message or "")


def test_18_unknown_never_becomes_filled(execution_core, repo):
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_environment=ExecutionEnvironment.LIVE,
        live_execution_enabled=True,
    )
    execution_core.update_policy(policy)
    execution_core.live_broker_adapter.set_force_timeout_for_testing(True)

    prop = make_valid_proposal(proposal_id="prop_unknown_check", env=ExecutionEnvironment.LIVE)
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_unknown_check", user_confirmed=True)
    assert success is False
    assert pos is None
    # Crucial safety invariant: UNKNOWN must NOT create open position or claim FILLED
    assert order.status == OrderStatus.UNKNOWN
    assert order.status != OrderStatus.FILLED
    open_positions = repo.get_open_positions()
    assert not any(p.proposal_id == "prop_unknown_check" for p in open_positions)


def test_19_duplicate_retry_is_blocked(execution_core, repo):
    prop = make_valid_proposal(proposal_id="prop_dedup", env=ExecutionEnvironment.PAPER)
    repo.save_trade_proposal(prop)

    # First execution succeeds
    success1, order1, pos1, errs1 = execution_core.execute_proposal("prop_dedup", user_confirmed=True)
    assert success1 is True

    # Immediate second execution attempt with same proposal must be blocked
    success2, order2, pos2, errs2 = execution_core.execute_proposal("prop_dedup", user_confirmed=True)
    assert success2 is False
    assert any("DUPLICATE" in e or "INVALID_STATE" in e for e in errs2)


# ==============================================================================
# 6. Audit & Representation (Tests 20 - 22)
# ==============================================================================

def test_20_execution_records_contain_environment(execution_core, repo):
    prop = make_valid_proposal(proposal_id="prop_audit_env", env=ExecutionEnvironment.PAPER)
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_audit_env", user_confirmed=True)
    assert success is True

    # Check Proposal
    p_rec = repo.get_trade_proposal("prop_audit_env")
    assert p_rec.execution_environment == ExecutionEnvironment.PAPER

    # Check Order
    o_rec = repo.get_order(order.order_id)
    assert o_rec.execution_environment == ExecutionEnvironment.PAPER

    # Check Position
    pos_rec = repo.get_position(pos.position_id)
    assert pos_rec.execution_environment == ExecutionEnvironment.PAPER


def test_21_paper_fill_is_marked_as_simulated(execution_core, repo):
    prop = make_valid_proposal(proposal_id="prop_audit_sim", env=ExecutionEnvironment.PAPER)
    repo.save_trade_proposal(prop)

    success, order, pos, errs = execution_core.execute_proposal("prop_audit_sim", user_confirmed=True)
    assert success is True
    assert order.is_simulated is True
    assert order.execution_environment == ExecutionEnvironment.PAPER


def test_22_live_fill_cannot_be_represented_without_verified_adapter_response(execution_core, repo):
    # Without a verified fill response from the live adapter, a LIVE order cannot be marked FILLED
    order = Order(
        intent_id="int_live_unverified",
        proposal_id="prop_live_unverified",
        symbol="EUR/USD",
        order_type=OrderType.MARKET,
        direction="BUY",
        quantity=100.0,
        execution_environment=ExecutionEnvironment.LIVE,
    )
    # Adapter is disconnected
    status, broker_id, fill_px, meta, err = execution_core.live_broker_adapter.submit_order(order)
    assert status != OrderStatus.FILLED
    assert status == OrderStatus.BROKER_ERROR
    assert fill_px is None
