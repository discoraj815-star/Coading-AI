"""Unit tests for Step 8 Position Manager: TP1 partial booking, SL-to-entry, trailing SL, and close."""

import pytest
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.execution.position_manager import PositionManager
from trading_ai.schemas.enums import (
    ExecutionEnvironment,
    PositionManagementAction,
    PositionStatus,
)
from trading_ai.schemas.execution import Position, TradeProposal, TradeThesis
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
def position_manager(repo, discipline_engine):
    return PositionManager(repo, discipline_engine)


def save_dummy_proposal(repo, proposal_id, symbol="EUR/USD"):
    prop = TradeProposal(
        proposal_id=proposal_id,
        symbol=symbol,
        direction="BUY",
        entry_price=1.0850,
        hard_stop_loss=1.0800,
        tp1=1.0925,
        final_target=1.0950,
        invalidation_condition="Below 1.0800",
        trade_thesis=TradeThesis(summary="Test thesis", invalidation_premise="Stop below 1.0800"),
    )
    repo.save_trade_proposal(prop)


def test_position_manager_tp1_partial_booking_and_sl_to_entry(position_manager, repo):
    save_dummy_proposal(repo, "prop_tp1")
    pos = Position(
        position_id="pos_tp1_test",
        proposal_id="prop_tp1",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        current_quantity=10000.0,
        initial_quantity=10000.0,
        initial_sl=1.0800,
        current_sl=1.0800,
        tp1=1.0925,  # 1.5R (0.0075 reward)
        final_target=1.0950,
        status=PositionStatus.OPEN,
    )
    repo.save_position(pos)

    # Price reaches TP1 (1.0925)
    updated_pos, events = position_manager.update_position_on_price(pos, current_price=1.0926)

    assert updated_pos.tp1_reached is True
    assert updated_pos.current_quantity == 5000.0  # 50% booked
    assert updated_pos.realized_pnl > 0
    # Stop loss moved to Entry
    assert updated_pos.sl_moved_to_entry is True
    assert updated_pos.current_sl == 1.0850

    actions = [e.action for e in events]
    assert PositionManagementAction.TP1_REACHED_PARTIAL_BOOKED in actions
    assert PositionManagementAction.SL_MOVED_TO_ENTRY in actions


def test_position_manager_stop_loss_hit(position_manager, repo):
    save_dummy_proposal(repo, "prop_sl")
    pos = Position(
        position_id="pos_sl_test",
        proposal_id="prop_sl",
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
        execution_environment=ExecutionEnvironment.LIVE,
    )
    repo.save_position(pos)

    # Price drops to 1.0795 (below SL 1.0800)
    updated_pos, events = position_manager.update_position_on_price(pos, current_price=1.0795)

    assert updated_pos.status == PositionStatus.CLOSED
    assert updated_pos.current_quantity == 0.0
    assert updated_pos.realized_pnl < 0
    assert PositionManagementAction.POSITION_CLOSED in [e.action for e in events]

    # Verify Discipline Engine tracked realized loss for LIVE position
    state = repo.get_discipline_state("default_user")
    assert state.daily_realized_loss > 0


def test_position_manager_paper_pnl_never_alters_live_discipline(position_manager, repo):
    save_dummy_proposal(repo, "prop_paper_sl")
    pos = Position(
        position_id="pos_paper_sl_test",
        proposal_id="prop_paper_sl",
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
        execution_environment=ExecutionEnvironment.PAPER,
    )
    repo.save_position(pos)

    # Price drops to 1.0795 (below SL 1.0800)
    updated_pos, events = position_manager.update_position_on_price(pos, current_price=1.0795)

    assert updated_pos.status == PositionStatus.CLOSED
    assert updated_pos.current_quantity == 0.0
    assert updated_pos.realized_pnl < 0

    # B7 Isolation: Discipline Engine state for live user must NOT be altered by paper loss
    state = repo.get_discipline_state("default_user")
    assert state is None or state.daily_realized_loss == 0.0


def test_position_manager_manual_close_and_update_sl(position_manager, repo):
    save_dummy_proposal(repo, "prop_man")
    pos = Position(
        position_id="pos_manual_test",
        proposal_id="prop_man",
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
    )
    repo.save_position(pos)

    # Update SL manually
    mod_pos, err = position_manager.manual_update_sl("pos_manual_test", new_sl=1.0825)
    assert err is None
    assert mod_pos.current_sl == 1.0825

    # Manual exit
    closed_pos, err = position_manager.manual_close_position("pos_manual_test", current_price=1.0890)
    assert err is None
    assert closed_pos.status == PositionStatus.CLOSED
    assert closed_pos.realized_pnl > 0
