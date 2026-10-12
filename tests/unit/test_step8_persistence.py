"""Unit tests for Step 8 restart recovery and database persistence."""

import os
import tempfile
import pytest
from trading_ai.schemas.enums import (
    EngagementMode,
    ExecutionMode,
    OrderStatus,
    OrderType,
    PositionManagementAction,
    PositionStatus,
    ProposalLifecycleState,
)
from trading_ai.schemas.execution import (
    ExecutionIntent,
    ExecutionPolicy,
    Order,
    Position,
    PositionEvent,
    TradeProposal,
    TradeThesis,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


def test_step8_restart_persistence():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # 1. Initialize first DB connection and write records
        db1 = DatabaseManager(db_path)
        repo1 = Repository(db1)

        prop = TradeProposal(
            proposal_id="prop_restart_1",
            symbol="BTC/USD",
            direction="BUY",
            entry_price=60000.0,
            hard_stop_loss=58500.0,
            tp1=62250.0,
            final_target=63000.0,
            invalidation_condition="Below 58500",
            engagement_mode=EngagementMode.CALCULATED_AGGRESSION,
            trade_thesis=TradeThesis(summary="BTC breakout thesis", invalidation_premise="Break below 58500"),
            lifecycle_state=ProposalLifecycleState.APPROVED,
        )
        repo1.save_trade_proposal(prop)

        policy = ExecutionPolicy(
            user_id="default_user",
            execution_mode=ExecutionMode.MANUAL,
            min_rr_ratio=2.5,
        )
        repo1.save_execution_policy(policy)

        order = Order(
            order_id="ord_restart_1",
            intent_id="int_restart_1",
            proposal_id="prop_restart_1",
            symbol="BTC/USD",
            order_type=OrderType.MARKET,
            direction="BUY",
            quantity=0.5,
            status=OrderStatus.FILLED,
            average_fill_price=60010.0,
        )
        repo1.save_order(order)

        pos = Position(
            position_id="pos_restart_1",
            proposal_id="prop_restart_1",
            symbol="BTC/USD",
            direction="BUY",
            entry_price=60010.0,
            current_quantity=0.5,
            initial_quantity=0.5,
            initial_sl=58500.0,
            current_sl=58500.0,
            tp1=62250.0,
            final_target=63000.0,
            status=PositionStatus.OPEN,
        )
        repo1.save_position(pos)

        ev = PositionEvent(
            event_id="ev_restart_1",
            position_id="pos_restart_1",
            action=PositionManagementAction.INITIAL_SL_SET,
            price=60010.0,
            quantity_affected=0.5,
            notes="Position opened on BTC/USD",
        )
        repo1.save_position_event(ev)

        # 2. Simulate complete restart: new DB connection and Repository instance
        db2 = DatabaseManager(db_path)
        repo2 = Repository(db2)

        recovered_prop = repo2.get_trade_proposal("prop_restart_1")
        assert recovered_prop is not None
        assert recovered_prop.symbol == "BTC/USD"
        assert recovered_prop.lifecycle_state == ProposalLifecycleState.APPROVED

        recovered_policy = repo2.get_execution_policy("default_user")
        assert recovered_policy is not None
        assert recovered_policy.min_rr_ratio == 2.5

        recovered_order = repo2.get_order("ord_restart_1")
        assert recovered_order is not None
        assert recovered_order.status == OrderStatus.FILLED
        assert recovered_order.average_fill_price == 60010.0

        recovered_pos = repo2.get_position("pos_restart_1")
        assert recovered_pos is not None
        assert recovered_pos.status == PositionStatus.OPEN
        assert recovered_pos.current_sl == 58500.0

        events = repo2.get_position_events("pos_restart_1")
        assert len(events) == 1
        assert events[0].action == PositionManagementAction.INITIAL_SL_SET

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
