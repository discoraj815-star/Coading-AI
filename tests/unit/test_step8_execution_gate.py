"""Unit tests for Step 8 Execution Gate."""

import pytest
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.execution.execution_gate import ExecutionGate
from trading_ai.schemas.enums import (
    ConfluenceCategory,
    EngagementMode,
    ExecutionMode,
    ProposalLifecycleState,
)
from trading_ai.schemas.execution import (
    ExecutionPolicy,
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
def execution_gate(repo, discipline_engine):
    return ExecutionGate(repo, discipline_engine)


def create_sample_proposal():
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
        entry_price=1.0850,
        stop_loss=1.0800,
        stop_distance=0.0050,
        stop_distance_pct=0.46,
        target_price=1.0950,
        reward_distance=0.0100,
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
        proposal_id="prop_sample_1",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        hard_stop_loss=1.0800,
        tp1=1.0925,
        final_target=1.0950,
        invalidation_condition="Below 1.0800",
        confluences=confluences,
        risk_assessment=risk,
        proposed_quantity=20000.0,
        trade_thesis=thesis,
        lifecycle_state=ProposalLifecycleState.RISK_VALIDATED,
    )


def test_execution_gate_manual_requires_user_confirmation(execution_gate, repo):
    proposal = create_sample_proposal()
    repo.save_trade_proposal(proposal)
    policy = ExecutionPolicy(execution_mode=ExecutionMode.MANUAL)

    # Without user confirmation: rejected
    auth, intent, reasons = execution_gate.validate_and_authorize_execution(
        proposal=proposal,
        policy=policy,
        user_confirmed=False,
    )
    assert auth is False
    assert any("USER_CONFIRMATION_REQUIRED" in r for r in reasons)

    # With user confirmation: approved
    auth, intent, reasons = execution_gate.validate_and_authorize_execution(
        proposal=proposal,
        policy=policy,
        user_confirmed=True,
    )
    assert auth is True
    assert intent is not None
    assert intent.status == "AUTHORIZED"


def test_execution_gate_auto_disabled_blocks_execution(execution_gate, repo):
    proposal = create_sample_proposal()
    repo.save_trade_proposal(proposal)
    policy = ExecutionPolicy(
        execution_mode=ExecutionMode.AUTO,
        auto_execution_enabled=False,  # Disabled
    )

    auth, intent, reasons = execution_gate.validate_and_authorize_execution(
        proposal=proposal,
        policy=policy,
    )
    assert auth is False
    assert any("AUTO_EXECUTION_DISABLED" in r for r in reasons)


def test_execution_gate_auto_enabled_passes(execution_gate, repo):
    proposal = create_sample_proposal()
    repo.save_trade_proposal(proposal)
    policy = ExecutionPolicy(
        execution_mode=ExecutionMode.AUTO,
        auto_execution_enabled=True,
        allowed_symbols=["EUR/USD"],
        max_auto_risk_pct=1.5,
    )

    auth, intent, reasons = execution_gate.validate_and_authorize_execution(
        proposal=proposal,
        policy=policy,
    )
    assert auth is True
    assert intent is not None


def test_execution_gate_anti_duplicate_token_blocks_reexecution(execution_gate, repo):
    proposal = create_sample_proposal()
    repo.save_trade_proposal(proposal)
    policy = ExecutionPolicy(execution_mode=ExecutionMode.MANUAL)

    # First authorization succeeds
    auth1, intent1, _ = execution_gate.validate_and_authorize_execution(
        proposal=proposal,
        policy=policy,
        user_confirmed=True,
    )
    assert auth1 is True

    # Second execution attempt with identical parameters is blocked
    auth2, intent2, reasons2 = execution_gate.validate_and_authorize_execution(
        proposal=proposal,
        policy=policy,
        user_confirmed=True,
    )
    assert auth2 is False
    assert any("DUPLICATE_EXECUTION_BLOCKED" in r for r in reasons2)
