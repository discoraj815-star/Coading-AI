"""Unit tests for Step 8: JARVIS & Chat Execution Core Grounding.

Verifies:
1. Grounded POSITIONS_STATUS responses with live open positions from HedgeFundExecutionCore / Repository.
2. Truthful POSITIONS_STATUS response when no positions are open (0 hallucination).
3. Grounded PROPOSALS_STATUS responses with live trade proposals.
4. Truthful PROPOSALS_STATUS response when no proposals exist.
5. Grounded EXECUTION_STATUS responses with live policy (MANUAL vs AUTO, PAPER vs LIVE, Min RR).
6. Disambiguation: Distinguishes position/proposal/status queries from execution commands (buy/sell).
7. Unified AI Brain parity across Text and Voice modalities.
"""

import pytest
from datetime import datetime, timezone

from trading_ai.chat.chat_service import AIChatService
from trading_ai.companion.core import TradingCompanionCore
from trading_ai.execution.core import HedgeFundExecutionCore
from trading_ai.schemas.enums import (
    EngagementMode,
    ExecutionEnvironment,
    ExecutionMode,
    PositionStatus,
    ProposalLifecycleState,
)
from trading_ai.schemas.execution import (
    ExecutionPolicy,
    Position,
    RiskAssessment,
    TradeProposal,
)
from trading_ai.schemas.unified import (
    InputModality,
    OutputModality,
    UnifiedAIRequest,
)
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    db_file = tmp_path / "test_step8_jarvis.db"
    return Repository(str(db_file))


@pytest.fixture
def companion_core(repo):
    return TradingCompanionCore(repository=repo)


@pytest.fixture
def execution_core(repo, companion_core):
    return HedgeFundExecutionCore(
        repository=repo,
        discipline_engine=companion_core.discipline_engine,
    )


@pytest.fixture
def chat_service(repo, companion_core, execution_core):
    service = AIChatService(
        repository=repo,
        companion_core=companion_core,
        execution_core=execution_core,
    )
    companion_core.set_chat_service(service)
    return service


def test_jarvis_positions_status_with_open_position(repo, execution_core, chat_service):
    """JARVIS must retrieve authoritative open positions and explain them naturally."""
    prop = TradeProposal(
        proposal_id="prop_test_1",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        hard_stop_loss=1.0800,
        tp1=1.0925,
        final_target=1.0950,
        proposed_quantity=10000.0,
        invalidation_condition="Price breaks structural stop loss",
        execution_environment=ExecutionEnvironment.PAPER,
    )
    repo.save_trade_proposal(prop)

    pos = Position(
        position_id="pos_step8_test_1",
        proposal_id="prop_test_1",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        current_quantity=10000.0,
        initial_quantity=10000.0,
        initial_sl=1.0800,
        current_sl=1.0820,
        tp1=1.0925,
        final_target=1.0950,
        status=PositionStatus.OPEN,
        execution_environment=ExecutionEnvironment.PAPER,
    )
    repo.save_position(pos)

    req = UnifiedAIRequest(
        input_text="What open positions do I have right now?",
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        session_id="test_session_step8",
        user_id="default_user",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "POSITIONS_STATUS"
    assert "EUR/USD" in res.response_text
    assert "1.0850" in res.response_text
    assert "1.0820" in res.response_text
    assert "Active Managed Positions" in res.response_text
    assert "EUR/USD" in res.speech_text
    assert "1.0850" in res.speech_text


def test_jarvis_positions_status_when_no_positions(repo, execution_core, chat_service):
    """JARVIS must truthfully state zero open positions without fabricating trades."""
    req = UnifiedAIRequest(
        input_text="Show my active positions",
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        session_id="test_session_step8_empty",
        user_id="default_user",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "POSITIONS_STATUS"
    assert "0 open positions" in res.response_text or "No Open Positions" in res.response_text
    assert "no open positions" in res.speech_text.lower()


def test_jarvis_proposals_status_with_active_proposals(repo, execution_core, chat_service):
    """JARVIS must retrieve recent trade proposals and report risk/reward parameters."""
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
        potential_loss=100.0,
        potential_profit=200.0,
        engagement_mode=EngagementMode.DEFENSIVE,
        is_risk_acceptable=True,
    )
    prop = TradeProposal(
        proposal_id="prop_step8_test_1",
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        hard_stop_loss=1.0800,
        tp1=1.0925,
        final_target=1.0950,
        proposed_quantity=20000.0,
        invalidation_condition="Price breaks structural stop loss",
        risk_assessment=risk,
        lifecycle_state=ProposalLifecycleState.RISK_VALIDATED,
        execution_environment=ExecutionEnvironment.PAPER,
    )
    repo.save_trade_proposal(prop)

    req = UnifiedAIRequest(
        input_text="What are my trade proposals?",
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        session_id="test_session_step8_prop",
        user_id="default_user",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "PROPOSALS_STATUS"
    assert "prop_step8_test_1" in res.response_text
    assert "EUR/USD" in res.response_text
    assert "1.0850" in res.response_text
    assert "1:2.0" in res.response_text
    assert "EUR/USD" in res.speech_text


def test_jarvis_proposals_status_when_no_proposals(repo, execution_core, chat_service):
    """JARVIS must truthfully report no active proposals."""
    req = UnifiedAIRequest(
        input_text="Any active trade proposals?",
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        session_id="test_session_step8_prop_empty",
        user_id="default_user",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "PROPOSALS_STATUS"
    assert "No Trade Proposals" in res.response_text
    assert "no trade proposals" in res.speech_text.lower()


def test_jarvis_execution_mode_and_status(repo, execution_core, chat_service):
    """JARVIS must explain execution mode, environment, and risk gates grounded in policy."""
    policy = ExecutionPolicy(
        user_id="default_user",
        execution_mode=ExecutionMode.MANUAL,
        execution_environment=ExecutionEnvironment.PAPER,
        auto_execution_enabled=False,
        live_execution_enabled=False,
        min_rr_ratio=2.5,
        max_auto_risk_pct=1.0,
    )
    repo.save_execution_policy(policy)

    req = UnifiedAIRequest(
        input_text="What is my current execution mode and risk status?",
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        session_id="test_session_step8_mode",
        user_id="default_user",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "EXECUTION_STATUS"
    assert "MANUAL" in res.response_text
    assert "PAPER" in res.response_text
    assert "2.5" in res.response_text
    assert "MANUAL" in res.speech_text
    assert "PAPER" in res.speech_text


def test_jarvis_voice_audio_synthesis_for_positions_query(repo, execution_core, chat_service):
    """Voice query produces audio_output_url and identical reasoning context as text."""
    prop = TradeProposal(
        proposal_id="prop_voice_1",
        symbol="BTC/USD",
        direction="BUY",
        entry_price=64000.0,
        hard_stop_loss=63000.0,
        tp1=65500.0,
        final_target=67000.0,
        proposed_quantity=0.5,
        invalidation_condition="Price breaks structural stop loss",
        execution_environment=ExecutionEnvironment.PAPER,
    )
    repo.save_trade_proposal(prop)

    pos = Position(
        position_id="pos_voice_test_1",
        proposal_id="prop_voice_1",
        symbol="BTC/USD",
        direction="BUY",
        entry_price=64000.0,
        current_quantity=0.5,
        initial_quantity=0.5,
        initial_sl=63000.0,
        current_sl=63500.0,
        tp1=65500.0,
        final_target=67000.0,
        status=PositionStatus.OPEN,
        execution_environment=ExecutionEnvironment.PAPER,
    )
    repo.save_position(pos)

    req = UnifiedAIRequest(
        input_text="List open positions",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        session_id="test_session_voice_step8",
        user_id="default_user",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "POSITIONS_STATUS"
    assert res.audio_output_url is not None
    assert "BTC/USD" in res.speech_text
    assert "64000" in res.speech_text


def test_disambiguation_positions_query_vs_trade_execution(repo, execution_core, chat_service):
    """Informational questions about positions must never trigger trade proposals."""
    queries = [
        ("What open positions do I have?", "POSITIONS_STATUS"),
        ("Show my positions", "POSITIONS_STATUS"),
        ("What are my trade proposals?", "PROPOSALS_STATUS"),
        ("What is my execution mode?", "EXECUTION_STATUS"),
    ]
    for q_text, expected_intent in queries:
        req = UnifiedAIRequest(
            input_text=q_text,
            input_modality=InputModality.TEXT,
            output_modality=OutputModality.TEXT,
            session_id=f"test_session_disambig_{expected_intent}",
            user_id="default_user",
        )
        res = chat_service.process_unified_request(req)
        assert res.intent == expected_intent, f"Query '{q_text}' expected intent {expected_intent}, got {res.intent}"
        assert res.action_intent is None, f"Query '{q_text}' should not have an action intent"
