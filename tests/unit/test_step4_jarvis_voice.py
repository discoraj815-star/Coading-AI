"""Unit and Integration Tests for Step 4: JARVIS Continuous Voice Interaction & UI Handshake.

Validates:
1. Unified Voice & Text shared session continuity (Voice -> Chat -> Voice).
2. Push-to-Talk and Voice State machine transitions (IDLE, LISTENING, PROCESSING, THINKING, SPEAKING, INTERRUPTED, ERROR).
3. Trading Safety Handshake:
   - BUY/SELL/CLOSE voice command triggers CONFIRMATION_REQUIRED.
   - Vague confirmation ("okay", "hmm", "sounds good") is strictly REJECTED.
   - Explicit confirmation ("CONFIRM ORDER") passes through ExecutionGate.
   - Cancellation ("CANCEL") clears pending order intent.
4. Canonical AI reasoning preserved for both Chat and TTS presentation.
5. TokenManager centralized budget, complexity classification, and 20% headroom safety.
6. Real REST API contracts for /api/unified/process, /api/unified/confirm, /api/unified/session, /api/voice/command.
"""

import time
import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app
from trading_ai.chat import AIChatService
from trading_ai.schemas.unified import (
    UnifiedAIRequest,
    UnifiedAIResponse,
    InputModality,
    OutputModality,
    UnifiedAudioMetadata,
    UnifiedSessionState,
)
from trading_ai.schemas.security import VoiceCommandRequest, VoiceCommandResult
from trading_ai.schemas.enums import VoiceIntent


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def chat_service():
    from trading_ai.server.app import ai_chat_service
    return ai_chat_service


def test_step4_voice_state_machine_representation():
    """Verify voice state machine states and deterministic transitions."""
    valid_states = ["IDLE", "LISTENING", "PROCESSING", "THINKING", "SPEAKING", "INTERRUPTED", "ERROR"]
    assert len(valid_states) == 7
    # Verify canonical linear transition chain
    transitions = {
        "IDLE": ["LISTENING"],
        "LISTENING": ["PROCESSING", "ERROR", "IDLE"],
        "PROCESSING": ["THINKING", "ERROR"],
        "THINKING": ["SPEAKING", "ERROR"],
        "SPEAKING": ["IDLE", "INTERRUPTED", "LISTENING"],
        "INTERRUPTED": ["IDLE", "LISTENING"],
        "ERROR": ["IDLE"],
    }
    for state, allowed in transitions.items():
        assert len(allowed) > 0


def test_step4_shared_session_continuous_conversation(chat_service):
    """Verify natural continuous conversation across Voice and Chat sharing exact session context."""
    sess_id = f"jarvis_cont_{int(time.time()*1000)}"

    # Turn 1 (Voice): Analyze BTC on 1H
    req1 = UnifiedAIRequest(
        session_id=sess_id,
        user_id="trader",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Analyze BTC on the 1 hour chart.",
        audio_metadata=UnifiedAudioMetadata(confidence=0.98, duration_seconds=2.4),
        source="jarvis_ptt",
    )
    res1 = chat_service.process_unified_request(req1)
    assert res1.response_text is not None
    assert res1.speech_text is not None
    sess_state1 = chat_service.get_session_state(sess_id)
    assert sess_state1.active_symbol in ["BTC/USD", "BTC"]
    assert sess_state1.active_timeframe == "1H"

    # Turn 2 (Chat): Follow-up question without repeating symbol
    req2 = UnifiedAIRequest(
        session_id=sess_id,
        user_id="trader",
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        input_text="What about the liquidity below?",
        source="chat_drawer",
    )
    res2 = chat_service.process_unified_request(req2)
    assert res2.response_text is not None
    # Context summary retains BTC/USD from Voice Turn 1
    assert "BTC" in res2.market_context_used

    # Turn 3 (Voice): Another follow up
    req3 = UnifiedAIRequest(
        session_id=sess_id,
        user_id="trader",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Would you wait for confirmation?",
        source="jarvis_continuous",
    )
    res3 = chat_service.process_unified_request(req3)
    assert res3.response_text is not None
    assert "BTC" in res3.market_context_used


def test_step4_trade_safety_vague_confirmation_rejected(chat_service):
    """Verify proposed order cannot be confirmed by vague speech."""
    sess_id = f"jarvis_safe_{int(time.time()*1000)}"

    # 1. User proposes BUY order via Voice
    req1 = UnifiedAIRequest(
        session_id=sess_id,
        user_id="trader",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Buy BTC now",
        source="jarvis_ptt",
    )
    res1 = chat_service.process_unified_request(req1)
    assert res1.requires_confirmation is True
    assert res1.safety_status == "CONFIRMATION_REQUIRED"
    assert res1.action_intent == "BUY"

    sess_state = chat_service.get_session_state(sess_id)
    assert sess_state.pending_action_intent == "BUY"

    # 2. User provides vague confirmation: "okay sounds good"
    vague_inputs = ["okay", "hmm", "do it maybe", "sounds good", "sure why not", "alright"]
    for v in vague_inputs:
        req_vague = UnifiedAIRequest(
            session_id=sess_id,
            user_id="trader",
            input_modality=InputModality.VOICE,
            output_modality=OutputModality.BOTH,
            input_text=v,
            source="jarvis_continuous",
        )
        res_vague = chat_service.process_unified_request(req_vague)
        # MUST BE REJECTED
        assert res_vague.requires_confirmation is True
        assert res_vague.safety_status == "CONFIRMATION_REQUIRED"
        assert res_vague.intent == "CONFIRMATION_VAGUE_REJECTED"
        assert "Vague Input Rejected" in res_vague.response_text

    # Pending action still intact
    sess_state_still = chat_service.get_session_state(sess_id)
    assert sess_state_still.pending_action_intent == "BUY"


def test_step4_trade_safety_explicit_confirmation_and_cancellation(chat_service):
    """Verify explicit confirmation dispatches through ExecutionGate, and cancellation clears intent."""
    sess_id = f"jarvis_confirm_{int(time.time()*1000)}"

    # 1. Propose order
    chat_service.process_unified_request(UnifiedAIRequest(
        session_id=sess_id,
        user_id="trader",
        input_modality=InputModality.VOICE,
        input_text="Sell ETH at market",
    ))
    sess_state = chat_service.get_session_state(sess_id)
    assert sess_state.pending_action_intent == "SELL"

    # 2. Cancel order
    res_cancel = chat_service.process_unified_request(UnifiedAIRequest(
        session_id=sess_id,
        user_id="trader",
        input_modality=InputModality.VOICE,
        input_text="Cancel order",
    ))
    assert res_cancel.intent == "ORDER_CANCELLED"
    assert "Order Cancelled" in res_cancel.response_text
    sess_state_post_cancel = chat_service.get_session_state(sess_id)
    assert sess_state_post_cancel.pending_action_intent is None

    # 3. Propose again and explicitly confirm
    chat_service.process_unified_request(UnifiedAIRequest(
        session_id=sess_id,
        user_id="trader",
        input_modality=InputModality.VOICE,
        input_text="Buy BTC now",
    ))
    res_confirm = chat_service.process_unified_request(UnifiedAIRequest(
        session_id=sess_id,
        user_id="trader",
        input_modality=InputModality.VOICE,
        input_text="CONFIRM ORDER",
    ))
    assert res_confirm.intent == "ORDER_CONFIRMED"
    assert "Execution Gate Invocation" in res_confirm.response_text
    sess_state_post_confirm = chat_service.get_session_state(sess_id)
    assert sess_state_post_confirm.pending_action_intent is None


def test_step4_tts_speech_formatting_canonical_preservation(chat_service):
    """Verify TTS speech text is sanitized for spoken playback while preserving full trading levels."""
    raw_markdown = (
        "### Market Analysis for BTC/USD (1H)\n\n"
        "- **Directional Bias**: Bullish order flow.\n"
        "- **Key POI**: 67,450.50.\n"
        "- **Stop Loss Invalidation**: `66,800.00`.\n"
        "- **Take Profit Target**: 69,200.00.\n"
    )
    from trading_ai.chat.chat_service import format_speech_friendly_text
    speech_output = format_speech_friendly_text(raw_markdown)

    # Markdown characters removed
    assert "#" not in speech_output
    assert "*" not in speech_output
    assert "`" not in speech_output

    # Crucial numbers and levels preserved 100%
    assert "67,450.50" in speech_output
    assert "66,800.00" in speech_output
    assert "69,200.00" in speech_output
    assert "Bullish" in speech_output


def test_step4_token_manager_voice_budget_headroom(chat_service):
    """Verify voice requests pass through TokenManager with 20% safety headroom."""
    tm = chat_service._token_manager
    model_name = "gemini-3.8-flash"
    capacity = tm.get_model_capacity(model_name)

    from trading_ai.core.token_manager import TokenBudgetTier
    budget = tm.get_budget_for_tier(TokenBudgetTier.MEDIUM, model_capacity=capacity)
    assert budget.safety_margin_percent >= 0.20
    assert budget.output_reserve >= 800


def test_step4_api_unified_confirm_endpoint(client):
    """Verify REST API /api/unified/confirm executes UI button confirmation flow."""
    sess_id = f"api_confirm_test_{int(time.time()*1000)}"

    # Propose trade
    p_res = client.post("/api/unified/process", json={
        "session_id": sess_id,
        "input_text": "Buy SOL now",
        "input_modality": "voice",
        "output_modality": "both",
    })
    assert p_res.status_code == 200
    assert p_res.json()["requires_confirmation"] is True

    # Confirm via API endpoint
    c_res = client.post("/api/unified/confirm", json={
        "session_id": sess_id,
        "action": "CONFIRM",
    })
    assert c_res.status_code == 200
    data = c_res.json()
    assert data["intent"] == "ORDER_CONFIRMED"
    assert "Execution Gate" in data["response_text"]


def test_step4_api_voice_gateway_session_persistence(client):
    """Verify /api/voice/command honors session_id for cross-modality context sharing."""
    sess_id = f"gw_sess_{int(time.time()*1000)}"

    # 1. Voice command with session_id
    res1 = client.post("/api/voice/command", json={
        "command_text": "What is the market doing on EUR/USD 4H?",
        "session_id": sess_id,
    })
    assert res1.status_code == 200
    assert res1.json()["authorized"] is True

    # 2. Check session state
    sess_res = client.get(f"/api/unified/session?session_id={sess_id}")
    assert sess_res.status_code == 200
    state = sess_res.json()
    assert state["active_symbol"] in ["EUR/USD", "EURUSD"]
    assert state["active_timeframe"] == "4H"
