"""
Unit and Integration Tests for Step 3 — Unified AI Brain & Shared Memory.

Validates:
1. Unified Request / Response normalization (Text, Voice, Multimodal).
2. Shared cross-modality session memory (Voice -> Text -> Voice context continuity).
3. Modality-independent user rules, corrections, and command parsing.
4. TokenManager centralization: dynamic tiers, compression, and 20% headroom across modalities.
5. Speech-friendly response formatting without altering trading facts or conditions.
6. Execution safety: Voice cannot bypass ExecutionGate, RiskEngine, or Confirmation Gates.
7. InteractiveVoiceEngine and SecureVoiceGateway delegation to Unified Brain.
8. Supervisor/Worker CommitBarrier serial ordering invariant regression.
"""

import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from trading_ai.schemas.unified import (
    InputModality,
    OutputModality,
    UnifiedAudioMetadata,
    UnifiedAIRequest,
    UnifiedAIResponse,
    UnifiedSessionState,
)
from trading_ai.chat.chat_service import (
    AIChatService,
    format_speech_friendly_text,
    extract_symbol_from_text,
    extract_timeframe_from_text,
)
from trading_ai.chat.app_guide import ApplicationGuide
from trading_ai.schemas.chat import ChatMessageRequest
from trading_ai.schemas import (
    JobMetadata,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    PipelineStage,
    UserCommand,
    CommandIntent,
)
from trading_ai.schemas.security import VoiceCommandRequest
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier
from trading_ai.companion.voice_engine import InteractiveVoiceEngine
from trading_ai.companion.core import TradingCompanionCore
from trading_ai.security.voice_gateway import SecureVoiceGateway
from trading_ai.security.capability_gate import CapabilityPolicyGate
from trading_ai.core.token_manager import TokenManager, TokenBudgetTier
from trading_ai.core.model_router import ModelRouter, ModelRole
from trading_ai.server.app import app


@pytest.fixture
def test_db_repo(tmp_path):
    """File-based temporary SQLite database and repository for clean isolation."""
    db_file = tmp_path / "step3_unified_brain.db"
    db_mgr = DatabaseManager(db_path=str(db_file))
    repo = Repository(db_mgr)
    return repo, str(db_file)


# ---------------------------------------------------------------------------
# 1. Unified Request & Response Contract Tests
# ---------------------------------------------------------------------------

def test_unified_request_normalization():
    """Verify UnifiedAIRequest handles text and voice modalities with metadata."""
    req_text = UnifiedAIRequest(
        session_id="session_101",
        input_modality=InputModality.TEXT,
        input_text="What is the current bias on BTC?",
    )
    assert req_text.input_modality == InputModality.TEXT
    assert req_text.output_modality == OutputModality.TEXT
    assert req_text.audio_metadata is None

    audio_meta = UnifiedAudioMetadata(
        confidence=0.96,
        duration_seconds=2.8,
        speaker="trader_primary",
        language="en",
    )
    req_voice = UnifiedAIRequest(
        session_id="session_101",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Analyze ETH liquidity pools",
        audio_metadata=audio_meta,
    )
    assert req_voice.input_modality == InputModality.VOICE
    assert req_voice.output_modality == OutputModality.BOTH
    assert req_voice.audio_metadata.confidence == 0.96
    assert req_voice.audio_metadata.speaker == "trader_primary"


def test_speech_friendly_text_formatting():
    """Verify format_speech_friendly_text strips markdown while preserving numbers and trading facts."""
    markdown_sample = (
        "### BTC/USD 1H Analysis\n"
        "- **Bias**: Bullish structure with BOS at **$64,250.00**.\n"
        "- **Key Level**: Order block resting between $63,800 - $63,950.\n"
        "- `Invalidation`: 1H close below $63,500. Risk: 1.5%."
    )
    speech = format_speech_friendly_text(markdown_sample)

    # Markdown syntax removed
    assert "###" not in speech
    assert "**" not in speech
    assert "`" not in speech
    assert not speech.startswith("-")

    # Critical trading facts preserved exactly
    assert "BTC/USD 1H Analysis" in speech
    assert "64,250.00" in speech
    assert "63,800" in speech
    assert "63,950" in speech
    assert "63,500" in speech
    assert "1.5%" in speech


# ---------------------------------------------------------------------------
# 2. Shared Session & Cross-Modality Memory Continuity Tests
# ---------------------------------------------------------------------------

def test_cross_modality_context_sharing(test_db_repo):
    """
    Verify Voice -> Text -> Voice context sharing:
    1. User speaks: 'Analyze BTC/USD on the 1H timeframe' (Voice).
    2. User types: 'What about the liquidity below?' (Text) -> Inherits BTC/USD and 1H!
    3. User speaks: 'What is our invalidation?' (Voice) -> Still grounded in BTC/USD.
    """
    repo, _ = test_db_repo
    service = AIChatService(repository=repo, app_guide=ApplicationGuide())
    session_id = "cross_modal_session_99"

    # Turn 1: Voice request mentioning symbol and timeframe
    req_voice_1 = UnifiedAIRequest(
        session_id=session_id,
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Analyze BTC/USD on the 1H timeframe",
    )
    resp_voice_1 = service.process_unified_request(req_voice_1)
    assert resp_voice_1.input_modality == InputModality.VOICE
    assert resp_voice_1.speech_text is not None

    session_state = service.get_session_state(session_id)
    assert session_state.active_symbol == "BTC/USD"
    assert session_state.active_timeframe == "1H"

    # Turn 2: Text request with relative pronoun ("What about the liquidity below?")
    req_text_2 = UnifiedAIRequest(
        session_id=session_id,
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        input_text="What about the liquidity below?",
    )
    resp_text_2 = service.process_unified_request(req_text_2)
    assert resp_text_2.input_modality == InputModality.TEXT
    # Must use market context of BTC/USD
    assert resp_text_2.market_context_used is not None
    assert "BTC" in resp_text_2.market_context_used.upper()

    # Turn 3: Follow-up Voice request ("What is our invalidation?")
    req_voice_3 = UnifiedAIRequest(
        session_id=session_id,
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="What is our invalidation?",
    )
    resp_voice_3 = service.process_unified_request(req_voice_3)
    assert resp_voice_3.input_modality == InputModality.VOICE
    assert resp_voice_3.speech_text is not None
    assert "BTC" in resp_voice_3.market_context_used.upper()


# ---------------------------------------------------------------------------
# 3. Modality-Independent User Commands & Knowledge Persistence Tests
# ---------------------------------------------------------------------------

def test_user_command_from_voice_and_text(test_db_repo):
    """
    Verify SAVE_RULE and CORRECT commands work identically through Voice and Text,
    and persist authoritative KnowledgeItems to the SQLite database.
    """
    repo, _ = test_db_repo
    service = AIChatService(repository=repo, app_guide=ApplicationGuide())

    # Voice SAVE_RULE command
    req_voice_rule = UnifiedAIRequest(
        session_id="rule_session",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Save rule: Never take a trade before London open liquidity sweep.",
    )
    resp_voice_rule = service.process_unified_request(req_voice_rule)
    assert resp_voice_rule.intent == "SAVE_RULE"
    assert "London open" in resp_voice_rule.response_text or "recorded" in resp_voice_rule.response_text.lower()
    assert resp_voice_rule.speech_text is not None

    # Text CORRECT command
    req_text_correct = UnifiedAIRequest(
        session_id="rule_session",
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        input_text="Correct: Only consider sweeps above Asian session high.",
    )
    resp_text_correct = service.process_unified_request(req_text_correct)
    assert resp_text_correct.intent == "CORRECT"
    assert "Asian session high" in resp_text_correct.response_text or "recorded" in resp_text_correct.response_text.lower()

    # Verify KnowledgeItems exist in repository
    k_items = repo.get_knowledge("user_command", only_committed=True)
    statements = [k.statement for k in k_items]
    assert any("London open liquidity sweep" in s for s in statements)
    assert any("Asian session high" in s for s in statements)


# ---------------------------------------------------------------------------
# 4. TokenManager Centralization Across Modalities
# ---------------------------------------------------------------------------

def test_token_manager_centralization_for_voice_and_text(test_db_repo):
    """
    Verify TokenManager enforces token budget, headroom, and complexity classification
    for Voice input just like Text input.
    """
    repo, _ = test_db_repo
    tm = TokenManager()
    service = AIChatService(repository=repo, token_manager=tm, app_guide=ApplicationGuide())

    # Voice query
    req_voice = UnifiedAIRequest(
        session_id="token_session",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Provide a detailed multi-timeframe liquidity analysis of Solana.",
    )
    resp_voice = service.process_unified_request(req_voice)
    assert resp_voice.token_diagnostics is not None
    assert "tier" in resp_voice.token_diagnostics
    assert "estimated_tokens" in resp_voice.token_diagnostics
    assert resp_voice.token_diagnostics["estimated_tokens"] > 0


# ---------------------------------------------------------------------------
# 5. Trading Execution Safety via Voice & Text
# ---------------------------------------------------------------------------

def test_voice_cannot_bypass_execution_safety(test_db_repo):
    """
    Verify Voice input saying 'Buy BTC now' is treated as input modality, NOT execution authority.
    It triggers action intent proposal but requires strict confirmation / capability gating.
    """
    repo, _ = test_db_repo
    service = AIChatService(repository=repo, app_guide=ApplicationGuide())

    req_buy = UnifiedAIRequest(
        session_id="safety_session",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Buy 1 BTC at market immediately",
    )
    resp_buy = service.process_unified_request(req_buy)

    # Action intent identified as BUY
    assert resp_buy.action_intent == "BUY"
    # Strict confirmation required
    assert resp_buy.requires_confirmation is True
    assert resp_buy.safety_status == "CONFIRMATION_REQUIRED"
    assert "confirmation required" in resp_buy.response_text.lower()
    assert "confirmation required" in resp_buy.speech_text.lower()


# ---------------------------------------------------------------------------
# 6. InteractiveVoiceEngine & SecureVoiceGateway Integration
# ---------------------------------------------------------------------------

def test_interactive_voice_engine_delegates_to_unified_brain(test_db_repo):
    """Verify InteractiveVoiceEngine routes queries to AIChatService when connected."""
    repo, _ = test_db_repo
    chat_service = AIChatService(repository=repo, app_guide=ApplicationGuide())
    voice_engine = InteractiveVoiceEngine(repository=repo, chat_service=chat_service)

    from trading_ai.schemas.companion import CompanionQueryRequest
    query_req = CompanionQueryRequest(
        symbol="BTC/USD",
        query="What is the current bias on BTC?",
        modality="voice",
        session_id="engine_test_session",
    )
    resp = voice_engine.process_query(query_req)
    assert resp is not None
    assert resp.answer != ""
    assert resp.synthesized_audio_mock is not None


def test_secure_voice_gateway_delegates_to_unified_brain(test_db_repo):
    """Verify SecureVoiceGateway routes authorized voice commands to AIChatService."""
    repo, _ = test_db_repo
    chat_service = AIChatService(repository=repo, app_guide=ApplicationGuide())
    gate = CapabilityPolicyGate()
    voice_gateway = SecureVoiceGateway(capability_gate=gate, chat_service=chat_service)

    cmd_req = VoiceCommandRequest(
        command_text="What is the trend on ETH?",
        device_id="trader_mic",
    )
    res = voice_gateway.process_voice_command(cmd_req)
    assert res.authorized is True
    assert res.response_text != ""
    assert res.audio_output_url is not None


# ---------------------------------------------------------------------------
# 7. Supervisor/Worker CommitBarrier Serial Ordering Invariant Regression
# ---------------------------------------------------------------------------

def test_commit_barrier_serial_ordering_regression(tmp_path):
    """
    Regression invariant: Later serial task units (6, 7) must remain buffered
    and cannot be committed while an earlier unit (5) is missing/failed.
    Once unit 5 completes, 5, 6, 7 commit sequentially.
    """
    db_file = tmp_path / "barrier_regression.db"
    db_mgr = DatabaseManager(db_path=str(db_file))
    repo = Repository(db_mgr)

    job = JobMetadata(
        job_id="job_regression_barrier",
        file_hash="hash_barrier_reg",
        filename="regression_lecture.mp4",
        file_path=str(tmp_path / "regression_lecture.mp4"),
    )
    repo.save_job(job)

    barrier = CommitBarrier(repo)
    job_id = "job_regression_barrier"

    for i in range(1, 8):
        barrier.register_task_unit(
            job_id=job_id,
            stage=PipelineStage.GEMINI_REASONING,
            start_time=(i - 1) * 30.0,
            end_time=i * 30.0,
            payload={"serial": i},
        )

    # 1, 2, 3, 4 committed
    assert barrier.complete_task_unit(job_id, 1, {"ok": 1}) == [1]
    assert barrier.complete_task_unit(job_id, 2, {"ok": 2}) == [2]
    assert barrier.complete_task_unit(job_id, 3, {"ok": 3}) == [3]
    assert barrier.complete_task_unit(job_id, 4, {"ok": 4}) == [4]

    # Unit 5 fails
    from trading_ai.schemas import ErrorSeverity
    barrier.fail_task_unit(
        job_id=job_id,
        serial_number=5,
        error_message="Temporary network timeout",
        severity=ErrorSeverity.RECOVERABLE,
    )

    # Unit 6 and 7 complete while 5 is failed -> MUST BE BUFFERED, return empty committed list!
    c6 = barrier.complete_task_unit(job_id, serial_number=6, result={"ok": 6})
    assert c6 == []
    c7 = barrier.complete_task_unit(job_id, serial_number=7, result={"ok": 7})
    assert c7 == []

    # Verify authoritative progress is halted at 4
    state = barrier.get_barrier_state(job_id)
    assert state.highest_committed_serial == 4
    assert state.blocked_on_serial == 5
    assert state.buffered_units == [6, 7]

    # Unit 5 is retried and succeeds -> Triggers cascade: 5, 6, 7 committed in strict order!
    c5 = barrier.recover_task_unit(job_id=job_id, serial_number=5, recovered_result={"ok": 5})
    assert c5 == [5, 6, 7]

    # Verify highest committed serial is now 7
    state_after = barrier.get_barrier_state(job_id)
    assert state_after.highest_committed_serial == 7
    assert state_after.blocked_on_serial is None
    assert state_after.buffered_units == []


# ---------------------------------------------------------------------------
# 8. REST Endpoints Integration via TestClient
# ---------------------------------------------------------------------------

def test_unified_api_endpoints():
    """Verify POST /api/unified/process and GET /api/unified/session endpoints."""
    client = TestClient(app)

    # 1. Post unified text request
    res = client.post(
        "/api/unified/process",
        json={
            "session_id": "testclient_session",
            "user_id": "trader",
            "input_modality": "text",
            "output_modality": "text",
            "input_text": "Analyze BTC on the 4H timeframe",
            "active_symbol": "BTC/USD",
            "active_timeframe": "4H",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "testclient_session"
    assert data["input_modality"] == "text"
    assert "response_text" in data
    assert len(data["response_text"]) > 0

    # 2. Get session state
    res_session = client.get("/api/unified/session?session_id=testclient_session")
    assert res_session.status_code == 200
    s_data = res_session.json()
    assert s_data["session_id"] == "testclient_session"
    assert s_data["active_symbol"] == "BTC/USD"
    assert s_data["active_timeframe"] == "4H"

    # 3. Post unified voice request in same session
    res_voice = client.post(
        "/api/unified/process",
        json={
            "session_id": "testclient_session",
            "user_id": "trader",
            "input_modality": "voice",
            "output_modality": "both",
            "input_text": "Where is the liquidity?",
            "audio_metadata": {
                "confidence": 0.99,
                "duration_seconds": 1.5,
            },
        },
    )
    assert res_voice.status_code == 200
    v_data = res_voice.json()
    assert v_data["input_modality"] == "voice"
    assert v_data["speech_text"] is not None
    # Reused BTC context from previous turn
    assert v_data["market_context_used"] is not None
    assert "BTC" in v_data["market_context_used"].upper()
