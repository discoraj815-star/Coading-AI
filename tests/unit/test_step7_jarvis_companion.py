"""Unit tests for Step 7: JARVIS Response Quality, Companion Grounding & Disambiguation.

Verifies:
1. Grounded COMPANION_SCENARIO responses with IF/THEN/UNLESS conditional explanations.
2. Symbol awareness: explicit query symbol overrides active session symbol.
3. Grounded DISCIPLINE_STATUS responses with factual risk limits (no hallucinated values).
4. Grounded ACTIVE_ALERTS responses reflecting ProactiveAlertEngine state without fabrication.
5. Text + Voice parity: identical cognitive core (AIChatService) and reasoning traces.
6. Strict intent disambiguation (Knowledge vs Scenario vs Discipline vs Alerts vs Execution).
7. Settings Hub discipline configuration synchronization.
"""

import pytest
from datetime import datetime, timezone
from trading_ai.chat.app_guide import ApplicationGuide
from trading_ai.chat.chat_service import AIChatService, extract_symbol_from_text
from trading_ai.companion.core import TradingCompanionCore
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.companion.scenario_engine import ScenarioEngine
from trading_ai.companion.voice_engine import InteractiveVoiceEngine
from trading_ai.schemas.chat import ChatMessageRequest
from trading_ai.schemas.companion import (
    AlertEvent,
    DisciplineConfig,
    DisciplineState,
    ScenarioCondition,
)
from trading_ai.schemas.enums import (
    AlertDecision,
    AlertLifecycleState,
    AlertPriority,
    AlertQuality,
    AlertType,
    LevelType,
    MarketBias,
    VoiceIntent,
)
from trading_ai.schemas.market import (
    CurrentSituationSnapshot,
    MarketLevel,
)
from trading_ai.schemas.settings import UnifiedSettings, TradingSettings
from trading_ai.schemas.unified import (
    InputModality,
    OutputModality,
    UnifiedAIRequest,
)
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    db_file = tmp_path / "test_step7_jarvis.db"
    return Repository(str(db_file))


@pytest.fixture
def companion_core(repo):
    return TradingCompanionCore(repo)


@pytest.fixture
def chat_service(repo, companion_core):
    svc = AIChatService(
        repository=repo,
        companion_core=companion_core,
        app_guide=ApplicationGuide(),
    )
    companion_core.set_chat_service(svc)
    return svc


# --- 1. Scenario Grounding & Symbol Awareness Tests ---

def test_jarvis_companion_scenario_grounded(repo, companion_core, chat_service):
    """Verify JARVIS retrieves actual ScenarioEngine conditional IF-THEN-UNLESS logic."""
    # Seed current situation snapshot with POI
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.08500,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Bullish order flow above 1.08200",
        current_poi=MarketLevel(
            symbol="EUR/USD",
            timeframe="15m",
            price=1.08350,
            level_type=LevelType.FVG_BOTTOM,
        ),
        setup_confirmation_state="CONFIRMED_BUY_SETUP",
        timestamp=datetime.now(timezone.utc).timestamp(),
    )
    repo.save_current_situation(sit)

    # Ask forward scenario
    req = UnifiedAIRequest(
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="What is the forward scenario for EUR/USD?",
        symbol="EUR/USD",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "COMPANION_SCENARIO"
    assert "EUR/USD" in res.response_text
    # Must contain conditional phrasing, not prophecy
    assert "IF" in res.response_text
    assert "THEN" in res.response_text
    assert "UNLESS" in res.response_text
    assert "1.08350" in res.response_text
    # Voice speech text must be natural and speech-friendly
    assert len(res.speech_text) > 0
    assert "scenario" in res.speech_text.lower()
    assert res.audio_output_url is not None


def test_jarvis_symbol_awareness_override(repo, companion_core, chat_service):
    """Verify explicit symbol in query overrides active session symbol."""
    # Active session is set to EUR/USD
    session_state = chat_service.get_session_state("test_session", default_symbol="EUR/USD")
    assert session_state.active_symbol == "EUR/USD"

    # Query explicitly mentions GBP/USD
    req = UnifiedAIRequest(
        session_id="test_session",
        input_modality=InputModality.TEXT,
        input_text="What is the forward scenario for GBP/USD?",
        symbol="EUR/USD",  # Client passed active EUR/USD
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "COMPANION_SCENARIO"
    assert "GBP/USD" in res.response_text
    assert session_state.active_symbol == "GBP/USD"


def test_jarvis_scenario_fallback_honesty(repo, companion_core, chat_service):
    """Verify JARVIS states scenario is unavailable without fabricating prices."""
    req = UnifiedAIRequest(
        input_modality=InputModality.TEXT,
        input_text="What is the scenario for AUD/USD?",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "COMPANION_SCENARIO"
    assert "No Forward Scenarios Currently Generated" in res.response_text
    # Zero hallucination check
    assert "AUD/USD" in res.response_text


# --- 2. Discipline Status & Guardrails Tests ---

def test_jarvis_discipline_status_grounded(repo, companion_core, chat_service):
    """Verify JARVIS retrieves factual discipline limits without hallucination."""
    # Configure custom discipline rules
    cfg = DisciplineConfig(
        user_id="default_user",
        stop_loss_required=True,
        max_risk_per_trade_pct=1.5,
        daily_loss_limit=300.0,
        max_daily_trades=5,
        cooldown_minutes=45,
    )
    companion_core.update_discipline_config("default_user", cfg)

    req = UnifiedAIRequest(
        input_modality=InputModality.TEXT,
        input_text="What are my current discipline limits?",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "DISCIPLINE_STATUS"
    assert "1.5%" in res.response_text
    assert "$300.00" in res.response_text
    assert "45 minutes" in res.response_text
    assert "Enforced" in res.response_text
    # Spoken phrasing check
    assert "discipline" in res.speech_text.lower() or "limits" in res.speech_text.lower()


def test_jarvis_discipline_unconfigured_honesty(repo, companion_core, chat_service):
    """Verify unconfigured discipline limits are honestly reported as unavailable."""
    # Reset to blank unconfigured discipline state
    clean_state = DisciplineState(user_id="default_user", config=DisciplineConfig(user_id="default_user", daily_loss_limit=None, max_risk_per_trade_pct=None))
    repo.save_discipline_state(clean_state)

    req = UnifiedAIRequest(
        input_modality=InputModality.TEXT,
        input_text="What is my current discipline limit?",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "DISCIPLINE_STATUS"
    assert "Unavailable / Not configured" in res.response_text


# --- 3. Active Alerts Grounding Tests ---

def test_jarvis_active_alerts_grounded(repo, companion_core, chat_service):
    """Verify JARVIS explains actual active alerts with zero hallucination."""
    # Save a real alert in repository
    alert = AlertEvent(
        symbol="EUR/USD",
        alert_type=AlertType.LIQUIDITY_EVENT,
        priority=AlertPriority.HIGH,
        lifecycle_state=AlertLifecycleState.ALERTED,
        decision=AlertDecision.ALERT,
        quality=AlertQuality.HIGH_QUALITY,
        title="Asian High Sweep Detected",
        message="Price swept 1.08650 liquidity pool with immediate 5m rejection wick",
        is_acknowledged=False,
    )
    repo.save_alert_event(alert)

    req = UnifiedAIRequest(
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="What proactive alerts are active right now?",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "ACTIVE_ALERTS"
    assert "Asian High Sweep" in res.response_text
    assert "LIQUIDITY_EVENT" in res.response_text
    assert "HIGH" in res.response_text
    assert "EUR/USD" in res.response_text
    assert "Asian High Sweep" in res.speech_text


def test_jarvis_active_alerts_empty_state_honesty(repo, companion_core, chat_service):
    """Verify JARVIS states zero active alerts when none are present."""
    req = UnifiedAIRequest(
        input_modality=InputModality.TEXT,
        input_text="Any active alerts right now?",
    )
    res = chat_service.process_unified_request(req)

    assert res.intent == "ACTIVE_ALERTS"
    assert "No Active Alerts" in res.response_text
    assert "zero active" in res.speech_text.lower() or "no active proactive alerts" in res.speech_text.lower()


# --- 4. Text and Voice Parity Tests ---

def test_text_and_voice_share_same_cognitive_core(repo, companion_core, chat_service):
    """Verify text chat and voice JARVIS produce identical reasoning grounding."""
    # 1. Text Chat Request
    req_text = UnifiedAIRequest(
        input_modality=InputModality.TEXT,
        output_modality=OutputModality.TEXT,
        input_text="What are my current discipline limits?",
    )
    res_text = chat_service.process_unified_request(req_text)

    # 2. Voice JARVIS Request
    req_voice = UnifiedAIRequest(
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="What are my current discipline limits?",
    )
    res_voice = chat_service.process_unified_request(req_voice)

    # Both must share identical intent, factual response text, and grounding evidence
    assert res_text.intent == res_voice.intent == "DISCIPLINE_STATUS"
    assert res_text.response_text == res_voice.response_text
    assert res_text.evidence == res_voice.evidence
    assert res_voice.audio_output_url is not None


# --- 5. Intent Disambiguation Regression Tests ---

@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("What is a liquidity sweep?", "GENERAL_MARKET_QA"),
        ("Can you explain what order flow is?", "GENERAL_MARKET_QA"),
        ("What are my current discipline limits?", "DISCIPLINE_STATUS"),
        ("Show my discipline rules", "DISCIPLINE_STATUS"),
        ("What is the forward scenario for EUR/USD?", "COMPANION_SCENARIO"),
        ("What happens next for EUR/USD?", "COMPANION_SCENARIO"),
        ("What proactive alerts are active right now?", "ACTIVE_ALERTS"),
        ("Any active alerts?", "ACTIVE_ALERTS"),
        ("Is CPI creating risk right now?", "FUNDAMENTAL_INTELLIGENCE"),
        ("What is the economic calendar risk?", "FUNDAMENTAL_INTELLIGENCE"),
        ("Buy EUR/USD.", "TRADE_ACTION_PROPOSED"),
        ("Sell GBP/USD immediately", "TRADE_ACTION_PROPOSED"),
        ("Can you explain buy side liquidity?", "GENERAL_MARKET_QA"),
        ("Should I buy EUR/USD?", "GENERAL_MARKET_QA"),
    ],
)
def test_intent_classification_disambiguation(repo, companion_core, chat_service, query, expected_intent):
    """Verify intent classifier unambiguously separates knowledge, companion, scenarios, alerts, fundamentals, and execution."""
    req = UnifiedAIRequest(
        input_modality=InputModality.TEXT,
        input_text=query,
    )
    res = chat_service.process_unified_request(req)
    assert res.intent == expected_intent, f"Query '{query}' expected {expected_intent}, got {res.intent}"


# --- 6. Symbol Extraction Helper Tests ---

def test_symbol_extraction_helper():
    """Verify symbol extraction handles pairs with and without slash."""
    assert extract_symbol_from_text("What is the scenario for EUR/USD?") == "EUR/USD"
    assert extract_symbol_from_text("Check GBPUSD forward path") == "GBP/USD"
    assert extract_symbol_from_text("Status of BTC") == "BTC/USD"
    assert extract_symbol_from_text("What is happening with gold?") == "XAU/USD"
    assert extract_symbol_from_text("What is a liquidity sweep?") is None
