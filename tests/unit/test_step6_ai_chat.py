"""Unit tests for Step 6 Fundamental Intelligence Integration in AIChatService.

Verifies that the Single AI Brain (AIChatService) actively ingests and utilizes
Step 6 Fundamental Data and Technical-Fundamental Correlation for conversational
grounding across Text and JARVIS Voice.
"""

from datetime import datetime, timezone
import pytest

from trading_ai.chat.chat_service import AIChatService
from trading_ai.schemas.chat import ChatMessageRequest
from trading_ai.schemas.enums import EventImpact, EventStatus, NewsRiskLevel, TechFundRelationship
from trading_ai.schemas.fundamental import FundamentalEvent, TechnicalFundamentalAnalysis, TechFundConflict
from trading_ai.schemas.unified import InputModality, OutputModality, UnifiedAIRequest
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    db_path = tmp_path / "test_step6_chat.db"
    db = DatabaseManager(db_path=str(db_path))
    repository = Repository(db)

    # Seed an event
    evt = FundamentalEvent(
        event_id="evt_test_nfp",
        source="forex_factory",
        title="Non-Farm Payrolls",
        currency="USD",
        event_time=datetime.now(timezone.utc),
        impact=EventImpact.HIGH,
        status=EventStatus.SCHEDULED,
    )
    repository.save_fundamental_event(evt)

    # Seed tech fund analysis for EUR/USD
    analysis = TechnicalFundamentalAnalysis(
        analysis_id="tfa_test_eurusd",
        symbol="EUR/USD",
        relationship=TechFundRelationship.NEWS_IMMINENT,
        technical_bias="BULLISH_BIAS",
        news_risk=NewsRiskLevel.HIGH_RISK,
        conflict=TechFundConflict(has_conflict=False),
        action_context="WAIT_FOR_CONFIRMATION",
        explanation="High impact USD event imminent; recommend waiting for post-release structure.",
        timestamp=datetime.now(timezone.utc).timestamp(),
        created_at=datetime.now(timezone.utc),
    )
    repository.save_tech_fund_analysis(analysis)

    return repository


def test_ai_chat_fundamental_intelligence_intent(repo):
    chat_svc = AIChatService(repository=repo)

    req = ChatMessageRequest(
        message="What is the fundamental news risk and economic events for EUR/USD?",
        symbol="EUR/USD",
        session_id="test_session_fund",
    )
    res = chat_svc.process_message(req)

    assert res.intent == "FUNDAMENTAL_INTELLIGENCE"
    assert "HIGH_RISK" in res.ai_message
    assert "WAIT_FOR_CONFIRMATION" in res.ai_message
    assert "Non-Farm Payrolls" in res.ai_message
    assert res.navigation_action is not None
    assert res.navigation_action.tab_id == "tab-market"


def test_ai_chat_voice_fundamental_speech_output(repo):
    chat_svc = AIChatService(repository=repo)

    req = UnifiedAIRequest(
        session_id="voice_fund_test",
        input_modality=InputModality.VOICE,
        output_modality=OutputModality.BOTH,
        input_text="Is there any FOMC or news conflict on EUR/USD?",
        symbol="EUR/USD",
    )
    res = chat_svc.process_unified_request(req)

    assert res.intent == "FUNDAMENTAL_INTELLIGENCE"
    assert res.speech_text is not None
    assert "high risk" in res.speech_text.lower()
    assert "wait for confirmation" in res.speech_text.lower()
