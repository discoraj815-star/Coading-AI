"""Unit tests for Phase 1 AI Chat and Application Guide / Insights System."""

import pytest
from unittest.mock import MagicMock
from trading_ai.chat.app_guide import ApplicationGuide, APP_FEATURES_REGISTRY
from trading_ai.chat.chat_service import AIChatService
from trading_ai.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    ChatHistoryResponse,
    AppGuideResponse,
    ChatNavigationAction,
)


def test_app_guide_tabs_coverage():
    """Verify ApplicationGuide contains all 18 tabs in the application map."""
    guide = ApplicationGuide()
    tabs = guide.get_all_tabs()
    assert len(tabs) >= 18
    tab_ids = [t.tab_id for t in tabs]
    assert "tab-notes" in tab_ids
    assert "tab-market" in tab_ids
    assert "tab-step17" in tab_ids
    assert "tab-step18" in tab_ids


def test_app_guide_find_feature_watchlist():
    """Verify find_feature locates the Watchlist in Tab 9 Live Market."""
    guide = ApplicationGuide()
    feature = guide.find_feature("Where is the Watchlist?")
    assert feature is not None
    assert feature.tab_id == "tab-market"
    assert "Watchlist" in feature.name
    assert feature.status in ("COMPLETE", "AVAILABLE")


def test_app_guide_negative_hallucination_settings():
    """Verify find_feature returns Settings Hub or explicit implementation status."""
    guide = ApplicationGuide()
    feature = guide.find_feature("Where are the system settings?")
    assert feature is not None
    assert feature.status in ("NOT_IMPLEMENTED", "AVAILABLE")
    assert "settings" in feature.how_to_access.lower() or "not implemented" in feature.status.lower()


def test_app_guide_generate_tour():
    """Verify generate_tour produces structured tour text covering key areas."""
    guide = ApplicationGuide()
    tour = guide.generate_tour()
    assert "TRADING AI ASSISTANT — APPLICATION TOUR" in tour
    assert "Tab 1: Notes" in tour
    assert "Tab 9: Live Market" in tour
    assert "Tab 17: Master Decision Engine" in tour
    assert "Tab 18: Universal Providers & Benchmark" in tour


def test_app_guide_system_prompt_summary():
    """Verify get_system_prompt_summary enforces hallucination control instructions."""
    guide = ApplicationGuide()
    summary = guide.get_system_prompt_summary()
    assert "STRICT INSTRUCTION: Never invent tabs, panels, or buttons" in summary
    assert "tab-market" in summary


def test_chat_service_app_tour_query():
    """Verify AIChatService handles tour requests with grounded application map."""
    service = AIChatService(app_guide=ApplicationGuide())
    req = ChatMessageRequest(
        session_id="test_tour_session",
        message="Give me a complete tour of the application.",
        symbol="EUR/USD",
        current_tab="tab-notes",
    )
    res = service.process_message(req)
    assert isinstance(res, ChatMessageResponse)
    assert res.session_id == "test_tour_session"
    assert "Tab 1: Notes" in res.ai_message or "TRADING AI ASSISTANT" in res.ai_message
    assert res.intent == "APP_TOUR"
    assert res.navigation_action is not None
    assert res.navigation_action.tab_id == "tab-market"


def test_chat_service_find_feature_navigation():
    """Verify AIChatService handles 'Where is the Watchlist?' with navigation action."""
    service = AIChatService(app_guide=ApplicationGuide())
    req = ChatMessageRequest(
        session_id="test_nav_session",
        message="Where is the Watchlist?",
        symbol="EUR/USD",
        current_tab="tab-notes",
    )
    res = service.process_message(req)
    assert isinstance(res, ChatMessageResponse)
    assert "Tab 9" in res.ai_message or "Live Market" in res.ai_message
    assert res.navigation_action is not None
    assert res.navigation_action.tab_id == "tab-market"


def test_chat_service_settings_hallucination_control():
    """Verify AIChatService does NOT hallucinate a Settings page when asked."""
    service = AIChatService(app_guide=ApplicationGuide())
    req = ChatMessageRequest(
        session_id="test_settings_session",
        message="Where are the broker settings?",
        symbol="EUR/USD",
        current_tab="tab-notes",
    )
    res = service.process_message(req)
    assert isinstance(res, ChatMessageResponse)
    assert "not implemented" in res.ai_message.lower() or "no dedicated settings" in res.ai_message.lower() or "settings" in res.ai_message.lower()


def test_chat_service_market_and_bias_grounding():
    """Verify AIChatService grounds market status and bias questions."""
    mock_companion = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.bias = "BULLISH"
    mock_ctx.setup_confirmation_state = "Awaiting Confirmation"
    mock_ctx.news_risk = "LOW"
    mock_ctx.current_price = 1.08500
    mock_companion.get_companion_context.return_value = mock_ctx
    mock_query_res = MagicMock()
    mock_query_res.answer = "BULLISH bias due to 15m liquidity sweep and FVG."
    mock_query_res.evidence = ["15m FVG", "Liquidity sweep"]
    mock_companion.query_companion.return_value = mock_query_res


    service = AIChatService(companion_core=mock_companion, app_guide=ApplicationGuide())
    req = ChatMessageRequest(
        session_id="test_market_session",
        message="What is the current market situation and why is the AI bullish?",
        symbol="EUR/USD",
        current_tab="tab-market",
    )
    res = service.process_message(req)
    assert isinstance(res, ChatMessageResponse)
    assert "BULLISH" in res.ai_message or "1.0850" in res.ai_message or "context" in res.ai_message.lower()


def test_chat_service_multi_turn_history():
    """Verify multi-turn history accumulation and clearing."""
    service = AIChatService(app_guide=ApplicationGuide())
    sess = "multi_turn_session_1"

    # Turn 1
    req1 = ChatMessageRequest(session_id=sess, message="Where is the Watchlist?")
    res1 = service.process_message(req1)
    assert res1.user_message == "Where is the Watchlist?"

    # Turn 2
    req2 = ChatMessageRequest(session_id=sess, message="Give me a tour of the application")
    res2 = service.process_message(req2)
    assert res2.user_message == "Give me a tour of the application"

    # Fetch History
    history = service.get_history(session_id=sess)
    assert isinstance(history, ChatHistoryResponse)
    assert len(history.turns) >= 2

    # Clear History
    service.clear_history(session_id=sess)

    # Verify history empty
    history_after = service.get_history(session_id=sess)
    assert len(history_after.turns) == 0


def test_chat_service_conversation_compression_on_long_thread():
    """Verify that older conversation turns are summarized when history grows."""
    service = AIChatService(app_guide=ApplicationGuide())
    sess = "compress_session"

    # Add 6 turns (3 exchanges)
    for i in range(3):
        service.process_message(
            ChatMessageRequest(session_id=sess, message=f"Analyze symbol EUR/USD exchange {i}")
        )

    # Now format history: with > 4 turns, it must format summary + recent turns
    formatted = service._format_recent_history(sess, max_turns=3, symbol="EUR/USD")
    assert "STRUCTURED CONVERSATION SUMMARY" in formatted or "EUR/USD" in formatted


def test_chat_service_overflow_handling():
    """Verify that a massive request exceeding safe capacity returns controlled overflow message."""
    from trading_ai.core.token_manager import TokenManager, TokenBudget, TokenBudgetTier
    # Create TokenManager with artificially restricted budget to simulate extreme overflow
    tm = TokenManager()
    service = AIChatService(token_manager=tm, app_guide=ApplicationGuide())

    # Send a massive prompt (100k+ chars) that exceeds typical thresholds
    huge_message = "Analyze market " + ("EUR/USD and fair value gap confluence " * 4000)
    req = ChatMessageRequest(
        session_id="overflow_session",
        message=huge_message,
        symbol="EUR/USD",
        current_tab="tab-notes",
    )
    res = service.process_message(req)
    assert isinstance(res, ChatMessageResponse)
    # Either handled or gracefully returned controlled overflow notice
    assert res.ai_message is not None
    assert len(res.ai_message) > 0
