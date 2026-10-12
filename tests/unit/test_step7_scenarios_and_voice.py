"""Unit tests for Step 7 ScenarioEngine and InteractiveVoiceEngine."""

import pytest
from trading_ai.companion.scenario_engine import ScenarioEngine
from trading_ai.companion.voice_engine import CompanionIntentRouter, InteractiveVoiceEngine
from trading_ai.schemas.companion import CompanionQueryRequest
from trading_ai.schemas.enums import AlertQuality, LevelType, MarketBias, MarketDirection, SetupStatus, VoiceIntent
from trading_ai.schemas.market import CurrentSituationSnapshot, MarketLevel, MarketSetup
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


def test_scenario_engine_if_then_unless_structure(repo):
    engine = ScenarioEngine(repo)
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Clean bullish continuation",
        current_poi=MarketLevel(
            level_id="p1",
            symbol="EUR/USD",
            timeframe="15m",
            price=1.0830,
            level_type=LevelType.OB_HIGH,
            name="15m Bullish OB",
        ),
        nearest_major_liquidity=MarketLevel(
            level_id="l1",
            symbol="EUR/USD",
            timeframe="15m",
            price=1.0910,
            level_type=LevelType.LIQUIDITY_BUY_SIDE,
            name="Buy-side Liquidity",
        ),
    )
    repo.save_current_situation(sit)

    scenarios = engine.generate_scenarios_for_symbol("EUR/USD")
    assert len(scenarios) >= 1
    s = scenarios[0]
    assert s.if_condition.startswith("IF")
    assert "THEN" in s.then_outcome
    assert s.unless_condition.startswith("UNLESS")
    assert "invalidates" in s.invalidated_when.lower()


def test_scenario_engine_custom_level(repo):
    engine = ScenarioEngine(repo)
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Bullish",
    )
    repo.save_current_situation(sit)

    scenarios = engine.generate_scenarios_for_symbol("EUR/USD", custom_level=1.0900)
    assert len(scenarios) == 1
    s = scenarios[0]
    assert s.level == 1.0900
    assert "1.09000" in s.if_condition


def test_voice_intent_routing_all_intents():
    cases = [
        ("Market mein abhi kya chal raha hai?", VoiceIntent.MARKET_STATUS),
        ("What is the market doing right now?", VoiceIntent.MARKET_STATUS),
        ("EURUSD bullish kyun hai?", VoiceIntent.EXPLAIN_BIAS),
        ("Why is the bias bearish?", VoiceIntent.EXPLAIN_BIAS),
        ("Explain the active setup", VoiceIntent.EXPLAIN_SETUP),
        ("Any upcoming high impact news?", VoiceIntent.NEWS_QUERY),
        ("Agar 1.0850 break hua to?", VoiceIntent.SCENARIO_ANALYSIS),
        ("What happens if price drops below 1.0800?", VoiceIntent.SCENARIO_ANALYSIS),
        ("Is this setup valid?", VoiceIntent.SETUP_VALIDATION),
        ("What did the market do yesterday?", VoiceIntent.MARKET_HISTORY),
        ("What is my current risk status?", VoiceIntent.RISK_STATUS),
        ("Discipline status and trade count?", VoiceIntent.DISCIPLINE_STATUS),
        ("Why was I alerted?", VoiceIntent.WHY_ALERT),
        ("Can you help me trade?", VoiceIntent.GENERAL_MARKET_QA),
    ]
    for text, expected in cases:
        assert CompanionIntentRouter.route_intent(text) == expected, f"Failed for '{text}'"


def test_interactive_voice_engine_market_status_and_scenario(repo):
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Trending up with clear market structure",
        setup_confirmation_state="CONFIRMED",
        active_confirmed_setup=MarketSetup(
            setup_id="s1",
            symbol="EUR/USD",
            setup_type="Fair Value Gap Long",
            timeframe="15m",
            direction=MarketDirection.BULLISH,
            status=SetupStatus.CONFIRMED,
        ),
        current_poi=MarketLevel(
            level_id="p1",
            symbol="EUR/USD",
            timeframe="15m",
            price=1.0830,
            level_type=LevelType.OB_HIGH,
            name="15m Bullish OB",
        ),
    )
    repo.save_current_situation(sit)

    voice_engine = InteractiveVoiceEngine(repo)

    # 1. Market Status Query
    res_status = voice_engine.process_query(
        CompanionQueryRequest(symbol="EUR/USD", query="Market mein abhi kya chal raha hai?")
    )
    assert res_status.intent == VoiceIntent.MARKET_STATUS
    assert "1.08500" in res_status.answer
    assert "BULLISH" in res_status.answer
    assert res_status.synthesized_audio_mock is not None

    # 2. Scenario Analysis Query
    res_scenario = voice_engine.process_query(
        CompanionQueryRequest(symbol="EUR/USD", query="Agar 1.0900 break hua to?")
    )
    assert res_scenario.intent == VoiceIntent.SCENARIO_ANALYSIS
    assert len(res_scenario.scenarios) > 0
    assert "1.09000" in res_scenario.scenarios[0].if_condition


def test_market_history_honesty(repo):
    voice_engine = InteractiveVoiceEngine(repo)
    # When no historical order flow is recorded in local memory:
    res = voice_engine.process_query(
        CompanionQueryRequest(symbol="ETH/USD", query="What did the market do yesterday?")
    )
    assert res.intent == VoiceIntent.MARKET_HISTORY
    assert "DATA_UNAVAILABLE" in res.answer or "cached" in res.answer.lower()
