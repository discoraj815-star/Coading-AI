"""
Unit tests for Step 5: Background Market Intelligence & Intelligent Selective Visualization.
Tests setup lifecycle transitions, confirmation evaluation, current situation extraction,
display selection policy, and Step 5 command intents.
"""

import pytest
from datetime import datetime, timezone

from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.command_parser import UserCommandParser
from trading_ai.core.command_service import UserCommandService
from trading_ai.core.lecture_validator import LectureValidator
from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
from trading_ai.core.mtf_engine import MultiTimeframeEngine
from trading_ai.core.situation_engine import CurrentSituationEngine
from trading_ai.pipeline.watchlist_monitor import WatchlistMonitor
from trading_ai.schemas.enums import (
    CommandIntent,
    KnowledgeType,
    LevelType,
    MarketBias,
    MarketDirection,
    MarkingType,
    MarkingValidity,
    SetupStatus,
    TimeframeHierarchy,
)
from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.schemas.market import (
    DisplayPolicy,
    MarketLevel,
    MarketSetup,
    TimeframeAnalysis,
    UnifiedMarketAnalysis,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_step5.db")
    db = DatabaseManager(db_file)
    return db, db_file


def test_setup_lifecycle_and_confirmation_promotion(temp_db):
    """Test setup transitions from DETECTED/VALID -> CONFIRMED -> ACTIVE -> INVALIDATED."""
    db, _ = temp_db
    repo = Repository(db)
    validator = LectureValidator(repo)

    # 1. Seed rule for OB
    rule = KnowledgeItem(
        job_id="test_job",
        item_type=KnowledgeType.RULE,
        title="Bullish Order Block",
        statement="Enter on retracement to order block with HTF alignment",
        conditions=["Displacement confirmed"],
        exceptions=["High impact news"],
        tags=["OB", "order block"],
    )
    repo.save_knowledge([rule])

    # 2. Build analysis
    htf = TimeframeAnalysis(
        symbol="BTC/USD",
        timeframe="1d",
        hierarchy_role=TimeframeHierarchy.HTF,
        direction=MarketDirection.BULLISH,
        structure="BOS_BULLISH",
        bias=MarketBias.BULLISH_BIAS,
    )
    ltf = TimeframeAnalysis(
        symbol="BTC/USD",
        timeframe="15m",
        hierarchy_role=TimeframeHierarchy.LTF,
        direction=MarketDirection.BULLISH,
        structure="SWING_HIGH",
        bias=MarketBias.BULLISH_BIAS,
    )
    analysis = UnifiedMarketAnalysis(
        symbol="BTC/USD",
        htf_analysis=htf,
        ltf_analysis=ltf,
        overall_bias=MarketBias.BULLISH_BIAS,
    )

    # 3. Initial detection & validation -> VALID
    setup = validator.validate_detected_pattern(
        symbol="BTC/USD",
        timeframe="15m",
        pattern_name="order block",
        direction=MarketDirection.BULLISH,
        market_analysis=analysis,
        pattern_context={"displacement": True},
    )
    assert setup.status == SetupStatus.VALID
    repo.save_market_setup(setup)

    # 4. Confirmation evaluation -> CONFIRMED
    confirmed = validator.evaluate_setup_confirmation(
        setup=setup,
        current_price=65000.0,
        market_analysis=analysis,
    )
    assert confirmed.status == SetupStatus.CONFIRMED
    assert confirmed.confirmation_timestamp is not None
    assert len(confirmed.confirmation_evidence) > 0

    # 5. Invalidation evaluation (HTF flips bearish)
    bearish_htf = htf.model_copy()
    bearish_htf.direction = MarketDirection.BEARISH
    bearish_analysis = analysis.model_copy()
    bearish_analysis.htf_analysis = bearish_htf

    invalidated = validator.evaluate_setup_confirmation(
        setup=confirmed,
        current_price=65000.0,
        market_analysis=bearish_analysis,
    )
    assert invalidated.status == SetupStatus.INVALIDATED
    assert invalidated.invalidation_timestamp is not None
    assert "flipped" in invalidated.status_transition_reason.lower()


def test_current_situation_engine_and_selective_display(temp_db):
    """Test distillation of 'What matters RIGHT NOW?' and display selection policy."""
    db, _ = temp_db
    repo = Repository(db)
    provider = SimulatedMarketDataProvider(seed=42)
    marker = ChartMarker()
    engine = CurrentSituationEngine(repository=repo, chart_marker=marker)

    snapshot = provider.get_snapshot("BTC/USD", "15m", limit=30)
    current_price = snapshot.latest_price

    # Seed levels: one nearby, one far away
    near_liq = MarketLevel(
        symbol="BTC/USD",
        timeframe="15m",
        price=current_price + 10.0,
        level_type=LevelType.LIQUIDITY_BUY_SIDE,
    )
    far_liq = MarketLevel(
        symbol="BTC/USD",
        timeframe="1d",
        price=current_price + 50000.0,
        level_type=LevelType.LIQUIDITY_BUY_SIDE,
    )
    poi = MarketLevel(
        symbol="BTC/USD",
        timeframe="15m",
        price=current_price - 15.0,
        level_type=LevelType.OB_HIGH,
    )
    repo.save_market_level(near_liq)
    repo.save_market_level(far_liq)
    repo.save_market_level(poi)

    # Seed active setup
    setup = MarketSetup(
        symbol="BTC/USD",
        timeframe="15m",
        setup_type="OB",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.CONFIRMED,
        confidence=0.9,
    )
    repo.save_market_setup(setup)

    analysis = UnifiedMarketAnalysis(
        symbol="BTC/USD",
        overall_bias=MarketBias.BULLISH_BIAS,
    )

    # Evaluate current situation with selective display
    situation = engine.evaluate_situation(
        symbol="BTC/USD",
        snapshot=snapshot,
        unified_analysis=analysis,
    )

    assert situation.current_price == current_price
    assert situation.overall_bias == MarketBias.BULLISH_BIAS
    assert situation.nearest_major_liquidity is not None
    assert situation.nearest_major_liquidity.price == near_liq.price
    assert situation.current_poi is not None
    assert situation.active_confirmed_setup is not None
    assert situation.setup_confirmation_state == "CONFIRMED"
    assert situation.invalidation_level is not None
    assert len(situation.visible_markings) <= 7

    # Far level must NOT be included in default selective visible markings
    visible_prices = [
        m.coordinates_json.get("price") for m in situation.visible_markings
        if "price" in m.coordinates_json
    ]
    assert far_liq.price not in visible_prices


def test_step5_commands_parsing_and_execution(temp_db):
    """Test user commands for selective visualization and background intelligence."""
    db, _ = temp_db
    repo = Repository(db)
    parser = UserCommandParser()
    service = UserCommandService(repository=repo)

    # 1. Parse Step 5 intents
    res1 = parser.parse("Show all liquidity")
    assert res1.command.intent == CommandIntent.SHOW_ALL_LIQUIDITY

    res2 = parser.parse("Show hidden setups")
    assert res2.command.intent == CommandIntent.SHOW_HIDDEN_SETUPS

    res3 = parser.parse("Show unconfirmed setups")
    assert res3.command.intent == CommandIntent.SHOW_UNCONFIRMED_SETUPS

    res4 = parser.parse("Show confirmation evidence")
    assert res4.command.intent == CommandIntent.SHOW_CONFIRMATION_EVIDENCE

    res5 = parser.parse("Show invalidation condition")
    assert res5.command.intent == CommandIntent.SHOW_INVALIDATION_CONDITION

    res6 = parser.parse("Hide background markings")
    assert res6.command.intent == CommandIntent.HIDE_BACKGROUND_MARKINGS

    res7 = parser.parse("Focus current setup")
    assert res7.command.intent == CommandIntent.FOCUS_CURRENT_SETUP

    # 2. Execute commands
    out1 = service.execute_command(res1.command)
    assert out1["status"] == "success"
    assert "liquidity_levels" in out1

    out2 = service.execute_command(res2.command)
    assert out2["status"] == "success"
    assert "hidden_setups" in out2

    out3 = service.execute_command(res3.command)
    assert out3["status"] == "success"
    assert "unconfirmed_setups" in out3

    out6 = service.execute_command(res6.command)
    assert out6["status"] == "success"
    assert "visible_markings_count" in out6
