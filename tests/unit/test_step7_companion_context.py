"""Unit tests for Step 7 TradingCompanionContext and ContextBuilder."""

import pytest
from trading_ai.companion.context_builder import CompanionContextBuilder
from trading_ai.companion.core import TradingCompanionCore
from trading_ai.schemas.companion import TradingCompanionContext
from trading_ai.schemas.enums import LevelType, MarketBias, MarketDirection, SetupStatus
from trading_ai.schemas.market import CurrentSituationSnapshot, MarketLevel, MarketSetup
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def in_memory_repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


def test_companion_context_builder_empty(in_memory_repo):
    builder = CompanionContextBuilder(in_memory_repo)
    ctx = builder.build_context("EUR/USD")
    assert isinstance(ctx, TradingCompanionContext)
    assert ctx.symbol == "EUR/USD"
    assert ctx.current_price == 0.0
    assert ctx.bias == "NEUTRAL"
    assert ctx.setup_confirmation_state == "NO_ACTIVE_SETUP"
    assert ctx.discipline_status["configured"] is False


def test_companion_context_builder_with_situation(in_memory_repo):
    # Save a realistic situation snapshot
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Clean higher-highs and order block reaction.",
        setup_confirmation_state="CONFIRMED",
        invalidation_level=1.0810,
        confirmation_level=1.0855,
        nearest_major_liquidity=MarketLevel(
            level_id="lvl_liq_1",
            symbol="EUR/USD",
            timeframe="15m",
            price=1.0920,
            level_type=LevelType.LIQUIDITY_BUY_SIDE,
            name="Buy-side Liquidity Pool",
        ),
        current_poi=MarketLevel(
            level_id="lvl_poi_1",
            symbol="EUR/USD",
            timeframe="15m",
            price=1.0845,
            level_type=LevelType.OB_HIGH,
            name="Bullish Order Block",
        ),
        active_confirmed_setup=MarketSetup(
            setup_id="setup_1",
            symbol="EUR/USD",
            setup_type="ICT Bullish Reversal",
            timeframe="15m",
            direction=MarketDirection.BULLISH,
            status=SetupStatus.CONFIRMED,
        ),
        news_risk="LOW_RISK",
        technical_fundamental_relationship="TECHNICALLY_VALID",
        news_time_distance_minutes=45.0,
    )
    in_memory_repo.save_current_situation(sit)

    builder = CompanionContextBuilder(in_memory_repo)
    ctx = builder.build_context("EUR/USD")

    assert ctx.symbol == "EUR/USD"
    assert ctx.current_price == 1.0850
    assert ctx.bias == "BULLISH_BIAS"
    assert ctx.setup_confirmation_state == "CONFIRMED"
    assert ctx.invalidation_level == 1.0810
    assert ctx.confirmation_level == 1.0855
    assert ctx.nearest_liquidity is not None
    assert ctx.nearest_liquidity["price"] == 1.0920
    assert ctx.current_poi is not None
    assert ctx.current_poi["price"] == 1.0845
    assert ctx.active_setup is not None
    assert ctx.active_setup["setup_type"] == "ICT Bullish Reversal"


def test_trading_companion_core_facade(in_memory_repo):
    core = TradingCompanionCore(in_memory_repo)
    ctx = core.get_companion_context("GBP/USD")
    assert ctx.symbol == "GBP/USD"
    assert ctx.bias == "NEUTRAL"


def test_companion_context_builder_with_tech_fund_upcoming_events(in_memory_repo):
    """Regression test: verify context building when situation has no next event but tech_fund has upcoming events."""
    from datetime import datetime, timezone
    from trading_ai.schemas.fundamental import FundamentalEvent, NewsTimingContext, TechnicalFundamentalAnalysis

    event = FundamentalEvent(
        title="CPI Release",
        currency="USD",
        event_time=datetime.now(timezone.utc),
    )
    timing_ctx = NewsTimingContext(
        symbol="EUR/USD",
        time_to_next_seconds=3600.0,
        upcoming_events=[event],
    )
    analysis = TechnicalFundamentalAnalysis(
        symbol="EUR/USD",
        technical_bias="BULLISH",
        timing_context=timing_ctx,
    )
    in_memory_repo.save_tech_fund_analysis(analysis)

    builder = CompanionContextBuilder(in_memory_repo)
    ctx = builder.build_context("EUR/USD")

    assert ctx.symbol == "EUR/USD"
    assert ctx.next_high_impact_event is not None
    assert ctx.next_high_impact_event["title"] == "CPI Release"
    assert ctx.news_time_distance_minutes == 60.0
