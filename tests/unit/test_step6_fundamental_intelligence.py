"""
Unit tests for Step 6 Fundamental Data Provider, News Timing Engine, and Correlation Engine.
Tests Forex Factory feed normalization, currency matching, timing categories,
risk level assessment, and technical-fundamental conflict detection.
"""

from datetime import datetime, timedelta, timezone
import pytest

from trading_ai.core.tech_fund_correlator import TechnicalFundamentalCorrelator
from trading_ai.fundamental.forex_factory import ForexFactoryProvider
from trading_ai.fundamental.news_timing import NewsTimingEngine
from trading_ai.schemas.enums import (
    EventImpact,
    EventStatus,
    MarketBias,
    MarketDirection,
    NewsRiskLevel,
    NewsTimingCategory,
    ProviderStatus,
    SetupStatus,
    TechFundRelationship,
)
from trading_ai.schemas.fundamental import (
    FundamentalEvent,
    FundamentalProviderState,
    NewsTimingContext,
)
from trading_ai.schemas.market import MarketSetup, UnifiedMarketAnalysis


MOCK_RAW_FEED = [
    {
        "title": "Non-Farm Employment Change",
        "country": "USD",
        "date": "2026-10-02T12:30:00-04:00",
        "impact": "High",
        "forecast": "180K",
        "previous": "142K",
        "actual": "254K",
    },
    {
        "title": "Unemployment Rate",
        "country": "USD",
        "date": "2026-10-02T12:30:00-04:00",
        "impact": "High",
        "forecast": "4.2%",
        "previous": "4.2%",
        "actual": "4.1%",
    },
    {
        "title": "ECB Monetary Policy Statement",
        "country": "EUR",
        "date": "2026-10-02T13:45:00-04:00",
        "impact": "High",
        "forecast": "",
        "previous": "",
        "actual": "",
    },
    {
        "title": "German Prelim CPI m/m",
        "country": "EUR",
        "date": "2026-10-02T08:00:00-04:00",
        "impact": "Medium",
        "forecast": "0.1%",
        "previous": "-0.1%",
        "actual": "0.0%",
    },
]


def test_forex_factory_normalization():
    provider = ForexFactoryProvider(calendar_url="http://invalid.local/calendar.json")
    events = provider.load_mock_events(MOCK_RAW_FEED)

    assert len(events) == 4
    nfp = next(e for e in events if "Non-Farm" in e.title)
    assert nfp.currency == "USD"
    assert nfp.impact == EventImpact.HIGH
    assert nfp.actual == "254K"
    assert nfp.status == EventStatus.RELEASED
    assert nfp.source == "forex_factory"

    ecb = next(e for e in events if "ECB" in e.title)
    assert ecb.currency == "EUR"
    assert ecb.impact == EventImpact.HIGH
    assert ecb.status == EventStatus.SCHEDULED


def test_forex_factory_offline_graceful_handling():
    provider = ForexFactoryProvider(calendar_url="http://127.0.0.1:9999/nonexistent.json", timeout_seconds=0.5)
    # Fetch from non-existent URL should not crash and should update provider state
    events = provider.fetch_events()
    assert isinstance(events, list)
    status = provider.get_provider_status()
    assert status.status in (ProviderStatus.UNAVAILABLE, ProviderStatus.ERROR)
    assert status.error_message is not None


def test_news_timing_engine_currencies():
    timing = NewsTimingEngine()
    assert timing.resolve_symbol_currencies("EURUSD") == ["EUR", "USD"]
    assert timing.resolve_symbol_currencies("GBP/USD") == ["GBP", "USD"]
    assert timing.resolve_symbol_currencies("USDJPY") == ["USD", "JPY"]
    assert timing.resolve_symbol_currencies("BTC/USD") == ["USD"]
    assert timing.resolve_symbol_currencies("XAUUSD") == ["USD"]


def test_news_timing_event_proximity_and_risk():
    timing = NewsTimingEngine(pre_event_window_seconds=1800, immediate_window_seconds=300)
    now = datetime(2026, 10, 2, 14, 0, 0, tzinfo=timezone.utc)

    # Event in 10 minutes (upcoming within 30m window)
    event_10m = FundamentalEvent(
        title="FOMC Statement",
        currency="USD",
        event_time=now + timedelta(minutes=10),
        impact=EventImpact.HIGH,
        status=EventStatus.SCHEDULED,
    )

    ctx = timing.evaluate_timing(symbol="EURUSD", events=[event_10m], current_time=now)
    assert ctx.category == NewsTimingCategory.UPCOMING_EVENT
    assert ctx.risk_level == NewsRiskLevel.HIGH_RISK
    assert ctx.next_high_impact_event is not None
    assert ctx.time_to_next_seconds == 600.0


def test_news_timing_event_imminent():
    timing = NewsTimingEngine(immediate_window_seconds=300)
    now = datetime(2026, 10, 2, 14, 0, 0, tzinfo=timezone.utc)

    # Event in 3 minutes (imminent)
    event_3m = FundamentalEvent(
        title="CPI Release",
        currency="USD",
        event_time=now + timedelta(minutes=3),
        impact=EventImpact.HIGH,
        status=EventStatus.SCHEDULED,
    )

    ctx = timing.evaluate_timing(symbol="EURUSD", events=[event_3m], current_time=now)
    assert ctx.category == NewsTimingCategory.EVENT_IMMINENT
    assert ctx.risk_level == NewsRiskLevel.EXTREME_RISK


def test_tech_fund_directional_conflict():
    correlator = TechnicalFundamentalCorrelator()
    now = datetime(2026, 10, 2, 14, 0, 0, tzinfo=timezone.utc)

    # Technical is BULLISH on EURUSD
    analysis = UnifiedMarketAnalysis(
        symbol="EURUSD",
        overall_bias=MarketBias.BULLISH_BIAS,
    )

    # But released USD event had strong positive surprise (e.g. NFP +74K), which is BEARISH for EURUSD
    strong_usd_event = FundamentalEvent(
        title="Non-Farm Payrolls",
        currency="USD",
        event_time=now - timedelta(minutes=15),
        impact=EventImpact.HIGH,
        forecast="180K",
        actual="254K",  # Strong beat for USD
        status=EventStatus.RELEASED,
    )

    timing_ctx = NewsTimingContext(
        symbol="EURUSD",
        relevant_currencies=["EUR", "USD"],
        category=NewsTimingCategory.RECENTLY_RELEASED,
        risk_level=NewsRiskLevel.MEDIUM_RISK,
        recent_released_event=strong_usd_event,
        time_since_last_seconds=900.0,
    )

    result = correlator.correlate(
        symbol="EURUSD",
        unified_analysis=analysis,
        active_setup=None,
        timing_context=timing_ctx,
    )

    assert result.conflict.has_conflict is True
    assert result.conflict.conflict_type == "FUNDAMENTAL_TECHNICAL_CONFLICT"
    assert result.relationship == TechFundRelationship.FUNDAMENTAL_CONFLICT
    assert "STAND_ASIDE" in result.action_context or "REQUIRE_HIGHER" in result.action_context


def test_tech_fund_imminent_event_with_setup():
    correlator = TechnicalFundamentalCorrelator()
    now = datetime(2026, 10, 2, 14, 0, 0, tzinfo=timezone.utc)

    analysis = UnifiedMarketAnalysis(
        symbol="EURUSD",
        overall_bias=MarketBias.BULLISH_BIAS,
    )
    setup = MarketSetup(
        symbol="EURUSD",
        timeframe="15m",
        setup_type="BULLISH_ORDER_BLOCK",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.CONFIRMED,
    )

    imminent_event = FundamentalEvent(
        title="FOMC Interest Rate Decision",
        currency="USD",
        event_time=now + timedelta(minutes=4),
        impact=EventImpact.HIGH,
        status=EventStatus.SCHEDULED,
    )

    timing_ctx = NewsTimingContext(
        symbol="EURUSD",
        relevant_currencies=["EUR", "USD"],
        category=NewsTimingCategory.EVENT_IMMINENT,
        risk_level=NewsRiskLevel.EXTREME_RISK,
        next_high_impact_event=imminent_event,
        time_to_next_seconds=240.0,
    )

    result = correlator.correlate(
        symbol="EURUSD",
        unified_analysis=analysis,
        active_setup=setup,
        timing_context=timing_ctx,
    )

    assert result.conflict.has_conflict is True
    assert result.conflict.conflict_type == "HIGH_IMPACT_EVENT_IMMINENT"
    assert result.relationship == TechFundRelationship.NEWS_IMMINENT
    assert "WAIT_FOR_CONFIRMATION" in result.action_context
