"""Unit tests for TradingView Market Data & Chart Context Provider.

Verifies:
- Category is MARKET_DATA / CHART_CONTEXT
- Initial status is ADAPTER_READY / NOT_CONFIGURED (never fabricates REST API)
- Webhook ingestion produces canonical NormalizedMarketData with is_live=True
- Never silently falls back to simulated data when live feed is absent
"""

import time
import pytest

from trading_ai.providers.tradingview import TradingViewMarketDataProvider
from trading_ai.schemas.provider import ProviderCategory, ProviderLifecycleState


def test_tradingview_initial_status_not_configured():
    """Verify TradingView is marked ADAPTER_READY / NOT_CONFIGURED by default."""
    tv = TradingViewMarketDataProvider()
    assert tv.provider_id == "tradingview"
    assert tv.metadata.provider_type == ProviderCategory.MARKET_DATA
    assert tv.metadata.status == ProviderLifecycleState.DISCOVERED

    # Test connection when unconfigured
    res = tv.test_connection()
    assert res.success is False
    assert "ADAPTER_READY" in res.details.get("status", "")
    assert res.details.get("real_rest_api_fabricated") is False


def test_tradingview_configured_connection_test():
    """Verify test_connection passes when webhook secret or session is provided."""
    tv = TradingViewMarketDataProvider(webhook_secret="sec_tv_12345")
    res = tv.test_connection()
    assert res.success is True
    assert tv.metadata.status == ProviderLifecycleState.CONNECTED
    assert "TradingView Webhook Connector operational" in res.message


def test_tradingview_webhook_ingestion_and_normalization():
    """Verify incoming TradingView payload is parsed into canonical NormalizedMarketData."""
    tv = TradingViewMarketDataProvider()
    payload = {
        "ticker": "BTCUSD",
        "timeframe": "15m",
        "close": 68450.25,
        "open": 68200.0,
        "high": 68500.0,
        "low": 68150.0,
        "volume": 250.4,
        "time": 1700000000.0,
        "exchange": "BINANCE",
        "strategy": "LiquiditySweepLong",
        "action": "BUY",
        "indicators": {"rsi": 32.5, "fvg_detected": True},
    }

    norm = tv.ingest_webhook_payload(payload)

    assert norm.provider_id == "tradingview"
    assert norm.canonical_symbol == "BTC/USD"
    assert norm.timeframe == "15m"
    assert norm.close == 68450.25
    assert norm.is_live is True
    assert norm.is_simulated is False
    assert norm.chart_context["indicators"]["fvg_detected"] is True

    # Latest price retrieval
    assert tv.get_latest_price("BTC/USD") == 68450.25
    assert tv.get_latest_price("BTCUSD") == 68450.25

    # Check chart context
    ctx = tv.get_chart_context("BTC/USD")
    assert ctx["action"] == "BUY"


def test_tradingview_no_silent_fallback_to_simulated():
    """Verify that get_latest_price returns None for un-cached symbol (no simulation fallback)."""
    tv = TradingViewMarketDataProvider()
    # No data ingested yet
    price = tv.get_latest_price("EUR/USD")
    assert price is None, "Must return None rather than silently faking simulated price for live feed"

    fresh, age = tv.check_data_freshness("EUR/USD")
    assert fresh is False
