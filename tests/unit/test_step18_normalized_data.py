"""Unit tests for Normalized Market Data Contract, Symbol Mapping, and Timeframe Normalization.

Verifies:
- NormalizedMarketData schema validation and default fields
- Known symbol mappings across Delta, TradingView, XM, and MT5
- Unknown symbol mapping rejection with SymbolMappingRequiredError
- Timeframe alias mapping into canonical timeframes
- Unsupported timeframe rejection with TimeframeUnsupportedError
"""

import pytest

from trading_ai.schemas.normalized_data import (
    NormalizedMarketData,
    SymbolMapping,
    SymbolMappingRequiredError,
    TimeframeUnsupportedError,
    normalize_timeframe,
    resolve_canonical_symbol,
)


def test_normalized_market_data_schema():
    """Verify NormalizedMarketData validation and live/simulated flags."""
    norm = NormalizedMarketData(
        provider_id="delta",
        provider_type="BROKER",
        symbol="BTCUSD",
        canonical_symbol="BTC/USD",
        exchange="DELTA",
        timestamp=1700000000.0,
        open=65000.0,
        high=65500.0,
        low=64800.0,
        close=65200.0,
        volume=100.5,
        timeframe="15m",
        is_live=False,
        is_simulated=True,
    )
    assert norm.canonical_symbol == "BTC/USD"
    assert norm.is_live is False
    assert norm.is_simulated is True
    assert norm.data_quality == 1.0


def test_symbol_mapping_resolution():
    """Verify symbol mapping for Delta, TradingView, and MT5."""
    m1 = resolve_canonical_symbol("delta", "BTCUSD")
    assert m1.canonical_symbol == "BTC/USD"
    assert m1.contract_type == "PERPETUAL"

    m2 = resolve_canonical_symbol("tradingview", "BTCUSDT")
    assert m2.canonical_symbol == "BTC/USDT"

    m3 = resolve_canonical_symbol("mt5", "EURUSD")
    assert m3.canonical_symbol == "EUR/USD"
    assert m3.price_precision == 5


def test_unknown_symbol_mapping_raises_error():
    """Verify that unknown symbol raises SymbolMappingRequiredError to prevent trading."""
    with pytest.raises(SymbolMappingRequiredError) as exc:
        resolve_canonical_symbol("unknown_provider", "UNKNOWN_TICKER_XYZ")
    assert "SYMBOL_MAPPING_REQUIRED" in str(exc.value)


def test_timeframe_normalization():
    """Verify timeframe normalization and alias translation."""
    assert normalize_timeframe("1m") == "1m"
    assert normalize_timeframe("1") == "1m"
    assert normalize_timeframe("15") == "15m"
    assert normalize_timeframe("60") == "1h"
    assert normalize_timeframe("240") == "4h"
    assert normalize_timeframe("1d") == "1d"
    assert normalize_timeframe("D") == "1d"
    assert normalize_timeframe("1w") == "1w"
    assert normalize_timeframe("1M") == "1M"


def test_unsupported_timeframe_raises_error():
    """Verify unsupported timeframe raises TimeframeUnsupportedError."""
    with pytest.raises(TimeframeUnsupportedError) as exc:
        normalize_timeframe("7m")
    assert "TIMEFRAME_UNSUPPORTED" in str(exc.value)
