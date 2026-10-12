"""Step 18 Correction: Normalized Market Data Contract, Symbol Mapping, and Timeframe Mapping.

Ensures all market providers normalize into a strictly typed, canonical schema.
Guarantees AI trading brain consumes normalized data, with explicit symbol & timeframe validation,
and never mixes simulated data as live data.
"""

from datetime import datetime, timezone
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SymbolMappingRequiredError(ValueError):
    """Raised when an unknown symbol mapping is encountered."""
    pass


class TimeframeUnsupportedError(ValueError):
    """Raised when an unsupported timeframe is requested or mapped."""
    pass


class SymbolMapping(BaseModel):
    """Canonical mapping definition for a financial instrument across providers."""

    provider_symbol: str = Field(..., description="Provider's proprietary ticker/symbol")
    canonical_symbol: str = Field(..., description="Canonical unified symbol (e.g. BTC/USD, EUR/USD)")
    asset_class: str = Field(default="CRYPTO", description="CRYPTO, FOREX, COMMODITY, INDEX, EQUITY")
    exchange: str = Field(default="UNKNOWN", description="Underlying exchange or venue")
    contract_type: str = Field(default="SPOT", description="SPOT, PERPETUAL, FUTURES, CFD")
    quote_currency: str = Field(default="USD", description="Quote currency (USD, USDT, EUR, etc.)")
    price_precision: int = Field(default=2, description="Decimal places for pricing")


class NormalizedMarketData(BaseModel):
    """Canonical market data contract.
    
    All external feeds (TradingView, TradingLite, Delta, MT5, etc.) must normalize into this model.
    The AI Brain must only consume NormalizedMarketData.
    """

    provider_id: str = Field(..., description="Identifier of the publishing provider")
    provider_type: str = Field(default="MARKET_DATA", description="Provider category")
    symbol: str = Field(..., description="Raw provider symbol")
    canonical_symbol: str = Field(..., description="Unified canonical symbol")
    exchange: str = Field(default="UNKNOWN", description="Exchange or data venue")
    timestamp: float = Field(..., description="Bar/tick POSIX timestamp in seconds")
    timezone_name: str = Field(default="UTC", description="Timezone name")
    bid: Optional[float] = Field(default=None, description="Bid price if quote")
    ask: Optional[float] = Field(default=None, description="Ask price if quote")
    last: Optional[float] = Field(default=None, description="Last traded price")
    open: float = Field(..., description="Bar open price")
    high: float = Field(..., description="Bar high price")
    low: float = Field(..., description="Bar low price")
    close: float = Field(..., description="Bar close price")
    volume: float = Field(default=0.0, description="Volume")
    timeframe: str = Field(default="1m", description="Canonical timeframe (e.g. 1m, 5m, 1h, 1d)")
    session: str = Field(default="ALL", description="Market session (ASIA, LONDON, NY, ALL)")
    data_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality confidence 0.0-1.0")
    is_live: bool = Field(default=False, description="True ONLY if real live market data")
    is_simulated: bool = Field(default=False, description="True if simulated, mock, or backtest data")
    sequence: int = Field(default=0, description="Monotonic sequence number for tick ordering")
    source_timestamp: Optional[float] = Field(default=None, description="Original publisher timestamp")
    received_timestamp: float = Field(
        default_factory=lambda: time.time(),
        description="Local arrival POSIX timestamp"
    )
    chart_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="TradingView / visual chart markers, annotations, indicator states"
    )
    raw_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific auxiliary data"
    )


# Canonical Supported Timeframes
CANONICAL_TIMEFRAMES = {
    "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"
}

# Raw timeframe aliases mapping to canonical
TIMEFRAME_ALIASES: Dict[str, str] = {
    "1": "1m",
    "1m": "1m",
    "1min": "1m",
    "5": "5m",
    "5m": "5m",
    "5min": "5m",
    "15": "15m",
    "15m": "15m",
    "15min": "15m",
    "30": "30m",
    "30m": "30m",
    "30min": "30m",
    "60": "1h",
    "1h": "1h",
    "60m": "1h",
    "240": "4h",
    "4h": "4h",
    "d": "1d",
    "1d": "1d",
    "D": "1d",
    "w": "1w",
    "1w": "1w",
    "W": "1w",
    "M": "1M",
    "1M": "1M",
    "month": "1M",
}


# Pre-configured Symbol Mapping Directory
# Maps (provider_id, provider_symbol) -> SymbolMapping
# Also allows generic fallback by provider_symbol if provider is generic
KNOWN_SYMBOL_MAPPINGS: Dict[str, SymbolMapping] = {
    # Delta Exchange
    "delta:BTCUSD": SymbolMapping(
        provider_symbol="BTCUSD",
        canonical_symbol="BTC/USD",
        asset_class="CRYPTO",
        exchange="DELTA",
        contract_type="PERPETUAL",
        quote_currency="USD",
        price_precision=2,
    ),
    "delta:BTCUSDT": SymbolMapping(
        provider_symbol="BTCUSDT",
        canonical_symbol="BTC/USDT",
        asset_class="CRYPTO",
        exchange="DELTA",
        contract_type="PERPETUAL",
        quote_currency="USDT",
        price_precision=2,
    ),
    "delta:ETHUSD": SymbolMapping(
        provider_symbol="ETHUSD",
        canonical_symbol="ETH/USD",
        asset_class="CRYPTO",
        exchange="DELTA",
        contract_type="PERPETUAL",
        quote_currency="USD",
        price_precision=2,
    ),
    # TradingView
    "tradingview:BTCUSD": SymbolMapping(
        provider_symbol="BTCUSD",
        canonical_symbol="BTC/USD",
        asset_class="CRYPTO",
        exchange="BINANCE",
        contract_type="SPOT",
        quote_currency="USD",
        price_precision=2,
    ),
    "tradingview:BTC/USD": SymbolMapping(
        provider_symbol="BTC/USD",
        canonical_symbol="BTC/USD",
        asset_class="CRYPTO",
        exchange="BINANCE",
        contract_type="SPOT",
        quote_currency="USD",
        price_precision=2,
    ),
    "tradingview:BTCUSDT": SymbolMapping(
        provider_symbol="BTCUSDT",
        canonical_symbol="BTC/USDT",
        asset_class="CRYPTO",
        exchange="BINANCE",
        contract_type="SPOT",
        quote_currency="USDT",
        price_precision=2,
    ),
    "tradingview:EURUSD": SymbolMapping(
        provider_symbol="EURUSD",
        canonical_symbol="EUR/USD",
        asset_class="FOREX",
        exchange="FXCM",
        contract_type="SPOT",
        quote_currency="USD",
        price_precision=5,
    ),
    "tradingview:XAUUSD": SymbolMapping(
        provider_symbol="XAUUSD",
        canonical_symbol="XAU/USD",
        asset_class="COMMODITY",
        exchange="OANDA",
        contract_type="SPOT",
        quote_currency="USD",
        price_precision=2,
    ),
    # TradingLite / Trading-Leads
    "trading_leads:BTCUSDT": SymbolMapping(
        provider_symbol="BTCUSDT",
        canonical_symbol="BTC/USDT",
        asset_class="CRYPTO",
        exchange="BINANCE",
        contract_type="PERPETUAL",
        quote_currency="USDT",
        price_precision=2,
    ),
    "trading_leads:BTCUSD": SymbolMapping(
        provider_symbol="BTCUSD",
        canonical_symbol="BTC/USD",
        asset_class="CRYPTO",
        exchange="BINANCE",
        contract_type="PERPETUAL",
        quote_currency="USD",
        price_precision=2,
    ),
    # MetaTrader 5 / XM
    "mt5:EURUSD": SymbolMapping(
        provider_symbol="EURUSD",
        canonical_symbol="EUR/USD",
        asset_class="FOREX",
        exchange="XM",
        contract_type="CFD",
        quote_currency="USD",
        price_precision=5,
    ),
    "mt5:GBPUSD": SymbolMapping(
        provider_symbol="GBPUSD",
        canonical_symbol="GBP/USD",
        asset_class="FOREX",
        exchange="XM",
        contract_type="CFD",
        quote_currency="USD",
        price_precision=5,
    ),
    "mt5:XAUUSD": SymbolMapping(
        provider_symbol="XAUUSD",
        canonical_symbol="XAU/USD",
        asset_class="COMMODITY",
        exchange="XM",
        contract_type="CFD",
        quote_currency="USD",
        price_precision=2,
    ),
    "mt5:BTCUSD": SymbolMapping(
        provider_symbol="BTCUSD",
        canonical_symbol="BTC/USD",
        asset_class="CRYPTO",
        exchange="XM",
        contract_type="CFD",
        quote_currency="USD",
        price_precision=2,
    ),
    # XM Broker direct
    "xm:EURUSD": SymbolMapping(
        provider_symbol="EURUSD",
        canonical_symbol="EUR/USD",
        asset_class="FOREX",
        exchange="XM",
        contract_type="CFD",
        quote_currency="USD",
        price_precision=5,
    ),
    "xm:XAUUSD": SymbolMapping(
        provider_symbol="XAUUSD",
        canonical_symbol="XAU/USD",
        asset_class="COMMODITY",
        exchange="XM",
        contract_type="CFD",
        quote_currency="USD",
        price_precision=2,
    ),
}


def normalize_timeframe(provider_timeframe: str) -> str:
    """Normalize raw provider timeframe into canonical timeframe.
    
    Raises TimeframeUnsupportedError if unmapped.
    """
    cleaned = str(provider_timeframe).strip()
    if cleaned in CANONICAL_TIMEFRAMES:
        return cleaned
    if cleaned in TIMEFRAME_ALIASES:
        return TIMEFRAME_ALIASES[cleaned]
    raise TimeframeUnsupportedError(
        f"TIMEFRAME_UNSUPPORTED: '{provider_timeframe}' does not map to canonical timeframes {sorted(CANONICAL_TIMEFRAMES)}"
    )


def resolve_canonical_symbol(provider_id: str, provider_symbol: str) -> SymbolMapping:
    """Resolve raw provider symbol to canonical SymbolMapping.
    
    Raises SymbolMappingRequiredError if mapping is undefined.
    """
    key = f"{provider_id.lower()}:{provider_symbol.strip()}"
    if key in KNOWN_SYMBOL_MAPPINGS:
        return KNOWN_SYMBOL_MAPPINGS[key]
    
    # Generic symbol lookup fallback if standard symbol without provider prefix is unambiguous
    # e.g. "BTC/USD"
    clean_sym = provider_symbol.strip()
    if clean_sym in ("BTC/USD", "EUR/USD", "GBP/USD", "XAU/USD", "BTC/USDT"):
        return SymbolMapping(
            provider_symbol=clean_sym,
            canonical_symbol=clean_sym,
            asset_class="CRYPTO" if "BTC" in clean_sym else ("COMMODITY" if "XAU" in clean_sym else "FOREX"),
            exchange="GENERIC",
            contract_type="SPOT",
            quote_currency="USD" if "USD" in clean_sym else "USDT",
            price_precision=2 if ("BTC" in clean_sym or "XAU" in clean_sym) else 5,
        )

    raise SymbolMappingRequiredError(
        f"SYMBOL_MAPPING_REQUIRED: Provider '{provider_id}' symbol '{provider_symbol}' has no canonical mapping. Trading prohibited until defined."
    )


def register_custom_symbol_mapping(mapping: SymbolMapping, provider_id: str = "custom") -> None:
    """Dynamically register or override a canonical symbol mapping."""
    key = f"{provider_id.lower()}:{mapping.provider_symbol.strip()}"
    KNOWN_SYMBOL_MAPPINGS[key] = mapping
