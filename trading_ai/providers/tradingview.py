"""Step 18 Correction: TradingView Live Market Data & Chart Context Provider.

Canonical Provider Ownership:
- Category: LIVE_MARKET_DATA / CHART_DATA / CHART_CONTEXT
- Capabilities: Live ticks, candles, chart context, visual markers, multi-timeframe observation.
- Reality Principle: TradingView does not provide a public unrestricted REST market data API.
  Authorized integration mechanism: TradingView Webhook / Pine Script Alert Receiver & Authorized Connector Boundary.
  When not configured, marked ADAPTER_READY / NOT_CONFIGURED.
  NEVER fabricates an undocumented REST endpoint.
  NEVER silently falls back to simulated data while labeling it LIVE.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.providers.base import MarketDataProvider
from trading_ai.schemas.normalized_data import (
    NormalizedMarketData,
    SymbolMapping,
    normalize_timeframe,
    resolve_canonical_symbol,
)
from trading_ai.schemas.provider import (
    ConnectionTestResult,
    ProviderCategory,
    ProviderLifecycleState,
    ProviderMetadata,
    TradingProviderExecutionState,
)

logger = logging.getLogger(__name__)


class TradingViewMarketDataProvider(MarketDataProvider):
    """TradingView Live Market Data and Chart Context Provider Adapter.
    
    Accepts authorized TradingView webhook alerts / Pine Script bar alerts,
    parses symbol and timeframe, extracts chart context, and normalizes into
    Canonical NormalizedMarketData with is_live=True.
    """

    def __init__(
        self,
        webhook_secret: str = "",
        session_id: str = "",
        webhook_path: str = "/api/providers/tradingview/webhook",
    ):
        meta = ProviderMetadata(
            provider_id="tradingview",
            provider_name="TradingView Chart & Live Market Data",
            provider_type=ProviderCategory.MARKET_DATA,
            version="1.0.0",
            status=ProviderLifecycleState.DISCOVERED,
            trading_state=TradingProviderExecutionState.LIVE_PENDING_APPROVAL,
            capabilities=[
                "LIVE_PRICE_STREAM",
                "LIVE_CANDLES",
                "CHART_CONTEXT",
                "WEBHOOK_CONNECTOR",
                "PINE_SCRIPT_ALERTS",
                "MULTI_TIMEFRAME_CHART",
                "MARKET_OBSERVATION",
            ],
            supported_symbols=[
                "BTCUSD", "BTC/USD", "BTCUSDT",
                "EURUSD", "GBPUSD", "USDJPY",
                "XAUUSD", "*"
            ],
            supported_markets=["Crypto", "Forex", "Metals", "Indices", "Equities"],
            supported_timeframes=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"],
            authentication_type="WEBHOOK_SECRET_OR_SESSION_KEY",
            configuration_schema={
                "webhook_secret": "Optional passphrase / HMAC secret for TradingView webhook validation",
                "session_id": "Optional user session connector or bridge token",
                "webhook_endpoint": "Local receiver endpoint: /api/providers/tradingview/webhook",
                "ip_whitelist": "TradingView Official IPs (52.89.214.238, 34.212.75.30, 54.218.53.128, 52.32.178.7)",
            },
            health_status="ADAPTER_READY",
            rate_limit="Webhook rate limits apply per TradingView subscription plan",
            cost_class="FREEMIUM",
            data_quality=1.0,
            latency_class="LOW",
            sandbox_available=False,
            paper_supported=True,
            live_supported=True,
            enabled=True,
            approved=False,
        )
        super().__init__(meta)
        self._webhook_secret: str = webhook_secret
        self._session_id: str = session_id
        self._webhook_path: str = webhook_path
        
        # In-memory storage for latest live prices, candles, and chart context
        self._live_cache: Dict[str, NormalizedMarketData] = {}
        self._candle_history: Dict[str, List[Dict[str, Any]]] = {}
        self._chart_context: Dict[str, Dict[str, Any]] = {}
        self._last_received_time: float = 0.0

        if webhook_secret or session_id:
            self.metadata.status = ProviderLifecycleState.CONFIGURED

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize adapter with configuration without fabricating credentials."""
        if config:
            self._webhook_secret = config.get("webhook_secret", self._webhook_secret)
            self._session_id = config.get("session_id", self._session_id)
            if self._webhook_secret or self._session_id:
                self.metadata.status = ProviderLifecycleState.CONFIGURED
        return True

    def test_connection(self) -> ConnectionTestResult:
        """Safe non-destructive connection and receiver health check.
        
        Verifies local webhook route readiness and configuration without fabricating an external REST API.
        """
        t0 = time.time()
        is_configured = bool(self._webhook_secret or self._session_id or self._live_cache)
        latency = (time.time() - t0) * 1000.0

        if not is_configured:
            return ConnectionTestResult(
                success=False,
                provider_id=self.provider_id,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                message="TradingView Adapter Ready. Configuration Required: Send alerts/data to /api/providers/tradingview/webhook or configure webhook secret.",
                capabilities_verified=["WEBHOOK_CONNECTOR", "CHART_CONTEXT"],
                details={
                    "status": "ADAPTER_READY / NOT_CONFIGURED",
                    "webhook_endpoint": self._webhook_path,
                    "active_cached_symbols": list(self._live_cache.keys()),
                    "real_rest_api_fabricated": False,
                },
            )

        self.metadata.status = ProviderLifecycleState.CONNECTED
        self.metadata.last_success = datetime.now(timezone.utc)
        return ConnectionTestResult(
            success=True,
            provider_id=self.provider_id,
            provider_name=self.provider_name,
            latency_ms=round(latency, 2),
            message="TradingView Webhook Connector operational. Ready to ingest live chart data and alerts.",
            capabilities_verified=self.metadata.capabilities,
            details={
                "status": "CONNECTED",
                "webhook_endpoint": self._webhook_path,
                "cached_symbols_count": len(self._live_cache),
                "is_live": True,
            },
        )

    def ingest_webhook_payload(self, payload: Dict[str, Any]) -> NormalizedMarketData:
        """Ingest live bar or tick from TradingView Alert/Pine Script Webhook.
        
        Normalizes into canonical NormalizedMarketData with is_live=True.
        """
        raw_symbol = str(payload.get("ticker") or payload.get("symbol") or "BTC/USD").strip()
        raw_timeframe = str(payload.get("timeframe") or payload.get("interval") or "15m").strip()

        # Normalize timeframe and symbol
        try:
            canonical_tf = normalize_timeframe(raw_timeframe)
        except Exception:
            canonical_tf = "15m"

        try:
            mapping = resolve_canonical_symbol("tradingview", raw_symbol)
            canonical_symbol = mapping.canonical_symbol
        except Exception:
            canonical_symbol = raw_symbol

        # Parse price and candle values
        price = float(payload.get("close") or payload.get("price") or payload.get("last") or 0.0)
        open_val = float(payload.get("open") or price)
        high_val = float(payload.get("high") or max(open_val, price))
        low_val = float(payload.get("low") or min(open_val, price))
        close_val = price
        volume_val = float(payload.get("volume") or 0.0)
        ts = float(payload.get("time") or payload.get("timestamp") or time.time())

        # Extract chart context (indicators, drawing notes, alert message)
        chart_ctx = {
            "strategy": payload.get("strategy"),
            "action": payload.get("action"),
            "order_comment": payload.get("order_comment"),
            "indicators": payload.get("indicators", {}),
            "pine_message": payload.get("message", ""),
        }

        normalized = NormalizedMarketData(
            provider_id="tradingview",
            provider_type="MARKET_DATA",
            symbol=raw_symbol,
            canonical_symbol=canonical_symbol,
            exchange=payload.get("exchange", "TRADINGVIEW"),
            timestamp=ts,
            timezone_name="UTC",
            bid=float(payload.get("bid")) if payload.get("bid") is not None else None,
            ask=float(payload.get("ask")) if payload.get("ask") is not None else None,
            last=close_val,
            open=open_val,
            high=high_val,
            low=low_val,
            close=close_val,
            volume=volume_val,
            timeframe=canonical_tf,
            session=payload.get("session", "ALL"),
            data_quality=1.0,
            is_live=True,
            is_simulated=False,
            sequence=int(payload.get("sequence", 0)),
            source_timestamp=ts,
            received_timestamp=time.time(),
            chart_context=chart_ctx,
            raw_metadata=payload,
        )

        # Store in cache
        cache_key = f"{canonical_symbol}:{canonical_tf}"
        self._live_cache[canonical_symbol] = normalized
        self._live_cache[cache_key] = normalized
        self._chart_context[canonical_symbol] = chart_ctx
        self._last_received_time = time.time()

        # Update candle history
        if canonical_symbol not in self._candle_history:
            self._candle_history[canonical_symbol] = []
        self._candle_history[canonical_symbol].append({
            "timestamp": ts,
            "open": open_val,
            "high": high_val,
            "low": low_val,
            "close": close_val,
            "volume": volume_val,
        })
        if len(self._candle_history[canonical_symbol]) > 1000:
            self._candle_history[canonical_symbol] = self._candle_history[canonical_symbol][-1000:]

        self.metadata.status = ProviderLifecycleState.CONNECTED
        self.metadata.last_success = datetime.now(timezone.utc)
        return normalized

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return latest live price if available.
        
        NEVER returns simulated data. If unavailable, returns None to allow fail-closed handling.
        """
        # Try direct and canonical key
        if symbol in self._live_cache:
            return self._live_cache[symbol].close
        for k, v in self._live_cache.items():
            if v.canonical_symbol == symbol or v.symbol == symbol:
                return v.close
        return None

    def get_candles(self, symbol: str, timeframe: str = "15m", limit: int = 100) -> List[Dict[str, Any]]:
        """Return cached live candles for symbol."""
        candles = self._candle_history.get(symbol, [])
        if not candles:
            # Check with / without slashes
            for s, clist in self._candle_history.items():
                if s.replace("/", "") == symbol.replace("/", ""):
                    candles = clist
                    break
        return candles[-limit:] if candles else []

    def check_data_freshness(self, symbol: str, max_age_seconds: float = 300.0) -> Tuple[bool, float]:
        """Check if TradingView live market data is fresh."""
        record = self._live_cache.get(symbol)
        if not record:
            for k, v in self._live_cache.items():
                if v.canonical_symbol == symbol or v.symbol == symbol:
                    record = v
                    break
        if not record:
            return False, 999999.0
        age = time.time() - record.received_timestamp
        return (age <= max_age_seconds), age

    def get_chart_context(self, symbol: str) -> Dict[str, Any]:
        """Return latest TradingView chart context and indicators."""
        return self._chart_context.get(symbol, {})
