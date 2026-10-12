"""Step 18: Economic News and Calendar Providers (Forex Factory & Generic)."""
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional
import logging

from trading_ai.fundamental.forex_factory import ForexFactoryProvider
from trading_ai.providers.base import NewsProvider
from trading_ai.schemas.provider import (
    ConnectionTestResult,
    ProviderCategory,
    ProviderLifecycleState,
    ProviderMetadata,
    TradingProviderExecutionState,
)

logger = logging.getLogger(__name__)


class ForexFactoryNewsProvider(NewsProvider):
    """
    Forex Factory Economic Calendar and News Provider.

    Hard Epistemic Rules:
      - Interacts via the official public JSON calendar feed.
      - Gracefully handles rate limiting (HTTP 429) and upstream network errors without crashing.
      - Never fabricates synthetic economic events.
      - Feeds directly into Fundamental Context -> Event Risk -> Master Decision Engine.
      - High impact adverse events trigger FUNDAMENTAL_NEWS_CONFLICT -> NO_TRADE.
    """

    def __init__(self, calendar_url: Optional[str] = None):
        meta = ProviderMetadata(
            provider_id="forex_factory",
            provider_name="Forex Factory Economic Calendar",
            provider_type=ProviderCategory.NEWS,
            version="1.0.0",
            status=ProviderLifecycleState.CONNECTED,
            trading_state=TradingProviderExecutionState.PAPER_ONLY,
            capabilities=[
                "ECONOMIC_CALENDAR",
                "HIGH_IMPACT_EVENT_DETECTION",
                "CURRENCY_FILTERING",
                "NEWS_TIMING_RISK",
            ],
            supported_symbols=["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "NZD", "XAU"],
            supported_markets=["Forex", "Global Macro"],
            supported_timeframes=["*"],
            authentication_type="PUBLIC_HTTP_JSON",
            configuration_schema={
                "calendar_url": "string (URL to Forex Factory calendar JSON feed)",
                "timeout_seconds": "float (Default: 5.0)",
            },
            health_status="HEALTHY",
            sandbox_available=True,
            paper_supported=True,
            live_supported=True,
            enabled=True,
            approved=True,
        )
        super().__init__(meta)
        self._provider = ForexFactoryProvider(calendar_url=calendar_url) if calendar_url else ForexFactoryProvider()

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        if config and config.get("calendar_url"):
            self._provider = ForexFactoryProvider(calendar_url=config["calendar_url"])
        self.metadata.status = ProviderLifecycleState.CONNECTED
        return True

    def test_connection(self) -> ConnectionTestResult:
        t0 = time.time()
        try:
            status = self._provider.get_provider_status()
            latency = (time.time() - t0) * 1000.0
            is_ok = status.status.value in ("OPERATIONAL", "STALE")
            return ConnectionTestResult(
                success=is_ok,
                provider_id=self.provider_id,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                message=f"Forex Factory calendar feed status: {status.status.value}.",
                capabilities_verified=self.metadata.capabilities,
                details={
                    "status": status.status.value,
                    "is_stale": status.is_stale,
                    "events_cached": len(self._provider._cached_events),
                },
            )
        except Exception as e:
            latency = (time.time() - t0) * 1000.0
            return ConnectionTestResult(
                success=False,
                provider_id=self.provider_id,
                provider_name=self.provider_name,
                latency_ms=round(latency, 2),
                message=f"Forex Factory test failed: {e}",
                capabilities_verified=[],
                details={"error": str(e)},
            )

    def get_upcoming_events(
        self,
        currencies: Optional[List[str]] = None,
        lookahead_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        events = self._provider.fetch_events(start_time=now, currencies=currencies)
        return [e.model_dump() for e in events]

    get_economic_calendar = get_upcoming_events

    def check_news_risk(self, symbol: str, window_minutes: float = 30.0) -> Dict[str, Any]:
        curr = symbol.split("/")[0] if "/" in symbol else symbol[:3]
        events = self._provider.fetch_events(currencies=[curr])
        high_impact = [e for e in events if getattr(e, "impact", None) and getattr(e.impact, "value", "") == "HIGH"]
        return {
            "symbol": symbol,
            "has_high_impact_near": len(high_impact) > 0,
            "event_count": len(events),
            "high_impact_count": len(high_impact),
        }
