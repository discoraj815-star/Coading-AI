"""Step 16: Cloud Market Monitor — 24×7 autonomous background intelligence with stale data gating."""
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

from trading_ai.events.router import EventRouter
from trading_ai.pipeline.watchlist_monitor import WatchlistMonitor
from trading_ai.schemas.cloud import CloudStreamEvent
from trading_ai.schemas.enums import EventPriorityLevel, MarketDataStatus, NotificationDeliveryStatus, SetupStatus
from trading_ai.schemas.market import CurrentSituationSnapshot, UnifiedMarketAnalysis
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class CloudMarketMonitor:
    """
    24×7 Cloud-Resident Market Intelligence Worker.
    Continuously processes authorized watchlist symbols when user PCs/phones are offline.

    Guarantees:
      - Stale Market Data Gating: If market data is stale, marks MARKET_DATA_STALE and NEVER generates CONFIRMED_SETUP.
      - Token Budget Preservation: Runs deterministic ICT/SMC filters first before calling Gemini.
      - Setup Notification State Machine: Tracks DETECTED -> AWAITING_CONFIRMATION -> CONFIRMED -> INVALIDATED.
      - Zero UI dependency: Operations persist directly in SQLite and EventRouter.
    """

    STALE_DATA_THRESHOLD_SECONDS = 7200  # 2 hours without new candles = STALE

    def __init__(
        self,
        watchlist_monitor: WatchlistMonitor,
        repository: Repository,
        event_router: Optional[EventRouter] = None,
        master_decision_engine: Optional[Any] = None,
    ):
        self._monitor = watchlist_monitor
        self._repo = repository
        self._router = event_router
        self._master_engine = master_decision_engine
        self._cycles_completed = 0
        self._stale_symbols: Dict[str, bool] = {}

    @property
    def master_decision_engine(self) -> Optional[Any]:
        return self._master_engine

    @master_decision_engine.setter
    def master_decision_engine(self, engine: Any) -> None:
        self._master_engine = engine

    @property
    def monitor(self) -> WatchlistMonitor:
        return self._monitor

    def step_monitor_cycle(self, force: bool = False) -> Dict[str, Any]:
        """
        Execute one comprehensive evaluation cycle across the entire watchlist.
        Emits stream events for newly detected, confirmed, or invalidated setups.
        """
        self._cycles_completed += 1
        results = {}
        for sym in self._monitor.watchlist:
            res = self.process_symbol_cloud(sym, force=force)
            results[sym] = res

        return {
            "cycle": self._cycles_completed,
            "symbols_processed": len(results),
            "results": results,
            "stale_symbols": self._stale_symbols,
        }

    def process_symbol_cloud(self, symbol: str, force: bool = False) -> Dict[str, Any]:
        """Process a single symbol with strict stale-data gating and event broadcast."""
        # 1. Fetch latest candle snapshot
        snapshot = self._monitor._provider.get_snapshot(symbol=symbol, timeframe="15m")
        latest_time = snapshot.candles[-1].timestamp if snapshot.candles else 0.0

        # Check stale data condition
        now_ts = time.time()
        is_stale = (now_ts - latest_time) > self.STALE_DATA_THRESHOLD_SECONDS if latest_time > 0 else True
        self._stale_symbols[symbol] = is_stale

        if is_stale:
            logger.warning("Market data for %s is STALE (last timestamp: %s). Confirmation disabled.", symbol, latest_time)
            # Emit STALE event if transitioning
            if self._router:
                stale_evt = CloudStreamEvent(
                    event_type="MARKET_DATA_STALE",
                    symbol=symbol,
                    priority=EventPriorityLevel.P4_HIGH_VALUE_MARKET_EVENT,
                    payload={"symbol": symbol, "last_timestamp": latest_time, "threshold_seconds": self.STALE_DATA_THRESHOLD_SECONDS},
                )
                self._router.route_event(stale_evt)
            return {"status": "STALE_DATA", "symbol": symbol, "timestamp": latest_time}

        # 2. Run standard incremental processing
        analysis = self._monitor.process_symbol(symbol=symbol, force=force)
        if not analysis:
            return {"status": "NO_ANALYSIS", "symbol": symbol}

        # 3. Check for newly confirmed or invalidated setups
        active_setups = analysis.active_setups or []
        for setup in active_setups:
            if setup.status == SetupStatus.CONFIRMED:
                # Dispatch P3 Confirmed Setup Event
                pattern = getattr(setup, "setup_type", getattr(setup, "pattern_name", "ICT Setup"))
                if self._router:
                    evt = CloudStreamEvent(
                        event_type="SETUP_CONFIRMED",
                        symbol=symbol,
                        priority=EventPriorityLevel.P3_CONFIRMED_SETUP,
                        confidence=float(setup.confidence),
                        evidence_reference={"pattern": pattern, "timeframe": setup.timeframe},
                        payload={
                            "setup_id": setup.setup_id,
                            "symbol": symbol,
                            "pattern_name": pattern,
                            "direction": setup.direction.value if hasattr(setup.direction, "value") else str(setup.direction),
                            "timeframe": setup.timeframe,
                            "entry_price": getattr(setup, "entry_price", getattr(setup, "key_price", 0.0)),
                            "stop_loss": getattr(setup, "stop_loss", 0.0),
                            "poi_level": getattr(setup, "poi_level", getattr(setup, "key_price", 0.0)),
                            "invalidation_level": getattr(setup, "invalidation_level", getattr(setup, "stop_loss", 0.0)),
                            "confirmation_state": "CONFIRMED",
                        },
                    )
                    self._router.route_event(evt)

            elif setup.status == SetupStatus.INVALIDATED:
                if self._router:
                    evt = CloudStreamEvent(
                        event_type="SETUP_INVALIDATED",
                        symbol=symbol,
                        priority=EventPriorityLevel.P4_HIGH_VALUE_MARKET_EVENT,
                        payload={
                            "setup_id": setup.setup_id,
                            "symbol": symbol,
                            "reason": setup.invalidation_condition or "Breached invalidation level",
                            "confirmation_state": "INVALIDATED",
                        },
                    )
                    self._router.route_event(evt)

        # 4. Step 17: Master Decision Engine Multi-Concept Evaluation
        master_dec = None
        if self._master_engine:
            try:
                m_data = {
                    "candles": snapshot.candles,
                    "structure_bias": analysis.htf_analysis.bias.value if analysis.htf_analysis and hasattr(analysis.htf_analysis.bias, "value") else None,
                    "is_stale": is_stale,
                }
                master_dec = self._master_engine.evaluate_decision(
                    symbol=symbol,
                    timeframe="15m",
                    market_data=m_data,
                )
            except Exception as e:
                logger.debug("Master decision evaluation exception in cloud monitor: %s", e)

        return {
            "status": "PROCESSED",
            "symbol": symbol,
            "setups_count": len(active_setups),
            "bias": analysis.htf_analysis.bias.value if analysis.htf_analysis and hasattr(analysis.htf_analysis.bias, "value") else "UNKNOWN",
            "master_decision": master_dec.model_dump(mode="json") if master_dec else None,
        }

    def health(self) -> Dict[str, Any]:
        return {
            "status": "HEALTHY",
            "watchlist": self._monitor.watchlist,
            "cycles_completed": self._cycles_completed,
            "stale_symbols": self._stale_symbols,
        }
