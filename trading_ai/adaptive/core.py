"""
Step 11 Adaptive AI Core Architecture.
Coordinates reactive event dispatching, tiered memory, incremental technical analysis,
adaptive compute allocation, performance telemetry, and controlled component upgrades.
Preserves existing Step 1-10 engines without duplication.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional
import uuid

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.adaptive.incremental_engine import IncrementalAnalysisEngine
from trading_ai.adaptive.memory_tier import TieredMemoryManager
from trading_ai.adaptive.priority_scheduler import AdaptiveComputeScheduler
from trading_ai.adaptive.upgrade_manager import ComponentUpgradeManager
from trading_ai.schemas import Candle, MarketBias, MarketSetup, Position
from trading_ai.schemas.enums import EventPriority, SystemEventType
from trading_ai.schemas.performance import (
    AdaptiveComputePolicy,
    IncrementalAnalysisDelta,
    LatencyTelemetry,
    SystemEvent,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class AdaptiveAICore:
    """
    Central orchestration facade for Step 11 High-Performance & Adaptive AI Architecture.
    """

    def __init__(self, repository: Repository):
        self.repository = repository
        self.event_bus = ReactiveEventBus()
        self.memory = TieredMemoryManager(repository=repository)
        self.incremental_engine = IncrementalAnalysisEngine()
        self.scheduler = AdaptiveComputeScheduler()
        self.upgrade_manager = ComponentUpgradeManager(repository=repository)

        # Wire internal event listeners
        self._setup_event_listeners()

    def _setup_event_listeners(self) -> None:
        """Register core event subscriptions."""
        # When a setup changes or is confirmed, update hot memory and emit high priority alert
        self.event_bus.subscribe(
            SystemEventType.SETUP_CONFIRMED,
            lambda evt: logger.info(f"High-priority setup confirmed for {evt.symbol}"),
        )
        self.event_bus.subscribe(
            SystemEventType.PRICE_UPDATE,
            lambda evt: self._handle_price_event(evt),
        )

    def _handle_price_event(self, evt: SystemEvent) -> None:
        """Handle incoming price update on the bus."""
        price = evt.payload.get("price")
        if price is not None:
            self.memory.update_hot_price(evt.symbol, float(price), evt.timestamp)

    def process_market_tick(
        self,
        symbol: str,
        price: float,
        timestamp: Optional[datetime] = None,
        source: str = "LIVE_FEED",
    ) -> Dict[str, Any]:
        """
        Process a raw price tick through the high-performance pipeline:
        Ingestion -> Normalization -> Hot Memory -> Event Dispatch -> Telemetry.
        """
        t0 = time.perf_counter()
        ts = timestamp or datetime.now(timezone.utc)

        # 1. Ingestion & Normalization
        t_ingest_start = time.perf_counter()
        normalized_price = float(price)
        t_norm_end = time.perf_counter()

        # 2. Update HOT Memory (<0.1ms)
        self.memory.update_hot_price(symbol, normalized_price, ts)
        t_mem_end = time.perf_counter()

        # 3. Publish PRICE_UPDATE event
        event = SystemEvent(
            event_id=f"tick_{symbol}_{int(ts.timestamp() * 1000)}",
            event_type=SystemEventType.PRICE_UPDATE,
            priority=EventPriority.MEDIUM,
            symbol=symbol,
            timestamp=ts,
            payload={"price": normalized_price, "source": source},
        )
        accepted = self.event_bus.publish(event)
        self.event_bus.dispatch_next()

        t_end = time.perf_counter()

        # Measure deterministic latencies in ms
        ingest_ms = (t_norm_end - t_ingest_start) * 1000.0
        norm_ms = (t_mem_end - t_norm_end) * 1000.0
        decision_ms = (t_end - t0) * 1000.0

        telemetry = LatencyTelemetry(
            ingestion_latency_ms=round(ingest_ms, 3),
            normalization_latency_ms=round(norm_ms, 3),
            technical_analysis_ms=0.0,
            mtf_synthesis_ms=0.0,
            gemini_reasoning_ms=0.0,
            end_to_end_decision_ms=round(decision_ms, 3),
            queue_wait_ms=0.0,
            measured_at=ts,
        )

        try:
            self.repository.save_latency_telemetry(telemetry, symbol=symbol)
        except Exception as e:
            logger.warning(f"Failed to persist latency telemetry: {e}")

        return {
            "status": "PROCESSED",
            "symbol": symbol,
            "price": normalized_price,
            "event_accepted": accepted,
            "latency_ms": round(decision_ms, 3),
            "timestamp": ts.isoformat(),
        }

    def process_candle(
        self,
        symbol: str,
        timeframe: str,
        candle: Candle,
    ) -> IncrementalAnalysisDelta:
        """
        Process a new candle incrementally:
        Updates warm cache -> runs delta analysis -> emits events if BOS/CHOCH detected.
        """
        t0 = time.perf_counter()

        # 1. Update Hot & Warm Memory
        self.memory.update_hot_candle(symbol, candle)
        self.memory.append_warm_candle(symbol, timeframe, candle)

        # 2. Run Incremental Delta Analysis
        delta = self.incremental_engine.process_new_candle(symbol, timeframe, candle)

        # 3. Publish high-priority events if structure changed
        ts_val = candle.timestamp if isinstance(candle.timestamp, (int, float)) else candle.timestamp.timestamp()
        if delta.new_choch_detected:
            self.event_bus.publish(
                SystemEvent(
                    event_id=f"choch_{symbol}_{timeframe}_{int(ts_val)}",
                    event_type=SystemEventType.CHOCH_DETECTED,
                    priority=EventPriority.HIGH,
                    symbol=symbol,
                    timestamp=ts_val,
                    payload={"delta": delta.__dict__},
                )
            )
        elif delta.new_bos_detected:
            self.event_bus.publish(
                SystemEvent(
                    event_id=f"bos_{symbol}_{timeframe}_{int(ts_val)}",
                    event_type=SystemEventType.BOS_DETECTED,
                    priority=EventPriority.MEDIUM,
                    symbol=symbol,
                    timestamp=ts_val,
                    payload={"delta": delta.__dict__},
                )
            )

        t_end = time.perf_counter()
        analysis_ms = (t_end - t0) * 1000.0

        telemetry = LatencyTelemetry(
            ingestion_latency_ms=0.0,
            normalization_latency_ms=0.0,
            technical_analysis_ms=round(analysis_ms, 3),
            mtf_synthesis_ms=0.0,
            gemini_reasoning_ms=0.0,
            end_to_end_decision_ms=round(analysis_ms, 3),
            queue_wait_ms=0.0,
            measured_at=datetime.now(timezone.utc),
        )
        try:
            self.repository.save_latency_telemetry(telemetry, symbol=symbol)
        except Exception as e:
            logger.warning(f"Failed to persist latency telemetry: {e}")

        return delta

    def get_system_status(self) -> Dict[str, Any]:
        """Return comprehensive Step 11 performance and runtime status."""
        recent_telemetry = self.repository.get_recent_latency_telemetry(limit=10)
        avg_latency = (
            sum(t.end_to_end_decision_ms for t in recent_telemetry) / len(recent_telemetry)
            if recent_telemetry
            else 0.0
        )

        return {
            "status": "HEALTHY",
            "event_bus": self.event_bus.get_queue_stats(),
            "memory_tiers": self.memory.get_tier_stats(),
            "average_latency_ms": round(avg_latency, 2),
            "recent_telemetry_count": len(recent_telemetry),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
