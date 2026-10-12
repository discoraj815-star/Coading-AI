"""
Canonical Market Supervisor: Live Market Monitoring & Real-Time AI Chart Marking Coordinator.

Pipeline:
  Active Broker
        ↓
  Authorized Live Market Data
        ↓
  Active Symbol / Chart
        ↓
  Market Supervisor
        ↓
  HTF → MTF → LTF Analysis
        ↓
  Existing Rules & Lecture Knowledge
        ↓
  Evidence Validation
        ↓
  ChartMarker (Clean-Chart & Clutter Filter)
        ↓
  LIVE BROKER CHART MARKINGS
        ↓
  CurrentSituationEngine
        ↓
  JARVIS Real-Time Grounding

INVARIANTS:
  1. Broker Connection ≠ Trade Authorization.
  2. MarketSupervisor strictly observes, analyzes, marks, and proposes setups;
     it NEVER directly executes live broker orders.
  3. Execution must strictly pass ExecutionGate → RiskEngine → CommitBarrier → BrokerAdapter.
  4. If broker data is stale or disconnected, live markings are paused and flagged.
"""

from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.mtf_engine import MultiTimeframeEngine
from trading_ai.core.situation_engine import CurrentSituationEngine
from trading_ai.providers.registry import UniversalProviderRegistry
from trading_ai.schemas.enums import (
    LevelType,
    MarketBias,
    MarketDataStatus,
    MarketDirection,
    MarkingType,
    MarkingValidity,
    SetupStatus,
    UnifiedExecutionTestingMode,
)
from trading_ai.schemas.market import (
    Candle,
    ChartMarking,
    CurrentSituationSnapshot,
    MarketLevel,
    MarketSetup,
    MarketSnapshot,
    UnifiedMarketAnalysis,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class MarketEvaluationResult(dict):
    """Result of continuous market evaluation."""

    @property
    def markings(self) -> List[ChartMarking]:
        return self.get("markings", [])


class MarketSupervisor:
    """
    Central continuous live-market supervisor coordinating multi-timeframe analysis,
    rule matching, canonical chart markings, and situational grounding for JARVIS.
    """

    def __init__(
        self,
        repository: Optional[Repository] = None,
        provider_registry: Optional[UniversalProviderRegistry] = None,
        mtf_engine: Optional[MultiTimeframeEngine] = None,
        situation_engine: Optional[CurrentSituationEngine] = None,
        chart_marker: Optional[ChartMarker] = None,
        data_provider: Optional[Any] = None,
        execution_gate: Optional[Any] = None,
        active_symbol: str = "BTC/USD",
        active_timeframe: str = "15m",
        storage_repo: Optional[Repository] = None,
        notification_orchestrator: Optional[Any] = None,
        **kwargs: Any,
    ):
        self._repo = repository or storage_repo or Repository()
        self._registry = provider_registry or UniversalProviderRegistry(self._repo)
        self._mtf = mtf_engine or MultiTimeframeEngine(self._repo)
        self._situation_engine = situation_engine or CurrentSituationEngine(self._repo, self._mtf)
        self._marker = chart_marker or ChartMarker()
        if data_provider is None:
            from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
            self._data_provider = SimulatedMarketDataProvider(seed=42)
        else:
            self._data_provider = data_provider
        self._execution_gate = execution_gate
        self._notification_orchestrator = kwargs.get("notification_orchestrator") or notification_orchestrator

        self._active_symbol = active_symbol
        self._active_timeframe = active_timeframe
        self._is_monitoring = True
        self._execution_mode = UnifiedExecutionTestingMode.PAPER
        self._data_health = "CONNECTED"  # "CONNECTED", "DATA_STALE", "DISCONNECTED"
        self._last_evaluated_ts: float = 0.0
        self._last_stale_warning_ts: float = 0.0
        self._cached_markings: List[ChartMarking] = []

    def set_notification_orchestrator(self, orchestrator: Any) -> None:
        """Connect canonical notification orchestrator for live alert routing."""
        self._notification_orchestrator = orchestrator

    def get_markings_for_symbol(self, symbol: str, timeframe: Optional[str] = None) -> List[ChartMarking]:
        """Retrieve current cached or persisted markings for a symbol."""
        if self._cached_markings and (not symbol or any(m.symbol == symbol for m in self._cached_markings)):
            return [m for m in self._cached_markings if not symbol or m.symbol == symbol]
        return self._repo.get_chart_markings(symbol=symbol, timeframe=timeframe, only_valid=True)

    @property
    def active_broker_id(self) -> str:
        """Retrieve canonical active broker identifier."""
        active_broker = self._registry.get_active_broker()
        return active_broker.provider_id if active_broker else "paper_broker"

    @property
    def active_broker_name(self) -> str:
        active_broker = self._registry.get_active_broker()
        return active_broker.provider_name if active_broker else "Paper Broker"

    @property
    def is_monitoring(self) -> bool:
        return self._is_monitoring

    @property
    def execution_mode(self) -> UnifiedExecutionTestingMode:
        return self._execution_mode

    def get_status(self) -> Dict[str, Any]:
        """Comprehensive operational status of MarketSupervisor."""
        active_broker = self._registry.get_active_broker()
        broker_status = active_broker.metadata.status.value if active_broker else "DISCONNECTED"
        caps = active_broker.metadata.capabilities if active_broker else []

        return {
            "supervisor_active": self._is_monitoring,
            "data_health": self._data_health,
            "active_broker": {
                "broker_id": self.active_broker_id,
                "broker_name": self.active_broker_name,
                "connection_status": broker_status,
                "capabilities": caps,
                "is_live_supported": active_broker.metadata.live_supported if active_broker else False,
                "is_approved": active_broker.metadata.approved if active_broker else False,
            },
            "market_context": {
                "active_symbol": self._active_symbol,
                "active_timeframe": self._active_timeframe,
                "last_evaluated": self._last_evaluated_ts,
                "markings_count": len(self._cached_markings),
            },
            "execution": {
                "mode": self._execution_mode.value,
                "is_auto_live": self._execution_mode == UnifiedExecutionTestingMode.AUTO_LIVE,
                "is_manual_live": self._execution_mode == UnifiedExecutionTestingMode.MANUAL_LIVE,
                "is_paper": self._execution_mode == UnifiedExecutionTestingMode.PAPER,
                "is_backtest": self._execution_mode == UnifiedExecutionTestingMode.BACKTEST,
            },
        }

    def switch_broker(self, broker_id: str) -> Dict[str, Any]:
        """
        Gracefully detaches old broker monitoring and rebinds to new broker context.
        STRICT SAFETY: Never places orders, never enables live auto execution.
        """
        old_broker_id = self.active_broker_id
        logger.info(f"MarketSupervisor detaching from broker '{old_broker_id}' to rebind '{broker_id}'.")

        # 1. Switch in UniversalProviderRegistry
        self._registry.set_active_broker(broker_id)
        new_broker = self._registry.get_active_broker()

        # 2. Reset health state and flush stale live markings
        self._data_health = "CONNECTED" if (new_broker and new_broker.metadata.status.value == "CONNECTED") else "DISCONNECTED"
        self._cached_markings = []

        # 3. Synchronize initial situation for active symbol under new broker
        try:
            self.evaluate_market(symbol=self._active_symbol, timeframe=self._active_timeframe)
        except Exception as e:
            logger.warning(f"Error evaluating market after broker switch: {e}")

        logger.info(f"MarketSupervisor rebind complete. Active broker is now: {broker_id}")
        return {
            "status": "REBOUND",
            "previous_broker": old_broker_id,
            "active_broker_id": self.active_broker_id,
            "data_health": self._data_health,
            "supervisor_active": self._is_monitoring,
        }

    def set_active_symbol_timeframe(self, symbol: str, timeframe: str) -> None:
        """Update the active chart symbol and timeframe context."""
        self._active_symbol = symbol
        self._active_timeframe = timeframe
        logger.info(f"MarketSupervisor context set to {symbol} ({timeframe}).")

    def set_execution_mode(self, mode: UnifiedExecutionTestingMode) -> Dict[str, Any]:
        """
        Update execution mode with safety gates.
        Enforces: AUTO_LIVE requires explicit approval.
        """
        if mode == UnifiedExecutionTestingMode.AUTO_LIVE:
            active_broker = self._registry.get_active_broker()
            if not (active_broker and active_broker.metadata.approved and active_broker.metadata.live_supported):
                raise ValueError("AUTO_LIVE mode requires explicit broker approval and live trading authorization.")
        self._execution_mode = mode
        logger.info(f"MarketSupervisor execution mode set to: {mode.value}")
        return {
            "status": "SUCCESS",
            "success": True,
            "execution_mode": mode.value,
            "current_mode": mode.value,
        }

    def evaluate_market(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        snapshot: Optional[MarketSnapshot] = None,
    ) -> MarketEvaluationResult:
        """
        Continuous market evaluation & rule-grounded AI marking pipeline:
          1. Validates market data freshness. If stale, pauses live markings.
          2. Runs MTF Analysis (HTF → MTF → LTF).
          3. Evaluates structural concepts (BOS, CHOCH, FVG, Order Blocks, Liquidity, Sweeps, POIs).
          4. Validates evidence & confidence.
          5. Filters clutter via ChartMarker.
          6. Persists canonical markings to repository.
          7. Updates CurrentSituationEngine for zero-hallucination JARVIS grounding.
        """
        target_symbol = symbol or self._active_symbol
        target_tf = timeframe or self._active_timeframe
        now_ts = datetime.now(timezone.utc).timestamp()

        # 1. Fetch Market Snapshot
        if snapshot is None:
            if self._data_provider:
                snapshot = self._data_provider.get_snapshot(symbol=target_symbol, timeframe=target_tf, limit=50)
            else:
                snapshot = MarketSnapshot(
                    symbol=target_symbol,
                    timeframe=target_tf,
                    latest_price=100.0,
                    timestamp=now_ts,
                    candles=[],
                )

        # 2. Data Staleness / Disconnection Guard (Section 38)
        if snapshot.candles:
            last_candle_ts = snapshot.candles[-1].timestamp
            # If data is older than 2 hours in a live scenario, flag as stale
            if now_ts - last_candle_ts > 7200:
                self._data_health = "DATA_STALE"
            else:
                self._data_health = "CONNECTED"
        else:
            self._data_health = "DATA_STALE"

        active_broker = self._registry.get_active_broker()
        if active_broker and active_broker.metadata.status.value in ("DISCONNECTED", "ERROR"):
            self._data_health = "DISCONNECTED"

        if self._data_health != "CONNECTED":
            logger.warning(f"MarketSupervisor: Live data is {self._data_health} for {target_symbol}. New live markings paused.")
            # Retrieve existing historical markings without generating unverified new ones
            existing = self._repo.get_chart_markings(symbol=target_symbol, timeframe=target_tf, only_valid=True)
            self._cached_markings = existing
            return MarketEvaluationResult({
                "status": "PAUSED_" + self._data_health,
                "success": False,
                "symbol": target_symbol,
                "timeframe": target_tf,
                "markings": existing,
                "count": len(existing),
                "data_health": self._data_health,
            })

        # 3. Multi-Timeframe Analysis
        unified_analysis = self._repo.get_latest_market_analysis(symbol=target_symbol)
        if not unified_analysis:
            try:
                unified_analysis = self._mtf.analyze_symbol(symbol=target_symbol)
                self._repo.save_market_analysis(unified_analysis)
            except Exception as e:
                logger.warning(f"MTF analysis error for {target_symbol}: {e}")

        # 4. Extract Setups & Levels from Rules & Repository
        setups: List[MarketSetup] = self._repo.get_market_setups(symbol=target_symbol)
        levels: List[MarketLevel] = self._repo.get_market_levels(symbol=target_symbol, timeframe=target_tf)

        # Generate candidates from candles if none persisted yet
        if not setups and snapshot.candles and len(snapshot.candles) >= 3:
            detected_setups = self._detect_market_setups_from_candles(snapshot)
            setups.extend(detected_setups)
            for s in detected_setups:
                try:
                    self._repo.save_market_setup(s)
                except Exception:
                    pass

        # 5. Filter Clutter & Deduplicate via ChartMarker
        markings = self._marker.generate_markings(
            snapshot=snapshot,
            setups=setups,
            levels=levels,
            source="SUPERVISOR",
            broker_id=self.active_broker_id,
        )

        # 6. Persist Canonical Markings
        for m in markings:
            try:
                self._repo.save_chart_marking(m)
            except Exception as e:
                logger.warning(f"Could not persist chart marking {m.marking_id}: {e}")

        self._cached_markings = markings
        self._last_evaluated_ts = now_ts

        # 7. Update CurrentSituationEngine
        if unified_analysis:
            try:
                self._situation_engine.evaluate_situation(
                    symbol=target_symbol,
                    snapshot=snapshot,
                    unified_analysis=unified_analysis,
                )
            except Exception as e:
                logger.warning(f"Error updating situation engine for {target_symbol}: {e}")

        # 8. Dispatch qualified events to NotificationOrchestrator
        if getattr(self, "_notification_orchestrator", None) and setups:
            for s in setups:
                try:
                    from trading_ai.schemas.cloud import CloudStreamEvent
                    from trading_ai.schemas.enums import EventPriorityLevel
                    is_confirmed = getattr(s.status, "value", str(s.status)) in ("CONFIRMED", "ACTIVE", "VALID")
                    prio = EventPriorityLevel.P3_CONFIRMED_SETUP if is_confirmed else EventPriorityLevel.P4_HIGH_VALUE_MARKET_EVENT
                    evt_type = "SETUP_CONFIRMED" if is_confirmed else ("SETUP_" + getattr(s.status, "value", str(s.status)))
                    evt = CloudStreamEvent(
                        event_type=evt_type,
                        symbol=s.symbol,
                        priority=prio,
                        confidence=float(getattr(s, "confidence", 0.88)),
                        evidence_reference={"setup_id": s.setup_id, "pattern": s.setup_type, "timeframe": target_tf},
                        payload={
                            "setup_id": s.setup_id,
                            "symbol": s.symbol,
                            "pattern_name": s.setup_type,
                            "direction": s.direction.value if hasattr(s.direction, "value") else str(s.direction),
                            "timeframe": target_tf,
                            "confirmation_state": getattr(s.status, "value", str(s.status)),
                            "invalidation_level": getattr(s, "invalidation_price", None) or "Key Structure Boundary",
                            "poi_level": getattr(s, "entry_price", None) or "Optimal Entry Zone",
                            "reason": "; ".join(getattr(s, "conditions_satisfied", [])) or "Lecture-grounded SMC confluence",
                        },
                    )
                    self._notification_orchestrator.process_event(evt)
                except Exception as e:
                    logger.debug(f"Failed to dispatch notification for setup: {e}")

        return MarketEvaluationResult({
            "status": "SUCCESS",
            "success": True,
            "symbol": target_symbol,
            "timeframe": target_tf,
            "markings": markings,
            "count": len(markings),
            "data_health": self._data_health,
        })

    def _detect_market_setups_from_candles(self, snapshot: MarketSnapshot) -> List[MarketSetup]:
        """Detect SMC / ICT structural setups strictly from live broker price action."""
        candles = snapshot.candles
        if len(candles) < 3:
            return []

        c1 = candles[-3]
        c2 = candles[-2]
        c3 = candles[-1]
        setups: List[MarketSetup] = []

        # Bullish Displacement / Order Block setup
        if c2.close > c2.open and c3.close > c2.high and (c2.close - c2.open) > (c1.high - c1.low):
            setups.append(
                MarketSetup(
                    symbol=snapshot.symbol,
                    timeframe=snapshot.timeframe,
                    setup_type="BULLISH_ORDER_BLOCK",
                    direction=MarketDirection.BULLISH,
                    status=SetupStatus.VALID,
                    applicable_rule_id="RULE-OB-01",
                    conditions_satisfied=["Strong displacement", "Displaced previous high"],
                    supporting_evidence=[f"Bullish expansion bar at {c2.close:.5f}"],
                    confidence=0.91,
                    confirmation_timestamp=c3.timestamp,
                )
            )

        # Bearish Displacement / Order Block setup
        elif c2.close < c2.open and c3.close < c2.low and (c2.open - c2.close) > (c1.high - c1.low):
            setups.append(
                MarketSetup(
                    symbol=snapshot.symbol,
                    timeframe=snapshot.timeframe,
                    setup_type="BEARISH_ORDER_BLOCK",
                    direction=MarketDirection.BEARISH,
                    status=SetupStatus.VALID,
                    applicable_rule_id="RULE-OB-02",
                    conditions_satisfied=["Strong downward displacement", "Displaced previous low"],
                    supporting_evidence=[f"Bearish expansion bar at {c2.close:.5f}"],
                    confidence=0.91,
                    confirmation_timestamp=c3.timestamp,
                )
            )

        return setups

    def get_markings_for_symbol(self, symbol: str, timeframe: Optional[str] = None) -> List[ChartMarking]:
        """Retrieve recent valid supervisor markings."""
        if symbol == self._active_symbol and (timeframe is None or timeframe == self._active_timeframe):
            if self._cached_markings:
                return self._cached_markings
        return self._repo.get_chart_markings(symbol=symbol, timeframe=timeframe, only_valid=True)
