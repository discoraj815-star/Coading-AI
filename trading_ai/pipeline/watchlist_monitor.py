"""Watchlist Background Monitoring Engine: Multi-symbol independent analysis."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.lecture_validator import LectureValidator
from trading_ai.core.market_data_provider import BaseMarketDataProvider
from trading_ai.core.mtf_engine import MultiTimeframeEngine
from trading_ai.core.situation_engine import CurrentSituationEngine
from trading_ai.schemas.enums import (
    JobStatus,
    MarketDataStatus,
    MarketDirection,
    MarkingValidity,
    PipelineStage,
    SetupStatus,
    WatchlistMonitorStatus,
)
from trading_ai.schemas.media import JobMetadata
from trading_ai.schemas.market import (
    ChartMarking,
    MarketLevel,
    MarketSetup,
    UnifiedMarketAnalysis,
    WatchlistMonitorState,
)
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier

logger = logging.getLogger(__name__)


class WatchlistMonitor:
    """
    Independently monitors and analyzes all symbols in a watchlist.

    Guarantees:
      - The visible chart is NOT the only symbol analyzed.
      - Every eligible symbol in the watchlist is queued and analyzed independently.
      - Uses incremental processing: unchanged candle timestamps are not reprocessed.
      - Interacts with CommitBarrier for task-unit tracking and crash recovery.
      - Can be started, paused, resumed, and stopped via user command or API.
    """

    DEFAULT_WATCHLIST = ["BTC/USD", "ETH/USD", "SOL/USD", "EUR/USD"]

    def __init__(
        self,
        repository: Repository,
        data_provider: BaseMarketDataProvider,
        mtf_engine: MultiTimeframeEngine,
        lecture_validator: LectureValidator,
        chart_marker: ChartMarker,
        commit_barrier: Optional[CommitBarrier] = None,
        watchlist: Optional[List[str]] = None,
        situation_engine: Optional[CurrentSituationEngine] = None,
        companion_core: Optional[Any] = None,
    ):
        self._repo = repository
        self._provider = data_provider
        self._mtf = mtf_engine
        self._validator = lecture_validator
        self._marker = chart_marker
        self._situation_engine = situation_engine or CurrentSituationEngine(repository=repository, chart_marker=chart_marker)
        self._companion_core = companion_core
        self._barrier = commit_barrier or CommitBarrier(repository)
        self._watchlist = list(watchlist or self.DEFAULT_WATCHLIST)
        self._status = WatchlistMonitorStatus.IDLE

        # Initialize watchlist state in DB if not present
        for sym in self._watchlist:
            if not self._repo.get_watchlist_monitor_state(sym):
                self._repo.save_watchlist_monitor_state(
                    WatchlistMonitorState(
                        symbol=sym,
                        timeframe="15m",
                        latest_status=WatchlistMonitorStatus.IDLE,
                        worker_state="IDLE",
                    )
                )

    @property
    def watchlist(self) -> List[str]:
        return self._watchlist

    @property
    def status(self) -> WatchlistMonitorStatus:
        return self._status

    def add_symbol(self, symbol: str) -> None:
        """Add symbol to active watchlist."""
        if symbol not in self._watchlist:
            self._watchlist.append(symbol)
            self._repo.save_watchlist_monitor_state(
                WatchlistMonitorState(
                    symbol=symbol,
                    timeframe="15m",
                    latest_status=self._status,
                    worker_state="IDLE",
                )
            )

    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from active watchlist."""
        if symbol in self._watchlist:
            self._watchlist.remove(symbol)

    def start(self) -> None:
        """Start watchlist background monitoring."""
        self._status = WatchlistMonitorStatus.MONITORING
        self._update_all_states(WatchlistMonitorStatus.MONITORING, "RUNNING")

    def pause(self) -> None:
        """Pause watchlist background monitoring."""
        self._status = WatchlistMonitorStatus.PAUSED
        self._update_all_states(WatchlistMonitorStatus.PAUSED, "PAUSED")

    def resume(self) -> None:
        """Resume watchlist background monitoring."""
        self._status = WatchlistMonitorStatus.MONITORING
        self._update_all_states(WatchlistMonitorStatus.MONITORING, "RUNNING")

    def stop(self) -> None:
        """Stop watchlist background monitoring."""
        self._status = WatchlistMonitorStatus.STOPPED
        self._update_all_states(WatchlistMonitorStatus.STOPPED, "STOPPED")

    def _update_all_states(self, status: WatchlistMonitorStatus, worker_state: str) -> None:
        for sym in self._watchlist:
            st = self._repo.get_watchlist_monitor_state(sym)
            if st:
                st.latest_status = status
                st.worker_state = worker_state
                st.updated_at = datetime.now(timezone.utc)
                self._repo.save_watchlist_monitor_state(st)

    def process_symbol(self, symbol: str, force: bool = False) -> Optional[UnifiedMarketAnalysis]:
        """
        Run full analysis pipeline for a single watchlist symbol.

        Incremental: Skips analysis if latest candle timestamp is unchanged
        unless force=True.
        """
        state = self._repo.get_watchlist_monitor_state(symbol)
        if not state:
            state = WatchlistMonitorState(symbol=symbol)

        # 1. Check Market Data
        ltf_tf = self._mtf.hierarchy.get(self._mtf.hierarchy.get("LTF", "15m"), "15m")
        snapshot = self._provider.get_snapshot(symbol=symbol, timeframe="15m")

        if snapshot.status in (MarketDataStatus.DATA_MISSING, MarketDataStatus.PROVIDER_ERROR):
            state.last_error = snapshot.error or "Market data unavailable"
            state.retry_count += 1
            state.worker_state = "ERROR"
            self._repo.save_watchlist_monitor_state(state)
            return None

        # Incremental check: has new candle arrived?
        latest_candle_time = snapshot.candles[-1].timestamp if snapshot.candles else 0.0
        if not force and latest_candle_time <= state.last_processed_timestamp and state.last_processed_timestamp > 0.0:
            # Unchanged data — skip expensive re-analysis (token & compute conservation)
            return self._repo.get_latest_market_analysis(symbol)

        state.worker_state = "ANALYZING"
        self._repo.save_watchlist_monitor_state(state)

        # 2. Multi-Timeframe Analysis (HTF -> MTF -> LTF)
        unified_analysis = self._mtf.analyze_symbol(symbol=symbol)

        # 3. Detect Potential Setups on LTF & MTF
        ltf_analysis = unified_analysis.ltf_analysis
        detected_setups: List[MarketSetup] = []

        if ltf_analysis and ltf_analysis.direction != MarketDirection.UNKNOWN:
            # Test pattern detection: Bullish/Bearish FVG or BOS
            pattern_name = "FVG"
            pattern_context = {
                "displacement": True,
                "is_invalidated": False,
            }
            validated_setup = self._validator.validate_detected_pattern(
                symbol=symbol,
                timeframe=ltf_analysis.timeframe,
                pattern_name=pattern_name,
                direction=ltf_analysis.direction,
                market_analysis=unified_analysis,
                pattern_context=pattern_context,
            )
            detected_setups.append(validated_setup)
            self._repo.save_market_setup(validated_setup)

        # 4. Generate Clean Chart Markings
        all_levels = (unified_analysis.htf_analysis.key_levels if unified_analysis.htf_analysis else []) + (
            unified_analysis.ltf_analysis.key_levels if unified_analysis.ltf_analysis else []
        )
        for lvl in all_levels:
            self._repo.save_market_level(lvl)

        markings = self._marker.generate_markings(
            snapshot=snapshot,
            setups=detected_setups,
            levels=all_levels,
        )
        for m in markings:
            self._repo.save_chart_marking(m)

        # 5. Commit Barrier Registration (Job tracking)
        # Ensure job exists for this symbol monitor run
        monitor_job_id = f"monitor_{symbol.replace('/', '_')}"
        if not self._repo.get_job(monitor_job_id):
            self._repo.save_job(
                JobMetadata(
                    job_id=monitor_job_id,
                    file_hash=f"hash_{monitor_job_id}",
                    filename=f"{symbol}_monitor",
                    file_path=f"market://{symbol}",
                    status=JobStatus.COMPLETED,
                    current_stage=PipelineStage.STORAGE,
                )
            )
        task_unit = self._barrier.register_task_unit(
            job_id=monitor_job_id,
            stage=PipelineStage.STORAGE,
            start_time=latest_candle_time,
            end_time=time.time(),
            payload={"symbol": symbol, "setups_count": len(detected_setups)},
        )
        self._barrier.complete_task_unit(
            job_id=monitor_job_id,
            serial_number=task_unit.serial_number,
            result={"symbol": symbol, "status": "COMPLETED"},
        )

        # 6. Step 5 Lifecycle Confirmation & Current Situation Evaluation
        confirmed_setups = []
        for s in detected_setups:
            s_evaluated = self._validator.evaluate_setup_confirmation(
                setup=s,
                current_price=snapshot.latest_price,
                market_analysis=unified_analysis,
            )
            self._repo.save_market_setup(s_evaluated)
            confirmed_setups.append(s_evaluated)

        # Fetch previous situation for bias change detection if available
        prev_situation = self._repo.get_latest_current_situation(symbol)

        # Generate lightweight decision-relevant CurrentSituationSnapshot
        current_situation = self._situation_engine.evaluate_situation(
            symbol=symbol,
            snapshot=snapshot,
            unified_analysis=unified_analysis,
        )

        # Step 7 Proactive Alert Trigger
        if self._companion_core:
            try:
                self._companion_core.evaluate_symbol_alerts(
                    symbol=symbol,
                    current_situation=current_situation,
                    previous_situation=prev_situation,
                )
            except Exception as e:
                logger.warning("Companion alert evaluation failed for %s: %s", symbol, e)

        # 7. Persist Unified Analysis
        unified_analysis.active_setups = confirmed_setups
        unified_analysis.active_markings = markings
        self._repo.save_market_analysis(unified_analysis)
        self._repo.save_market_snapshot(snapshot)

        # 8. Update Watchlist State
        state.last_processed_timestamp = latest_candle_time
        state.worker_state = "IDLE" if self._status != WatchlistMonitorStatus.MONITORING else "MONITORING"
        state.latest_status = self._status
        state.updated_at = datetime.now(timezone.utc)
        self._repo.save_watchlist_monitor_state(state)

        return unified_analysis

    def process_all_watchlist(self, force: bool = False) -> Dict[str, Optional[UnifiedMarketAnalysis]]:
        """
        Iterate over and process all symbols in the watchlist independently.
        The visible chart is NOT the only symbol analyzed!
        """
        results: Dict[str, Optional[UnifiedMarketAnalysis]] = {}
        for sym in self._watchlist:
            results[sym] = self.process_symbol(sym, force=force)
        return results
