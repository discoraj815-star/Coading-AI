"""
Incremental Technical Analysis Engine.
Optimizes market structure, swings, BOS, CHOCH, FVG, and liquidity updates by computing
deltas upon new candle/tick arrival without expensive full historical recomputations.
Maintains timeframe separation and anti-flapping hysteresis.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from trading_ai.core.technical_analysis import DeepTechnicalAnalyzer, SwingPoint
from trading_ai.schemas.enums import (
    DealingRangeZone,
    MarketBias,
    MarketDirection,
    SwingClassification,
)
from trading_ai.schemas.market import Candle, MarketSnapshot
from trading_ai.schemas.performance import IncrementalAnalysisDelta

logger = logging.getLogger(__name__)


def _to_float_timestamp(ts: Union[float, int, datetime]) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, datetime):
        return ts.timestamp()
    return 0.0


@dataclass
class TimeframeState:
    """Persistent state for a specific symbol and timeframe."""
    symbol: str
    timeframe: str
    candles: List[Candle] = field(default_factory=list)
    swings: List[SwingPoint] = field(default_factory=list)
    recent_bos: List[Dict[str, Any]] = field(default_factory=list)
    recent_choch: List[Dict[str, Any]] = field(default_factory=list)
    active_fvgs: List[Dict[str, Any]] = field(default_factory=list)
    liquidity_pools: List[float] = field(default_factory=list)
    current_bias: MarketBias = MarketBias.NEUTRAL_BIAS
    bias_confidence: float = 0.5
    bias_consecutive_count: int = 0
    last_candle_time: Optional[float] = None


class IncrementalAnalysisEngine:
    """
    Computes incremental delta changes per candle or tick.
    Ensures O(1) or O(window) efficiency instead of O(N^2) reprocessing.
    """

    def __init__(self, full_analyzer: Optional[DeepTechnicalAnalyzer] = None, max_history: int = 150):
        self._full_analyzer = full_analyzer or DeepTechnicalAnalyzer()
        self._max_history = max_history
        # Key: f"{symbol}:{timeframe}"
        self._states: Dict[str, TimeframeState] = {}
        # Hysteresis threshold to prevent flapping
        self._flapping_threshold: int = 2

    def get_state(self, symbol: str, timeframe: str) -> TimeframeState:
        """Get or initialize state for symbol and timeframe."""
        key = f"{symbol}:{timeframe}"
        if key not in self._states:
            self._states[key] = TimeframeState(symbol=symbol, timeframe=timeframe)
        return self._states[key]

    def process_new_candle(self, symbol: str, timeframe: str, new_candle: Candle) -> IncrementalAnalysisDelta:
        """
        Process a newly completed or updated candle incrementally.
        Evaluates swings delta, BOS, CHOCH, FVG, and micro-changes.
        """
        state = self.get_state(symbol, timeframe)
        affected_components: List[str] = ["candles"]
        ts_float = _to_float_timestamp(new_candle.timestamp)

        # Check duplicate or out-of-order candle
        if state.last_candle_time is not None and ts_float < state.last_candle_time:
            # Out of order: drop or ignore without corrupting state
            return IncrementalAnalysisDelta(
                symbol=symbol,
                timeframe=timeframe,
                new_candle_timestamp=ts_float,
                timestamp=ts_float,
                price_delta=0.0,
                affected_components=["out_of_order_ignored"],
            )

        # Append to sliding history
        prev_close = state.candles[-1].close if state.candles else new_candle.open
        state.candles.append(new_candle)
        if len(state.candles) > self._max_history:
            state.candles.pop(0)

        state.last_candle_time = ts_float
        price_delta = new_candle.close - prev_close

        bos_detected = False
        choch_detected = False
        liquidity_swept = False
        displacement_inc = False
        rejection_wick = False
        volatility_exp = False
        structure_weakened = False

        # Need at least 3 candles to evaluate swings / FVGs
        if len(state.candles) >= 3:
            # Check if candle[-2] forms a swing point (fractal pivot high/low)
            c_prev = state.candles[-3]
            c_mid = state.candles[-2]
            c_cur = state.candles[-1]
            c_mid_ts = _to_float_timestamp(c_mid.timestamp)

            # 1. Swing High Detection at c_mid
            if c_mid.high > c_prev.high and c_mid.high > c_cur.high:
                idx = len(state.candles) - 2
                classification = SwingClassification.HIGHER_HIGH
                if state.swings:
                    last_high = next((s for s in reversed(state.swings) if s.is_high), None)
                    if last_high:
                        classification = SwingClassification.HIGHER_HIGH if c_mid.high > last_high.price else SwingClassification.LOWER_HIGH
                sp = SwingPoint(index=idx, timestamp=c_mid_ts, price=c_mid.high, is_high=True, classification=classification)
                state.swings.append(sp)
                state.liquidity_pools.append(c_mid.high)
                affected_components.append("swings")

            # 2. Swing Low Detection at c_mid
            elif c_mid.low < c_prev.low and c_mid.low < c_cur.low:
                idx = len(state.candles) - 2
                classification = SwingClassification.LOWER_LOW
                if state.swings:
                    last_low = next((s for s in reversed(state.swings) if not s.is_high), None)
                    if last_low:
                        classification = SwingClassification.HIGHER_LOW if c_mid.low > last_low.price else SwingClassification.LOWER_LOW
                sp = SwingPoint(index=idx, timestamp=c_mid_ts, price=c_mid.low, is_high=False, classification=classification)
                state.swings.append(sp)
                state.liquidity_pools.append(c_mid.low)
                affected_components.append("swings")

            # 3. BOS / CHOCH delta detection against active swings
            last_swing_high = next((s for s in reversed(state.swings) if s.is_high), None)
            last_swing_low = next((s for s in reversed(state.swings) if not s.is_high), None)

            if last_swing_high and new_candle.close > last_swing_high.price:
                if state.current_bias == MarketBias.BULLISH_BIAS:
                    bos_detected = True
                    state.recent_bos.append({"type": "BULLISH_BOS", "price": last_swing_high.price, "time": ts_float})
                    affected_components.append("bos")
                else:
                    choch_detected = True
                    state.recent_choch.append({"type": "BULLISH_CHOCH", "price": last_swing_high.price, "time": ts_float})
                    affected_components.append("choch")
                    self._update_bias_with_hysteresis(state, MarketBias.BULLISH_BIAS)

            elif last_swing_low and new_candle.close < last_swing_low.price:
                if state.current_bias == MarketBias.BEARISH_BIAS:
                    bos_detected = True
                    state.recent_bos.append({"type": "BEARISH_BOS", "price": last_swing_low.price, "time": ts_float})
                    affected_components.append("bos")
                else:
                    choch_detected = True
                    state.recent_choch.append({"type": "BEARISH_CHOCH", "price": last_swing_low.price, "time": ts_float})
                    affected_components.append("choch")
                    self._update_bias_with_hysteresis(state, MarketBias.BEARISH_BIAS)

            # 4. FVG (Imbalance) Delta Check across last 3 candles
            # Bullish FVG: c_prev.high < c_cur.low
            if c_cur.low > c_prev.high:
                gap_size = c_cur.low - c_prev.high
                if gap_size > 0.0001:
                    state.active_fvgs.append({"type": "BULLISH_FVG", "bottom": c_prev.high, "top": c_cur.low, "time": ts_float})
                    affected_components.append("fvg")
            # Bearish FVG: c_prev.low > c_cur.high
            elif c_prev.low > c_cur.high:
                gap_size = c_prev.low - c_cur.high
                if gap_size > 0.0001:
                    state.active_fvgs.append({"type": "BEARISH_FVG", "top": c_prev.low, "bottom": c_cur.high, "time": ts_float})
                    affected_components.append("fvg")

            # 5. Liquidity Sweep Detection (wick penetrates a pool then closes back inside)
            for liq in list(state.liquidity_pools[-10:]):
                if (new_candle.high > liq and new_candle.close < liq) or (new_candle.low < liq and new_candle.close > liq):
                    liquidity_swept = True
                    affected_components.append("liquidity_sweep")
                    break

            # 6. Micro-changes: Displacement, Rejection Wick, Volatility Expansion
            body = abs(new_candle.close - new_candle.open)
            candle_range = new_candle.high - new_candle.low
            upper_wick = new_candle.high - max(new_candle.open, new_candle.close)
            lower_wick = min(new_candle.open, new_candle.close) - new_candle.low

            # Calculate rolling averages of last up to 10 candles
            recent = state.candles[-10:]
            avg_body = sum(abs(c.close - c.open) for c in recent) / len(recent) if recent else 0.001
            avg_range = sum(c.high - c.low for c in recent) / len(recent) if recent else 0.001

            if body > 1.8 * avg_body and candle_range > 0:
                displacement_inc = True
                affected_components.append("displacement")

            if (upper_wick > 2.0 * max(body, 0.0001) and upper_wick > 0.4 * candle_range) or \
               (lower_wick > 2.0 * max(body, 0.0001) and lower_wick > 0.4 * candle_range):
                rejection_wick = True
                affected_components.append("rejection_wick")

            if candle_range > 1.8 * avg_range:
                volatility_exp = True
                affected_components.append("volatility_expansion")

            # Structure weakening: small body following multiple pushes in same direction
            if len(recent) >= 4:
                consecutive_same_dir = all(c.close > c.open for c in recent[-3:]) or all(c.close < c.open for c in recent[-3:])
                if consecutive_same_dir and body < 0.4 * avg_body:
                    structure_weakened = True
                    affected_components.append("structure_weakened")

        return IncrementalAnalysisDelta(
            symbol=symbol,
            timeframe=timeframe,
            new_candle_timestamp=ts_float,
            timestamp=ts_float,
            price_delta=price_delta,
            swings_changed="swings" in affected_components,
            bos_detected=bos_detected,
            choch_detected=choch_detected,
            new_bos_detected=bos_detected,
            new_choch_detected=choch_detected,
            liquidity_swept=liquidity_swept,
            displacement_increase=displacement_inc,
            rejection_wick_detected=rejection_wick,
            volatility_expansion=volatility_exp,
            structure_weakened=structure_weakened,
            affected_components=affected_components,
        )

    def _update_bias_with_hysteresis(self, state: TimeframeState, candidate_bias: MarketBias) -> None:
        """Apply hysteresis to avoid noisy bias flapping."""
        if state.current_bias == candidate_bias:
            state.bias_consecutive_count += 1
            state.bias_confidence = min(1.0, 0.6 + 0.1 * state.bias_consecutive_count)
        else:
            state.bias_consecutive_count += 1
            if state.bias_consecutive_count >= self._flapping_threshold:
                state.current_bias = candidate_bias
                state.bias_consecutive_count = 1
                state.bias_confidence = 0.7

    def notify_htf_change(self, symbol: str, htf: str, new_htf_bias: MarketBias, dependent_ltfs: List[str]) -> List[str]:
        """
        When HTF bias changes, invalidate dependent LTF assumptions without recomputing HTF unnecessarily.
        Returns list of invalidated LTF keys.
        """
        invalidated = []
        for ltf in dependent_ltfs:
            key = f"{symbol}:{ltf}"
            if key in self._states:
                # Reset bias confirmation count to trigger re-alignment
                self._states[key].bias_consecutive_count = 0
                invalidated.append(key)
        return invalidated
