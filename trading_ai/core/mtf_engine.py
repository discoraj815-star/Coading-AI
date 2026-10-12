"""Multi-Timeframe Engine: HTF -> MTF -> LTF structure, bias, and level analysis."""

import uuid
from typing import Dict, List, Optional, Tuple

from trading_ai.core.market_data_provider import BaseMarketDataProvider
from trading_ai.schemas.enums import (
    LevelType,
    MarketBias,
    MarketDataStatus,
    MarketDirection,
    MarketStructureType,
    MarkingValidity,
    TimeframeHierarchy,
)
from trading_ai.schemas.market import (
    Candle,
    MarketLevel,
    MarketSnapshot,
    TimeframeAnalysis,
    UnifiedMarketAnalysis,
)


class MultiTimeframeEngine:
    """
    Analyzes market structure hierarchically across multiple timeframes.

    Execution Flow:
      1. HTF: Macro direction, key swing levels, liquidity boundaries
      2. MTF: Intermediate confirmation or conflict detection
      3. LTF: Execution structure, micro-imbalances, triggers
      4. Synthesis: Resolves overall bias or flags MULTI_TIMEFRAME_CONFLICT.
    """

    DEFAULT_HIERARCHY = {
        TimeframeHierarchy.HTF: "1d",
        TimeframeHierarchy.MTF: "1h",
        TimeframeHierarchy.LTF: "15m",
    }

    def __init__(
        self,
        data_provider: BaseMarketDataProvider,
        hierarchy: Optional[Dict[TimeframeHierarchy, str]] = None,
    ):
        self._provider = data_provider
        self._hierarchy = hierarchy or dict(self.DEFAULT_HIERARCHY)

    @property
    def hierarchy(self) -> Dict[TimeframeHierarchy, str]:
        return self._hierarchy

    def set_timeframe(self, role: TimeframeHierarchy, timeframe: str) -> None:
        """Configure timeframe for a specific hierarchy tier."""
        self._hierarchy[role] = timeframe

    def analyze_single_timeframe(
        self,
        snapshot: MarketSnapshot,
        role: TimeframeHierarchy = TimeframeHierarchy.MTF,
    ) -> TimeframeAnalysis:
        """
        Extract market structure, swing levels, and directional bias for a single timeframe.
        """
        candles = snapshot.candles
        symbol = snapshot.symbol
        tf = snapshot.timeframe

        if not candles or len(candles) < 5:
            return TimeframeAnalysis(
                symbol=symbol,
                timeframe=tf,
                hierarchy_role=role,
                direction=MarketDirection.UNKNOWN,
                structure=MarketStructureType.RANGE,
                key_levels=[],
                bias=MarketBias.NEUTRAL_BIAS,
                notes=["Insufficient candles for structure analysis"],
                timestamp=snapshot.timestamp,
            )

        # 1. Identify Swings (local extrema with 2-bar lookaround)
        swing_highs, swing_lows = self._find_swings(candles, lookaround=2)

        # 2. Determine Market Direction & Structure Breaks (BOS / CHOCH)
        direction, structure, notes = self._determine_structure(candles, swing_highs, swing_lows)

        # 3. Extract Critical Levels
        key_levels = self._extract_key_levels(symbol, tf, candles, swing_highs, swing_lows)

        # 4. Map direction to timeframe bias
        if direction == MarketDirection.BULLISH:
            bias = MarketBias.BULLISH_BIAS
        elif direction == MarketDirection.BEARISH:
            bias = MarketBias.BEARISH_BIAS
        else:
            bias = MarketBias.NEUTRAL_BIAS

        return TimeframeAnalysis(
            symbol=symbol,
            timeframe=tf,
            hierarchy_role=role,
            direction=direction,
            structure=structure,
            key_levels=key_levels,
            bias=bias,
            notes=notes,
            timestamp=snapshot.timestamp,
        )

    def analyze_symbol(
        self,
        symbol: str,
        limit_per_tf: int = 60,
    ) -> UnifiedMarketAnalysis:
        """
        Run multi-timeframe analysis across HTF, MTF, and LTF.

        Detects time-frame alignment or MULTI_TIMEFRAME_CONFLICT without forcing bias.
        """
        htf_tf = self._hierarchy[TimeframeHierarchy.HTF]
        mtf_tf = self._hierarchy[TimeframeHierarchy.MTF]
        ltf_tf = self._hierarchy[TimeframeHierarchy.LTF]

        htf_snap = self._provider.get_snapshot(symbol, timeframe=htf_tf, limit=limit_per_tf)
        mtf_snap = self._provider.get_snapshot(symbol, timeframe=mtf_tf, limit=limit_per_tf)
        ltf_snap = self._provider.get_snapshot(symbol, timeframe=ltf_tf, limit=limit_per_tf)

        # Check for stale or missing data
        conflict_notes: List[str] = []
        if self._provider.is_stale(htf_snap) or self._provider.is_stale(mtf_snap):
            conflict_notes.append("Market data is stale or delayed; analysis flagged.")

        htf_analysis = self.analyze_single_timeframe(htf_snap, role=TimeframeHierarchy.HTF)
        mtf_analysis = self.analyze_single_timeframe(mtf_snap, role=TimeframeHierarchy.MTF)
        ltf_analysis = self.analyze_single_timeframe(ltf_snap, role=TimeframeHierarchy.LTF)

        return self._synthesize_multi_timeframe(
            symbol=symbol,
            htf_analysis=htf_analysis,
            mtf_analysis=mtf_analysis,
            ltf_analysis=ltf_analysis,
            conflict_notes=conflict_notes,
            timestamp=ltf_snap.timestamp,
        )

    def _synthesize_multi_timeframe(
        self,
        symbol: str,
        htf_analysis: TimeframeAnalysis,
        mtf_analysis: TimeframeAnalysis,
        ltf_analysis: TimeframeAnalysis,
        conflict_notes: Optional[List[str]] = None,
        timestamp: Optional[float] = None,
    ) -> UnifiedMarketAnalysis:
        """
        Synthesize multi-timeframe bias and detect conflicts between HTF, MTF, and LTF.
        """
        notes = list(conflict_notes or [])
        directions = [htf_analysis.direction, mtf_analysis.direction, ltf_analysis.direction]

        has_bullish = any(d == MarketDirection.BULLISH for d in directions)
        has_bearish = any(d == MarketDirection.BEARISH for d in directions)

        if has_bullish and has_bearish:
            overall_bias = MarketBias.MULTI_TIMEFRAME_CONFLICT
            notes.append(
                f"Multi-timeframe conflict detected across timeframes: "
                f"HTF ({htf_analysis.timeframe})={htf_analysis.direction.value}, "
                f"MTF ({mtf_analysis.timeframe})={mtf_analysis.direction.value}, "
                f"LTF ({ltf_analysis.timeframe})={ltf_analysis.direction.value}."
            )
        elif htf_analysis.direction == MarketDirection.BULLISH and mtf_analysis.direction == MarketDirection.BULLISH:
            overall_bias = MarketBias.BULLISH_BIAS
        elif htf_analysis.direction == MarketDirection.BEARISH and mtf_analysis.direction == MarketDirection.BEARISH:
            overall_bias = MarketBias.BEARISH_BIAS
        elif htf_analysis.direction == MarketDirection.BULLISH:
            overall_bias = MarketBias.BULLISH_BIAS
        elif htf_analysis.direction == MarketDirection.BEARISH:
            overall_bias = MarketBias.BEARISH_BIAS
        else:
            overall_bias = MarketBias.NEUTRAL_BIAS

        return UnifiedMarketAnalysis(
            symbol=symbol,
            htf_analysis=htf_analysis,
            mtf_analysis=mtf_analysis,
            ltf_analysis=ltf_analysis,
            overall_bias=overall_bias,
            active_setups=[],
            active_markings=[],
            conflict_notes=notes,
            timestamp=timestamp or ltf_analysis.timestamp,
        )

    def _find_swings(
        self, candles: List[Candle], lookaround: int = 2
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find local swing highs and swing lows index and price."""
        highs: List[Tuple[int, float]] = []
        lows: List[Tuple[int, float]] = []

        n = len(candles)
        for i in range(lookaround, n - lookaround):
            curr_high = candles[i].high
            curr_low = candles[i].low

            # High check
            if all(curr_high >= candles[j].high for j in range(i - lookaround, i + lookaround + 1) if j != i):
                highs.append((i, curr_high))

            # Low check
            if all(curr_low <= candles[j].low for j in range(i - lookaround, i + lookaround + 1) if j != i):
                lows.append((i, curr_low))

        return highs, lows

    def _determine_structure(
        self,
        candles: List[Candle],
        swing_highs: List[Tuple[int, float]],
        swing_lows: List[Tuple[int, float]],
    ) -> Tuple[MarketDirection, MarketStructureType, List[str]]:
        """Determine structural direction and identify BOS / CHOCH breaks."""
        notes: List[str] = []

        if not swing_highs or not swing_lows:
            # Fall back to candle close trend
            first_close = candles[0].close
            last_close = candles[-1].close
            if last_close > first_close * 1.002:
                return MarketDirection.BULLISH, MarketStructureType.RANGE, ["Trend up based on candle close"]
            elif last_close < first_close * 0.998:
                return MarketDirection.BEARISH, MarketStructureType.RANGE, ["Trend down based on candle close"]
            return MarketDirection.NEUTRAL, MarketStructureType.CONSOLIDATION, ["Consolidation in range"]

        # Examine sequence of last 2 swing highs and lows
        last_high = swing_highs[-1][1]
        last_low = swing_lows[-1][1]
        prev_high = swing_highs[-2][1] if len(swing_highs) >= 2 else last_high
        prev_low = swing_lows[-2][1] if len(swing_lows) >= 2 else last_low

        latest_price = candles[-1].close

        # Break of Structure (BOS) checks
        if latest_price > last_high:
            notes.append(f"Bullish BOS above swing high at {last_high}")
            return MarketDirection.BULLISH, MarketStructureType.BOS_BULLISH, notes
        if latest_price < last_low:
            notes.append(f"Bearish BOS below swing low at {last_low}")
            return MarketDirection.BEARISH, MarketStructureType.BOS_BEARISH, notes

        # Higher Highs & Higher Lows vs Lower Highs & Lower Lows
        higher_high = last_high > prev_high
        higher_low = last_low > prev_low
        lower_high = last_high < prev_high
        lower_low = last_low < prev_low

        if higher_high and higher_low:
            notes.append("Market structure: Higher Highs and Higher Lows")
            return MarketDirection.BULLISH, MarketStructureType.SWING_HIGH, notes
        elif lower_high and lower_low:
            notes.append("Market structure: Lower Highs and Lower Lows")
            return MarketDirection.BEARISH, MarketStructureType.SWING_LOW, notes
        elif higher_high and lower_low:
            notes.append("Expanding range / potential CHOCH")
            return MarketDirection.RANGING, MarketStructureType.CHOCH_BULLISH, notes
        else:
            notes.append("Internal range / consolidation")
            return MarketDirection.RANGING, MarketStructureType.RANGE, notes

    def _extract_key_levels(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Candle],
        swing_highs: List[Tuple[int, float]],
        swing_lows: List[Tuple[int, float]],
    ) -> List[MarketLevel]:
        """Extract swing points and recent high/low levels."""
        levels: List[MarketLevel] = []

        # Recent Swing High
        if swing_highs:
            last_high_idx, last_high_val = swing_highs[-1]
            levels.append(
                MarketLevel(
                    symbol=symbol,
                    timeframe=timeframe,
                    price=last_high_val,
                    level_type=LevelType.SWING_HIGH,
                    evidence_json={"candle_index": last_high_idx, "timestamp": candles[last_high_idx].timestamp},
                    confidence=0.9,
                    validity=MarkingValidity.VALID,
                    invalidation_condition="Candle close above swing high",
                )
            )

        # Recent Swing Low
        if swing_lows:
            last_low_idx, last_low_val = swing_lows[-1]
            levels.append(
                MarketLevel(
                    symbol=symbol,
                    timeframe=timeframe,
                    price=last_low_val,
                    level_type=LevelType.SWING_LOW,
                    evidence_json={"candle_index": last_low_idx, "timestamp": candles[last_low_idx].timestamp},
                    confidence=0.9,
                    validity=MarkingValidity.VALID,
                    invalidation_condition="Candle close below swing low",
                )
            )

        # Session High & Low from the snapshot
        highest_candle = max(candles, key=lambda c: c.high)
        lowest_candle = min(candles, key=lambda c: c.low)

        levels.append(
            MarketLevel(
                symbol=symbol,
                timeframe=timeframe,
                price=highest_candle.high,
                level_type=LevelType.SESSION_HIGH,
                evidence_json={"timestamp": highest_candle.timestamp},
                confidence=0.95,
                validity=MarkingValidity.VALID,
            )
        )
        levels.append(
            MarketLevel(
                symbol=symbol,
                timeframe=timeframe,
                price=lowest_candle.low,
                level_type=LevelType.SESSION_LOW,
                evidence_json={"timestamp": lowest_candle.timestamp},
                confidence=0.95,
                validity=MarkingValidity.VALID,
            )
        )

        return levels
