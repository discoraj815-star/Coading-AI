"""
Step 6 Deep Technical Analysis Engine.
Provides comprehensive market structure, swing classification (HH, HL, LH, LL, EQH, EQL),
BOS, CHOCH, displacement, dealing range equilibrium, liquidity pool mapping,
and POI confluence evaluation strictly constrained by learned lecture rules.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.core.lecture_validator import LectureValidator
from trading_ai.schemas.enums import (
    DealingRangeZone,
    LevelType,
    MarketBias,
    MarketDirection,
    MarketStructureType,
    MarkingType,
    MarkingValidity,
    SetupStatus,
    SwingClassification,
)
from trading_ai.schemas.market import (
    Candle,
    ChartMarking,
    MarketLevel,
    MarketSetup,
    MarketSnapshot,
)

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    """Classified structural swing high or low."""
    index: int
    timestamp: float
    price: float
    is_high: bool
    classification: SwingClassification


@dataclass
class DealingRange:
    """Current active structural dealing range."""
    high: float
    low: float
    equilibrium: float
    current_zone: DealingRangeZone


@dataclass
class TechnicalStructureReport:
    """Comprehensive technical analysis summary."""
    symbol: str
    timeframe: str
    current_price: float
    swings: List[SwingPoint]
    recent_bos: List[Dict[str, Any]]
    recent_choch: List[Dict[str, Any]]
    dealing_range: Optional[DealingRange]
    displacement_detected: bool
    displacement_direction: Optional[MarketDirection]
    liquidity_sweeps: List[Dict[str, Any]]
    active_fvgs: List[Dict[str, Any]]
    active_order_blocks: List[Dict[str, Any]]
    confluence_score: float
    structure_bias: MarketBias
    evidence: List[str]


class DeepTechnicalAnalyzer:
    """
    Analyzes candlestick price action with institutional SMC / Price Action concepts:
    Swings, BOS, CHOCH, Displacement, Liquidity, POIs, and Confluence.
    """

    def __init__(self, lecture_validator: Optional[LectureValidator] = None):
        self._validator = lecture_validator

    def analyze_candles(self, snapshot: MarketSnapshot) -> TechnicalStructureReport:
        """Execute complete technical analysis over snapshot candles."""
        candles = snapshot.candles
        symbol = snapshot.symbol
        timeframe = snapshot.timeframe
        current_price = snapshot.latest_price
        evidence: List[str] = []

        if len(candles) < 5:
            return TechnicalStructureReport(
                symbol=symbol,
                timeframe=timeframe,
                current_price=current_price,
                swings=[],
                recent_bos=[],
                recent_choch=[],
                dealing_range=None,
                displacement_detected=False,
                displacement_direction=None,
                liquidity_sweeps=[],
                active_fvgs=[],
                active_order_blocks=[],
                confluence_score=0.5,
                structure_bias=MarketBias.NEUTRAL_BIAS,
                evidence=["Insufficient candles for deep technical structure (< 5)."],
            )

        # 1. Identify and Classify Swings
        swings = self._identify_swings(candles)
        evidence.append(f"Identified {len(swings)} structural swing points.")

        # 2. Break of Structure (BOS) & Change of Character (CHOCH)
        recent_bos, recent_choch = self._detect_structure_breaks(candles, swings)

        # 3. Displacement
        displacement_detected, disp_dir = self._detect_recent_displacement(candles)
        if displacement_detected and disp_dir:
            evidence.append(f"Displacement detected in {disp_dir.value} direction.")

        # 4. Dealing Range & Equilibrium
        dealing_range = self._calculate_dealing_range(candles, current_price)
        if dealing_range:
            evidence.append(
                f"Dealing range: [{dealing_range.low:.4f} - {dealing_range.high:.4f}], "
                f"Eq: {dealing_range.equilibrium:.4f} ({dealing_range.current_zone.value})."
            )

        # 5. Liquidity Pools & Sweeps
        liquidity_sweeps = self._detect_liquidity_sweeps(candles, swings)

        # 6. POIs: Fair Value Gaps & Order Blocks
        active_fvgs = self._detect_fvgs(candles)
        active_order_blocks = self._detect_order_blocks(candles, displacement_detected, disp_dir)

        # 7. Structure Bias
        structure_bias = self._determine_structure_bias(swings, recent_bos, recent_choch)

        # 8. Confluence Score (0.0 to 1.0)
        confluence_score = self._calculate_confluence(
            structure_bias=structure_bias,
            displacement=displacement_detected,
            sweeps=liquidity_sweeps,
            fvgs=active_fvgs,
            obs=active_order_blocks,
            dealing_range=dealing_range,
        )

        return TechnicalStructureReport(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            swings=swings,
            recent_bos=recent_bos,
            recent_choch=recent_choch,
            dealing_range=dealing_range,
            displacement_detected=displacement_detected,
            displacement_direction=disp_dir,
            liquidity_sweeps=liquidity_sweeps,
            active_fvgs=active_fvgs,
            active_order_blocks=active_order_blocks,
            confluence_score=confluence_score,
            structure_bias=structure_bias,
            evidence=evidence,
        )

    def _identify_swings(self, candles: List[Candle], window: int = 1) -> List[SwingPoint]:
        """Detect local swing highs and lows and classify as HH/HL/LH/LL/EQH/EQL."""
        swings: List[SwingPoint] = []
        n = len(candles)
        if n < (window * 2 + 1):
            return swings

        prev_high: Optional[float] = None
        prev_low: Optional[float] = None

        for i in range(window, n - window):
            c = candles[i]
            is_high = all(c.high >= candles[j].high for j in range(i - window, i + window + 1) if j != i)
            is_low = all(c.low <= candles[j].low for j in range(i - window, i + window + 1) if j != i)

            if is_high:
                classification = SwingClassification.HIGHER_HIGH
                if prev_high is not None:
                    diff_pct = abs(c.high - prev_high) / prev_high
                    if diff_pct < 0.0006:
                        classification = SwingClassification.EQUAL_HIGH
                    elif c.high > prev_high:
                        classification = SwingClassification.HIGHER_HIGH
                    else:
                        classification = SwingClassification.LOWER_HIGH
                prev_high = c.high
                swings.append(
                    SwingPoint(
                        index=i,
                        timestamp=c.timestamp,
                        price=c.high,
                        is_high=True,
                        classification=classification,
                    )
                )

            elif is_low:
                classification = SwingClassification.LOWER_LOW
                if prev_low is not None:
                    diff_pct = abs(c.low - prev_low) / prev_low
                    if diff_pct < 0.0006:
                        classification = SwingClassification.EQUAL_LOW
                    elif c.low < prev_low:
                        classification = SwingClassification.LOWER_LOW
                    else:
                        classification = SwingClassification.HIGHER_LOW
                prev_low = c.low
                swings.append(
                    SwingPoint(
                        index=i,
                        timestamp=c.timestamp,
                        price=c.low,
                        is_high=False,
                        classification=classification,
                    )
                )

        return swings

    def _detect_structure_breaks(
        self, candles: List[Candle], swings: List[SwingPoint]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detect BOS (trend continuation) and CHOCH (trend change)."""
        bos_list: List[Dict[str, Any]] = []
        choch_list: List[Dict[str, Any]] = []

        high_swings = [s for s in swings if s.is_high]
        low_swings = [s for s in swings if not s.is_high]

        if not high_swings or not low_swings:
            return bos_list, choch_list

        last_high = high_swings[-1]
        last_low = low_swings[-1]
        latest_candle = candles[-1]

        # Bullish break
        if latest_candle.close > last_high.price:
            if last_high.classification in (SwingClassification.HIGHER_HIGH, SwingClassification.EQUAL_HIGH):
                bos_list.append({
                    "type": "BOS_BULLISH",
                    "broken_level": last_high.price,
                    "candle_close": latest_candle.close,
                    "timestamp": latest_candle.timestamp,
                })
            else:
                choch_list.append({
                    "type": "CHOCH_BULLISH",
                    "broken_level": last_high.price,
                    "candle_close": latest_candle.close,
                    "timestamp": latest_candle.timestamp,
                })

        # Bearish break
        if latest_candle.close < last_low.price:
            if last_low.classification in (SwingClassification.LOWER_LOW, SwingClassification.EQUAL_LOW):
                bos_list.append({
                    "type": "BOS_BEARISH",
                    "broken_level": last_low.price,
                    "candle_close": latest_candle.close,
                    "timestamp": latest_candle.timestamp,
                })
            else:
                choch_list.append({
                    "type": "CHOCH_BEARISH",
                    "broken_level": last_low.price,
                    "candle_close": latest_candle.close,
                    "timestamp": latest_candle.timestamp,
                })

        return bos_list, choch_list

    def _detect_recent_displacement(
        self, candles: List[Candle], lookback: int = 14
    ) -> Tuple[bool, Optional[MarketDirection]]:
        """Identify energetic institutional displacement candles."""
        if len(candles) < 5:
            return False, None

        recent = candles[-min(lookback, len(candles)):]
        ranges = [abs(c.high - c.low) for c in recent[:-1]]
        if not ranges:
            return False, None

        avg_range = sum(ranges) / len(ranges)
        last_c = candles[-1]
        last_range = abs(last_c.high - last_c.low)
        body = abs(last_c.close - last_c.open)

        # Displacement threshold: range >= 1.4x average and body is dominant (>= 50% of range)
        if avg_range > 0 and last_range >= (1.4 * avg_range) and (body / last_range) >= 0.50:
            if last_c.close > last_c.open:
                return True, MarketDirection.BULLISH
            elif last_c.close < last_c.open:
                return True, MarketDirection.BEARISH

        return False, None

    def _calculate_dealing_range(
        self, candles: List[Candle], current_price: float, lookback: int = 30
    ) -> Optional[DealingRange]:
        """Compute the dealing range high, low, and equilibrium (50%)."""
        if not candles:
            return None

        recent = candles[-min(lookback, len(candles)):]
        highest_high = max(c.high for c in recent)
        lowest_low = min(c.low for c in recent)

        if highest_high <= lowest_low:
            return None

        eq = (highest_high + lowest_low) / 2.0
        zone = DealingRangeZone.EQUILIBRIUM
        if current_price > (eq * 1.002):
            zone = DealingRangeZone.PREMIUM
        elif current_price < (eq * 0.998):
            zone = DealingRangeZone.DISCOUNT

        return DealingRange(
            high=highest_high,
            low=lowest_low,
            equilibrium=eq,
            current_zone=zone,
        )

    def _detect_liquidity_sweeps(
        self, candles: List[Candle], swings: List[SwingPoint]
    ) -> List[Dict[str, Any]]:
        """Detect false breakouts / wicks taking liquidity and closing back inside."""
        sweeps: List[Dict[str, Any]] = []
        if len(candles) < 3 or not swings:
            return sweeps

        last_c = candles[-1]
        for s in swings[-4:]:
            if s.is_high and last_c.high > s.price and last_c.close < s.price:
                sweeps.append({
                    "type": "BUY_SIDE_LIQUIDITY_SWEEP",
                    "level": s.price,
                    "sweep_high": last_c.high,
                    "candle_close": last_c.close,
                    "timestamp": last_c.timestamp,
                })
            elif (not s.is_high) and last_c.low < s.price and last_c.close > s.price:
                sweeps.append({
                    "type": "SELL_SIDE_LIQUIDITY_SWEEP",
                    "level": s.price,
                    "sweep_low": last_c.low,
                    "candle_close": last_c.close,
                    "timestamp": last_c.timestamp,
                })

        return sweeps

    def _detect_fvgs(self, candles: List[Candle]) -> List[Dict[str, Any]]:
        """Detect active Fair Value Gaps across recent bars."""
        fvgs: List[Dict[str, Any]] = []
        if len(candles) < 3:
            return fvgs

        for i in range(2, min(len(candles), 15)):
            c1 = candles[-i - 1]
            c2 = candles[-i]
            c3 = candles[-i + 1]

            # Bullish FVG: c1.high < c3.low
            if c3.low > c1.high:
                fvgs.append({
                    "type": "FVG_BULLISH",
                    "top": c3.low,
                    "bottom": c1.high,
                    "size": c3.low - c1.high,
                    "timestamp": c2.timestamp,
                })
            # Bearish FVG: c1.low > c3.high
            elif c1.low > c3.high:
                fvgs.append({
                    "type": "FVG_BEARISH",
                    "top": c1.low,
                    "bottom": c3.high,
                    "size": c1.low - c3.high,
                    "timestamp": c2.timestamp,
                })

        return fvgs

    def _detect_order_blocks(
        self,
        candles: List[Candle],
        displacement: bool,
        disp_dir: Optional[MarketDirection],
    ) -> List[Dict[str, Any]]:
        """Identify institutional Order Blocks preceding displacement."""
        obs: List[Dict[str, Any]] = []
        if len(candles) < 4 or not displacement or not disp_dir:
            return obs

        # Find the last opposing candle prior to displacement
        for i in range(2, min(len(candles), 6)):
            c = candles[-i]
            if disp_dir == MarketDirection.BULLISH and c.close < c.open:
                obs.append({
                    "type": "BULLISH_ORDER_BLOCK",
                    "high": c.high,
                    "low": c.low,
                    "timestamp": c.timestamp,
                })
                break
            elif disp_dir == MarketDirection.BEARISH and c.close > c.open:
                obs.append({
                    "type": "BEARISH_ORDER_BLOCK",
                    "high": c.high,
                    "low": c.low,
                    "timestamp": c.timestamp,
                })
                break

        return obs

    def _determine_structure_bias(
        self,
        swings: List[SwingPoint],
        bos: List[Dict[str, Any]],
        choch: List[Dict[str, Any]],
    ) -> MarketBias:
        """Evaluate structural bias from classified swings and breaks."""
        if not swings:
            return MarketBias.NEUTRAL_BIAS

        bullish_score = 0
        bearish_score = 0

        # Recent swings
        for s in swings[-3:]:
            if s.classification in (SwingClassification.HIGHER_HIGH, SwingClassification.HIGHER_LOW):
                bullish_score += 1
            elif s.classification in (SwingClassification.LOWER_HIGH, SwingClassification.LOWER_LOW):
                bearish_score += 1

        for b in bos:
            if "BULLISH" in b["type"]:
                bullish_score += 2
            else:
                bearish_score += 2

        for c in choch:
            if "BULLISH" in c["type"]:
                bullish_score += 1.5
            else:
                bearish_score += 1.5

        if bullish_score > bearish_score + 1:
            return MarketBias.BULLISH_BIAS
        elif bearish_score > bullish_score + 1:
            return MarketBias.BEARISH_BIAS
        return MarketBias.NEUTRAL_BIAS

    def _calculate_confluence(
        self,
        structure_bias: MarketBias,
        displacement: bool,
        sweeps: List[Dict[str, Any]],
        fvgs: List[Dict[str, Any]],
        obs: List[Dict[str, Any]],
        dealing_range: Optional[DealingRange],
    ) -> float:
        """Synthesize technical confluence score (0.0 to 1.0)."""
        score = 0.35  # Base

        if structure_bias in (MarketBias.BULLISH_BIAS, MarketBias.BEARISH_BIAS):
            score += 0.20
        if displacement:
            score += 0.15
        if sweeps:
            score += 0.15
        if fvgs or obs:
            score += 0.10
        if dealing_range and dealing_range.current_zone in (DealingRangeZone.DISCOUNT, DealingRangeZone.PREMIUM):
            score += 0.05

        return min(round(score, 2), 1.0)
