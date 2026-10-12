"""Clean Chart Marking Engine: Validates, filters clutter, and plots high-priority chart overlays."""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from trading_ai.schemas.enums import (
    MarkingType,
    MarkingValidity,
    SetupStatus,
)
from trading_ai.schemas.market import (
    Candle,
    ChartMarking,
    MarketLevel,
    MarketSetup,
    MarketSnapshot,
)


class ChartMarker:
    """
    Manages automatic chart markings while enforcing clean-chart principles.

    Pipeline:
      DETECT -> VALIDATE -> RELEVANCE FILTER -> DUPLICATE CHECK -> CLUTTER CHECK -> PLOT

    Rules:
      1. Weak/uncertain setups are NOT plotted on the chart (they remain in analysis/evidence).
      2. Duplicate markings within price tolerance are suppressed.
      3. Mitigated or invalidated zones are marked INVALIDATED or EXPIRED.
      4. Total visible markings are capped to max_markings to prevent clutter.
    """

    def __init__(self, max_markings: int = 15, price_duplicate_tolerance: float = 0.001):
        self._max_markings = max_markings
        self._price_duplicate_tolerance = price_duplicate_tolerance

    def generate_markings(
        self,
        snapshot: MarketSnapshot,
        setups: List[MarketSetup],
        levels: List[MarketLevel],
        source: str = "SUPERVISOR",
        broker_id: Optional[str] = None,
        backtest_run_id: Optional[str] = None,
    ) -> List[ChartMarking]:
        """
        Generate and filter markings for a snapshot.
        Applies relevance filtering, duplicate suppression, and clutter limits.
        """
        raw_markings: List[ChartMarking] = []
        symbol = snapshot.symbol
        timeframe = snapshot.timeframe
        candles = snapshot.candles
        now_ts = datetime.now(timezone.utc).timestamp()

        # 1. Plot Valid Setups (e.g., FVG, Order Block, BOS, POI, Entry/Stop/Target)
        for setup in setups:
            # Relevance Filter: Only plot VALID setups
            if setup.status != SetupStatus.VALID:
                continue

            marking_type = self._map_setup_to_marking_type(setup.setup_type)
            priority = 10 if marking_type in (MarkingType.SETUP_ZONE, MarkingType.ORDER_BLOCK, MarkingType.ENTRY) else 8

            raw_markings.append(
                ChartMarking(
                    symbol=symbol,
                    timeframe=timeframe,
                    marking_type=marking_type,
                    coordinates_json={"direction": setup.direction.value, "timeframe": timeframe},
                    rule_id=setup.applicable_rule_id,
                    evidence_json={
                        "setup_id": setup.setup_id,
                        "conditions": setup.conditions_satisfied,
                        "evidence": setup.supporting_evidence,
                    },
                    validity=MarkingValidity.VALID,
                    confidence=setup.confidence,
                    priority=priority,
                    concept=setup.setup_type,
                    direction=setup.direction.value,
                    source=source,
                    rule_reference=setup.applicable_rule_id,
                    rule_version="1.0",
                    status="ACTIVE",
                    visibility="VISIBLE",
                    created_by="MARKET_SUPERVISOR" if source == "SUPERVISOR" else "BACKTEST_ENGINE",
                    broker_id=broker_id,
                    backtest_run_id=backtest_run_id,
                    timestamp=now_ts,
                )
            )

        # 2. Plot Key Swing Levels & Session Levels
        for lvl in levels:
            if lvl.validity != MarkingValidity.VALID:
                continue

            raw_markings.append(
                ChartMarking(
                    symbol=symbol,
                    timeframe=timeframe,
                    marking_type=MarkingType.IMPORTANT_LEVEL,
                    coordinates_json={"price": lvl.price, "level_type": lvl.level_type.value},
                    rule_id=lvl.rule_id,
                    evidence_json=lvl.evidence_json,
                    validity=MarkingValidity.VALID,
                    confidence=lvl.confidence,
                    priority=5,
                    concept=lvl.level_type.value,
                    price=lvl.price,
                    source=source,
                    rule_reference=lvl.rule_id,
                    status="ACTIVE",
                    visibility="VISIBLE",
                    created_by="MARKET_SUPERVISOR" if source == "SUPERVISOR" else "BACKTEST_ENGINE",
                    broker_id=broker_id,
                    backtest_run_id=backtest_run_id,
                    timestamp=now_ts,
                )
            )

        # 3. Detect and Plot Fair Value Gaps (FVGs) directly from candles with mitigation check
        fvg_markings = self._detect_fair_value_gaps(
            symbol, timeframe, candles, source=source, broker_id=broker_id, backtest_run_id=backtest_run_id
        )
        raw_markings.extend(fvg_markings)

        # 4. Duplicate Suppression (filter out duplicate price levels within tolerance)
        deduped = self._suppress_duplicates(raw_markings)

        # 5. Clutter Check: Limit to max visible markings by priority
        deduped.sort(key=lambda m: (m.priority, m.confidence), reverse=True)
        final_markings = deduped[: self._max_markings]

        return final_markings

    def _detect_fair_value_gaps(
        self,
        symbol: str,
        timeframe: str,
        candles: List[Candle],
        source: str = "SUPERVISOR",
        broker_id: Optional[str] = None,
        backtest_run_id: Optional[str] = None,
    ) -> List[ChartMarking]:
        """Identify 3-candle Fair Value Gaps and check for mitigation."""
        fvgs: List[ChartMarking] = []
        if len(candles) < 3:
            return fvgs

        latest_price = candles[-1].close
        now_ts = datetime.now(timezone.utc).timestamp()

        for i in range(len(candles) - 2):
            c1 = candles[i]
            c2 = candles[i + 1]
            c3 = candles[i + 2]

            # Bullish FVG: c1.high < c3.low
            if c3.low > c1.high:
                gap_top = c3.low
                gap_bottom = c1.high
                # Check mitigation: has subsequent price closed below gap_bottom?
                is_invalidated = latest_price < gap_bottom
                if is_invalidated:
                    continue

                fvgs.append(
                    ChartMarking(
                        symbol=symbol,
                        timeframe=timeframe,
                        marking_type=MarkingType.FVG,
                        coordinates_json={
                            "top": gap_top,
                            "bottom": gap_bottom,
                            "mid": (gap_top + gap_bottom) / 2.0,
                            "direction": "BULLISH",
                            "start_ts": c1.timestamp,
                            "end_ts": c3.timestamp,
                        },
                        validity=MarkingValidity.VALID,
                        confidence=0.88,
                        invalidation_condition="Candle close below gap bottom",
                        priority=7,
                        concept="FVG_BULLISH",
                        price=(gap_top + gap_bottom) / 2.0,
                        price_range={"top": gap_top, "bottom": gap_bottom},
                        direction="BULLISH",
                        source=source,
                        status="ACTIVE",
                        visibility="VISIBLE",
                        created_by="MARKET_SUPERVISOR" if source == "SUPERVISOR" else "BACKTEST_ENGINE",
                        broker_id=broker_id,
                        backtest_run_id=backtest_run_id,
                        timestamp=now_ts,
                    )
                )

            # Bearish FVG: c1.low > c3.high
            elif c1.low > c3.high:
                gap_top = c1.low
                gap_bottom = c3.high
                is_invalidated = latest_price > gap_top
                if is_invalidated:
                    continue

                fvgs.append(
                    ChartMarking(
                        symbol=symbol,
                        timeframe=timeframe,
                        marking_type=MarkingType.FVG,
                        coordinates_json={
                            "top": gap_top,
                            "bottom": gap_bottom,
                            "mid": (gap_top + gap_bottom) / 2.0,
                            "direction": "BEARISH",
                            "start_ts": c1.timestamp,
                            "end_ts": c3.timestamp,
                        },
                        validity=MarkingValidity.VALID,
                        confidence=0.88,
                        invalidation_condition="Candle close above gap top",
                        priority=7,
                        concept="FVG_BEARISH",
                        price=(gap_top + gap_bottom) / 2.0,
                        price_range={"top": gap_top, "bottom": gap_bottom},
                        direction="BEARISH",
                        source=source,
                        status="ACTIVE",
                        visibility="VISIBLE",
                        created_by="MARKET_SUPERVISOR" if source == "SUPERVISOR" else "BACKTEST_ENGINE",
                        broker_id=broker_id,
                        backtest_run_id=backtest_run_id,
                        timestamp=now_ts,
                    )
                )

        return fvgs

    def _suppress_duplicates(self, markings: List[ChartMarking]) -> List[ChartMarking]:
        """Suppress duplicate markings with near-identical price levels."""
        unique_markings: List[ChartMarking] = []

        for m in markings:
            m_price = self._extract_marking_price(m)
            is_dup = False

            if m_price is not None:
                for existing in unique_markings:
                    if existing.marking_type == m.marking_type:
                        ex_price = self._extract_marking_price(existing)
                        if ex_price is not None:
                            diff = abs(m_price - ex_price) / max(ex_price, 1e-6)
                            if diff < self._price_duplicate_tolerance:
                                is_dup = True
                                break
            if not is_dup:
                unique_markings.append(m)

        return unique_markings

    def _extract_marking_price(self, m: ChartMarking) -> Optional[float]:
        """Helper to extract reference price from coordinates or price attribute."""
        if m.price is not None:
            return float(m.price)
        coords = m.coordinates_json
        if "price" in coords:
            return float(coords["price"])
        if "top" in coords and "bottom" in coords:
            return float((coords["top"] + coords["bottom"]) / 2.0)
        return None

    def _map_setup_to_marking_type(self, setup_type: str) -> MarkingType:
        """Map setup pattern name to ChartMarking type."""
        st = setup_type.lower()
        if "sweep" in st:
            return MarkingType.SWEEP
        if "swing_high" in st or "swing high" in st:
            return MarkingType.SWING_HIGH
        if "swing_low" in st or "swing low" in st:
            return MarkingType.SWING_LOW
        if "poi" in st:
            return MarkingType.POI
        if "support" in st:
            return MarkingType.SUPPORT
        if "resistance" in st:
            return MarkingType.RESISTANCE
        if "entry" in st:
            return MarkingType.ENTRY
        if "stop" in st:
            return MarkingType.STOP
        if "target" in st:
            return MarkingType.TARGET
        if "confirm" in st:
            return MarkingType.CONFIRMATION
        if "invalid" in st:
            return MarkingType.INVALIDATED_SETUP
        if "fvg" in st:
            return MarkingType.FVG
        if "ob" in st or "order block" in st:
            return MarkingType.ORDER_BLOCK
        if "liquidity" in st:
            return MarkingType.LIQUIDITY
        if "bos" in st:
            return MarkingType.BOS
        if "choch" in st:
            return MarkingType.CHOCH
        if "structure" in st:
            return MarkingType.MARKET_STRUCTURE
        return MarkingType.SETUP_ZONE
