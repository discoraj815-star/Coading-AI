"""Scenario Engine: Conditional IF-THEN-UNLESS market scenario generation and analysis."""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from trading_ai.schemas.companion import ScenarioCondition
from trading_ai.schemas.enums import AlertQuality
from trading_ai.schemas.market import CurrentSituationSnapshot
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class ScenarioEngine:
    """
    Generates conditional IF-THEN-UNLESS scenarios based on market structure and POIs.

    Guarantees:
      - Always framed conditionally (IF - THEN - UNLESS - INVALIDATED WHEN).
      - NEVER predicts future prices with absolute certainty or fabricated statistical odds.
      - Uses available evidence: S/R, POIs, liquidity pools, and structure boundaries.
    """

    def __init__(self, repository: Repository):
        self._repo = repository

    def generate_scenarios_for_symbol(
        self,
        symbol: str,
        custom_level: Optional[float] = None,
        situation: Optional[CurrentSituationSnapshot] = None,
    ) -> List[ScenarioCondition]:
        """
        Generate conditional scenarios for a symbol based on nearest key levels
        or an explicitly queried hypothetical level.
        """
        sit = situation or self._repo.get_latest_current_situation(symbol)
        if not sit:
            return []

        scenarios: List[ScenarioCondition] = []
        current_price = sit.current_price
        bias = sit.overall_bias.value if hasattr(sit.overall_bias, "value") else str(sit.overall_bias)

        # 1. Evaluate Explicit Custom Level Scenario
        if custom_level is not None and custom_level > 0:
            sc = self._build_custom_level_scenario(symbol, current_price, custom_level, sit)
            scenarios.append(sc)
            self._repo.save_scenario_analysis(sc)
            return scenarios

        # 2. Evaluate POI Scenario
        if sit.current_poi:
            if isinstance(sit.current_poi, dict):
                poi_p = sit.current_poi.get("price", 0.0)
                poi_name = sit.current_poi.get("name") or sit.current_poi.get("level_type", "POI")
            else:
                poi_p = getattr(sit.current_poi, "price", 0.0)
                poi_type_obj = getattr(sit.current_poi, "level_type", "POI")
                poi_type_str = poi_type_obj.value if hasattr(poi_type_obj, "value") else str(poi_type_obj)
                poi_name = getattr(sit.current_poi, "name", None) or poi_type_str

            if poi_p > 0:
                is_above = current_price >= poi_p
                direction = "BULLISH_CONTINUATION" if is_above else "BEARISH_CONTINUATION"

                sc_poi = ScenarioCondition(
                    symbol=symbol,
                    level=poi_p,
                    direction=direction,
                    if_condition=(
                        f"IF price holds support above {poi_name} ({poi_p:.5f}) with lower-timeframe rejection wicks"
                        if is_above
                        else f"IF price encounters resistance at {poi_name} ({poi_p:.5f}) and fails to close above"
                    ),
                    then_outcome=(
                        f"THEN expect continuation toward nearest liquidity pool at "
                        f"{sit.nearest_major_liquidity.price if sit.nearest_major_liquidity else 'untested highs'}."
                    ),
                    unless_condition=(
                        f"UNLESS price prints a strong 15m body close breaching {poi_p:.5f}"
                    ),
                    invalidated_when=(
                        f"A sustained close past {poi_p:.5f} invalidates this scenario, signaling potential displacement."
                    ),
                    next_poi_or_liquidity=(
                        sit.nearest_major_liquidity.price if sit.nearest_major_liquidity else None
                    ),
                    confidence_quality=AlertQuality.HIGH_QUALITY if sit.active_confirmed_setup else AlertQuality.MEDIUM_QUALITY,
                    evidence=[
                        f"Current Price: {current_price:.5f}",
                        f"POI: {poi_name} at {poi_p:.5f}",
                        f"Overall Bias: {bias}",
                    ],
                )
                scenarios.append(sc_poi)
                self._repo.save_scenario_analysis(sc_poi)

        # 3. Evaluate Liquidity Sweep / Break Scenario
        if sit.nearest_major_liquidity:
            if isinstance(sit.nearest_major_liquidity, dict):
                liq_p = sit.nearest_major_liquidity.get("price", 0.0)
                liq_name = sit.nearest_major_liquidity.get("name") or sit.nearest_major_liquidity.get("level_type", "Liquidity Pool")
            else:
                liq_p = getattr(sit.nearest_major_liquidity, "price", 0.0)
                liq_type_obj = getattr(sit.nearest_major_liquidity, "level_type", "Liquidity Pool")
                liq_type_str = liq_type_obj.value if hasattr(liq_type_obj, "value") else str(liq_type_obj)
                liq_name = getattr(sit.nearest_major_liquidity, "name", None) or liq_type_str

            if liq_p > 0:
                is_above = current_price < liq_p  # Target is above
                sc_liq = ScenarioCondition(
                    symbol=symbol,
                    level=liq_p,
                    direction="LIQUIDITY_RUN" if is_above else "LIQUIDITY_SWEEP",
                    if_condition=(
                        f"IF upward momentum breaks and sweeps buy-side liquidity at {liq_name} ({liq_p:.5f})"
                        if is_above
                        else f"IF downward momentum sweeps sell-side liquidity at {liq_name} ({liq_p:.5f})"
                    ),
                    then_outcome=(
                        "THEN look for either immediate reversal displacement (turtle soup) or impulsive continuation to next HTF pool."
                    ),
                    unless_condition=(
                        f"UNLESS price stalls or rejects before reaching {liq_p:.5f}"
                    ),
                    invalidated_when=(
                        f"Reversal structure forming prior to sweeping {liq_p:.5f} invalidates the immediate run."
                    ),
                    next_poi_or_liquidity=liq_p,
                    confidence_quality=AlertQuality.MEDIUM_QUALITY,
                    evidence=[
                        f"Current Price: {current_price:.5f}",
                        f"Liquidity Pool: {liq_name} at {liq_p:.5f}",
                        f"Proximity: {abs(current_price - liq_p):.5f}",
                    ],
                )
                scenarios.append(sc_liq)
                self._repo.save_scenario_analysis(sc_liq)

        return scenarios

    def _build_custom_level_scenario(
        self,
        symbol: str,
        current_price: float,
        level: float,
        situation: CurrentSituationSnapshot,
    ) -> ScenarioCondition:
        """Construct scenario for a specific user-queried price level."""
        is_above = level > current_price
        direction = "UPSIDE_BREAKOUT" if is_above else "DOWNSIDE_BREAKOUT"

        return ScenarioCondition(
            symbol=symbol,
            level=level,
            direction=direction,
            if_condition=(
                f"IF price surges and confirms a candle body close above {level:.5f}"
                if is_above
                else f"IF price breaks down and closes firmly below {level:.5f}"
            ),
            then_outcome=(
                f"THEN market structure shifts toward next liquidity target, likely triggering stop orders."
            ),
            unless_condition=(
                f"UNLESS this breach is accompanied by low volume/wick-only rejection without displacement"
            ),
            invalidated_when=(
                f"Failure to sustain above {level:.5f} with immediate re-entry into prior range invalidates the breakout."
            ),
            next_poi_or_liquidity=(
                situation.nearest_major_liquidity.price if situation.nearest_major_liquidity else None
            ),
            confidence_quality=AlertQuality.MEDIUM_QUALITY,
            evidence=[
                f"Current Price: {current_price:.5f}",
                f"Queried Level: {level:.5f}",
                f"Distance: {abs(current_price - level):.5f}",
                f"HTF Bias: {situation.overall_bias.value if hasattr(situation.overall_bias, 'value') else situation.overall_bias}",
            ],
        )
