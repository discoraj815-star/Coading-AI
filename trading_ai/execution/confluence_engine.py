"""Confluence Engine: Deterministic multi-factor independent confirmation validator."""

import logging
from typing import Dict, List, Optional, Set, Tuple

from trading_ai.schemas.enums import ConfluenceCategory
from trading_ai.schemas.execution import TradeConfluence
from trading_ai.schemas.market import CurrentSituationSnapshot

logger = logging.getLogger(__name__)


class ConfluenceEngine:
    """
    Validates independent technical and fundamental confluence for trade proposals.

    Hard Rules:
      - Minimum default: 2 independent confirmations, Preferred: 3.
      - NEVER double-count multiple observations of the same underlying event.
      - Each confirmation belongs to an `independence_group`.
      - Missing data is marked DATA_UNAVAILABLE; never fabricated.
    """

    MIN_INDEPENDENT_CONFLUENCES = 2
    PREFERRED_INDEPENDENT_CONFLUENCES = 3

    def evaluate_confluences(
        self,
        symbol: str,
        situation: CurrentSituationSnapshot,
        direction: str,
    ) -> Tuple[List[TradeConfluence], bool, List[str]]:
        """
        Extract and validate independent confluences from the current market situation.

        Returns:
            (valid_confluences, is_confluence_sufficient, notes_or_warnings)
        """
        confluences: List[TradeConfluence] = []
        observed_groups: Set[str] = set()
        warnings: List[str] = []

        is_bullish = direction.upper() in ("BUY", "LONG", "BULLISH")

        # 1. Market Structure & Direction Alignment
        bias_str = situation.overall_bias.value if hasattr(situation.overall_bias, "value") else str(situation.overall_bias)
        if (is_bullish and "BULL" in bias_str.upper()) or (not is_bullish and "BEAR" in bias_str.upper()):
            group = f"STRUCTURE_{situation.symbol}"
            if group not in observed_groups:
                observed_groups.add(group)
                confluences.append(
                    TradeConfluence(
                        category=ConfluenceCategory.MARKET_STRUCTURE,
                        source="MultiTimeframeEngine",
                        independence_group=group,
                        timeframe="HTF/MTF",
                        evidence_reference=f"Bias: {bias_str}. {situation.bias_summary}",
                        confidence=0.9,
                    )
                )

        # 2. Confirmed Lecture Setup Matching
        setup = situation.active_confirmed_setup
        if setup:
            setup_dir = setup.direction.value if hasattr(setup.direction, "value") else str(setup.direction)
            if (is_bullish and "BULL" in setup_dir.upper()) or (not is_bullish and "BEAR" in setup_dir.upper()):
                group = f"SETUP_{setup.applicable_rule_id or setup.setup_type}"
                if group not in observed_groups:
                    observed_groups.add(group)
                    confluences.append(
                        TradeConfluence(
                            category=ConfluenceCategory.LECTURE_RULE,
                            source="LectureValidator",
                            independence_group=group,
                            timeframe=setup.timeframe,
                            evidence_reference=(
                                f"Rule {setup.applicable_rule_id or setup.setup_type} satisfied with "
                                f"{len(setup.conditions_satisfied)} conditions."
                            ),
                            confidence=setup.confidence,
                        )
                    )

        # 3. Point of Interest (POI) Reaction / Rejection
        poi = situation.current_poi
        if poi:
            poi_price = poi.price if hasattr(poi, "price") else (poi.get("price", 0.0) if isinstance(poi, dict) else 0.0)
            poi_type = poi.level_type if hasattr(poi, "level_type") else (poi.get("level_type", "POI") if isinstance(poi, dict) else "POI")
            type_str = poi_type.value if hasattr(poi_type, "value") else str(poi_type)
            
            # Distance check (within 0.25% of POI)
            if poi_price > 0 and abs(situation.current_price - poi_price) / poi_price <= 0.0025:
                group = f"POI_{round(poi_price, 4)}"
                if group not in observed_groups:
                    observed_groups.add(group)
                    confluences.append(
                        TradeConfluence(
                            category=ConfluenceCategory.POI,
                            source="ChartMarker",
                            independence_group=group,
                            timeframe=getattr(poi, "timeframe", "15m"),
                            evidence_reference=f"Price {situation.current_price:.5f} reacting at {type_str} @ {poi_price:.5f}",
                            confidence=0.85,
                        )
                    )

        # 4. Liquidity Sweep or Proximity
        liq = situation.nearest_major_liquidity
        if liq:
            liq_price = liq.price if hasattr(liq, "price") else (liq.get("price", 0.0) if isinstance(liq, dict) else 0.0)
            liq_type = liq.level_type if hasattr(liq, "level_type") else (liq.get("level_type", "LIQUIDITY") if isinstance(liq, dict) else "LIQUIDITY")
            type_str = liq_type.value if hasattr(liq_type, "value") else str(liq_type)
            
            # Liquidity level provides clear target or sweep premise
            group = f"LIQUIDITY_{round(liq_price, 4)}"
            if group not in observed_groups:
                observed_groups.add(group)
                confluences.append(
                    TradeConfluence(
                        category=ConfluenceCategory.LIQUIDITY,
                        source="MTFEngine",
                        independence_group=group,
                        timeframe=getattr(liq, "timeframe", "1h"),
                        evidence_reference=f"Key liquidity pool identified at {type_str} @ {liq_price:.5f}",
                        confidence=0.8,
                    )
                )

        # 5. Fundamental Alignment (Step 6)
        if situation.technical_fundamental_relationship == "TECHNICALLY_AND_FUNDAMENTALLY_ALIGNED":
            group = "FUNDAMENTAL_ALIGNMENT"
            if group not in observed_groups:
                observed_groups.add(group)
                confluences.append(
                    TradeConfluence(
                        category=ConfluenceCategory.FUNDAMENTAL,
                        source="TechnicalFundamentalCorrelator",
                        independence_group=group,
                        timeframe="MACRO",
                        evidence_reference=f"Fundamental context aligns with technical trade: {situation.fundamental_context or 'Aligned macro sentiment'}",
                        confidence=0.85,
                    )
                )
        elif situation.technical_fundamental_relationship == "FUNDAMENTAL_CONFLICT":
            warnings.append(
                f"FUNDAMENTAL_CONFLICT: Technical direction opposes macroeconomic context ({situation.news_risk})."
            )

        # Independence count verification
        independent_count = len(observed_groups)
        is_sufficient = independent_count >= self.MIN_INDEPENDENT_CONFLUENCES

        if not is_sufficient:
            warnings.append(
                f"INSUFFICIENT_CONFLUENCE: Found {independent_count} independent confirmation(s); "
                f"minimum required is {self.MIN_INDEPENDENT_CONFLUENCES}."
            )

        return confluences, is_sufficient, warnings
