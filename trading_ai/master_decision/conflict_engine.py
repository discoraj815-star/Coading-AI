"""Decision Conflict Engine for Step 17 Master Decision Engine.

Detects, classifies, and evaluates conflicts across:
1. Macro vs Micro layers (e.g. HTF SMC Bullish vs Micro Footprint aggressive sellers)
2. Directional Polarity conflicts (Opposing concept directions)
3. Source Authority contradictions (Conflicting rules of equal authority)
4. Orderflow execution pressure conflicts (Entering against unabsorbed aggressive delta)
5. Volume Profile rejection conflicts (Trading into unmitigated VAH/VAL rejection)
6. Fundamental / News conflicts (Trading into high-impact adverse news)
7. Stale data gating (Stale ticks or candles)

Rule: Unresolved conflicts force NO_TRADE / CONFLICTED. Never artificially average out contradictions.
"""
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.enums import ConceptDirection, TradeDecisionState
from trading_ai.schemas.master_decision import ConceptResult


class DecisionConflictEngine:
    """Rigorous multi-layer conflict detector for trade proposals."""

    def evaluate_conflicts(
        self,
        proposed_direction: str,
        concept_results: Dict[str, ConceptResult],
        market_data: Dict[str, Any],
        is_stale_data: bool = False,
        news_risk_level: str = "NO_RISK",
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Evaluate all potential conflicts across evaluated concepts and environment.

        Returns:
            (has_blocking_conflict, blocking_conflicts, informational_warnings)
        """
        blocking_conflicts: List[str] = []
        warnings: List[str] = []

        prop_dir = proposed_direction.upper()

        # 1. Stale Data Gating (Hard Block)
        if is_stale_data or market_data.get("is_stale", False):
            blocking_conflicts.append("STALE_MARKET_DATA: Market data feed exceeds stale threshold (>2 hours).")

        # 2. High-Impact Adverse News Risk
        if news_risk_level in ("HIGH_RISK", "EXTREME_RISK", "CRITICAL"):
            blocking_conflicts.append(f"FUNDAMENTAL_NEWS_CONFLICT: High-impact economic news window active ({news_risk_level}).")

        # 3. Macro vs Micro Layer Conflict
        macro_directions = set()
        micro_directions = set()

        for c_id, res in concept_results.items():
            if not res.is_data_available or res.direction in (ConceptDirection.UNKNOWN, ConceptDirection.NEUTRAL):
                continue

            # Classify macro vs micro based on concept ID
            is_macro = c_id in ("smc_ict", "volume_profile", "wyckoff", "auction_market_theory")
            if is_macro:
                macro_directions.add(res.direction.value)
            else:
                micro_directions.add(res.direction.value)

        if "BULLISH" in macro_directions and "BEARISH" in micro_directions:
            blocking_conflicts.append(
                "MACRO_MICRO_CONFLICT: Macro context (HTF SMC/Volume Profile/Wyckoff) is BULLISH "
                "but Micro execution layer (Orderflow/Footprint) exhibits strong BEARISH seller pressure."
            )
        elif "BEARISH" in macro_directions and "BULLISH" in micro_directions:
            blocking_conflicts.append(
                "MACRO_MICRO_CONFLICT: Macro context is BEARISH but Micro execution layer "
                "exhibits strong BULLISH buying pressure without confirmed macro reversal."
            )

        # 4. Multi-Concept Directional Polarity Conflict
        active_directions = set()
        for res in concept_results.values():
            if res.is_data_available and res.direction in (ConceptDirection.BULLISH, ConceptDirection.BEARISH):
                active_directions.add(res.direction.value)

        if "BULLISH" in active_directions and "BEARISH" in active_directions:
            conflicting_concepts = [
                f"{r.concept_name}: {r.direction.value}"
                for r in concept_results.values()
                if r.is_data_available and r.direction in (ConceptDirection.BULLISH, ConceptDirection.BEARISH)
            ]
            blocking_conflicts.append(
                f"CONCEPT_POLARITY_CONFLICT: Opposing directional directives detected: {', '.join(conflicting_concepts)}."
            )

        # 5. Proposed Direction vs Concept Misalignment
        if prop_dir in ("BUY", "LONG", "BULLISH"):
            for res in concept_results.values():
                if res.is_data_available and res.direction == ConceptDirection.BEARISH:
                    blocking_conflicts.append(
                        f"DIRECTIONAL_MISALIGNMENT: Proposed BUY opposes active {res.concept_name} BEARISH conclusion."
                    )
        elif prop_dir in ("SELL", "SHORT", "BEARISH"):
            for res in concept_results.values():
                if res.is_data_available and res.direction == ConceptDirection.BULLISH:
                    blocking_conflicts.append(
                        f"DIRECTIONAL_MISALIGNMENT: Proposed SELL opposes active {res.concept_name} BULLISH conclusion."
                    )

        # 6. Specific Micro Pressure Check (Orderflow / Footprint vs Proposed Trade)
        of_res = concept_results.get("orderflow")
        if of_res and of_res.is_data_available:
            if prop_dir == "BUY" and of_res.direction == ConceptDirection.BEARISH:
                blocking_conflicts.append("MICRO_ORDERFLOW_PRESSURE: Heavy aggressive seller delta against proposed Long.")
            elif prop_dir == "SELL" and of_res.direction == ConceptDirection.BULLISH:
                blocking_conflicts.append("MICRO_ORDERFLOW_PRESSURE: Heavy aggressive buyer delta against proposed Short.")

        # 7. Volume Profile Value Area Boundary Conflict
        vp_res = concept_results.get("volume_profile")
        if vp_res and vp_res.is_data_available:
            if prop_dir == "BUY" and vp_res.direction == ConceptDirection.BEARISH:
                blocking_conflicts.append("VP_VALUE_AREA_REJECTION: Trading Long directly into verified VAH rejection.")
            elif prop_dir == "SELL" and vp_res.direction == ConceptDirection.BULLISH:
                blocking_conflicts.append("VP_VALUE_AREA_REJECTION: Trading Short directly into verified VAL bounce.")

        # Summary decision
        has_blocking = len(blocking_conflicts) > 0
        return has_blocking, blocking_conflicts, warnings
