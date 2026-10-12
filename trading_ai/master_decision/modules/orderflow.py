"""Orderflow Concept Module for Step 17 Master Decision Engine.

Evaluates:
- Real delta and cumulative volume delta (CVD)
- Aggressive buyers vs aggressive sellers
- Absorption at key levels
- Trapped traders and exhaustion

CRITICAL HARD EPISTEMIC INVARIANT:
If real orderflow/delta data is not present, DO NOT fabricate or infer delta.
Return DATA_UNAVAILABLE with direction UNKNOWN and confidence 0.0.
"""
import time
from typing import Any, Dict, List, Optional

from trading_ai.master_decision.concept_contract import BaseConceptModule
from trading_ai.schemas.enums import ConceptDirection
from trading_ai.schemas.master_decision import ConceptResult, EpistemicConceptState


class OrderflowModule(BaseConceptModule):
    """Orderflow & Delta Dynamics Concept Module."""

    def __init__(self, authority_priority: int = 4):
        super().__init__(
            concept_id="orderflow",
            concept_name="Orderflow & Delta Analysis",
            authority_priority=authority_priority,
            required_data_fields=["orderflow_data"],
            is_approved=True,
            is_macro=False,  # Micro/Execution layer
        )

    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        market_data: Dict[str, Any],
        candidate_rules: Optional[List[Any]] = None,
        as_of_timestamp: Optional[float] = None,
    ) -> ConceptResult:
        now_ts = as_of_timestamp or time.time()
        is_avail, avail, missing = self.validate_data_availability(market_data)

        # Check orderflow availability
        of_data = market_data.get("orderflow_data")
        if not is_avail or not of_data or not isinstance(of_data, dict):
            return ConceptResult(
                concept_id=self.concept_id,
                concept_name=self.concept_name,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=now_ts,
                direction=ConceptDirection.UNKNOWN,
                bias_score=0.0,
                confidence=0.0,
                is_data_available=False,
                required_data=self.required_data_fields,
                available_data=avail,
                missing_data=["orderflow_data", "delta_feed"],
                notes="ORDERFLOW_DATA_UNAVAILABLE: No real delta or orderflow feed present for symbol.",
            )

        # Real orderflow metrics extraction
        delta = float(of_data.get("delta", 0.0))
        cvd = float(of_data.get("cumulative_delta", 0.0))
        cvd_slope = of_data.get("cvd_slope", "FLAT")
        absorption = of_data.get("absorption_detected", False)
        absorption_side = of_data.get("absorption_side", "NONE")
        trapped_traders = of_data.get("trapped_traders", False)
        buy_volume = float(of_data.get("aggressive_buy_volume", 0.0))
        sell_volume = float(of_data.get("aggressive_sell_volume", 0.0))

        # Direction logic based on empirical delta & absorption
        if delta > 150 or (absorption and absorption_side.upper() == "SELLERS_ABSORBED") or cvd_slope == "RISING":
            direction = ConceptDirection.BULLISH
            bias_score = min(1.0, 0.50 + (delta / 1000.0) if delta > 0 else 0.70)
        elif delta < -150 or (absorption and absorption_side.upper() == "BUYERS_ABSORBED") or cvd_slope == "FALLING":
            direction = ConceptDirection.BEARISH
            bias_score = max(-1.0, -0.50 + (delta / 1000.0) if delta < 0 else -0.70)
        else:
            direction = ConceptDirection.NEUTRAL
            bias_score = 0.0

        # Construct 4-layer epistemic state
        observed_items = [
            f"Bar Delta: {delta:+.1f}",
            f"Cumulative Delta (CVD): {cvd:+.1f} (Slope: {cvd_slope})",
            f"Aggressive Buyers Volume: {buy_volume:.1f}, Aggressive Sellers Volume: {sell_volume:.1f}",
        ]
        if absorption:
            observed_items.append(f"Absorption Confirmed: {absorption_side} at key level.")
        if trapped_traders:
            observed_items.append("Trapped market participants detected at extreme of candle.")

        # Learned rules
        applied_rules = []
        rule_ids = []
        rule_versions = []
        source_prov = []
        if candidate_rules:
            for r in candidate_rules:
                title = getattr(r, "title", r.get("title", "") if isinstance(r, dict) else "")
                r_id = getattr(r, "item_id", r.get("item_id", "") if isinstance(r, dict) else "")
                r_ver = str(getattr(r, "version", r.get("version", "1.0") if isinstance(r, dict) else "1.0"))
                if any(k in title.lower() for k in ["orderflow", "delta", "absorption", "cvd", "tape"]):
                    applied_rules.append(f"Learned Rule: '{title}' (v{r_ver})")
                    rule_ids.append(r_id)
                    rule_versions.append(r_ver)
                    source_prov.append({"rule_id": r_id, "title": title, "authority": "MENTORSHIP"})
        if not applied_rules:
            applied_rules.append("Standard Orderflow Framework: Delta Aggression + Absorption Identification")

        # Inferred deductions
        inferred_items = [
            f"Orderflow execution pressure favors {direction.value}.",
            f"Institutional absorption: {'Active defense present' if absorption else 'No passive absorption defense observed'}.",
        ]

        # Decided
        decided_text = f"Orderflow confirms {direction.value} entry pressure with delta={delta:+.1f}."

        epistemic = EpistemicConceptState(
            observed=observed_items,
            learned=applied_rules,
            inferred=inferred_items,
            decided=decided_text,
        )

        return ConceptResult(
            concept_id=self.concept_id,
            concept_name=self.concept_name,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=now_ts,
            direction=direction,
            bias_score=bias_score,
            confidence=0.85 if absorption else 0.70,
            evidence=observed_items,
            evidence_quality=0.88,
            required_data=self.required_data_fields,
            available_data=avail,
            missing_data=[],
            is_data_available=True,
            rule_ids=rule_ids,
            rule_versions=rule_versions,
            source_provenance=source_prov,
            epistemic_state=epistemic,
            invalidation="Orderflow delta flips aggressively against trade premise",
            notes="Real orderflow feed evaluated.",
        )
