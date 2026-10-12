"""Footprint Concept Module for Step 17 Master Decision Engine.

Evaluates:
- Bid/Ask volume clusters
- Stacked buy/sell imbalances (e.g. 300% imbalance)
- Absorption and exhaustion at price extremes
- Unfinished auction / single prints

CRITICAL HARD EPISTEMIC INVARIANT:
If footprint cluster data is missing, DO NOT fabricate footprint confirmation.
Return FOOTPRINT_DATA_UNAVAILABLE with direction UNKNOWN and confidence 0.0.
"""
import time
from typing import Any, Dict, List, Optional

from trading_ai.master_decision.concept_contract import BaseConceptModule
from trading_ai.schemas.enums import ConceptDirection
from trading_ai.schemas.master_decision import ConceptResult, EpistemicConceptState


class FootprintModule(BaseConceptModule):
    """Footprint & Bid/Ask Imbalance Chart Concept Module."""

    def __init__(self, authority_priority: int = 4):
        super().__init__(
            concept_id="footprint",
            concept_name="Footprint Chart & Imbalance",
            authority_priority=authority_priority,
            required_data_fields=["footprint_data"],
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

        fp_data = market_data.get("footprint_data")
        if not is_avail or not fp_data or not isinstance(fp_data, dict):
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
                missing_data=["footprint_data", "bid_ask_clusters"],
                notes="FOOTPRINT_DATA_UNAVAILABLE: No footprint cluster feed or bid/ask profile available.",
            )

        # Footprint metrics extraction
        stacked_imbalance_side = fp_data.get("stacked_imbalance_side", "NONE")
        stacked_count = int(fp_data.get("stacked_imbalance_count", 0))
        cluster_high_volume = float(fp_data.get("point_of_control_volume", 0.0))
        unfinished_auction = fp_data.get("unfinished_auction", False)
        exhaustion_at_extreme = fp_data.get("exhaustion_at_extreme", False)

        if stacked_imbalance_side.upper() == "BUY" and stacked_count >= 3:
            direction = ConceptDirection.BULLISH
            bias_score = min(1.0, 0.60 + (stacked_count * 0.08))
        elif stacked_imbalance_side.upper() == "SELL" and stacked_count >= 3:
            direction = ConceptDirection.BEARISH
            bias_score = max(-1.0, -0.60 - (stacked_count * 0.08))
        elif exhaustion_at_extreme:
            side = fp_data.get("exhaustion_side", "SELLERS")
            direction = ConceptDirection.BULLISH if "SELL" in side.upper() else ConceptDirection.BEARISH
            bias_score = 0.65 if direction == ConceptDirection.BULLISH else -0.65
        else:
            direction = ConceptDirection.NEUTRAL
            bias_score = 0.0

        # Epistemic layers
        observed_items = [
            f"Stacked Imbalance: {stacked_imbalance_side} (Level count: {stacked_count})",
            f"Footprint Node POC Volume: {cluster_high_volume:.1f}",
        ]
        if unfinished_auction:
            observed_items.append("Unfinished auction identified at swing boundary.")
        if exhaustion_at_extreme:
            observed_items.append("Volume exhaustion noted at candle extreme.")

        applied_rules = []
        rule_ids = []
        rule_versions = []
        source_prov = []
        if candidate_rules:
            for r in candidate_rules:
                title = getattr(r, "title", r.get("title", "") if isinstance(r, dict) else "")
                r_id = getattr(r, "item_id", r.get("item_id", "") if isinstance(r, dict) else "")
                r_ver = str(getattr(r, "version", r.get("version", "1.0") if isinstance(r, dict) else "1.0"))
                if any(k in title.lower() for k in ["footprint", "imbalance", "stacked", "cluster"]):
                    applied_rules.append(f"Learned Rule: '{title}' (v{r_ver})")
                    rule_ids.append(r_id)
                    rule_versions.append(r_ver)
                    source_prov.append({"rule_id": r_id, "title": title, "authority": "LECTURE"})
        if not applied_rules:
            applied_rules.append("Standard Footprint Framework: 3+ Stacked Imbalance Clusters & Extreme Exhaustion")

        inferred_items = [
            f"Footprint micro-structure indicates aggressive {stacked_imbalance_side} conviction.",
            f"Auction completion: {'Unfinished auction target' if unfinished_auction else 'Balanced auction resolution'}.",
        ]

        decided_text = f"Footprint resolves {direction.value} based on {stacked_count} stacked {stacked_imbalance_side} levels."

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
            confidence=0.85 if stacked_count >= 3 else 0.65,
            evidence=observed_items,
            evidence_quality=0.85,
            required_data=self.required_data_fields,
            available_data=avail,
            missing_data=[],
            is_data_available=True,
            rule_ids=rule_ids,
            rule_versions=rule_versions,
            source_provenance=source_prov,
            epistemic_state=epistemic,
            invalidation="Stacked imbalance zone breached by closing auction",
            notes="Valid footprint cluster analysis executed.",
        )
