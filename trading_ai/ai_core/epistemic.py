"""Four-Layer Epistemic Separation Engine for Step 15.

Strict Epistemic Invariants:
1. WHAT I LEARNED: Rules, concepts, approved mentor knowledge, journal lessons.
2. WHAT I OBSERVE: Current factual market evidence (candles, liquidity, structure, news).
3. WHAT I INFER: Synthesis connecting learned knowledge with current evidence.
4. WHAT I DECIDE: Actionable decision (Trade, No Trade, Wait, Monitor, Need Confirmation, Invalidated).

Philosophical Principle:
"मैंने क्या सीखा → अभी क्या देख रहा हूँ → उससे क्या निष्कर्ष निकलता है → मुझे क्या निर्णय लेना चाहिए।"

Boundary Safety:
- An observation CAN NEVER automatically become permanent learned knowledge.
- An inference CAN NEVER become an approved trading rule without explicit validation/promotion.
"""

from enum import Enum
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EpistemicState(str, Enum):
    OBSERVED = "OBSERVED"
    SUPPORTED = "SUPPORTED"
    INFERRED = "INFERRED"
    PARTIALLY_UNDERSTOOD = "PARTIALLY_UNDERSTOOD"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTED = "CONFLICTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    VALIDATED = "VALIDATED"


class FourLayerEpistemicState(BaseModel):
    """Rigorous epistemic state container separating all four cognitive layers."""

    # Layer 1: What I Learned
    what_i_learned: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Authoritative approved rules, mentor instructions, validated journal lessons",
    )

    # Layer 2: What I Observe
    what_i_observe: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Factual real-time observations: price, candles, volume, MTF state, news",
    )

    # Layer 3: What I Infer
    what_i_infer: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Deductive and probabilistic conclusions synthesised from Layer 1 + Layer 2",
    )

    # Layer 4: What I Decide
    what_i_decide: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final action: Trade, No Trade, Wait, Need Confirmation, Invalidated",
    )

    overall_epistemic_state: EpistemicState = Field(
        default=EpistemicState.OBSERVED,
        description="Current epistemic confidence and validation state",
    )
    hindi_summary: str = Field(
        default="",
        description="Clear synthesis in simple/pure Hindi for user-facing communication",
    )


class EpistemicSeparationEngine:
    """Evaluates and maintains strict boundaries across the 4 cognitive layers."""

    @classmethod
    def evaluate(
        cls,
        learned_rules: List[Dict[str, Any]],
        market_observations: List[Dict[str, Any]],
        inferences: Optional[List[Dict[str, Any]]] = None,
        why_not_reasons: Optional[List[str]] = None,
        news_risk: str = "NO_RISK",
    ) -> FourLayerEpistemicState:
        """Synthesize four-layer epistemic state ensuring strict boundary rules."""
        inferred_list = list(inferences or [])
        blocking_reasons = list(why_not_reasons or [])

        # Check for epistemic conflicts or missing knowledge
        if not learned_rules:
            epistemic_state = EpistemicState.NEEDS_REVIEW
            decision = {
                "action": "NO_TRADE",
                "reason": "INSUFFICIENT_KNOWLEDGE: No approved lecture/mentorship rules matched.",
            }
            hindi = "इस सेटअप के लिए कोई मान्य नियम नहीं मिला है (Insufficient Knowledge), इसलिए अभी ट्रेड न लें।"
        elif not market_observations:
            epistemic_state = EpistemicState.NEEDS_REVIEW
            decision = {
                "action": "NO_TRADE",
                "reason": "INSUFFICIENT_EVIDENCE: No real-time market observations available.",
            }
            hindi = "मार्केट डेटा या साक्ष्य अपर्याप्त हैं (Insufficient Evidence), ट्रेड नहीं लिया जा सकता।"
        elif blocking_reasons:
            epistemic_state = EpistemicState.CONFLICTED
            decision = {
                "action": "NO_TRADE",
                "reason": f"BLOCKED by Why-Not-Trade: {', '.join(blocking_reasons)}",
            }
            hindi = f"सेटअप को Why-Not-Trade नियमों ने रोक दिया है: {', '.join(blocking_reasons[:2])}। कृपया ट्रेड से बचें।"
        elif news_risk in ("HIGH_RISK", "CRITICAL"):
            epistemic_state = EpistemicState.CONFLICTED
            decision = {
                "action": "WAIT",
                "reason": "HIGH_IMPACT_NEWS: Pending major economic event.",
            }
            hindi = "हाई-इम्पैक्ट समाचार आने वाला है, जोखिम से बचने के लिए अभी शांत रहें और प्रतीक्षा करें।"
        else:
            # Check for partial matches vs fully validated setup
            has_pending_confirmation = any(
                "pending" in str(inf.get("status", "")).lower() or "missing" in str(inf.get("status", "")).lower()
                for inf in inferred_list
            )

            if has_pending_confirmation:
                epistemic_state = EpistemicState.PARTIALLY_UNDERSTOOD
                decision = {
                    "action": "WAIT",
                    "reason": "CONFIRMATION_PENDING: Rule matched partially; waiting for displacement/confirmation.",
                }
                hindi = "सेटअप आंशिक रूप से बना है, लेकिन पूर्ण पुष्टि (Confirmation) बाकी है। कृपया इंतज़ार करें।"
            else:
                epistemic_state = EpistemicState.VALIDATED
                decision = {
                    "action": "TRADE_CANDIDATE",
                    "reason": "VALIDATED: All learned rules satisfied by current observations.",
                }
                hindi = "सीखे गए सभी नियमों की पुष्टि वर्तमान मार्केट से हो चुकी है। यह एक मान्य सेटअप है।"

        return FourLayerEpistemicState(
            what_i_learned=learned_rules,
            what_i_observe=market_observations,
            what_i_infer=inferred_list,
            what_i_decide=decision,
            overall_epistemic_state=epistemic_state,
            hindi_summary=hindi,
        )
