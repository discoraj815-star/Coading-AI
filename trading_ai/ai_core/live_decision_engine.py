"""Proactive Live Market Decision & Scenario Forecasting Engine for Step 15."""

import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.ai_core.epistemic import EpistemicSeparationEngine, EpistemicState, FourLayerEpistemicState
from trading_ai.ai_core.multilingual import MultilingualNormalizer
from trading_ai.storage.repository import Repository
from trading_ai.trade_logic.core_trade_logic import CoreTradeLogic
from trading_ai.trade_logic.why_not_trade import WhyNotTradeEngine

logger = logging.getLogger(__name__)


class ForecastScenario(BaseModel):
    """Conditional forward-looking market scenario (not a prediction)."""

    scenario_name: str
    condition: str
    expected_outcome: str
    invalidation_level: Optional[float] = None
    confidence: float = 0.80
    uncertainty_notes: str
    hindi_explanation: str


class LiveDecisionResult(BaseModel):
    """Consolidated proactive live decision output."""

    symbol: str
    timeframe: str
    current_price: float
    active_concept: str
    concept_match_status: str  # FULL_MATCH, PARTIAL_MATCH, NO_MATCH
    missing_confirmations: List[str]
    scenarios: List[ForecastScenario]
    epistemic_state: FourLayerEpistemicState
    why_not_trade_result: Dict[str, Any]
    final_decision: str  # TRADE_CANDIDATE, WAIT, NO_TRADE, INVALIDATED
    hindi_speech: str


class LiveMarketDecisionEngine:
    """Enterprise Proactive Decision Engine for Step 15.

    Synthesizes:
    - Current Market Observations (Price, Liquidity, FVG, BOS, News)
    - Learned Lecture & Mentorship Rules
    - Active Concept Mapping & Gap Analysis
    - Scenario Forecasting (If-Then-Unless)
    - Mandatory Why-Not-Trade Checks
    - Four-Layer Epistemic Output
    - User-Facing Simple Hindi Synthesis
    """

    def __init__(
        self,
        repository: Optional[Repository] = None,
        trade_logic: Optional[CoreTradeLogic] = None,
        why_not_engine: Optional[WhyNotTradeEngine] = None,
    ):
        self._repo = repository
        self._trade_logic = trade_logic or CoreTradeLogic()
        self._why_not = why_not_engine or WhyNotTradeEngine()

    def evaluate_live_market(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "15m",
        current_price: float = 1.0850,
        direction: str = "BUY",
        observed_sweep: bool = True,
        observed_displacement: bool = False,
        observed_fvg: bool = False,
        htf_bias: str = "BULLISH",
        news_risk_level: str = "NO_RISK",
        structural_sl: Optional[float] = None,
        target_tp: Optional[float] = None,
    ) -> LiveDecisionResult:
        """Execute complete live market analysis and scenario generation."""
        sl_price = structural_sl or (current_price - 0.0025 if direction == "BUY" else current_price + 0.0025)
        tp_price = target_tp or (current_price + 0.0060 if direction == "BUY" else current_price - 0.0060)

        # 1. Learned Knowledge (Layer 1)
        learned_rules = [
            {
                "rule_id": "rule_ict_sweep_disp",
                "title": "Liquidity Sweep & Displacement Confirmation Rule",
                "conditions": ["Liquidity sweep of session high/low", "Energetic displacement candle", "Fair Value Gap created"],
                "authority": "APPROVED_LECTURE_RULE",
            }
        ]

        # 2. What I Observe (Layer 2)
        observations = [
            {"type": "CURRENT_PRICE", "value": current_price, "symbol": symbol, "timeframe": timeframe},
            {"type": "HTF_BIAS", "value": htf_bias},
            {"type": "LIQUIDITY_SWEEP", "detected": observed_sweep, "level": current_price - 0.0010 if direction == "BUY" else current_price + 0.0010},
            {"type": "DISPLACEMENT", "detected": observed_displacement},
            {"type": "FAIR_VALUE_GAP", "detected": observed_fvg},
            {"type": "NEWS_RISK", "level": news_risk_level},
        ]

        # 3. Concept Mapping & Gap Analysis
        active_concept = "Liquidity Sweep + Displacement + FVG Entry"
        missing_confirmations = []
        if not observed_sweep:
            missing_confirmations.append("Liquidity Sweep pending")
        if not observed_displacement:
            missing_confirmations.append("Displacement candle confirmation missing")
        if not observed_fvg:
            missing_confirmations.append("FVG mitigation pending")

        if not missing_confirmations:
            match_status = "FULL_MATCH"
        elif observed_sweep and (not observed_displacement or not observed_fvg):
            match_status = "PARTIAL_MATCH"
        else:
            match_status = "NO_MATCH"

        # 4. What I Infer (Layer 3)
        inferences = [
            {
                "concept": active_concept,
                "match_status": match_status,
                "status": "pending_displacement" if match_status == "PARTIAL_MATCH" else ("confirmed" if match_status == "FULL_MATCH" else "no_setup"),
                "confidence": 0.88 if match_status == "FULL_MATCH" else (0.65 if match_status == "PARTIAL_MATCH" else 0.20),
            }
        ]

        # 5. Why-Not-Trade Checks (Mandatory)
        wnt_blocking = []
        wnt_warnings = []

        if news_risk_level in ("HIGH_RISK", "CRITICAL"):
            wnt_blocking.append("High-impact news event is pending within the risk window.")
        is_bias_aligned = (
            (direction.upper() == "BUY" and htf_bias.upper() in ("BULLISH", "BUY")) or
            (direction.upper() == "SELL" and htf_bias.upper() in ("BEARISH", "SELL")) or
            htf_bias.upper() == "NEUTRAL"
        )
        if not is_bias_aligned:
            wnt_blocking.append(f"HTF bias ({htf_bias}) conflicts with trade direction ({direction}).")
        if match_status == "PARTIAL_MATCH":
            wnt_warnings.append("Displacement confirmation candle has not closed yet.")

        why_not_res = {
            "has_blocking": len(wnt_blocking) > 0,
            "blocking_reasons": wnt_blocking,
            "warnings": wnt_warnings,
        }

        # 6. Forecast Scenarios (Non-deterministic, conditional)
        scenarios = [
            ForecastScenario(
                scenario_name="Scenario A: Bullish Continuation on Displacement",
                condition="If energetic displacement candle closes above swing high with FVG formation",
                expected_outcome=f"Price is likely to continue towards target {tp_price:.4f}",
                invalidation_level=sl_price,
                confidence=0.82,
                uncertainty_notes="Requires displacement confirmation; does not guarantee execution.",
                hindi_explanation=f"यदि displacement कैंडल मजबूत बनती है, तो कीमत {tp_price:.4f} तक जाने की संभावना है।",
            ),
            ForecastScenario(
                scenario_name="Scenario B: Invalidation on Structure Failure",
                condition=f"If price breaks and closes below invalidation level {sl_price:.4f}",
                expected_outcome="Setup is completely invalidated; cancel all order ideas.",
                invalidation_level=sl_price,
                confidence=0.90,
                uncertainty_notes="Hard structural invalidation prevents unwarranted risk.",
                hindi_explanation=f"यदि मूल्य {sl_price:.4f} के नीचे बंद होता है, तो यह सेटअप पूरी तरह अमान्य (Invalidated) हो जाएगा।",
            ),
        ]

        # 7. Four-Layer Epistemic Evaluation
        epistemic_state = EpistemicSeparationEngine.evaluate(
            learned_rules=learned_rules,
            market_observations=observations,
            inferences=inferences,
            why_not_reasons=wnt_blocking,
            news_risk=news_risk_level,
        )

        # 8. Final Decision Selection
        if wnt_blocking:
            final_decision = "NO_TRADE"
            hindi_speech = f"ट्रेड नहीं लेना चाहिए। कारण: {wnt_blocking[0]}"
        elif match_status == "PARTIAL_MATCH":
            final_decision = "WAIT"
            hindi_speech = "Liquidity sweep तो हुआ है, लेकिन displacement की पुष्टि अभी बाकी है। इसलिए अभी प्रतीक्षा (WAIT) करें, जल्दबाजी में ट्रेड न लें।"
        elif match_status == "FULL_MATCH":
            final_decision = "TRADE_CANDIDATE"
            hindi_speech = f"सेटअप को सभी आवश्यक नियमों की पुष्टि मिल चुकी है। Stop Loss {sl_price:.4f} के साथ ट्रेड योजना बनाई जा सकती है।"
        else:
            final_decision = "NO_TRADE"
            hindi_speech = "वर्तमान मार्केट में कोई मान्य सेटअप नहीं बन रहा है। अनुशासन बनाए रखें।"

        result = LiveDecisionResult(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            active_concept=active_concept,
            concept_match_status=match_status,
            missing_confirmations=missing_confirmations,
            scenarios=scenarios,
            epistemic_state=epistemic_state,
            why_not_trade_result=why_not_res,
            final_decision=final_decision,
            hindi_speech=hindi_speech,
        )

        # Persist scenario to db if available
        if self._repo and hasattr(self._repo, "save_live_decision_scenario"):
            try:
                self._repo.save_live_decision_scenario({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "active_concept": active_concept,
                    "learned_rule_id": "rule_ict_sweep_disp",
                    "scenarios": [s.model_dump() for s in scenarios],
                    "confirmation_conditions": missing_confirmations,
                    "invalidation_conditions": [f"Price reaches {sl_price:.4f}"],
                    "decision_state": final_decision,
                    "hindi_explanation": hindi_speech,
                })
            except Exception as e:
                logger.debug(f"Could not persist live decision scenario: {e}")

        return result
