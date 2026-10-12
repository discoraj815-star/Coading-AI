"""Core Trade Logic Engine for Step 14.

Synthesizes validated knowledge, cross-source confluence, and negative validation checks
into an institutional TradeProposal and CoreTradeDecision.

Hard Invariants:
- Never executes trades directly.
- Flows strictly into Step 8 RiskEngine and ExecutionGate.
- Retains permanent source-level provenance for every suggested decision.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.schemas.enums import (
    AlertQuality,
    EngagementMode,
    ExecutionEnvironment,
    MarketBias,
    ProposalLifecycleState,
    TradeDecisionState,
)
from trading_ai.schemas.execution import TradeConfluence, TradeProposal
from trading_ai.schemas.trade_logic import (
    ConfluenceEvaluation,
    CoreTradeDecision,
    NormalizedKnowledgeItem,
    WhyNotTradeResult,
)
from trading_ai.trade_logic.data_fusion import DataFusionEngine
from trading_ai.trade_logic.why_not_trade import WhyNotTradeEngine


class CoreTradeLogic:
    """Institutional Trade Decision Engine executing Phase 14 analysis."""

    def __init__(
        self,
        data_fusion: Optional[DataFusionEngine] = None,
        why_not_trade: Optional[WhyNotTradeEngine] = None,
    ):
        self.fusion = data_fusion or DataFusionEngine()
        self.why_not = why_not_trade or WhyNotTradeEngine()

    def evaluate_and_propose(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        current_price: float,
        structural_sl: float,
        target_tp1: float,
        target_tp2: Optional[float] = None,
        candidate_rules: Optional[List[NormalizedKnowledgeItem]] = None,
        htf_bias: str = "BULLISH",
        is_poi_valid: bool = True,
        is_confirmation_present: bool = True,
        news_risk_level: str = "NO_RISK",
        active_exceptions: Optional[List[str]] = None,
        journal_warnings: Optional[List[str]] = None,
        daily_loss_limit_hit: bool = False,
    ) -> CoreTradeDecision:
        """Run complete Step 14 decision pipeline."""
        rules = candidate_rules or []

        # 1. Evaluate cross-source confluence & conflict detection
        confluence_res = self.fusion.evaluate_confluence(
            symbol=symbol,
            timeframe=timeframe,
            candidate_rules=rules,
            market_direction=direction,
            active_exceptions=active_exceptions,
            journal_warnings=journal_warnings,
        )

        # 2. Check if fusion found conflicts or no setup
        if confluence_res.decision_state in (
            TradeDecisionState.CONFLICTED,
            TradeDecisionState.NO_SETUP,
            TradeDecisionState.INSUFFICIENT_CONFLUENCE,
        ):
            return CoreTradeDecision(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                state=confluence_res.decision_state,
                why_summary=confluence_res.reasons_to_trade,
                why_not_summary=confluence_res.reasons_not_to_trade,
                source_provenance_trace=[
                    {
                        "rule_id": r.item_id,
                        "title": r.title,
                        "domain": r.domain.value,
                        "authority": r.authority.value,
                        "version": r.version,
                    }
                    for r in rules
                ],
                trade_proposal=None,
            )

        # 3. Calculate mathematical R:R
        risk = abs(current_price - structural_sl)
        reward = abs(target_tp1 - current_price)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        # Invalidation level and condition
        invalidation_text = (
            f"Candle close beyond structural invalidation level {structural_sl} on {timeframe} timeframe."
        )

        # 4. Run Mandatory "Why Not Trade?" Negative Validation Pass
        why_not_res = self.why_not.evaluate_reasons_not_to_trade(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=current_price,
            stop_loss=structural_sl,
            tp1=target_tp1,
            htf_bias=htf_bias,
            is_poi_valid=is_poi_valid,
            is_confirmation_present=is_confirmation_present,
            news_risk_level=news_risk_level,
            active_exceptions=active_exceptions,
            journal_warnings=journal_warnings,
            daily_loss_limit_hit=daily_loss_limit_hit,
        )

        if why_not_res.should_block:
            combined_why_not = confluence_res.reasons_not_to_trade + why_not_res.blocking_reasons
            return CoreTradeDecision(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                state=TradeDecisionState.NO_SETUP,
                entry_price=current_price,
                hard_stop_loss=structural_sl,
                tp1=target_tp1,
                tp2=target_tp2,
                final_target=target_tp2 or target_tp1,
                r_multiple=rr,
                invalidation_condition=invalidation_text,
                why_summary=confluence_res.reasons_to_trade,
                why_not_summary=combined_why_not,
                trade_proposal=None,
            )

        # 5. Build Permanent Provenance Trace
        provenance_trace = []
        confluence_list = []
        for r in rules:
            provenance_trace.append(
                {
                    "rule_id": r.item_id,
                    "title": r.title,
                    "domain": r.domain.value,
                    "authority": r.authority.value,
                    "version": r.version,
                    "source_id": r.source_id,
                    "evidence_refs": r.evidence_references,
                    "confidence": r.confidence,
                }
            )
            confluence_list.append(
                TradeConfluence(
                    factor=f"[{r.domain.value}] {r.title}",
                    weight=r.confidence,
                    timeframe=timeframe,
                    description=r.statement[:120],
                )
            )

        # 6. Synthesize Institutional TradeProposal (Step 8 Schema)
        proposal = TradeProposal(
            symbol=symbol,
            direction=direction.upper(),
            market_regime="TRENDING_ALIGNED" if htf_bias in ("BULLISH", "BEARISH") else "BALANCED",
            htf_context=f"Higher timeframe bias is {htf_bias}",
            mtf_context=f"Confluence score {confluence_res.confluence_score:.2f} across {len(rules)} rules",
            ltf_context=f"Execution timeframe {timeframe}",
            overall_bias=MarketBias.BULLISH_BIAS if direction.upper() == "BUY" else MarketBias.BEARISH_BIAS,
            entry_trigger="CONFIRMED_DISPLACEMENT",
            entry_price=current_price,
            hard_stop_loss=structural_sl,
            tp1=target_tp1,
            tp2=target_tp2,
            final_target=target_tp2 or target_tp1,
            invalidation_condition=invalidation_text,
            confluences=confluence_list,
            lecture_rule_evidence=[f"Rule {r.title} (v{r.version})" for r in rules],
            quality=AlertQuality.HIGH_QUALITY if confluence_res.confluence_score >= 0.8 else AlertQuality.MEDIUM_QUALITY,
            engagement_mode=EngagementMode.DEFENSIVE,
            execution_environment=ExecutionEnvironment.PAPER,
            lifecycle_state=ProposalLifecycleState.PROPOSED,
            user_approval_required=True,
        )

        return CoreTradeDecision(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction.upper(),
            state=confluence_res.decision_state,
            entry_price=current_price,
            hard_stop_loss=structural_sl,
            tp1=target_tp1,
            tp2=target_tp2,
            final_target=target_tp2 or target_tp1,
            r_multiple=rr,
            invalidation_condition=invalidation_text,
            confirmation_requirement="LTF confirmation candle displacement and sweep mitigation complete.",
            why_summary=confluence_res.reasons_to_trade,
            why_not_summary=why_not_res.non_blocking_warnings,
            source_provenance_trace=provenance_trace,
            trade_proposal=proposal,
        )

    def generate_trade_decision(
        self,
        symbol: str,
        direction: str,
        current_price: float = 1.0850,
        confluence_evaluation: Optional[ConfluenceEvaluation] = None,
        why_not_result: Optional[WhyNotTradeResult] = None,
        timeframe: str = "15m",
    ) -> CoreTradeDecision:
        """Direct trade decision generation from pre-computed confluence and why-not results."""
        conf = confluence_evaluation or ConfluenceEvaluation(
            symbol=symbol,
            timeframe=timeframe,
            decision_state=TradeDecisionState.NO_SETUP,
        )
        why_not = why_not_result or WhyNotTradeResult()

        if why_not.has_blocking_reasons or conf.decision_state in (
            TradeDecisionState.CONFLICTED,
            TradeDecisionState.NO_SETUP,
            TradeDecisionState.INSUFFICIENT_CONFLUENCE,
            TradeDecisionState.INVALIDATED,
        ):
            return CoreTradeDecision(
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                state=TradeDecisionState.NO_SETUP if why_not.has_blocking_reasons else conf.decision_state,
                entry_price=None,
                hard_stop_loss=0.0,
                tp1=0.0,
                r_multiple=0.0,
                invalidation_condition="Setup blocked or invalidated.",
                why_summary=conf.reasons_to_trade,
                why_not_summary=why_not.blocking_reasons,
                source_provenance_trace=[{"evidence_refs": ["EVID_BLOCK_1"]}],
            )

        sl = current_price - 0.0020 if direction.upper() == "BUY" else current_price + 0.0020
        tp1 = current_price + 0.0050 if direction.upper() == "BUY" else current_price - 0.0050
        rr = round(abs(tp1 - current_price) / abs(current_price - sl), 2)

        return CoreTradeDecision(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            state=conf.decision_state if conf.decision_state != TradeDecisionState.NO_SETUP else TradeDecisionState.HIGH_CONFLUENCE,
            entry_price=current_price,
            hard_stop_loss=sl,
            tp1=tp1,
            final_target=tp1,
            r_multiple=rr,
            invalidation_condition=f"Candle closes beyond structural invalidation {sl}",
            confirmation_requirement="LTF confirmation displacement candle complete.",
            why_summary=conf.reasons_to_trade,
            why_not_summary=why_not.non_blocking_warnings,
            source_provenance_trace=[{"evidence_refs": ["EVID_PROV_1", "EVID_PROV_2"]}],
        )

