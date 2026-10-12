"""
Step 6 Technical + Fundamental Correlation Engine.
Correlates deep technical analysis, active lecture-constrained setups, economic calendar timing,
and event risk without conflating correlation with causation.
Explicitly detects and categorizes technical vs fundamental conflicts.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.enums import (
    EventImpact,
    MarketBias,
    MarketDirection,
    NewsRiskLevel,
    NewsTimingCategory,
    ProviderStatus,
    SetupStatus,
    TechFundRelationship,
)
from trading_ai.schemas.fundamental import (
    FundamentalEvent,
    FundamentalProviderState,
    NewsTimingContext,
    TechFundConflict,
    TechnicalFundamentalAnalysis,
)
from trading_ai.schemas.market import MarketSetup, UnifiedMarketAnalysis

logger = logging.getLogger(__name__)


class TechnicalFundamentalCorrelator:
    """
    Synthesizes technical market context with fundamental news timing and event outcomes.
    Evaluates:
      TECHNICAL CONTEXT + FUNDAMENTAL CONTEXT + TIMING + LECTURE LOGIC = CURRENT MARKET SITUATION
    """

    def correlate(
        self,
        symbol: str,
        unified_analysis: UnifiedMarketAnalysis,
        active_setup: Optional[MarketSetup],
        timing_context: NewsTimingContext,
        provider_state: Optional[FundamentalProviderState] = None,
    ) -> TechnicalFundamentalAnalysis:
        """
        Execute reasoned deterministic correlation between technical structure and fundamental events.
        """
        evidence: List[str] = []
        now = datetime.now(timezone.utc)
        technical_bias_str = unified_analysis.overall_bias.value

        evidence.append(f"Technical Bias: {technical_bias_str}")
        if active_setup:
            evidence.append(
                f"Active Setup: {active_setup.setup_type} ({active_setup.direction.value}) "
                f"with status {active_setup.status.value}"
            )

        # 1. Check Data Freshness / Provider Health
        if provider_state:
            if provider_state.status in (ProviderStatus.UNAVAILABLE, ProviderStatus.BLOCKED, ProviderStatus.ERROR):
                if not timing_context.active_events and not timing_context.upcoming_events:
                    return TechnicalFundamentalAnalysis(
                        symbol=symbol,
                        relationship=TechFundRelationship.FUNDAMENTAL_UNAVAILABLE,
                        technical_bias=technical_bias_str,
                        news_risk=NewsRiskLevel.NO_RISK,
                        timing_context=timing_context,
                        conflict=TechFundConflict(has_conflict=False),
                        action_context="PROCEED_WITH_TECHNICAL_ANALYSIS_FUNDAMENTAL_UNAVAILABLE",
                        explanation=f"Fundamental data is unavailable ({provider_state.error_message or 'offline'}). Technical analysis remains valid.",
                        evidence=evidence + [f"Provider status: {provider_state.status.value}"],
                    )

            if provider_state.is_stale:
                evidence.append(f"Warning: Fundamental data is stale (> {provider_state.stale_after_seconds}s).")

        # 2. Assess Fundamental Directional Bias (from released events with actual vs forecast)
        fund_bias, fund_reason, fund_evidence = self._assess_fundamental_bias(symbol, timing_context)
        evidence.extend(fund_evidence)

        # 3. Evaluate Conflict Conditions
        conflict = self._detect_conflicts(
            symbol=symbol,
            technical_bias=unified_analysis.overall_bias,
            active_setup=active_setup,
            fund_bias=fund_bias,
            fund_reason=fund_reason,
            timing_context=timing_context,
        )

        # 4. Determine Overall Relationship & Action Context
        relationship, action_context, explanation = self._determine_relationship_and_action(
            unified_analysis=unified_analysis,
            active_setup=active_setup,
            timing_context=timing_context,
            conflict=conflict,
            fund_bias=fund_bias,
        )

        return TechnicalFundamentalAnalysis(
            symbol=symbol,
            relationship=relationship,
            technical_bias=technical_bias_str,
            news_risk=timing_context.risk_level,
            timing_context=timing_context,
            conflict=conflict,
            action_context=action_context,
            explanation=explanation,
            evidence=evidence + conflict.evidence,
            timestamp=now.timestamp(),
            created_at=now,
        )

    def _assess_fundamental_bias(
        self, symbol: str, timing_context: NewsTimingContext
    ) -> Tuple[str, str, List[str]]:
        """
        Derive directional fundamental tone from recently released economic data.
        Returns: (bias_string, reason_summary, evidence_list)
        """
        evidence: List[str] = []
        recent_event = timing_context.recent_released_event
        if not recent_event or not recent_event.actual:
            return "NEUTRAL", "No recently released economic data with numeric outcome.", evidence

        # Extract currencies
        parts = symbol.replace("/", "").upper()
        base_curr = parts[:3] if len(parts) >= 6 else "EUR"
        quote_curr = parts[3:6] if len(parts) >= 6 else "USD"

        # Check surprise if forecast and actual are parseable numbers
        actual_num = self._clean_number(recent_event.actual)
        forecast_num = self._clean_number(recent_event.forecast) if recent_event.forecast else None

        if actual_num is not None and forecast_num is not None:
            surprise = actual_num - forecast_num
            evidence.append(
                f"Recent event: {recent_event.title} ({recent_event.currency}) "
                f"Actual: {actual_num}, Forecast: {forecast_num} (Surprise: {surprise:+.2f})"
            )

            if abs(surprise) > 0.0001:
                is_positive = surprise > 0
                event_curr = recent_event.currency.upper()

                if event_curr == quote_curr:
                    # Positive quote currency outcome is BEARISH for base/quote pair (e.g. Strong USD -> Bearish EURUSD)
                    pair_bias = "BEARISH" if is_positive else "BULLISH"
                elif event_curr == base_curr:
                    pair_bias = "BULLISH" if is_positive else "BEARISH"
                else:
                    pair_bias = "NEUTRAL"

                reason = f"{recent_event.title} surprise for {event_curr} implies {pair_bias} fundamental pressure on {symbol}."
                return pair_bias, reason, evidence

        evidence.append(f"Recent event {recent_event.title} released; qualitative impact noted.")
        return "NEUTRAL", "Outcome released without directional numeric surprise.", evidence

    def _clean_number(self, val_str: str) -> Optional[float]:
        """Strip non-numeric characters (%, K, M, B) and parse float."""
        if not val_str:
            return None
        cleaned = val_str.replace("%", "").replace("K", "").replace("M", "").replace("B", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _detect_conflicts(
        self,
        symbol: str,
        technical_bias: MarketBias,
        active_setup: Optional[MarketSetup],
        fund_bias: str,
        fund_reason: str,
        timing_context: NewsTimingContext,
    ) -> TechFundConflict:
        """Explicitly detect directional, event-risk, or temporal conflicts."""
        evidence: List[str] = []

        # 1. Direct Directional Conflict
        is_tech_bullish = technical_bias == MarketBias.BULLISH_BIAS
        is_tech_bearish = technical_bias == MarketBias.BEARISH_BIAS

        if (is_tech_bullish and fund_bias == "BEARISH") or (is_tech_bearish and fund_bias == "BULLISH"):
            evidence.append(
                f"Directional clash: Technical is {technical_bias.value} but Fundamental outcome is {fund_bias}."
            )
            return TechFundConflict(
                has_conflict=True,
                conflict_type="FUNDAMENTAL_TECHNICAL_CONFLICT",
                technical_bias=technical_bias.value,
                fundamental_bias=fund_bias,
                confidence=0.85,
                reason=f"Technical structure is {technical_bias.value} while {fund_reason}",
                evidence=evidence,
            )

        # 2. Active Setup vs Imminent High-Impact Event Conflict
        if active_setup and active_setup.status in (SetupStatus.VALID, SetupStatus.CONFIRMED, SetupStatus.ACTIVE):
            if timing_context.risk_level in (NewsRiskLevel.EXTREME_RISK, NewsRiskLevel.HIGH_RISK):
                event_name = (
                    timing_context.next_high_impact_event.title
                    if timing_context.next_high_impact_event
                    else "High-Impact Economic Event"
                )
                time_dist = (
                    f"{int(timing_context.time_to_next_seconds // 60)}m"
                    if timing_context.time_to_next_seconds
                    else "imminent"
                )
                evidence.append(
                    f"Setup '{active_setup.setup_type}' active but {event_name} occurs in {time_dist}."
                )
                return TechFundConflict(
                    has_conflict=True,
                    conflict_type="HIGH_IMPACT_EVENT_IMMINENT",
                    technical_bias=technical_bias.value,
                    fundamental_bias="HIGH_RISK",
                    confidence=0.90,
                    reason=f"Valid setup exists, but high-impact event ({event_name}) is scheduled in {time_dist}. Elevated slippage and spread risk.",
                    evidence=evidence,
                )

        # 3. Multi-timeframe Technical Conflict
        if technical_bias == MarketBias.MULTI_TIMEFRAME_CONFLICT:
            evidence.append("Technical timeframes in conflict: higher and medium timeframes disagree.")
            return TechFundConflict(
                has_conflict=True,
                conflict_type="TECHNICAL_TIMEFRAME_CONFLICT",
                technical_bias=technical_bias.value,
                fundamental_bias=fund_bias,
                confidence=0.80,
                reason="Timeframe disagreement prevents confident technical execution.",
                evidence=evidence,
            )

        return TechFundConflict(has_conflict=False)

    def _determine_relationship_and_action(
        self,
        unified_analysis: UnifiedMarketAnalysis,
        active_setup: Optional[MarketSetup],
        timing_context: NewsTimingContext,
        conflict: TechFundConflict,
        fund_bias: str,
    ) -> Tuple[TechFundRelationship, str, str]:
        """Classify relationship enum, actionable trading context, and explanatory rationale."""
        # 1. Handled Conflicts First
        if conflict.has_conflict:
            if conflict.conflict_type == "HIGH_IMPACT_EVENT_IMMINENT":
                return (
                    TechFundRelationship.NEWS_IMMINENT,
                    "WAIT_FOR_CONFIRMATION_POST_EVENT",
                    f"Setup is technically formed, but upcoming event presents extreme volatility risk. {conflict.reason}",
                )
            elif conflict.conflict_type == "FUNDAMENTAL_TECHNICAL_CONFLICT":
                return (
                    TechFundRelationship.FUNDAMENTAL_CONFLICT,
                    "STAND_ASIDE_OR_REQUIRE_HIGHER_TIMEFRAME_CONFIRMATION",
                    f"Fundamental release opposes technical trend. {conflict.reason}",
                )
            elif conflict.conflict_type == "TECHNICAL_TIMEFRAME_CONFLICT":
                return (
                    TechFundRelationship.CONTEXT_CONFLICTED,
                    "NO_TRADE_MTF_CONFLICT",
                    "Multiple timeframes disagree. Lecture rules mandate waiting for alignment.",
                )

        # 2. News Active Right Now
        if timing_context.category == NewsTimingCategory.EVENT_HAPPENING_NOW:
            return (
                TechFundRelationship.NEWS_ACTIVE,
                "STAND_ASIDE_EVENT_IN_PROGRESS",
                "High or medium impact economic event is releasing right now. Extreme volatility expected.",
            )

        # 3. Post News Reaction with Supporting Technicals
        if timing_context.category == NewsTimingCategory.RECENTLY_RELEASED:
            if active_setup and active_setup.status in (SetupStatus.CONFIRMED, SetupStatus.ACTIVE):
                return (
                    TechFundRelationship.POST_NEWS_REACTION,
                    "PROCEED_WITH_VALIDATED_SETUP",
                    "News occurred near the observed price reaction; post-event technical structure confirms lecture setup.",
                )
            return (
                TechFundRelationship.POST_NEWS_REACTION,
                "OBSERVE_POST_EVENT_DISPLACEMENT",
                "Economic event recently released. Price action is establishing post-release structure.",
            )

        # 4. Aligned Fundamentals + Technicals
        if (
            (unified_analysis.overall_bias == MarketBias.BULLISH_BIAS and fund_bias == "BULLISH")
            or (unified_analysis.overall_bias == MarketBias.BEARISH_BIAS and fund_bias == "BEARISH")
        ):
            return (
                TechFundRelationship.CONTEXT_ALIGNED,
                "PROCEED_WITH_STANDARD_CONFIRMATION",
                f"Technical bias ({unified_analysis.overall_bias.value}) and fundamental surprise ({fund_bias}) are mutually aligned.",
            )

        # 5. Default Valid / Neutral Technicals
        if active_setup and active_setup.status in (SetupStatus.CONFIRMED, SetupStatus.ACTIVE, SetupStatus.VALID):
            return (
                TechFundRelationship.TECHNICALLY_VALID,
                "PROCEED_WITH_STANDARD_CONFIRMATION",
                f"Valid lecture setup ({active_setup.setup_type}) with acceptable fundamental event risk.",
            )

        return (
            TechFundRelationship.TECHNICALLY_VALID,
            "WAIT_FOR_CONFIRMATION",
            "Market structure analyzed. No immediate event risk; waiting for confirmed lecture setup.",
        )
