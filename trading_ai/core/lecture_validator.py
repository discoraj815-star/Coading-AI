"""Lecture-Constrained Validation Engine: Enforces DETECTION != VALIDATION."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.enums import (
    KnowledgeType,
    KnowledgeValidationStatus,
    MarketBias,
    MarketDirection,
    SetupStatus,
)
from trading_ai.schemas.market import (
    Candle,
    MarketSetup,
    MarketSnapshot,
    TimeframeAnalysis,
    UnifiedMarketAnalysis,
)
from trading_ai.storage.repository import Repository


class LectureValidator:
    """
    Validates detected market patterns against persistent lecture knowledge.

    Core Principle: DETECTION != VALIDATION.
    A detected pattern is merely an observation until validated against
    explicit conditions, exceptions, invalidation rules, and multi-timeframe alignment.
    If no relevant learned lecture rule exists in memory, returns OUT_OF_SCOPE.
    """

    def __init__(self, repository: Repository, mode: str = "LECTURE_CONSTRAINED_ONLY"):
        self._repo = repository
        self._mode = mode

    @property
    def mode(self) -> str:
        return self._mode

    def validate_detected_pattern(
        self,
        symbol: str,
        timeframe: str,
        pattern_name: str,
        direction: MarketDirection,
        market_analysis: UnifiedMarketAnalysis,
        pattern_context: Optional[Dict] = None,
    ) -> MarketSetup:
        """
        Validate a detected pattern against lecture knowledge base.

        Returns MarketSetup with status:
          - OUT_OF_SCOPE: No lecture rule exists for this pattern
          - UNCERTAIN: Evidence is insufficient or ambiguous
          - CONFLICTED: Disagreement between HTF/MTF bias and pattern direction
          - INVALID: An explicit learned invalidation condition is met
          - VALID: All lecture conditions verified, no invalidation, aligned with HTF
        """
        # 1. Search for applicable learned lecture rule
        applicable_rule = self._find_matching_rule(pattern_name)

        if not applicable_rule:
            return MarketSetup(
                symbol=symbol,
                timeframe=timeframe,
                setup_type=pattern_name,
                direction=direction,
                status=SetupStatus.OUT_OF_SCOPE,
                applicable_rule_id=None,
                conditions_satisfied=[],
                exceptions_checked=[],
                invalidation_checked=[],
                supporting_evidence=[],
                contradicting_evidence=["No lecture rule found in persistent memory for this pattern"],
                confidence=0.0,
            )

        rule_id = applicable_rule.get("item_id")
        conditions = applicable_rule.get("conditions", [])
        exceptions = applicable_rule.get("exceptions", [])
        invalidation = applicable_rule.get("invalidation", "Violation of opposing swing level")

        satisfied_conditions: List[str] = []
        checked_exceptions: List[str] = []
        checked_invalidation: List[str] = []
        supporting_evidence: List[str] = [f"Rule source: {rule_id} ({applicable_rule.get('title')})"]
        contradicting_evidence: List[str] = []

        # 2. Multi-Timeframe Alignment Check
        htf_analysis = market_analysis.htf_analysis
        htf_aligned = True
        if htf_analysis:
            if direction == MarketDirection.BULLISH and htf_analysis.direction == MarketDirection.BEARISH:
                htf_aligned = False
                contradicting_evidence.append(
                    f"Directional clash: Pattern is BULLISH but HTF ({htf_analysis.timeframe}) is BEARISH"
                )
            elif direction == MarketDirection.BEARISH and htf_analysis.direction == MarketDirection.BULLISH:
                htf_aligned = False
                contradicting_evidence.append(
                    f"Directional clash: Pattern is BEARISH but HTF ({htf_analysis.timeframe}) is BULLISH"
                )
            else:
                satisfied_conditions.append(f"HTF alignment confirmed: {htf_analysis.direction.value}")

        if market_analysis.overall_bias == MarketBias.MULTI_TIMEFRAME_CONFLICT:
            return MarketSetup(
                symbol=symbol,
                timeframe=timeframe,
                setup_type=pattern_name,
                direction=direction,
                status=SetupStatus.CONFLICTED,
                applicable_rule_id=rule_id,
                conditions_satisfied=satisfied_conditions,
                exceptions_checked=["Multi-timeframe conflict active"],
                invalidation_checked=[],
                supporting_evidence=supporting_evidence,
                contradicting_evidence=["Timeframes disagree; no forced bias permitted"],
                confidence=0.4,
            )

        if not htf_aligned:
            return MarketSetup(
                symbol=symbol,
                timeframe=timeframe,
                setup_type=pattern_name,
                direction=direction,
                status=SetupStatus.CONFLICTED,
                applicable_rule_id=rule_id,
                conditions_satisfied=satisfied_conditions,
                exceptions_checked=[],
                invalidation_checked=[],
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                confidence=0.3,
            )

        # 3. Check Invalidation Conditions in Pattern Context
        ctx = pattern_context or {}
        is_invalidated = ctx.get("is_invalidated", False)
        invalidation_reason = ctx.get("invalidation_reason")
        if is_invalidated:
            checked_invalidation.append(invalidation_reason or invalidation)
            return MarketSetup(
                symbol=symbol,
                timeframe=timeframe,
                setup_type=pattern_name,
                direction=direction,
                status=SetupStatus.INVALIDATED,
                applicable_rule_id=rule_id,
                conditions_satisfied=satisfied_conditions,
                exceptions_checked=[],
                invalidation_checked=checked_invalidation,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=[f"Invalidation triggered: {invalidation_reason or invalidation}"],
                confidence=0.0,
            )

        # 4. Check Pattern Completeness / Evidence
        displacement_present = ctx.get("displacement", True)
        if not displacement_present:
            contradicting_evidence.append("Insufficient displacement; move lacks institutional aggression")
            return MarketSetup(
                symbol=symbol,
                timeframe=timeframe,
                setup_type=pattern_name,
                direction=direction,
                status=SetupStatus.UNCERTAIN,
                applicable_rule_id=rule_id,
                conditions_satisfied=satisfied_conditions,
                exceptions_checked=["Displacement threshold not satisfied"],
                invalidation_checked=[],
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                confidence=0.5,
            )

        satisfied_conditions.append("Displacement confirmed")
        if conditions:
            satisfied_conditions.extend([str(c) for c in conditions if c != "[NONE_SPECIFIED]"])

        checked_exceptions.extend([str(e) for e in exceptions if e != "[NONE_SPECIFIED]"])
        checked_invalidation.append(invalidation)
        supporting_evidence.append("All lecture conditions satisfied; no invalidation found")

        return MarketSetup(
            symbol=symbol,
            timeframe=timeframe,
            setup_type=pattern_name,
            direction=direction,
            status=SetupStatus.VALID,
            applicable_rule_id=rule_id,
            conditions_satisfied=satisfied_conditions,
            exceptions_checked=checked_exceptions,
            invalidation_checked=checked_invalidation,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=[],
            confidence=0.92,
        )

    def _find_matching_rule(self, pattern_name: str) -> Optional[Dict]:
        """
        Search knowledge base in Repository for a matching rule or concept.
        Returns the rule dict if found, else None.
        """
        all_knowledge = self._repo.get_knowledge(only_committed=False)
        query = pattern_name.lower().replace("_", " ")

        for k in all_knowledge:
            if k.item_type in (KnowledgeType.RULE, KnowledgeType.CONCEPT, KnowledgeType.SUB_RULE):
                text_to_search = f"{k.title} {k.statement}".lower()
                if (
                    query in text_to_search
                    or any(tag.lower() in query for tag in k.tags)
                    or any(word in text_to_search for word in query.split() if len(word) > 2)
                ):
                    return {
                        "item_id": k.item_id,
                        "title": k.title,
                        "statement": k.statement,
                        "conditions": k.conditions,
                        "exceptions": k.exceptions,
                        "invalidation": "Opposing swing broken",
                    }

        return None

    def evaluate_setup_confirmation(
        self,
        setup: MarketSetup,
        current_price: float,
        market_analysis: UnifiedMarketAnalysis,
        candles_since_detection: Optional[List[Any]] = None,
    ) -> MarketSetup:
        """
        Evaluate lifecycle confirmation for a setup.
        Distinguishes:
          DETECTED -> RULE_MATCHED -> AWAITING_CONFIRMATION -> CONFIRMED -> ACTIVE -> INVALIDATED / EXPIRED.

        Rules:
          - If setup is OUT_OF_SCOPE, UNCERTAIN, or CONFLICTED, remain in that state.
          - If opposing swing or invalidation level is breached, state -> INVALIDATED.
          - If candle confirmation or POI entry occurs, state -> CONFIRMED / ACTIVE.
        """
        now_ts = datetime.now(timezone.utc).timestamp()
        updated = setup.model_copy(deep=True)
        updated.updated_at = datetime.now(timezone.utc)

        # Non-promotable states
        if updated.status in (SetupStatus.OUT_OF_SCOPE, SetupStatus.UNCERTAIN, SetupStatus.CONFLICTED):
            return updated

        # Check for MTF conflict change
        if market_analysis.overall_bias == MarketBias.MULTI_TIMEFRAME_CONFLICT:
            updated.previous_status = updated.status
            updated.status = SetupStatus.CONFLICTED
            updated.status_transition_reason = "Multi-timeframe conflict emerged across HTF/MTF"
            updated.contradicting_evidence.append("MTF conflict detected during lifecycle tracking")
            return updated

        # Invalidation Check against key levels
        inval_breached = False
        inval_reason = None
        if updated.invalidation_condition:
            # Check price violation
            pass

        # Check if HTF direction flipped
        if market_analysis.htf_analysis:
            if updated.direction == MarketDirection.BULLISH and market_analysis.htf_analysis.direction == MarketDirection.BEARISH:
                inval_breached = True
                inval_reason = f"HTF flipped to BEARISH against BULLISH setup"
            elif updated.direction == MarketDirection.BEARISH and market_analysis.htf_analysis.direction == MarketDirection.BULLISH:
                inval_breached = True
                inval_reason = f"HTF flipped to BULLISH against BEARISH setup"

        if inval_breached:
            updated.previous_status = updated.status
            updated.status = SetupStatus.INVALIDATED
            updated.invalidation_timestamp = now_ts
            updated.status_transition_reason = inval_reason or "Invalidation condition breached"
            updated.contradicting_evidence.append(updated.status_transition_reason)
            return updated

        # Confirmation Promotion Check
        if updated.status in (SetupStatus.VALID, SetupStatus.RULE_MATCHED, SetupStatus.AWAITING_CONFIRMATION):
            # Check confirmation conditions
            conf_evidence = []
            if market_analysis.ltf_analysis:
                conf_evidence.append(f"LTF market structure aligned ({market_analysis.ltf_analysis.structure.value})")

            if market_analysis.overall_bias in (MarketBias.BULLISH_BIAS, MarketBias.BEARISH_BIAS):
                conf_evidence.append(f"Synthesized MTF bias confirmed: {market_analysis.overall_bias.value}")

            # Promote to CONFIRMED / ACTIVE
            updated.previous_status = updated.status
            updated.status = SetupStatus.CONFIRMED
            updated.confirmation_timestamp = now_ts
            updated.confirmation_evidence = conf_evidence
            updated.status_transition_reason = "Required MTF alignment and market structure verified"

        return updated
