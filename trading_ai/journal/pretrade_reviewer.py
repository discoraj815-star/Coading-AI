"""
Pre-Trade Journal Reviewer for Step 10.
Evaluates proposed trades against historical journal memory, past mistakes,
applicable lessons, and discipline history using bounded, smallest-sufficient context.
"""
from typing import List, Optional
from datetime import datetime, timezone

from trading_ai.schemas.journal import (
    PreTradeReviewRequest,
    PreTradeReviewResult,
    TradeJournalEntry,
    JournalLesson,
    PlannedTradeValues,
)
from trading_ai.schemas.enums import (
    PreTradeReviewStatus,
    LessonLifecycleState,
    TradeOutcome,
)
from trading_ai.storage.repository import Repository
from trading_ai.companion.discipline_engine import TradingDisciplineEngine


class PreTradeReviewer:
    """
    Decision-support memory engine that reviews candidate trade plans before Step 8 execution.
    Does NOT duplicate Step 8 RiskEngine or ConfluenceEngine.
    """

    def __init__(self, repository: Repository, discipline_engine: Optional[TradingDisciplineEngine] = None):
        self.repository = repository
        self.discipline_engine = discipline_engine

    def review_trade(self, request: PreTradeReviewRequest) -> PreTradeReviewResult:
        warnings: List[str] = []
        rule_violations: List[str] = []
        discipline_issues: List[str] = []

        # 1. Validation of trade plan completeness (Phase 5)
        # Entry, SL, TP1, Direction, Risk, RR, Thesis
        plan = request.planned_values
        if plan.entry_price <= 0:
            warnings.append("Invalid or missing planned entry price.")
        if plan.stop_loss <= 0:
            warnings.append("Missing stop loss price.")
        if plan.planned_tp1 <= 0:
            warnings.append("Missing planned TP1 price.")
        if plan.planned_rr < 1.0:
            warnings.append(f"Planned RR {plan.planned_rr:.2f} is below 1.0 threshold.")
        if not plan.trade_thesis or len(plan.trade_thesis.strip()) < 5:
            warnings.append("Trade thesis is missing or insufficient.")

        # Directional SL logic
        if request.direction.upper() == "BUY" and plan.stop_loss >= plan.entry_price:
            rule_violations.append(f"Long trade stop loss ({plan.stop_loss}) must be below entry price ({plan.entry_price}).")
        elif request.direction.upper() == "SELL" and plan.stop_loss <= plan.entry_price:
            rule_violations.append(f"Short trade stop loss ({plan.stop_loss}) must be above entry price ({plan.entry_price}).")

        # 2. Retrieve bounded historical trades (smallest sufficient context)
        # Filter by symbol and environment
        history = self.repository.list_journal_entries(
            environment=request.environment,
            symbol=request.symbol,
            limit=10,
        )

        similar_trades: List[TradeJournalEntry] = []
        past_losses_count = 0
        past_wins_count = 0
        consecutive_losses = 0

        for t in history:
            # Match setup or rule if available
            is_similar = False
            if request.setup and t.setup and request.setup.lower() in t.setup.lower():
                is_similar = True
            elif request.rule_version_id and t.rule_version_id == request.rule_version_id:
                is_similar = True
            elif t.symbol == request.symbol:
                is_similar = True

            if is_similar:
                similar_trades.append(t)
                if t.actual.outcome == TradeOutcome.LOSS:
                    past_losses_count += 1
                elif t.actual.outcome == TradeOutcome.WIN:
                    past_wins_count += 1

                # Collect past observed behaviors / mistakes from similar trades
                for b in t.actual.observed_behaviors:
                    warnings.append(f"Historical issue on {t.symbol}: observed {b.behavior.value} ({b.details}).")

        # Check recent losing streak on this symbol
        for t in history[:3]:
            if t.actual.outcome == TradeOutcome.LOSS:
                consecutive_losses += 1
            else:
                break

        if consecutive_losses >= 3:
            warnings.append(f"Caution: Last {consecutive_losses} trades on {request.symbol} were losses.")

        # 3. Retrieve active lessons (User-validated lessons applicable to rule or symbol)
        all_lessons = self.repository.list_journal_lessons(
            state=LessonLifecycleState.VALIDATED,
            limit=20,
        )
        applicable_lessons: List[JournalLesson] = []
        for l in all_lessons:
            if l.affected_rule_id and request.rule_version_id and l.affected_rule_id == request.rule_version_id:
                applicable_lessons.append(l)
                warnings.append(f"Applicable Lesson ({l.lesson_id}): {l.observation} -> {l.proposed_action}")
            elif request.setup and request.setup.lower() in l.observation.lower():
                applicable_lessons.append(l)
                warnings.append(f"Applicable Setup Lesson ({l.lesson_id}): {l.observation}")

        # 4. Integrate Step 7 Discipline Engine checks if wired
        if self.discipline_engine:
            # Query recent discipline events or warnings
            discipline_status = getattr(self.discipline_engine, "last_status", None)
            if discipline_status and getattr(discipline_status, "is_locked", False):
                discipline_issues.append("Trading discipline lock is active: cooldown or max daily loss reached.")

        # 5. Determine overall decision status
        if rule_violations or discipline_issues:
            status = PreTradeReviewStatus.REJECTED
        elif warnings:
            status = PreTradeReviewStatus.WARNING
        else:
            status = PreTradeReviewStatus.APPROVED

        result = PreTradeReviewResult(
            proposal_id=request.proposal_id,
            status=status,
            planned_values=request.planned_values,
            similar_trades_count=len(similar_trades),
            similar_trades_sample=[
                {
                    "journal_id": t.journal_id,
                    "symbol": t.symbol,
                    "outcome": t.actual.outcome.value,
                    "realized_r": t.actual.realized_r,
                    "exit_reason": t.actual.exit_reason.value,
                }
                for t in similar_trades[:5]
            ],
            applicable_lessons=applicable_lessons[:5],
            warnings=warnings,
            rule_violations=rule_violations,
            discipline_issues=discipline_issues,
            recommendation=self._generate_recommendation(status, warnings, rule_violations, discipline_issues),
            reviewed_at=datetime.now(timezone.utc),
        )

        # Persist review result
        self.repository.save_pretrade_review(result)
        return result

    def _generate_recommendation(
        self,
        status: PreTradeReviewStatus,
        warnings: List[str],
        rule_violations: List[str],
        discipline_issues: List[str],
    ) -> str:
        if status == PreTradeReviewStatus.REJECTED:
            reasons = rule_violations + discipline_issues
            return f"Execution Blocked by Pre-Trade Journal Review: {'; '.join(reasons)}"
        elif status == PreTradeReviewStatus.WARNING:
            return f"Caution advised. Review warnings before sizing or entry: {'; '.join(warnings[:2])}"
        else:
            return "Trade plan fully compliant with journal rules and historical context. Approved to proceed to Step 8 RiskEngine."
