"""Trading Journal Feedback Engine for Step 14.

Closes the learning loop:
Learned Rule -> Trade Decision -> Actual Trade -> Journal -> Outcome -> Lesson -> Candidate Improvement.

Absolute Invariant:
Journal outcomes NEVER silently mutate live trading rules.
Any adjustments are produced as insights / candidate recommendations requiring explicit user approval.
"""
from typing import List, Optional

from trading_ai.schemas.journal import JournalLesson, TradeJournalEntry
from trading_ai.schemas.trade_logic import JournalFeedbackInsight
from trading_ai.storage.repository import Repository


class JournalFeedbackEngine:
    """Extracts actionable feedback from historical trade journal logs to inform trade proposals."""

    def __init__(self, repository: Optional[Repository] = None, repo: Optional[Repository] = None):
        self.repository = repository or repo

    def get_feedback_for_setup(
        self,
        symbol: str,
        setup_type: str = "liquidity_sweep",
        direction: Optional[str] = None,
    ) -> JournalFeedbackInsight:
        """Analyze past trades and lessons for a given symbol and setup type."""
        lessons: List[JournalLesson] = []
        entries: List[TradeJournalEntry] = []

        if self.repository and hasattr(self.repository, "list_journal_lessons"):
            lessons = self.repository.list_journal_lessons(limit=50)
        if self.repository and hasattr(self.repository, "list_journal_entries"):
            entries = self.repository.list_journal_entries(symbol=symbol, limit=50)

        # Filter relevant trades
        matching_entries = [
            e for e in entries
            if setup_type.lower() in (e.setup_type or "").lower()
            or (direction and e.planned_values.direction.upper() == direction.upper())
        ]

        total_trades = len(matching_entries)
        wins = sum(1 for e in matching_entries if e.actual_values and e.actual_values.outcome.value == "WIN")
        win_rate = round(wins / total_trades, 2) if total_trades > 0 else 0.50

        # Calculate average RR from historical trades
        r_multiples = [
            e.actual_values.r_multiple
            for e in matching_entries
            if e.actual_values and e.actual_values.r_multiple is not None
        ]
        avg_rr = round(sum(r_multiples) / len(r_multiples), 2) if r_multiples else 2.0

        # Match relevant lessons
        applicable_lessons = []
        warning_flags = []
        recommended_adjustments = []

        for lesson in lessons:
            content_lower = (lesson.title + " " + lesson.statement).lower()
            if setup_type.lower() in content_lower or symbol.lower() in content_lower:
                applicable_lessons.append(f"[{lesson.mistake_category.value}] {lesson.statement}")
                if "loss" in content_lower or "mistake" in content_lower or "failed" in content_lower:
                    warning_flags.append(f"Caution: Previous failure recorded — {lesson.statement[:100]}")
                    recommended_adjustments.append(f"Require higher-timeframe displacement confirmation before entry.")

        return JournalFeedbackInsight(
            symbol=symbol,
            setup_type=setup_type,
            total_historical_trades=total_trades,
            win_rate=win_rate,
            average_rr=avg_rr,
            applicable_lessons=applicable_lessons[:5],
            warning_flags=warning_flags[:3],
            recommended_condition_adjustments=recommended_adjustments[:3],
        )
