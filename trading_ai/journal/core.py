"""
Unified Step 10 Trading Journal Core Facade.
Coordinates PreTradeReviewer, PostTradeEvaluator, JournalAnalyticsEngine,
JournalLessonEngine, and JournalResilienceManager into a cohesive API.
Ensures Step 1–9 wiring, environmental isolation, and non-bypassable risk gates.
"""
from typing import List, Optional, Dict, Any

from trading_ai.storage.repository import Repository
from trading_ai.journal.pretrade_reviewer import PreTradeReviewer
from trading_ai.journal.posttrade_engine import PostTradeEvaluator
from trading_ai.journal.analytics import JournalAnalyticsEngine
from trading_ai.journal.lesson_engine import JournalLessonEngine
from trading_ai.journal.resilience import JournalResilienceManager
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.backtest.promotion_manager import RulePromotionManager


class TradingJournalCore:
    """
    Central Step 10 Journal Engine facade.
    Reuses existing Step 7 Discipline and Step 9 Promotion architectures without duplicating them.
    """

    def __init__(
        self,
        repository: Repository,
        discipline_engine: Optional[TradingDisciplineEngine] = None,
        promotion_manager: Optional[RulePromotionManager] = None,
    ):
        self.repository = repository
        self.discipline_engine = discipline_engine
        self.promotion_manager = promotion_manager

        # Sub-engines
        self.pretrade_reviewer = PreTradeReviewer(repository, discipline_engine)
        self.posttrade_evaluator = PostTradeEvaluator(repository)
        self.analytics_engine = JournalAnalyticsEngine(repository)
        self.lesson_engine = JournalLessonEngine(repository, promotion_manager)
        self.resilience_manager = JournalResilienceManager(repository)

    def get_status(self) -> Dict[str, Any]:
        """
        Returns high-level status of the Journal subsystem.
        """
        all_entries = self.repository.list_journal_entries(limit=1000)
        lessons = self.repository.list_journal_lessons(limit=100)

        env_counts = {"BACKTEST": 0, "PAPER": 0, "LIVE": 0}
        for e in all_entries:
            env_val = e.environment.value
            env_counts[env_val] = env_counts.get(env_val, 0) + 1

        return {
            "status": "ONLINE",
            "total_journal_entries": len(all_entries),
            "environments": env_counts,
            "total_lessons": len(lessons),
            "active_validated_lessons": len([l for l in lessons if l.status.value == "VALIDATED"]),
            "token_resilience": "ACTIVE",
            "safety_rule": "Lessons require user review; no silent live rule changes.",
        }
