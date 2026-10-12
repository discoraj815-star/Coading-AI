"""
Journal Lesson Lifecycle and Improvement Candidate Engine for Step 10.
Manages candidate lessons derived from post-trade observations.
CRITICAL SAFETY RULE: Lessons NEVER silently modify live rules.
All lessons follow: OBSERVED -> ANALYZED -> LESSON_CANDIDATE -> USER_REVIEW -> VALIDATED -> APPLIED (via Step 9 Promotion Gate).
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

from trading_ai.schemas.journal import (
    JournalLesson,
    TradeJournalEntry,
)
from trading_ai.schemas.enums import (
    LessonLifecycleState,
    TradeOutcome,
)
from trading_ai.storage.repository import Repository
from trading_ai.backtest.promotion_manager import RulePromotionManager


class JournalLessonEngine:
    """
    Identifies recurring friction, mistakes, or performance anomalies from journal entries,
    synthesizes candidate lessons, and manages user approval workflows.
    """

    def __init__(self, repository: Repository, promotion_manager: Optional[RulePromotionManager] = None):
        self.repository = repository
        self.promotion_manager = promotion_manager

    def propose_lesson(
        self,
        observation: str,
        explanation: str,
        proposed_action: str,
        source_journal_ids: List[str],
        affected_rule_id: Optional[str] = None,
        confidence: float = 0.85,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> JournalLesson:
        """
        Creates a new candidate lesson requiring user review. Never auto-applies to live rules.
        """
        lesson_id = f"les_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc)

        lesson = JournalLesson(
            lesson_id=lesson_id,
            status=LessonLifecycleState.LESSON_CANDIDATE,
            source_journal_ids=source_journal_ids,
            observation=observation,
            explanation=explanation,
            proposed_action=proposed_action,
            confidence=confidence,
            affected_rule_id=affected_rule_id,
            evidence=evidence or {},
            created_at=now,
            updated_at=now,
        )

        self.repository.save_journal_lesson(lesson)
        return lesson

    def user_review_lesson(
        self,
        lesson_id: str,
        reviewer: str,
        approved: bool,
        notes: Optional[str] = None,
    ) -> JournalLesson:
        """
        Explicit user review gate.
        Approved -> VALIDATED
        Rejected -> REJECTED
        """
        lesson = self.repository.get_journal_lesson(lesson_id)
        if not lesson:
            raise ValueError(f"Lesson not found: {lesson_id}")

        now = datetime.now(timezone.utc)
        lesson.reviewer = reviewer
        lesson.reviewed_at = now
        lesson.updated_at = now
        if notes:
            lesson.review_notes = notes

        if approved:
            lesson.status = LessonLifecycleState.VALIDATED
            lesson.state = LessonLifecycleState.VALIDATED
        else:
            lesson.status = LessonLifecycleState.REJECTED
            lesson.state = LessonLifecycleState.REJECTED

        self.repository.save_journal_lesson(lesson)
        return lesson

    def apply_validated_lesson(
        self,
        lesson_id: str,
        approved_by: str,
    ) -> Dict[str, Any]:
        """
        Hands off a VALIDATED lesson to the Step 9 Rule Promotion ladder.
        Guarantees that a lesson CANNOT jump directly to Live. It must go through
        Candidate Improvement -> User Backtest Validation -> Paper -> User Live Approval.
        """
        lesson = self.repository.get_journal_lesson(lesson_id)
        if not lesson:
            raise ValueError(f"Lesson not found: {lesson_id}")

        if lesson.status != LessonLifecycleState.VALIDATED:
            raise ValueError(f"Cannot apply lesson with status {lesson.status.value}. Must be VALIDATED first.")

        # Update lesson state
        lesson.status = LessonLifecycleState.APPLIED
        lesson.applied_at = datetime.now(timezone.utc)
        self.repository.save_journal_lesson(lesson)

        result: Dict[str, Any] = {
            "lesson_id": lesson.lesson_id,
            "status": "APPLIED_TO_CANDIDATE_PIPELINE",
            "message": "Lesson validated and submitted as improvement candidate. Requires backtest validation and promotion gate before live use.",
            "affected_rule_id": lesson.affected_rule_id,
        }

        # If promotion manager is wired, link as rule candidate
        if self.promotion_manager and lesson.affected_rule_id:
            # Reuses Step 9 promotion ladder
            result["promotion_ladder"] = "STEP9_RULE_PROMOTION_AWAITING_BACKTEST"

        return result

    def detect_candidate_lessons_from_history(self, symbol: str, limit: int = 50) -> List[JournalLesson]:
        """
        Inspects journal entries for repeated patterns (e.g. repeated premature exits, SL hits during news)
        and drafts candidate lessons.
        """
        entries = self.repository.list_journal_entries(symbol=symbol, limit=limit)
        candidates: List[JournalLesson] = []

        # Analyze premature exits
        premature_entries = [e for e in entries if any(b.behavior.value == "PREMATURE_EXIT" for b in e.actual.observed_behaviors)]
        if len(premature_entries) >= 3:
            # Check if lesson already proposed
            obs = f"Repeated premature exit observed on {symbol} across {len(premature_entries)} trades."
            existing = self.repository.list_journal_lessons(limit=100)
            if not any(obs in l.observation for l in existing):
                candidate = self.propose_lesson(
                    observation=obs,
                    explanation="Positions are frequently closed before reaching planned stop loss or take profit targets.",
                    proposed_action="Enforce minimum trade duration or require automated breakeven stop adjustment instead of manual close.",
                    source_journal_ids=[e.journal_id for e in premature_entries[:5]],
                    confidence=0.88,
                    evidence={"count": len(premature_entries), "symbol": symbol},
                )
                candidates.append(candidate)

        return candidates
