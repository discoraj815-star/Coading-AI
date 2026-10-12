"""Knowledge Normalization Layer for Step 14.

Transforms raw source artifacts, lecture items, mentorship rules, journal lessons,
and technical work findings into standardized, provenance-backed NormalizedKnowledgeItem records.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.schemas.enums import (
    KnowledgeType,
    SourceAuthority,
    SourceDomain,
    SourceType,
)
from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.schemas.journal import JournalLesson
from trading_ai.schemas.trade_logic import NormalizedKnowledgeItem


class KnowledgeNormalizer:
    """Normalizes heterogeneous source items into canonical NormalizedKnowledgeItem objects."""

    @staticmethod
    def from_lecture_item(
        item: KnowledgeItem,
        source_location: str = "01_YouTube_Lectures/lecture_video.mp4",
        authority: SourceAuthority = SourceAuthority.LECTURE_RULE,
    ) -> NormalizedKnowledgeItem:
        """Normalize a Step 2 KnowledgeItem from a lecture."""
        return NormalizedKnowledgeItem(
            item_id=f"norm_lec_{item.item_id}",
            domain=SourceDomain.YOUTUBE_LECTURES,
            source_type=SourceType.LOCAL_VIDEO,
            source_id=item.job_id,
            source_document_id=item.item_id,
            source_location=source_location,
            concept_type=item.item_type.value if hasattr(item.item_type, "value") else str(item.item_type),
            title=item.title,
            statement=getattr(item, "statement", getattr(item, "content", "")),
            conditions=list(item.conditions),
            exceptions=list(item.exceptions),
            confidence=item.confidence,
            authority=authority,
            version=1,
            evidence_references=[e.evidence_id for e in item.evidence if hasattr(e, "evidence_id")],
            tags=list(item.tags) or ["lecture", "price_action"],
        )

    @staticmethod
    def from_mentorship_rule(
        title: str,
        statement: str,
        conditions: List[str],
        exceptions: Optional[List[str]] = None,
        mentor_name: str = "Lead Mentor",
        session_id: str = "session_01",
        confidence: float = 1.0,
        source_location: str = "02_Mentorship_Sessions/session_notes.md",
    ) -> NormalizedKnowledgeItem:
        """Normalize a mentorship instruction or rule."""
        return NormalizedKnowledgeItem(
            item_id=f"norm_mentor_{uuid.uuid4().hex[:8]}",
            domain=SourceDomain.MENTORSHIP_SESSIONS,
            source_type=SourceType.USER_TEXT,
            source_id=f"mentor_{mentor_name.lower().replace(' ', '_')}",
            source_session_id=session_id,
            source_location=source_location,
            concept_type="MentorInstruction",
            title=title,
            statement=statement,
            conditions=conditions,
            exceptions=exceptions or [],
            confidence=confidence,
            authority=SourceAuthority.MENTORSHIP_RULE,
            version=1,
            evidence_references=[f"session:{session_id}"],
            tags=["mentorship", "discipline", "filter"],
        )

    @staticmethod
    def from_journal_lesson(
        lesson: JournalLesson,
        source_location: str = "03_Live_Trade_Logs/trade_journal.json",
    ) -> NormalizedKnowledgeItem:
        """Normalize a Step 10 JournalLesson derived from actual trade outcomes."""
        return NormalizedKnowledgeItem(
            item_id=f"norm_jour_{lesson.lesson_id}",
            domain=SourceDomain.LIVE_TRADE_LOGS,
            source_type=SourceType.USER_TEXT,
            source_id=f"trade_{lesson.derived_from_trade_id or 'unknown'}",
            source_document_id=lesson.lesson_id,
            source_location=source_location,
            concept_type="JournalLesson",
            title=lesson.title,
            statement=lesson.statement,
            conditions=[f"applies_to_setup:{lesson.target_setup_type}"] if lesson.target_setup_type else [],
            exceptions=[],
            confidence=lesson.confidence_weight,
            authority=SourceAuthority.JOURNAL_LESSON,
            version=lesson.version,
            evidence_references=[lesson.derived_from_trade_id] if lesson.derived_from_trade_id else [],
            tags=["journal", "live_trade", lesson.mistake_category.value if hasattr(lesson.mistake_category, "value") else str(lesson.mistake_category)],
        )

    @staticmethod
    def from_tech_work(
        title: str,
        finding_statement: str,
        script_or_algorithm_id: str,
        conditions: Optional[List[str]] = None,
        exceptions: Optional[List[str]] = None,
        confidence: float = 0.90,
        source_location: str = "04_Trading_Tech_Work/backtest_script.py",
    ) -> NormalizedKnowledgeItem:
        """Normalize a technical/quantitative finding from engineering work."""
        return NormalizedKnowledgeItem(
            item_id=f"norm_tech_{uuid.uuid4().hex[:8]}",
            domain=SourceDomain.TRADING_TECH_WORK,
            source_type=SourceType.WEB_DOCUMENT,
            source_id=script_or_algorithm_id,
            source_location=source_location,
            concept_type="TechnicalFinding",
            title=title,
            statement=finding_statement,
            conditions=conditions or [],
            exceptions=exceptions or [],
            confidence=confidence,
            authority=SourceAuthority.GENERAL_KNOWLEDGE,
            version=1,
            evidence_references=[f"script:{script_or_algorithm_id}"],
            tags=["technical", "quantitative", "algorithm"],
        )

    @staticmethod
    def normalize_rule(
        title: str,
        content: str,
        domain: SourceDomain = SourceDomain.YOUTUBE_LECTURES,
        rule_type: str = "SETUP_RULE",
        authority: SourceAuthority = SourceAuthority.LECTURE_RULE,
        confidence: float = 0.85,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        direction: Optional[str] = None,
        evidence_reference: Optional[str] = None,
        version: str = "1.0",
    ) -> NormalizedKnowledgeItem:
        """Create a canonical normalized knowledge item with explicit domain and provenance."""
        ev_refs = [evidence_reference] if evidence_reference else []
        conds = []
        if symbols:
            conds.append(f"symbols:{','.join(symbols)}")
        if timeframes:
            conds.append(f"timeframes:{','.join(timeframes)}")
        if direction:
            conds.append(f"direction:{direction}")

        dom = domain if isinstance(domain, SourceDomain) else SourceDomain(domain)
        auth = authority if isinstance(authority, SourceAuthority) else SourceAuthority(authority)

        ver_int = 1
        if isinstance(version, str):
            try:
                ver_int = int(version.split(".")[0])
            except Exception:
                ver_int = 1
        elif isinstance(version, int):
            ver_int = version

        return NormalizedKnowledgeItem(
            item_id=f"norm_{uuid.uuid4().hex[:10]}",
            domain=dom,
            source_type=SourceType.LOCAL_VIDEO if dom == SourceDomain.YOUTUBE_LECTURES else SourceType.USER_TEXT,
            source_id=f"{dom.value}_rule",
            source_location=f"{dom.value}/rules.txt",
            concept_type=rule_type,
            title=title,
            statement=content,
            conditions=conds,
            exceptions=[],
            confidence=confidence,
            authority=auth,
            version=ver_int,
            evidence_references=ev_refs,
            tags=[dom.value, rule_type.lower()],
        )

