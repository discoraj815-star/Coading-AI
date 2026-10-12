"""Schemas for Step 14 Separate File Architecture, Unified Multi-Source Analysis, and Core Trade Logic."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    RelationshipType,
    SourceAuthority,
    SourceDomain,
    SourceType,
    TradeDecisionState,
)
from trading_ai.schemas.execution import TradeProposal


AUTHORITY_PRIORITY_MAP: Dict[SourceAuthority, int] = {
    SourceAuthority.USER_OVERRIDE: 1,
    SourceAuthority.MENTORSHIP_RULE: 2,
    SourceAuthority.JOURNAL_LESSON: 3,
    SourceAuthority.LECTURE_RULE: 4,
    SourceAuthority.GENERAL_KNOWLEDGE: 5,
}


class KnowledgeRelationship(BaseModel):
    """Semantic or epistemic link between two knowledge units across sources."""

    relationship_id: str = Field(default_factory=lambda: f"rel_{uuid.uuid4().hex[:10]}")
    source_item_id: str = Field(description="Originating knowledge item ID")
    target_item_id: str = Field(description="Target knowledge item ID being supported, refined, or contradicted")
    relationship_type: RelationshipType = Field(description="Semantic nature of link")
    description: str = Field(default="", description="Human-readable rationale for this cross-source link")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NormalizedKnowledgeItem(BaseModel):
    """Normalized, provenance-backed knowledge unit representation across all 4 physical domains."""

    item_id: str = Field(default_factory=lambda: f"norm_{uuid.uuid4().hex[:10]}")
    domain: SourceDomain = Field(description="Physical source isolation domain")
    source_type: SourceType = Field(default=SourceType.LOCAL_VIDEO, description="Technical source medium")
    source_id: str = Field(description="Unique source identifier")
    source_document_id: Optional[str] = Field(default=None, description="Optional document or video file ID")
    source_session_id: Optional[str] = Field(default=None, description="Optional mentorship or trading session ID")
    source_location: str = Field(description="Canonical path or URI within physical domain")
    concept_type: str = Field(
        default="Rule",
        description="Concept type: Concept, Rule, Condition, Exception, Setup, RiskRule, MarketObservation, TradeLesson, TechnicalFinding, MentorInstruction, JournalLesson",
    )
    title: str = Field(description="Concise label or title")
    statement: str = Field(description="Canonical formulation of knowledge or instruction")
    conditions: List[str] = Field(default_factory=list, description="Mandatory prerequisite conditions")
    exceptions: List[str] = Field(default_factory=list, description="Invalidating exceptions")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Epistemic confidence score")
    authority: SourceAuthority = Field(default=SourceAuthority.LECTURE_RULE, description="Precedence authority")
    authority_priority: int = Field(default=4, description="Priority integer (1 = highest, 5 = lowest)")
    version: int = Field(default=1, ge=1, description="Immutable version number")
    evidence_references: List[str] = Field(default_factory=list, description="Linked evidence or trace IDs")
    tags: List[str] = Field(default_factory=list, description="Categorization and search tags")
    status: str = Field(default="ACTIVE", description="ACTIVE, DEPRECATED, SUPERSEDED, or UNDER_REVIEW")
    relationships: List[KnowledgeRelationship] = Field(default_factory=list, description="Cross-source links")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def source_domain(self) -> str:
        return self.domain.value if hasattr(self.domain, "value") else str(self.domain)

    @property
    def rule_version(self) -> str:
        return f"{self.version}.0"

    @property
    def rule_type(self) -> str:
        return self.concept_type

    @property
    def content(self) -> str:
        return self.statement

    def __init__(self, **data):
        if "source_domain" in data and "domain" not in data:
            data["domain"] = data["source_domain"]
        if "content" in data and "statement" not in data:
            data["statement"] = data["content"]
        if "rule_type" in data and "concept_type" not in data:
            data["concept_type"] = data["rule_type"]
        super().__init__(**data)
        if "authority" in data and "authority_priority" not in data:
            auth = self.authority
            if isinstance(auth, str):
                try:
                    auth = SourceAuthority(auth)
                except Exception:
                    auth = SourceAuthority.LECTURE_RULE
            self.authority_priority = AUTHORITY_PRIORITY_MAP.get(auth, 4)



class ConfluenceEvaluation(BaseModel):
    """Result of cross-source data fusion and multi-source confluence analysis."""

    confluence_id: str = Field(default_factory=lambda: f"conf_{uuid.uuid4().hex[:10]}")
    symbol: str
    timeframe: str
    decision_state: TradeDecisionState = Field(default=TradeDecisionState.NO_SETUP)
    confluence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence/confluence grade")
    matched_rules: List[NormalizedKnowledgeItem] = Field(default_factory=list)
    conflicting_rules: List[Dict[str, Any]] = Field(default_factory=list)
    missing_confirmations: List[str] = Field(default_factory=list)
    supporting_lessons: List[Dict[str, Any]] = Field(default_factory=list)
    mentor_filters_passed: List[str] = Field(default_factory=list)
    reasons_to_trade: List[str] = Field(default_factory=list)
    reasons_not_to_trade: List[str] = Field(default_factory=list)
    conflict_summary: Optional[str] = Field(default=None)
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def state(self) -> str:
        return self.decision_state.value if hasattr(self.decision_state, "value") else str(self.decision_state)

    def __init__(self, **data):
        if "state" in data and "decision_state" not in data:
            data["decision_state"] = data["state"]
        super().__init__(**data)


class WhyNotTradeResult(BaseModel):
    """Result of the mandatory negative validation pass."""

    should_block: bool = Field(default=False, description="True if a blocking reason prevents any trade suggestion")
    blocking_reasons: List[str] = Field(default_factory=list)
    non_blocking_warnings: List[str] = Field(default_factory=list)
    evaluated_checks_count: int = Field(default=0)
    has_htf_contradiction: bool = False
    has_news_risk: bool = False
    has_missing_confirmation: bool = False
    has_journal_warning: bool = False
    has_mentorship_exception: bool = False

    @property
    def has_blocking_reasons(self) -> bool:
        return self.should_block or len(self.blocking_reasons) > 0

    def __init__(self, **data):
        if "has_blocking_reasons" in data and "should_block" not in data:
            data["should_block"] = data["has_blocking_reasons"]
        super().__init__(**data)


class CoreTradeDecision(BaseModel):
    """Deterministic output of Core Trade Logic including exact prices, RR, and thesis."""

    decision_id: str = Field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:10]}")
    symbol: str
    timeframe: str
    direction: str = "NONE"  # BUY / SELL / NONE
    state: TradeDecisionState = TradeDecisionState.NO_SETUP
    entry_price: Optional[float] = 0.0
    hard_stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: Optional[float] = None
    final_target: float = 0.0
    r_multiple: float = 0.0
    invalidation_condition: str = ""
    confirmation_requirement: str = ""
    why_summary: List[str] = Field(default_factory=list)
    why_not_summary: List[str] = Field(default_factory=list)
    source_provenance_trace: List[Dict[str, Any]] = Field(default_factory=list)
    trade_proposal: Optional[TradeProposal] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def decision_state(self) -> str:
        return self.state.value if hasattr(self.state, "value") else str(self.state)

    @property
    def stop_loss(self) -> float:
        return self.hard_stop_loss

    @property
    def take_profits(self) -> List[float]:
        tps = [self.tp1]
        if self.tp2 is not None:
            tps.append(self.tp2)
        return tps

    @property
    def risk_reward_ratio(self) -> float:
        return self.r_multiple

    @property
    def invalidation(self) -> str:
        return self.invalidation_condition

    @property
    def evidence_references(self) -> List[str]:
        refs = []
        for p in self.source_provenance_trace:
            if isinstance(p, dict):
                r = p.get("evidence_refs") or p.get("evidence_references") or []
                if isinstance(r, list):
                    refs.extend(r)
                elif isinstance(r, str):
                    refs.append(r)
        return refs or ["TRACE_RECORD_1"]

    def __init__(self, **data):
        if "decision_state" in data and "state" not in data:
            data["state"] = data["decision_state"]
        if "stop_loss" in data and "hard_stop_loss" not in data:
            data["hard_stop_loss"] = data["stop_loss"]
        if "take_profits" in data:
            tps = data["take_profits"]
            if len(tps) > 0 and "tp1" not in data:
                data["tp1"] = tps[0]
            if len(tps) > 1 and "tp2" not in data:
                data["tp2"] = tps[1]
        if "risk_reward_ratio" in data and "r_multiple" not in data:
            data["r_multiple"] = data["risk_reward_ratio"]
        if "invalidation" in data and "invalidation_condition" not in data:
            data["invalidation_condition"] = data["invalidation"]
        super().__init__(**data)


class JournalFeedbackInsight(BaseModel):
    """Insight derived from trading journal history to inform future setups."""

    symbol: str
    setup_type: str
    total_historical_trades: int = 0
    win_rate: float = 0.0
    average_rr: float = 0.0
    applicable_lessons: List[str] = Field(default_factory=list)
    warning_flags: List[str] = Field(default_factory=list)
    recommended_condition_adjustments: List[str] = Field(default_factory=list)

    @property
    def validated_lessons(self) -> List[str]:
        return self.applicable_lessons

    @property
    def mistakes_to_avoid(self) -> List[str]:
        return self.warning_flags

    def __init__(self, **data):
        if "validated_lessons" in data and "applicable_lessons" not in data:
            data["applicable_lessons"] = data["validated_lessons"]
        if "mistakes_to_avoid" in data and "warning_flags" not in data:
            data["warning_flags"] = data["mistakes_to_avoid"]
        super().__init__(**data)

