"""
Step 6 Fundamental Data & Economic Event Schemas.
Normalized internal representations of economic calendar events, provider health,
and technical-fundamental correlation contexts.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    EventImpact,
    EventStatus,
    NewsRiskLevel,
    NewsTimingCategory,
    ProviderStatus,
    TechFundRelationship,
)


class FundamentalEvent(BaseModel):
    """
    Normalized internal representation of an economic news / calendar event.
    Decoupled from raw source providers (e.g. Forex Factory).
    """

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = "forex_factory"
    title: str
    currency: str  # e.g., "USD", "EUR", "GBP"
    event_time: datetime
    impact: EventImpact = EventImpact.LOW
    previous: Optional[str] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None
    status: EventStatus = EventStatus.SCHEDULED
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_url: Optional[str] = None
    confidence: float = 1.0
    evidence: List[str] = Field(default_factory=list)


class FundamentalProviderState(BaseModel):
    """Health, staleness, and rate-limit tracking for fundamental sources."""

    provider_name: str
    status: ProviderStatus = ProviderStatus.OPERATIONAL
    last_successful_fetch: Optional[datetime] = None
    stale_after_seconds: int = 86400  # Default 24 hours
    is_stale: bool = False
    error_message: Optional[str] = None
    unavailable_reason: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class NewsTimingContext(BaseModel):
    """Timing analysis of relevant news events for a particular instrument."""

    symbol: str
    relevant_currencies: List[str] = Field(default_factory=list)
    category: NewsTimingCategory = NewsTimingCategory.NO_EVENT
    risk_level: NewsRiskLevel = NewsRiskLevel.NO_RISK
    next_high_impact_event: Optional[FundamentalEvent] = None
    time_to_next_seconds: Optional[float] = None
    recent_released_event: Optional[FundamentalEvent] = None
    time_since_last_seconds: Optional[float] = None
    has_event_clustering: bool = False
    active_events: List[FundamentalEvent] = Field(default_factory=list)
    upcoming_events: List[FundamentalEvent] = Field(default_factory=list)


class TechFundConflict(BaseModel):
    """Structured disclosure of a conflict between technical setups and fundamentals."""

    has_conflict: bool = False
    conflict_type: str = "NONE"
    technical_bias: str = "NEUTRAL"
    fundamental_bias: str = "NEUTRAL"
    confidence: float = 1.0
    reason: str = ""
    evidence: List[str] = Field(default_factory=list)


class TechnicalFundamentalAnalysis(BaseModel):
    """
    Synthesized holistic market analysis correlating technical setup validity,
    lecture-constrained rules, economic events, and timing.
    """

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    relationship: TechFundRelationship = TechFundRelationship.TECHNICALLY_VALID
    technical_bias: str
    news_risk: NewsRiskLevel = NewsRiskLevel.NO_RISK
    timing_context: Optional[NewsTimingContext] = None
    conflict: TechFundConflict = Field(default_factory=TechFundConflict)
    action_context: str = "PROCEED_WITH_STANDARD_CONFIRMATION"
    explanation: str = ""
    evidence: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
