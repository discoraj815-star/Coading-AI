"""Step 7 Domain Schemas: AI Trading Companion, Proactive Alerts, Discipline, Scenarios, and Interactive Voice."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    AlertDecision,
    AlertLifecycleState,
    AlertPriority,
    AlertQuality,
    AlertType,
    DisciplineViolationType,
    VoiceIntent,
)


class AlertEvent(BaseModel):
    """Proactive market and trading alert record."""

    alert_id: str = Field(default_factory=lambda: f"alt_{uuid.uuid4().hex[:12]}")
    symbol: str
    alert_type: AlertType
    priority: AlertPriority
    lifecycle_state: AlertLifecycleState = AlertLifecycleState.ALERTED
    decision: AlertDecision = AlertDecision.ALERT
    quality: AlertQuality = AlertQuality.MEDIUM_QUALITY
    title: str
    message: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    is_acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    dedup_key: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AlertStateRecord(BaseModel):
    """Deduplication and tracking state across alert evaluation cycles."""

    dedup_key: str
    symbol: str
    last_alerted_at: float
    last_lifecycle_state: AlertLifecycleState
    alert_count: int = 1
    suppression_reason: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TradingCompanionContext(BaseModel):
    """
    Unified minimal sufficient decision context consumed by AlertEngine,
    VoiceEngine, ScenarioEngine, and DisciplineEngine.
    """

    context_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    current_price: float
    bias: str
    bias_summary: str
    nearest_liquidity: Optional[Dict[str, Any]] = None
    current_poi: Optional[Dict[str, Any]] = None
    active_setup: Optional[Dict[str, Any]] = None
    setup_confirmation_state: str = "NO_ACTIVE_SETUP"
    invalidation_level: Optional[float] = None
    confirmation_level: Optional[float] = None
    news_risk: str = "NO_RISK"
    tech_fund_relationship: str = "TECHNICALLY_VALID"
    next_high_impact_event: Optional[Dict[str, Any]] = None
    news_time_distance_minutes: Optional[float] = None
    recent_changes: List[str] = Field(default_factory=list)
    discipline_status: Dict[str, Any] = Field(default_factory=dict)
    active_alerts_count: int = 0
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class ScenarioCondition(BaseModel):
    """
    Conditional 'IF-THEN-UNLESS' market scenario analysis.
    Explicitly framed as conditional logic, NOT guaranteed prophecy.
    """

    scenario_id: str = Field(default_factory=lambda: f"scn_{uuid.uuid4().hex[:10]}")
    symbol: str
    level: float
    direction: str  # BULLISH / BEARISH / BREAKOUT / REJECTION
    if_condition: str
    then_outcome: str
    unless_condition: str
    invalidated_when: str
    next_poi_or_liquidity: Optional[float] = None
    confidence_quality: AlertQuality = AlertQuality.MEDIUM_QUALITY
    evidence: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DisciplineConfig(BaseModel):
    """
    User-defined discipline and risk boundary configuration.
    Unconfigured fields remain None/RULE_NOT_CONFIGURED (never fabricated).
    """

    user_id: str = "default_user"
    stop_loss_required: bool = True
    daily_loss_limit: Optional[float] = None
    max_daily_trades: Optional[int] = None
    max_risk_per_trade_pct: Optional[float] = None
    overtrading_trades_threshold: Optional[int] = None
    trades_window_minutes: Optional[int] = None
    cooldown_minutes: Optional[int] = None
    allowed_sessions: Optional[List[str]] = None


class DisciplineState(BaseModel):
    """Current discipline metrics and violation status for a trader."""

    user_id: str = "default_user"
    config: DisciplineConfig = Field(default_factory=DisciplineConfig)
    trades_today_count: int = 0
    daily_realized_loss: float = 0.0
    last_trade_timestamp: Optional[float] = None
    cooldown_active_until: Optional[datetime] = None
    recent_trades_timestamps: List[float] = Field(default_factory=list)
    active_violations: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TradeIntentRequest(BaseModel):
    """Trade intent presented to Trade Confirmation Gate for validation."""

    symbol: str
    direction: str
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_amount: Optional[float] = None
    account_balance: float = 10000.0


class TradeConfirmationGateResult(BaseModel):
    """
    Validation gate result before trade commitment.
    Live broker execution is NEVER performed here: stops at USER_CONFIRMATION_REQUIRED.
    """

    status: str = "USER_CONFIRMATION_REQUIRED"
    is_discipline_passed: bool
    is_setup_aligned: bool
    is_risk_acceptable: bool
    can_proceed: bool
    warnings: List[str] = Field(default_factory=list)
    checklist: Dict[str, bool] = Field(default_factory=dict)
    recommended_action: str


class CompanionQueryRequest(BaseModel):
    """User prompt or spoken audio transcript sent to the companion."""

    symbol: str
    query: str
    audio_base64: Optional[str] = None
    user_id: str = "default_user"


class CompanionQueryResponse(BaseModel):
    """Structured response from Trading Companion with intent classification and smallest sufficient context."""

    intent: VoiceIntent
    symbol: str
    title: str
    answer: str
    context_summary: str
    evidence: List[str] = Field(default_factory=list)
    scenarios: List[ScenarioCondition] = Field(default_factory=list)
    audio_transcript: Optional[str] = None
    synthesized_audio_mock: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompanionInteractionRecord(BaseModel):
    """Persistent companion query & response audit log."""

    interaction_id: str = Field(default_factory=lambda: f"int_{uuid.uuid4().hex[:12]}")
    symbol: str
    user_query: str
    intent: VoiceIntent
    answer: str
    context_summary: str
    evidence: List[str] = Field(default_factory=list)
    scenarios: List[ScenarioCondition] = Field(default_factory=list)
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
