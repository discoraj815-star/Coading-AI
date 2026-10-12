"""Step 8: Hedge Fund Execution & Risk Management Framework Schemas."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    AlertQuality,
    ConfluenceCategory,
    EngagementMode,
    ExecutionEnvironment,
    ExecutionMode,
    MarketBias,
    MarketDirection,
    OrderStatus,
    OrderType,
    PositionManagementAction,
    PositionStatus,
    ProposalLifecycleState,
)


class TradeConfluence(BaseModel):
    """Independent technical and fundamental confirmation evidence."""

    confluence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    category: ConfluenceCategory
    source: str
    independence_group: str  # Independent cluster (e.g. HTF_STRUCTURE, LIQUIDITY_SWEEP, LECTURE_OB)
    timeframe: str
    evidence_reference: str
    confidence: float = 1.0
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class TradeThesis(BaseModel):
    """Auditable trade rationale and invalidation premise."""

    thesis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    summary: str
    catalyst: Optional[str] = None
    invalidation_premise: str
    market_regime: str = "TRENDING"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RiskAssessment(BaseModel):
    """Deterministic mathematical risk, position size, and RR calculation."""

    assessment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    account_equity: float
    risk_percentage: float
    risk_amount: float
    entry_price: float
    stop_loss: float
    stop_distance: float
    stop_distance_pct: float
    target_price: float
    reward_distance: float
    risk_reward_ratio: float
    calculated_quantity: float
    rounded_quantity: float
    engagement_mode: EngagementMode
    sizing_factor: float = 1.0
    is_rr_acceptable: bool = True
    is_risk_acceptable: bool = True
    warnings: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TradeProposal(BaseModel):
    """
    Complete, auditable institutional trade proposal.
    Unifies Market Intelligence, Confluence, Risk, and Thesis before Execution.
    """

    proposal_id: str = Field(default_factory=lambda: f"prop_{uuid.uuid4().hex[:12]}")
    symbol: str
    direction: str  # BUY / SELL or BULLISH / BEARISH
    market_regime: str = "BALANCED"
    htf_context: str = ""
    mtf_context: str = ""
    ltf_context: str = ""
    overall_bias: MarketBias = MarketBias.NEUTRAL_BIAS
    liquidity_context: str = ""
    poi_context: str = ""
    lecture_rule_evidence: List[str] = Field(default_factory=list)
    confluences: List[TradeConfluence] = Field(default_factory=list)
    entry_trigger: str = "CONFIRMED_STRUCTURE"
    entry_price: float
    hard_stop_loss: float
    tp1: float
    tp2: Optional[float] = None
    final_target: float
    invalidation_condition: str
    risk_assessment: Optional[RiskAssessment] = None
    proposed_quantity: float = 0.0
    engagement_mode: EngagementMode = EngagementMode.DEFENSIVE
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.PAPER
    news_event_context: str = "NO_IMMINENT_NEWS"
    trade_thesis: Optional[TradeThesis] = None
    quality: AlertQuality = AlertQuality.MEDIUM_QUALITY
    lifecycle_state: ProposalLifecycleState = ProposalLifecycleState.PROPOSED
    user_approval_required: bool = True
    is_approved: bool = False
    rejection_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionPolicy(BaseModel):
    """User-controlled execution constraints and automation limits."""

    user_id: str = "default_user"
    execution_mode: ExecutionMode = ExecutionMode.MANUAL
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.PAPER
    live_execution_enabled: bool = False
    auto_execution_enabled: bool = False
    max_auto_risk_pct: float = 1.0
    min_rr_ratio: float = 2.0
    preferred_rr_ratio: float = 3.0
    default_tp1_r_multiple: float = 1.5
    default_tp1_close_pct: float = 0.5
    move_sl_to_entry_at_tp1: bool = True
    min_confluence_count: int = 2
    allowed_symbols: List[str] = Field(default_factory=lambda: ["EUR/USD", "BTC/USD", "ETH/USD", "SOL/USD", "XAU/USD"])
    defensive_size_pct: float = 0.25
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionIntent(BaseModel):
    """Authorized instruction passed from ExecutionGate to BrokerAdapter."""

    intent_id: str = Field(default_factory=lambda: f"int_{uuid.uuid4().hex[:12]}")
    proposal_id: str
    symbol: str
    execution_mode: ExecutionMode
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.PAPER
    order_type: OrderType = OrderType.MARKET
    direction: str  # BUY or SELL
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    tp1: float
    status: str = "CREATED"
    anti_duplicate_token: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Order(BaseModel):
    """Broker order tracking entity with execution lifecycle."""

    order_id: str = Field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:12]}")
    broker_order_id: Optional[str] = None
    intent_id: str
    proposal_id: str
    symbol: str
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.PAPER
    is_simulated: bool = True
    order_type: OrderType
    direction: str
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    broker_response: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Position(BaseModel):
    """Actively managed market position."""

    position_id: str = Field(default_factory=lambda: f"pos_{uuid.uuid4().hex[:12]}")
    proposal_id: str
    symbol: str
    direction: str  # BUY or SELL
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.PAPER
    entry_price: float
    current_quantity: float
    initial_quantity: float
    initial_sl: float
    current_sl: float
    tp1: float
    tp2: Optional[float] = None
    final_target: float
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    tp1_reached: bool = False
    sl_moved_to_entry: bool = False
    engagement_mode: EngagementMode = EngagementMode.DEFENSIVE
    opened_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: Optional[datetime] = None


class PositionEvent(BaseModel):
    """Audit log entry for position management actions."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    position_id: str
    action: PositionManagementAction
    price: float
    quantity_affected: float
    realized_pnl_delta: float = 0.0
    notes: str = ""
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
