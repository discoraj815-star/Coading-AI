"""Pydantic schemas for Market Intelligence, Multi-Timeframe Analysis, Markings, and Screen Privacy."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    LevelType,
    MarketBias,
    MarketDataStatus,
    MarketDirection,
    MarketStructureType,
    MarkingType,
    MarkingValidity,
    ScreenAccessStatus,
    SetupStatus,
    TimeframeHierarchy,
    WatchlistMonitorStatus,
)
from trading_ai.schemas.fundamental import FundamentalEvent


class Candle(BaseModel):
    """Single OHLCV bar."""

    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


class MarketSnapshot(BaseModel):
    """Market price snapshot for a symbol and timeframe."""

    symbol: str
    timeframe: str
    candles: List[Candle] = Field(default_factory=list)
    latest_price: float
    timestamp: float
    status: MarketDataStatus = MarketDataStatus.DATA_AVAILABLE
    error: Optional[str] = None


class MarketLevel(BaseModel):
    """Identified critical price level with provenance and invalidation conditions."""

    level_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timeframe: str
    price: float
    level_type: LevelType
    rule_id: Optional[str] = None
    evidence_json: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0
    validity: MarkingValidity = MarkingValidity.VALID
    invalidation_condition: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChartMarking(BaseModel):
    """Visual chart overlay element validated against lecture knowledge and market supervisor."""

    marking_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timeframe: str
    marking_type: MarkingType
    coordinates_json: Dict[str, Any] = Field(default_factory=dict)
    rule_id: Optional[str] = None
    evidence_json: Dict[str, Any] = Field(default_factory=dict)
    validity: MarkingValidity = MarkingValidity.VALID
    confidence: float = 1.0
    invalidation_condition: Optional[str] = None
    priority: int = 1
    # Canonical Live Broker & Backtest Marking Fields
    label: Optional[str] = None
    concept: Optional[str] = None
    price: Optional[float] = None
    price_range: Optional[Union[Dict[str, Any], List[float]]] = None
    direction: Optional[str] = None
    source: str = "SUPERVISOR"  # "SUPERVISOR", "BACKTEST", "TECHNICAL_ANALYSIS"
    rule_reference: Optional[str] = None
    rule_version: Optional[str] = "1.0"
    status: str = "VALIDATED"  # "DETECTED", "CANDIDATE", "VALIDATED", "CONFIRMED", "INVALIDATED", "EXPIRED", "UNCERTAIN"
    visibility: str = "VISIBLE"  # "VISIBLE", "SUPPRESSED", "FILTERED"
    created_by: str = "MARKET_SUPERVISOR"
    broker_id: Optional[str] = None
    market_context: Dict[str, Any] = Field(default_factory=dict)
    backtest_run_id: Optional[str] = None
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MarketSetup(BaseModel):
    """Lecture-constrained trading setup with persistent lifecycle and confirmation tracking."""

    setup_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    timeframe: str
    setup_type: str
    direction: MarketDirection
    status: SetupStatus
    applicable_rule_id: Optional[str] = None
    conditions_satisfied: List[str] = Field(default_factory=list)
    exceptions_checked: List[str] = Field(default_factory=list)
    invalidation_checked: List[str] = Field(default_factory=list)
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    confidence: float = 1.0
    # Step 5 Lifecycle & Confirmation Tracking
    detection_timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    confirmation_timestamp: Optional[float] = None
    invalidation_timestamp: Optional[float] = None
    invalidation_condition: Optional[str] = None
    confirmation_evidence: List[str] = Field(default_factory=list)
    previous_status: Optional[SetupStatus] = None
    status_transition_reason: Optional[str] = None
    is_displayed: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TimeframeAnalysis(BaseModel):
    """Analysis result for a specific timeframe within the hierarchy."""

    symbol: str
    timeframe: str
    hierarchy_role: TimeframeHierarchy
    direction: MarketDirection
    structure: MarketStructureType
    key_levels: List[MarketLevel] = Field(default_factory=list)
    bias: MarketBias = MarketBias.NEUTRAL_BIAS
    notes: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())


class UnifiedMarketAnalysis(BaseModel):
    """Synthesized multi-timeframe analysis across HTF, MTF, and LTF."""

    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    htf_analysis: Optional[TimeframeAnalysis] = None
    mtf_analysis: Optional[TimeframeAnalysis] = None
    ltf_analysis: Optional[TimeframeAnalysis] = None
    overall_bias: MarketBias = MarketBias.NEUTRAL_BIAS
    active_setups: List[MarketSetup] = Field(default_factory=list)
    active_markings: List[ChartMarking] = Field(default_factory=list)
    conflict_notes: List[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScreenPermission(BaseModel):
    """Privacy consent boundary for screen access."""

    permission_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    app_name: str
    window_title: Optional[str] = None
    region_box: Optional[List[int]] = None  # [x, y, w, h]
    status: ScreenAccessStatus = ScreenAccessStatus.SCREEN_ACCESS_DISABLED
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    granted_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


class WatchlistMonitorState(BaseModel):
    """State of an independently monitored symbol in the watchlist queue."""

    symbol: str
    timeframe: str = "15m"
    last_processed_timestamp: float = 0.0
    latest_status: WatchlistMonitorStatus = WatchlistMonitorStatus.IDLE
    worker_state: str = "IDLE"
    retry_count: int = 0
    last_error: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DisplayPolicy(BaseModel):
    """Configuration for intelligent selective chart visualization."""

    show_all_liquidity: bool = False
    show_hidden_setups: bool = False
    show_all_fvgs: bool = False
    show_all_order_blocks: bool = False
    show_unconfirmed: bool = False
    focus_active_setup_only: bool = False
    max_visible_markings: int = 7
    proximity_threshold_pct: float = 0.05  # Within 5% of current price


class CurrentSituationSnapshot(BaseModel):
    """
    Lightweight decision-context layer answering 'What matters RIGHT NOW?'
    Distilled minimal sufficient context for visible chart without noise.
    """

    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    current_price: float
    overall_bias: MarketBias
    bias_summary: str
    nearest_major_liquidity: Optional[MarketLevel] = None
    current_poi: Optional[MarketLevel] = None
    active_confirmed_setup: Optional[MarketSetup] = None
    setup_confirmation_state: str = "NO_ACTIVE_SETUP"
    invalidation_level: Optional[float] = None
    confirmation_level: Optional[float] = None
    timeframe_conflicts: List[str] = Field(default_factory=list)
    recent_changes: List[str] = Field(default_factory=list)
    visible_markings: List[ChartMarking] = Field(default_factory=list)
    background_tracking_count: int = 0
    # Step 6 Fundamental Market Intelligence Extension
    fundamental_context: Optional[str] = None
    relevant_news: List[FundamentalEvent] = Field(default_factory=list)
    next_high_impact_event: Optional[FundamentalEvent] = None
    news_time_distance_minutes: Optional[float] = None
    news_risk: str = "NO_RISK"
    technical_fundamental_relationship: str = "TECHNICALLY_VALID"
    technical_fundamental_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_action_context: str = "PROCEED_WITH_STANDARD_CONFIRMATION"
    timestamp: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
