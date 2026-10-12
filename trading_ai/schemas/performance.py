"""
Step 11 Adaptive AI Architecture Schemas.
Event Bus payloads, Tiered Memory States, Incremental Analysis Deltas,
Adaptive Compute Policies, Performance/Latency Telemetry, and Controlled Upgrade Records.
"""
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    AssetClass,
    EventPriority,
    MemoryTier,
    MarketBias,
    SystemEventType,
    UpgradeStatus,
)
from trading_ai.schemas.market import Candle, MarketSetup
from trading_ai.schemas.execution import Position


class SystemEvent(BaseModel):
    """Immutable, timestamped event delivered through the high-performance event bus."""

    event_id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    event_type: SystemEventType
    priority: EventPriority = EventPriority.MEDIUM
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    timestamp: Union[float, datetime] = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    payload: Dict[str, Any] = Field(default_factory=dict)
    provenance: str = "internal_runtime"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IncrementalAnalysisDelta(BaseModel):
    """Represents differential technical structure changes without full recalculation."""

    symbol: str
    timeframe: str
    previous_candle_timestamp: Optional[float] = None
    new_candle_timestamp: float = 0.0
    timestamp: Optional[Union[float, datetime]] = None
    price_delta: float = 0.0
    swings_changed: bool = False
    bos_detected: bool = False
    choch_detected: bool = False
    new_bos_detected: bool = False
    new_choch_detected: bool = False
    liquidity_swept: bool = False
    displacement_increase: bool = False
    rejection_wick_detected: bool = False
    volatility_expansion: bool = False
    structure_weakened: bool = False
    affected_components: List[str] = Field(default_factory=list)
    fvg_created_count: int = 0
    fvg_mitigated_count: int = 0
    nearest_liquidity_price: Optional[float] = None
    computation_time_ms: float = 0.0
    is_incremental: bool = True


class HotStateMemory(BaseModel):
    """Ultra-fast, lowest latency hot memory holding active trade and market decision context."""

    symbol: str
    latest_price: float = 0.0
    current_price: float = 0.0
    latest_tick_time: float = 0.0
    active_bias: str = "NEUTRAL"
    current_bias: Optional[MarketBias] = None
    active_regime: str = "BALANCED"
    nearest_liquidity: Optional[float] = None
    active_setup_id: Optional[str] = None
    active_setup: Optional[MarketSetup] = None
    active_position_id: Optional[str] = None
    active_position: Optional[Position] = None
    active_risk_breached: bool = False
    active_risk_score: float = 0.0
    active_candle: Optional[Candle] = None
    last_journal_entry_id: Optional[str] = None
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class LatencyTelemetry(BaseModel):
    """Deterministic latency measurement across the pipeline stages in milliseconds."""

    ingestion_latency_ms: float = 0.0
    normalization_latency_ms: float = 0.0
    technical_analysis_ms: float = 0.0
    mtf_synthesis_ms: float = 0.0
    gemini_reasoning_ms: float = 0.0
    end_to_end_decision_ms: float = 0.0
    queue_wait_ms: float = 0.0
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AdaptiveComputePolicy(BaseModel):
    """Dynamically allocated compute limits based on market regime and active position presence."""

    symbol: str = "ALL"
    regime: str = "BALANCED"
    has_active_position: bool = False
    has_confirmed_setup: bool = False
    poll_interval_seconds: float = 1.0
    monitoring_interval_seconds: float = 1.0
    technical_recalc_depth: str = "INCREMENTAL"
    gemini_skip_ratio: float = 0.5
    reasoning_depth: str = "FAST"
    news_check_interval_seconds: float = 10.0
    gemini_reasoning_allowed: bool = True
    priority: EventPriority = EventPriority.MEDIUM
    cache_ttl_seconds: float = 3.0


class ComponentUpgradeRecord(BaseModel):
    """Audit record governing the controlled candidate -> benchmark -> user approval ladder."""

    upgrade_id: str = Field(default_factory=lambda: f"upg_{uuid.uuid4().hex[:10]}")
    component_name: str
    current_version: str
    candidate_version: str
    status: UpgradeStatus = UpgradeStatus.CANDIDATE
    benchmark_accuracy: Optional[float] = None
    benchmark_latency_ms: Optional[float] = None
    regression_pass_rate: Optional[float] = None
    user_approved: bool = False
    approved_by: Optional[str] = None
    notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
