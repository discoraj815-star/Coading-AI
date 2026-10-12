"""
Step 9 Schemas: Backtesting, Trading-Brain Validation, Rule Versioning, and Live-Market Promotion.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    AssetClass,
    BacktestMode,
    BacktestStatus,
    RuleLifecycleState,
    TradeSource,
)


class RuleVersion(BaseModel):
    """
    Immutable versioned trading rule derived from a lecture or user formulation.
    Never overwritten; new modifications spawn new versions (e.g. v1, v2, v3).
    """

    rule_id: str = Field(default_factory=lambda: f"rule_{uuid.uuid4().hex[:10]}")
    version: int = Field(default=1, ge=1)
    title: str = Field(description="Short human-readable rule label")
    rule_text: str = Field(description="Exact formulated logic of the trading rule")
    conditions: List[str] = Field(default_factory=list, description="Mandatory market conditions for applicability")
    exceptions: List[str] = Field(default_factory=list, description="Invalidation conditions or exceptions")
    source_lecture_id: Optional[str] = Field(default=None, description="Source lecture job ID or citation")
    source_timestamp: Optional[float] = Field(default=None, description="Video/audio timestamp where rule was taught")
    provenance: str = Field(default="user_lecture", description="Provenance tracking origin")
    state: RuleLifecycleState = Field(default=RuleLifecycleState.LEARNED)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StrategyVersion(BaseModel):
    """
    Composite trading strategy binding one or more specific rule versions together.
    Maintains exact provenance of each underlying rule version.
    """

    strategy_id: str = Field(default_factory=lambda: f"strat_{uuid.uuid4().hex[:10]}")
    version: int = Field(default=1, ge=1)
    name: str = Field(description="Strategy name")
    description: str = Field(default="")
    rule_version_ids: List[str] = Field(default_factory=list, description="Exact rule_id list bound to this strategy")
    asset_classes: List[AssetClass] = Field(default_factory=lambda: [AssetClass.FOREX])
    symbols: List[str] = Field(default_factory=lambda: ["EUR/USD"])
    timeframes: List[str] = Field(default_factory=lambda: ["15m"])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BacktestConfig(BaseModel):
    """Configuration constraints and execution assumptions for a backtest run."""

    asset_class: AssetClass = Field(default=AssetClass.FOREX)
    symbol: str = Field(default="EUR/USD")
    timeframe: str = Field(default="15m")
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    initial_equity: float = Field(default=10000.0, gt=0)
    risk_per_trade_pct: float = Field(default=1.0, gt=0, le=5.0)
    spread_pips: float = Field(default=1.0, ge=0.0)
    slippage_pips: float = Field(default=0.5, ge=0.0)
    fee_pct: float = Field(default=0.0, ge=0.0)
    mode: BacktestMode = Field(default=BacktestMode.AI_BACKTEST)
    max_candles: int = Field(default=200, ge=20, le=5000)


class BacktestTrade(BaseModel):
    """Auditable individual trade executed during a backtest or manual replay."""

    trade_id: str = Field(default_factory=lambda: f"btt_{uuid.uuid4().hex[:10]}")
    backtest_id: str
    symbol: str
    asset_class: AssetClass
    direction: str = Field(description="BUY or SELL")
    entry_time: float
    entry_price: float
    exit_time: float
    exit_price: float
    stop_loss: float
    target_price: float
    quantity: float
    pnl: float
    r_multiple: float
    result: str = Field(description="WIN, LOSS, or BREAKEVEN")
    rule_version_id: Optional[str] = None
    source: TradeSource = Field(default=TradeSource.AI)
    confluence_summary: str = ""
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BacktestMetrics(BaseModel):
    """
    Deterministic quantitative metrics computed from historical backtest trades.
    Strictly mathematical (never fabricated by LLM).
    """

    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_r: float = 0.0
    expectancy: float = 0.0
    max_drawdown_dollars: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    average_win: float = 0.0
    average_loss: float = 0.0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    symbol_breakdown: Dict[str, Any] = Field(default_factory=dict)
    timeframe_breakdown: Dict[str, Any] = Field(default_factory=dict)
    regime_breakdown: Dict[str, Any] = Field(default_factory=dict)


class BacktestRun(BaseModel):
    """Persistent, checkpointed backtest execution job."""

    backtest_id: str = Field(default_factory=lambda: f"bt_{uuid.uuid4().hex[:10]}")
    mode: BacktestMode = Field(default=BacktestMode.AI_BACKTEST)
    asset_class: AssetClass = Field(default=AssetClass.FOREX)
    symbol: str = Field(default="EUR/USD")
    timeframe: str = Field(default="15m")
    strategy_version_id: Optional[str] = None
    rule_version_id: Optional[str] = None
    config: BacktestConfig = Field(default_factory=BacktestConfig)
    status: BacktestStatus = Field(default=BacktestStatus.PENDING)
    progress_pct: float = 0.0
    current_candle_index: int = 0
    total_candles: int = 0
    metrics: Optional[BacktestMetrics] = None
    is_reproducible: bool = True
    dataset_identifier: str = "simulated_historical_ohlcv"
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RulePromotionRecord(BaseModel):
    """
    Audit record tracking a rule's journey along the promotion ladder:
    LEARNED -> CANDIDATE -> BACKTESTING -> BACKTEST_COMPLETE -> USER_REVIEW ->
    APPROVED_FOR_PAPER -> PAPER_VALIDATION -> USER_LIVE_APPROVAL -> LIVE_ELIGIBLE -> LIVE_ACTIVE
    """

    promotion_id: str = Field(default_factory=lambda: f"prm_{uuid.uuid4().hex[:10]}")
    rule_version_id: str
    current_state: RuleLifecycleState = Field(default=RuleLifecycleState.LEARNED)
    backtest_id: Optional[str] = None
    user_backtest_approved: bool = False
    paper_validation_approved: bool = False
    user_live_approved: bool = False
    paper_trade_count: int = 0
    paper_pnl: float = 0.0
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Request / Response Schemas for APIs

class CreateRuleVersionRequest(BaseModel):
    title: str
    rule_text: str
    conditions: List[str] = Field(default_factory=list)
    exceptions: List[str] = Field(default_factory=list)
    source_lecture_id: Optional[str] = None
    source_timestamp: Optional[float] = None
    provenance: str = "user_input"
    tags: List[str] = Field(default_factory=list)


class StartBacktestRequest(BaseModel):
    mode: BacktestMode = Field(default=BacktestMode.AI_BACKTEST)
    asset_class: AssetClass = Field(default=AssetClass.FOREX)
    symbol: str = Field(default="EUR/USD")
    timeframe: str = Field(default="15m")
    rule_version_id: Optional[str] = None
    strategy_version_id: Optional[str] = None
    initial_equity: float = 10000.0
    risk_per_trade_pct: float = 1.0
    max_candles: int = 100


class ManualBacktestTradeRequest(BaseModel):
    backtest_id: str = ""
    symbol: str
    asset_class: AssetClass
    direction: str
    entry_time: float
    entry_price: float
    exit_time: float
    exit_price: float
    stop_loss: float
    target_price: float
    quantity: float
    rule_version_id: Optional[str] = None
    confluence_summary: str = ""
    notes: Optional[str] = None


class PromoteRuleRequest(BaseModel):
    user_confirmed: bool = True
    notes: str = "User promotion authorization"
