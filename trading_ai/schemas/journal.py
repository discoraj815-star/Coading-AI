"""
Step 10 Schemas: Professional Trading Journal, Behavioral Observation, Lessons, and Pre-Trade Review.
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import (
    AssetClass,
    EngagementMode,
    ExitReasonCategory,
    JournalBacktestType,
    JournalEnvironment,
    LessonLifecycleState,
    ObservedBehaviorType,
    PreTradeReviewStatus,
    TradeOutcome,
)


class PlannedTradeValues(BaseModel):
    """Immutable plan established prior to trade execution."""

    entry_price: float
    stop_loss: float
    tp1: float = 0.0
    final_target: float = 0.0
    planned_tp1: Optional[float] = None
    planned_final_tp: Optional[float] = None
    quantity: float = 0.0
    risk_amount: float = 0.0
    planned_risk_amount: Optional[float] = None
    risk_reward_ratio: float = 0.0
    planned_rr: Optional[float] = None
    invalidation_condition: str = ""
    invalidation_level: Optional[float] = None
    direction: str = "BUY"  # BUY / SELL
    trade_thesis: str = ""
    # Explicit Step 10 & 11 reasons
    sl_reason: str = ""
    tp_reason: str = ""
    tp1_reason: str = ""
    tp2_reason: str = ""
    invalidation_reason: str = ""
    planned_timestamp: Optional[datetime] = None

    def __init__(self, **data):
        # Synchronize synonyms
        if "planned_tp1" in data and ("tp1" not in data or data.get("tp1") == 0.0):
            data["tp1"] = data["planned_tp1"]
        elif "tp1" in data and ("planned_tp1" not in data or data.get("planned_tp1") is None):
            data["planned_tp1"] = data["tp1"]

        if "planned_final_tp" in data and ("final_target" not in data or data.get("final_target") == 0.0):
            data["final_target"] = data["planned_final_tp"]
        elif "final_target" in data and ("planned_final_tp" not in data or data.get("planned_final_tp") is None):
            data["planned_final_tp"] = data["final_target"]

        if "planned_risk_amount" in data and ("risk_amount" not in data or data.get("risk_amount") == 0.0):
            data["risk_amount"] = data["planned_risk_amount"]
        elif "risk_amount" in data and ("planned_risk_amount" not in data or data.get("planned_risk_amount") is None):
            data["planned_risk_amount"] = data["risk_amount"]

        if data.get("final_target") is None:
            data["final_target"] = 0.0
        if data.get("risk_amount") is None:
            data["risk_amount"] = 0.0
        if data.get("tp1") is None:
            data["tp1"] = 0.0
        if data.get("risk_reward_ratio") is None:
            data["risk_reward_ratio"] = 0.0

        super().__init__(**data)


class ActualTradeValues(BaseModel):
    """Empirical outcomes recorded upon trade closure. Never overwrites Planned."""

    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    actual_sl: Optional[float] = None
    actual_tp: Optional[float] = None
    actual_quantity: Optional[float] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    quantity: Optional[float] = None
    realized_pnl: float = 0.0
    realized_r: float = 0.0
    fees: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    entry_timestamp: Optional[float] = None
    exit_timestamp: Optional[float] = None
    # Step 10 & 11 Timestamps chain
    signal_timestamp: Optional[datetime] = None
    order_timestamp: Optional[datetime] = None
    fill_timestamp: Optional[datetime] = None
    exit_datetime: Optional[datetime] = None
    # Latencies in milliseconds
    analysis_latency_ms: Optional[float] = None
    signal_to_order_latency_ms: Optional[float] = None
    order_to_fill_latency_ms: Optional[float] = None
    total_execution_latency_ms: Optional[float] = None
    # Execution quality
    broker_status: str = "FILLED"
    partial_fills_count: int = 0
    execution_errors: List[str] = Field(default_factory=list)
    # Explicit post-trade why reasons
    why_won_or_lost_detail: str = ""
    why_sl_hit: str = ""
    why_tp_hit: str = ""
    why_manually_closed: str = ""
    why_target_modified: str = ""
    why_sl_modified: str = ""
    outcome: TradeOutcome = TradeOutcome.OPEN
    exit_reason: ExitReasonCategory = ExitReasonCategory.INSUFFICIENT_EVIDENCE
    exit_reason_notes: str = ""
    observed_behaviors: List["ObservedBehavior"] = Field(default_factory=list)
    rule_adhered: bool = True

    def __init__(self, **data):
        # Sync synonyms
        if "entry_price" in data and "actual_entry" not in data:
            data["actual_entry"] = data["entry_price"]
        elif "actual_entry" in data and "entry_price" not in data:
            data["entry_price"] = data["actual_entry"]

        if "exit_price" in data and "actual_exit" not in data:
            data["actual_exit"] = data["exit_price"]
        elif "actual_exit" in data and "exit_price" not in data:
            data["exit_price"] = data["actual_exit"]

        if "stop_loss" in data and "actual_sl" not in data:
            data["actual_sl"] = data["stop_loss"]
        elif "actual_sl" in data and "stop_loss" not in data:
            data["stop_loss"] = data["actual_sl"]

        if "quantity" in data and "actual_quantity" not in data:
            data["actual_quantity"] = data["quantity"]
        elif "actual_quantity" in data and "quantity" not in data:
            data["quantity"] = data["actual_quantity"]

        super().__init__(**data)


class ObservedBehavior(BaseModel):
    """Observable human and execution behavior backed strictly by empirical evidence."""

    behavior_id: str = Field(default_factory=lambda: f"beh_{uuid.uuid4().hex[:10]}")
    behavior_type: ObservedBehaviorType = ObservedBehaviorType.REPEATED_RULE_VIOLATION
    behavior: Optional[ObservedBehaviorType] = None
    description: str = ""
    details: Optional[str] = None
    evidence_source: Optional[str] = None
    evidence_sources: List[str] = Field(default_factory=list)
    severity: str = "MEDIUM"  # HIGH, MEDIUM, LOW, INFO
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    observed_at: Optional[datetime] = None

    def __init__(self, **data):
        if "behavior" in data and "behavior_type" not in data:
            data["behavior_type"] = data["behavior"]
        elif "behavior_type" in data and "behavior" not in data:
            data["behavior"] = data["behavior_type"]

        if "details" in data and "description" not in data:
            data["description"] = data["details"]
        elif "description" in data and "details" not in data:
            data["details"] = data["description"]

        if "evidence_source" in data and "evidence_sources" not in data:
            data["evidence_sources"] = [data["evidence_source"]]
        elif "evidence_sources" in data and data["evidence_sources"] and "evidence_source" not in data:
            data["evidence_source"] = data["evidence_sources"][0]

        if "observed_at" in data and "detected_at" not in data:
            data["detected_at"] = data["observed_at"]
        elif "detected_at" in data and "observed_at" not in data:
            data["observed_at"] = data["detected_at"]

        super().__init__(**data)


class TradeJournalEntry(BaseModel):
    """
    Comprehensive auditable trade journal record.
    Preserves planned vs actual values, context, confluences, behaviors, and provenance.
    """

    journal_id: str = Field(default_factory=lambda: f"jrn_{uuid.uuid4().hex[:12]}")
    trade_id: str = Field(description="Associated position_id, backtest_trade_id, or proposal_id")
    environment: JournalEnvironment = JournalEnvironment.PAPER
    backtest_type: JournalBacktestType = JournalBacktestType.NOT_APPLICABLE
    asset_class: AssetClass = AssetClass.FOREX
    symbol: str = "EUR/USD"
    timeframe: str = "15m"
    direction: str = "BUY"
    rule_version_id: Optional[str] = None
    strategy_version_id: Optional[str] = None
    lecture_source_id: Optional[str] = None
    setup_name: str = "ICT Liquidity Sweep"
    setup: Optional[str] = None
    market_regime: str = "BALANCED"
    htf_context: str = ""
    mtf_context: str = ""
    ltf_context: str = ""
    news_context: str = "NO_IMMINENT_NEWS"
    confluences: List[str] = Field(default_factory=list)
    engagement_mode: EngagementMode = EngagementMode.DEFENSIVE
    trade_thesis: str = ""
    planned: PlannedTradeValues
    actual: ActualTradeValues = Field(default_factory=ActualTradeValues)
    rule_adherence_score: float = 1.0  # 0.0 to 1.0
    rule_violations: List[str] = Field(default_factory=list)
    behaviors: List[ObservedBehavior] = Field(default_factory=list)
    notes: str = ""
    evidence_references: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    backtest_id: Optional[str] = None
    dataset_id: Optional[str] = None
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __init__(self, **data):
        if "setup" in data and "setup_name" not in data:
            data["setup_name"] = data["setup"]
        elif "setup_name" in data and "setup" not in data:
            data["setup"] = data["setup_name"]

        if "asset_class" in data and isinstance(data["asset_class"], str):
            val = data["asset_class"].upper()
            if val in ("FOREX", "FX"):
                data["asset_class"] = AssetClass.FOREX
            elif val in ("CRYPTO", "CRYPTOCURRENCY"):
                data["asset_class"] = AssetClass.CRYPTO
            else:
                data["asset_class"] = AssetClass.FOREX

        super().__init__(**data)


class JournalLesson(BaseModel):
    """
    Structured lesson extracted from journal review.
    Governed by strict user review before affecting rule candidates or strategies.
    """

    lesson_id: str = Field(default_factory=lambda: f"lsn_{uuid.uuid4().hex[:10]}")
    source_journal_ids: List[str] = Field(default_factory=list)
    symbol: Optional[str] = None
    asset_class: Optional[AssetClass] = None
    affected_rule_version_id: Optional[str] = None
    affected_rule_id: Optional[str] = None
    title: str = "Empirical Journal Lesson"
    observation: str
    explanation: str
    proposed_action: str
    confidence: float = 1.0
    state: LessonLifecycleState = LessonLifecycleState.OBSERVED
    status: Optional[LessonLifecycleState] = None
    reviewer_notes: Optional[str] = None
    review_notes: Optional[str] = None
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    evidence: Dict[str, Any] = Field(default_factory=dict)
    candidate_rule_version_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __init__(self, **data):
        if "affected_rule_id" in data and "affected_rule_version_id" not in data:
            data["affected_rule_version_id"] = data["affected_rule_id"]
        elif "affected_rule_version_id" in data and "affected_rule_id" not in data:
            data["affected_rule_id"] = data["affected_rule_version_id"]

        if "status" in data and "state" not in data:
            data["state"] = data["status"]
        elif "state" in data and "status" not in data:
            data["status"] = data["state"]

        if "review_notes" in data and "reviewer_notes" not in data:
            data["reviewer_notes"] = data["review_notes"]
        elif "reviewer_notes" in data and "review_notes" not in data:
            data["review_notes"] = data["reviewer_notes"]

        if "observation" in data and ("title" not in data or not data["title"]):
            data["title"] = data["observation"][:60]

        super().__init__(**data)


class PreTradeReviewRequest(BaseModel):
    """Input payload to evaluate a prospective trade candidate against historical journal memory."""

    proposal_id: Optional[str] = None
    symbol: str = "EUR/USD"
    asset_class: AssetClass = AssetClass.FOREX
    timeframe: str = "15m"
    direction: str = "BUY"
    setup_name: str = "ICT Liquidity Sweep"
    setup: Optional[str] = None
    rule_version_id: Optional[str] = None
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    risk_amount: float = 100.0
    environment: JournalEnvironment = JournalEnvironment.PAPER
    thesis: str = ""
    confluences: List[str] = Field(default_factory=list)
    planned_values: Optional[PlannedTradeValues] = None

    def __init__(self, **data):
        if "setup" in data and "setup_name" not in data:
            data["setup_name"] = data["setup"]
        elif "setup_name" in data and "setup" not in data:
            data["setup"] = data["setup_name"]

        if "planned_values" in data and data["planned_values"]:
            pv = data["planned_values"]
            if pv.entry_price and not data.get("entry_price"):
                data["entry_price"] = pv.entry_price
            if pv.stop_loss and not data.get("stop_loss"):
                data["stop_loss"] = pv.stop_loss
            if pv.tp1 and not data.get("target_price"):
                data["target_price"] = pv.tp1
        elif "entry_price" in data and "stop_loss" in data:
            data["planned_values"] = PlannedTradeValues(
                entry_price=data.get("entry_price", 0.0),
                stop_loss=data.get("stop_loss", 0.0),
                tp1=data.get("target_price", 0.0),
                final_target=data.get("target_price", 0.0),
                risk_amount=data.get("risk_amount", 100.0),
                direction=data.get("direction", "BUY"),
                trade_thesis=data.get("thesis", ""),
            )

        super().__init__(**data)


class PreTradeReviewResult(BaseModel):
    """Decision-support output rendered before execution authorization."""

    review_id: str = Field(default_factory=lambda: f"ptr_{uuid.uuid4().hex[:10]}")
    proposal_id: Optional[str] = None
    symbol: str = "EUR/USD"
    status: PreTradeReviewStatus = PreTradeReviewStatus.APPROVED
    plan_is_complete: bool = True
    missing_plan_elements: List[str] = Field(default_factory=list)
    historical_similar_trades_count: int = 0
    similar_trades_count: int = 0
    similar_trades_sample: List[Dict[str, Any]] = Field(default_factory=list)
    historical_win_rate: float = 0.0
    relevant_lessons: List[JournalLesson] = Field(default_factory=list)
    applicable_lessons: List[JournalLesson] = Field(default_factory=list)
    relevant_mistakes: List[str] = Field(default_factory=list)
    discipline_warnings: List[str] = Field(default_factory=list)
    discipline_issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    rule_violations: List[str] = Field(default_factory=list)
    risk_reward_ratio: float = 0.0
    planned_values: Optional[PlannedTradeValues] = None
    recommendation: str = ""
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __init__(self, **data):
        if "similar_trades_count" in data and "historical_similar_trades_count" not in data:
            data["historical_similar_trades_count"] = data["similar_trades_count"]
        elif "historical_similar_trades_count" in data and "similar_trades_count" not in data:
            data["similar_trades_count"] = data["historical_similar_trades_count"]

        if "applicable_lessons" in data and "relevant_lessons" not in data:
            data["relevant_lessons"] = data["applicable_lessons"]
        elif "relevant_lessons" in data and "applicable_lessons" not in data:
            data["applicable_lessons"] = data["relevant_lessons"]

        if "discipline_issues" in data and "discipline_warnings" not in data:
            data["discipline_warnings"] = data["discipline_issues"]
        elif "discipline_warnings" in data and "discipline_issues" not in data:
            data["discipline_issues"] = data["discipline_warnings"]

        super().__init__(**data)


class JournalAnalyticsSummary(BaseModel):
    """Deterministic mathematical statistics across isolated journal environments."""

    environment: JournalEnvironment
    asset_class: Optional[AssetClass] = None
    symbol: Optional[str] = None
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_r: float = 0.0
    expectancy: float = 0.0
    total_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    max_drawdown_dollars: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    rule_adherence_rate: float = 1.0
    total_rule_violations: int = 0
    observed_behaviors_count: Dict[str, int] = Field(default_factory=dict)
    setup_performance: Dict[str, Any] = Field(default_factory=dict)
    timeframe_performance: Dict[str, Any] = Field(default_factory=dict)
    long_vs_short_performance: Dict[str, Any] = Field(default_factory=dict)
    status_message: str = "OK"


class JournalProcessingCheckpoint(BaseModel):
    """Resilient checkpoint record enabling crash recovery during long journal evaluations."""

    job_id: str = Field(default_factory=lambda: f"jchk_{uuid.uuid4().hex[:10]}")
    last_processed_trade_id: Optional[str] = None
    last_processed_journal_id: Optional[str] = None
    processed_count: int = 0
    total_count: int = 0
    total_entries: int = 0
    context_token_estimate: int = 0
    tokens_consumed: int = 0
    status: str = "COMPLETED"
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_checkpoint_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        if "last_processed_journal_id" in data and "last_processed_trade_id" not in data:
            data["last_processed_trade_id"] = data["last_processed_journal_id"]
        elif "last_processed_trade_id" in data and "last_processed_journal_id" not in data:
            data["last_processed_journal_id"] = data["last_processed_trade_id"]

        if "total_entries" in data and "total_count" not in data:
            data["total_count"] = data["total_entries"]
        elif "total_count" in data and "total_entries" not in data:
            data["total_entries"] = data["total_count"]

        if "tokens_consumed" in data and "context_token_estimate" not in data:
            data["context_token_estimate"] = data["tokens_consumed"]
        elif "context_token_estimate" in data and "tokens_consumed" not in data:
            data["tokens_consumed"] = data["context_token_estimate"]

        if "last_checkpoint_at" in data and "updated_at" not in data:
            data["updated_at"] = data["last_checkpoint_at"]
        elif "updated_at" in data and "last_checkpoint_at" not in data:
            data["last_checkpoint_at"] = data["updated_at"]

        super().__init__(**data)
