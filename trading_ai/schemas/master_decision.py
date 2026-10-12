"""Step 17: Master Decision Engine Schemas & Data Contracts.

Defines:
- EpistemicConceptState: 4-layer epistemic separation (OBSERVED, LEARNED, INFERRED, DECIDED)
- ConceptResult: Normalized output contract for all trading concept modules
- MasterDecisionResult: Central deterministic trade decision outcome
- DecisionAuditRecord: Immutable persistence trace for audit & reproducibility
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import ConceptDirection, TradeDecisionState


class EpistemicConceptState(BaseModel):
    """Four-layer epistemic separation for a single concept's analysis."""

    observed: List[str] = Field(default_factory=list, description="Raw factual observations (metrics, prices, levels)")
    learned: List[str] = Field(default_factory=list, description="Approved user rules / lecture citations")
    inferred: List[str] = Field(default_factory=list, description="Logical deductions from observed + learned")
    decided: str = Field(default="UNKNOWN", description="Concrete conclusion for this concept")


class ConceptResult(BaseModel):
    """
    Normalized result contract produced by any trading concept module
    (SMC/ICT, Orderflow, Footprint, Volume Profile, Wyckoff, or Plugins).
    """

    concept_id: str = Field(..., description="Unique concept identifier (e.g. 'smc_ict', 'orderflow')")
    concept_name: str = Field(..., description="Human readable concept name")
    symbol: str = Field(..., description="Trading pair or asset symbol")
    timeframe: str = Field(default="15m", description="Primary timeframe evaluated")
    timestamp: float = Field(default=0.0, description="Evaluation timestamp (epoch seconds)")
    direction: ConceptDirection = Field(default=ConceptDirection.UNKNOWN, description="BULLISH, BEARISH, NEUTRAL, UNKNOWN")
    bias_score: float = Field(default=0.0, description="Normalized bias score from -1.0 (bearish) to +1.0 (bullish)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Concept model confidence (0.0 to 1.0)")
    evidence: List[str] = Field(default_factory=list, description="Key factual evidence items")
    evidence_quality: float = Field(default=0.0, ge=0.0, le=1.0, description="Factual robustness & recency (0.0 to 1.0)")
    required_data: List[str] = Field(default_factory=list, description="Data points required for concept activation")
    available_data: List[str] = Field(default_factory=list, description="Data points successfully verified")
    missing_data: List[str] = Field(default_factory=list, description="Required data points that are absent")
    is_data_available: bool = Field(default=True, description="False if any mandatory data is missing")
    rule_ids: List[str] = Field(default_factory=list, description="IDs of approved rules applied")
    rule_versions: List[str] = Field(default_factory=list, description="Versions of applied rules")
    source_provenance: List[Dict[str, Any]] = Field(default_factory=list, description="Authority provenance trace")
    epistemic_state: EpistemicConceptState = Field(default_factory=EpistemicConceptState)
    conflict_state: Optional[str] = Field(default=None, description="Identified internal conflict if any")
    invalidation: Optional[str] = Field(default=None, description="Structural invalidation condition or level")
    notes: Optional[str] = Field(default=None, description="Diagnostic commentary")


class MasterDecisionResult(BaseModel):
    """
    Final deterministic output from MasterDecisionEngine.
    Combines multi-concept confluence, conflict checking, risk verification, and bilingual presentation.
    """

    decision_id: str = Field(..., description="Unique UUID for this decision evaluation")
    symbol: str = Field(..., description="Trading symbol evaluated")
    timeframe: str = Field(default="15m", description="Execution timeframe")
    timestamp: float = Field(default=0.0, description="Evaluation timestamp")
    final_state: TradeDecisionState = Field(
        default=TradeDecisionState.NO_TRADE,
        description="Deterministic final decision state (CONFIRMED_SETUP, NO_TRADE, BLOCKED, etc.)",
    )
    direction: str = Field(default="NEUTRAL", description="BUY, SELL, or NEUTRAL")
    confluence_score: float = Field(default=0.0, ge=0.0, le=10.0, description="Deterministic confluence score (0.0 to 10.0)")
    threshold: float = Field(default=8.0, description="Configurable threshold required for trade authorization (e.g. 8.0/10)")
    participating_concepts: List[str] = Field(default_factory=list, description="List of concept IDs contributing positive confluence")
    concept_results: Dict[str, ConceptResult] = Field(default_factory=dict, description="Normalized outputs from all evaluated concepts")
    macro_bias: str = Field(default="NEUTRAL", description="Macro context layer bias (HTF SMC, HTF Volume Profile, Wyckoff)")
    micro_confirmation: str = Field(default="NEUTRAL", description="Micro execution layer confirmation (Orderflow, Footprint, LTF structure)")
    conflicts: List[str] = Field(default_factory=list, description="Active conflicts discovered between concepts or timeframes")
    blocking_reasons: List[str] = Field(default_factory=list, description="Mandatory conditions that blocked this setup")
    missing_evidence: List[str] = Field(default_factory=list, description="Mandatory evidence missing from analysis")
    entry_zone: Optional[str] = Field(default=None, description="Entry zone or trigger level")
    invalidation_level: Optional[float] = Field(default=None, description="Hard stop loss / structural invalidation price")
    target_level: Optional[float] = Field(default=None, description="Target take profit level")
    risk_reward_ratio: Optional[float] = Field(default=None, description="Mathematical R:R")
    risk_engine_status: Optional[str] = Field(default=None, description="Step 8 RiskEngine outcome (APPROVED, REJECTED, etc.)")
    execution_gate_status: Optional[str] = Field(default=None, description="Step 8 ExecutionGate outcome")
    epistemic_summary: Dict[str, Any] = Field(default_factory=dict, description="Aggregated 4-layer epistemic summary")
    user_summary_en: str = Field(default="", description="Structured user-facing output in English")
    user_summary_hi: str = Field(default="", description="Simple user-facing output in Hindi")
    notification_event_id: Optional[str] = Field(default=None, description="Dispatched CloudStreamEvent ID if confirmed")


class DecisionAuditRecord(BaseModel):
    """Immutable audit record saved in SQLite for historical reproducibility."""

    decision_id: str
    symbol: str
    timeframe: str
    timestamp: float
    final_state: str
    direction: str
    confluence_score: float
    threshold: float
    concept_results_json: str
    evidence_references_json: str
    rule_ids_json: str
    source_provenance_json: str
    epistemic_state_json: str
    conflicts_json: str
    missing_evidence_json: str
    risk_result_json: str
    entry_zone: Optional[str] = None
    invalidation_level: Optional[str] = None
    target_level: Optional[str] = None
    primary_reason: Optional[str] = None
    user_summary_en: Optional[str] = None
    user_summary_hi: Optional[str] = None
    notification_event_id: Optional[str] = None
    created_at: str
