"""Schemas for evidence-grounded trading knowledge items."""

import uuid
from typing import List, Optional
from pydantic import BaseModel, Field
from trading_ai.schemas.enums import KnowledgeType, KnowledgeValidationStatus
from trading_ai.schemas.evidence import EvidenceTrace


class KnowledgeItem(BaseModel):
    """Structured, evidence-grounded knowledge unit."""

    item_id: str = Field(
        default_factory=lambda: f"know_{uuid.uuid4().hex[:8]}",
        description="Unique knowledge item identifier",
    )
    job_id: str = Field(description="Associated lecture job ID")
    item_type: KnowledgeType = Field(
        description="Type: concept, definition, rule, sub_rule, example, counterexample, exception, condition, relationship, warning, terminology"
    )
    title: str = Field(description="Short human-readable title or label")
    statement: str = Field(description="The canonical formulation of this knowledge item")
    conditions: List[str] = Field(
        default_factory=list, description="Mandatory conditions for rule applicability"
    )
    exceptions: List[str] = Field(
        default_factory=list, description="Exceptions where rule must not trigger"
    )
    reasoning: str = Field(
        default="", description="Epistemic reasoning establishing why this item is valid or uncertain"
    )
    validation_status: KnowledgeValidationStatus = Field(
        default=KnowledgeValidationStatus.OBSERVED,
        description="Epistemic status: OBSERVED, INFERRED, UNCERTAIN, CONTRADICTED, VALIDATED, VERIFIED"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall confidence score"
    )
    start_time: float = Field(
        default=0.0, description="Earliest timestamp associated with this knowledge item"
    )
    end_time: float = Field(
        default=0.0, description="Latest timestamp associated with this knowledge item"
    )
    evidence: List[EvidenceTrace] = Field(
        default_factory=list, description="Linked traces proving this knowledge item"
    )
    parent_id: Optional[str] = Field(
        default=None, description="Optional parent knowledge ID (e.g. for sub-rules or conditions)"
    )
    tags: List[str] = Field(
        default_factory=list, description="Categorization tags (e.g. ['liquidity', 'FVG', 'risk'])"
    )
