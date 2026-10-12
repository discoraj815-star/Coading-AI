"""Schemas for user commands, corrections, and knowledge versioning."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import CommandIntent, KnowledgeValidationStatus


class UserCommand(BaseModel):
    """Structured representation of a user command instruction."""

    command_id: str = Field(description="Unique ID for this command invocation")
    raw_text: str = Field(description="Raw user command string")
    intent: CommandIntent = Field(description="Interpreted intent")
    target_knowledge_id: Optional[str] = Field(
        default=None, description="Target knowledge item ID if referring to an existing item"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured extracted entities: rule, conditions, exceptions, replacement, reason",
    )
    source: str = Field(default="user", description="Provenance marker")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CommandParseResult(BaseModel):
    """Result of parsing and routing a user command."""

    command: UserCommand
    parsed_success: bool = True
    suggested_action: str = Field(description="Action name: e.g. 'save_rule', 'apply_correction', 'explain'")
    explanation: str = Field(description="Brief explanation of how the command was understood")


class RuleSaveRequest(BaseModel):
    """Explicit request payload to save or remember a trading rule."""

    rule: str = Field(description="The primary formulation of the trading rule")
    conditions: Optional[List[str]] = Field(
        default=None,
        description="Specific conditions required for the rule to apply. Marked [MISSING] if unspecified.",
    )
    exceptions: Optional[List[str]] = Field(
        default=None,
        description="Specific invalidation or exception scenarios. Marked [NONE_SPECIFIED] if unspecified.",
    )
    source: str = Field(default="user", description="Origin source reference")
    evidence: Optional[str] = Field(default=None, description="Direct supporting evidence text or citation")
    job_id: Optional[str] = Field(default=None, description="Associated job ID if linked to a study/lecture")
    created_by: str = Field(default="user", description="Author or actor")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: KnowledgeValidationStatus = Field(default=KnowledgeValidationStatus.VALIDATED)
    version: int = Field(default=1)
    relationships: List[str] = Field(default_factory=list, description="Related concepts, tags or rules")
    tags: List[str] = Field(default_factory=list)


class CorrectionRequest(BaseModel):
    """User correction request modifying an existing or ambiguous knowledge item."""

    target_knowledge_id: Optional[str] = Field(
        default=None, description="Specific item ID to correct. If omitted, target will be inferred from query."
    )
    query: Optional[str] = Field(
        default=None, description="Search hint or concept name if item ID is not directly known"
    )
    correction_statement: str = Field(description="The corrected canonical formulation")
    reason: Optional[str] = Field(default=None, description="Why the previous interpretation was flawed")
    conditions: Optional[List[str]] = Field(default=None, description="Refined application conditions")
    exceptions: Optional[List[str]] = Field(default=None, description="Refined exception cases")
    job_id: Optional[str] = Field(default=None, description="Associated job ID")
    corrected_by: str = Field(default="user", description="Actor who submitted correction")


class KnowledgeVersion(BaseModel):
    """
    Immutable version audit entry tracking knowledge evolution and corrections.

    Preserves:
    - what was previously believed
    - what the user corrected
    - when it was corrected
    - what source/evidence caused the correction
    - current active interpretation
    - confidence and status
    """

    version_id: str = Field(description="Unique ID for this revision entry")
    item_id: str = Field(description="Canonical knowledge item ID")
    version_number: int = Field(description="Sequential revision number (1, 2, 3...)")
    job_id: str = Field(description="Associated job ID")
    title: str = Field(description="Title or label at this revision")
    statement: str = Field(description="Rule or concept statement at this revision")
    conditions: List[str] = Field(default_factory=list, description="Associated conditions")
    exceptions: List[str] = Field(default_factory=list, description="Associated exceptions")
    reasoning: str = Field(default="", description="Epistemic reasoning at this revision")
    validation_status: KnowledgeValidationStatus = Field(default=KnowledgeValidationStatus.VALIDATED)
    confidence: float = Field(default=1.0)
    evidence_json: str = Field(default="[]", description="JSON serialized evidence traces")
    correction_notes: Optional[str] = Field(
        default=None, description="User explanation or diff notes that triggered this revision"
    )
    corrected_by: str = Field(default="system", description="User or system actor who created this revision")
    corrected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    superseded_by_version: Optional[int] = Field(
        default=None, description="Version number that superseded this entry, if inactive"
    )
    is_active: bool = Field(default=True, description="True if this is the active interpretation")
    provenance: Dict[str, Any] = Field(
        default_factory=dict, description="Full audit trail linking source, prior versions, and evidence"
    )
