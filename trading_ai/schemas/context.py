"""Step 2 schemas: ContextItem, ContextWindow, and ContextSnapshot."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ContextItemSource(str, Enum):
    """Where a context item originated from."""

    TRANSCRIPT = "transcript"
    VISUAL = "visual"
    ALIGNMENT = "alignment"
    EVIDENCE = "evidence"
    KNOWLEDGE = "knowledge"
    NOTES = "notes"
    SUMMARY = "summary"


class ContextItem(BaseModel):
    """A single token-budgeted entry inside the active working context window."""

    item_id: str = Field(description="ID of the source artifact (segment_id, event_id, item_id, etc.)")
    source_type: ContextItemSource = Field(description="Origin table / stage of this context entry")
    serial_number: int = Field(default=1, description="Pipeline serial at which this item was added")
    start_time: float = Field(default=0.0, description="Timestamp this context entry covers (seconds)")
    end_time: float = Field(default=0.0, description="End timestamp for time-bounded items")
    text_content: str = Field(description="Token-countable text representation of the item")
    importance_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Scoring used by context pruner (higher = keep longer)"
    )
    estimated_tokens: int = Field(default=0, description="Pre-computed token estimate")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra key-value pairs (e.g. confidence, speaker, symbol) for retrieval filters"
    )


class ContextWindow(BaseModel):
    """
    Active working context window maintained by ContextManager.

    Tracks which items are in-window (included), which were pruned to save tokens
    (deferred), and carries a running compression summary of pruned content.
    """

    job_id: str
    serial_number: int = 1
    max_tokens: int = Field(default=24000, description="Hard token ceiling for this window")
    current_tokens: int = Field(default=0, description="Sum of estimated_tokens for included items")
    included: List[ContextItem] = Field(
        default_factory=list,
        description="Items currently inside the token budget"
    )
    deferred: List[ContextItem] = Field(
        default_factory=list,
        description="Items pruned from active window but retained in DB"
    )
    compression_summary: str = Field(
        default="",
        description="Running LLM-generated digest of pruned content (token-efficient stand-in)"
    )

    @property
    def headroom_tokens(self) -> int:
        return max(0, self.max_tokens - self.current_tokens)

    @property
    def utilization_pct(self) -> float:
        if self.max_tokens == 0:
            return 0.0
        return round(self.current_tokens / self.max_tokens * 100, 1)


class ContextSnapshot(BaseModel):
    """
    Persisted checkpoint of a ContextWindow for restart-recovery.

    Written to the `context_snapshots` table after every serial commit
    so the ContextManager can reconstruct working context across crashes.
    """

    snapshot_id: str = Field(description="Unique snapshot identifier")
    job_id: str
    serial_number: int
    window_tokens: int
    window_items_json: List[Dict[str, Any]] = Field(default_factory=list)
    pruned_items_json: List[Dict[str, Any]] = Field(default_factory=list)
    compression_summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
