"""Schemas for Supervisor/Worker coordination, serial ordering, and commit barriers."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from trading_ai.schemas.enums import ErrorSeverity, PipelineStage, TaskUnitStatus


class TaskUnit(BaseModel):
    """Atomic unit of work assigned a strictly increasing serial number."""

    unit_id: str = Field(description="Unique task unit ID")
    job_id: str = Field(description="Associated lecture job ID")
    serial_number: int = Field(description="Strict monotonic serial number: 1, 2, 3...")
    stage: PipelineStage = Field(description="Pipeline stage this unit executes")
    start_time: float = Field(description="Media segment start time in seconds")
    end_time: float = Field(description="Media segment end time in seconds")
    status: TaskUnitStatus = Field(
        default=TaskUnitStatus.PENDING, description="pending, processing, completed, buffered, committed, failed"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Input arguments and context for this unit"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Output payload produced upon completion"
    )
    retry_count: int = Field(default=0, description="Number of retries attempted")
    max_retries: int = Field(default=3, description="Maximum allowed retry attempts")
    last_error: Optional[str] = Field(default=None, description="Last error message if failed")
    error_severity: Optional[ErrorSeverity] = Field(
        default=None, description="recoverable or permanent"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CommitBarrierState(BaseModel):
    """Tracks the state of the persistent Knowledge Commit Barrier."""

    job_id: str
    highest_committed_serial: int = Field(
        default=0, description="Highest serial number whose results have been committed to permanent knowledge"
    )
    buffered_units: List[int] = Field(
        default_factory=list, description="List of serial numbers waiting behind an unrecovered barrier"
    )
    blocked_on_serial: Optional[int] = Field(
        default=None, description="Failed serial number currently blocking progression"
    )
