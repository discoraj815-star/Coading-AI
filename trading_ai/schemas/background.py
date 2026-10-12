"""Schemas for background jobs and process tracking."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import JobStatus, PipelineStage


class BackgroundJobState(BaseModel):
    """
    Persisted state of a background processing job.

    Enables crash-recovery, process tracking, pausing, resuming, and progress monitoring.
    """

    job_id: str = Field(description="Unique job identifier")
    source_id: str = Field(description="Identifier of source being processed")
    status: JobStatus = Field(default=JobStatus.QUEUED, description="Lifecycle status")
    progress: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Calculated completion percentage (0-100)"
    )
    current_stage: PipelineStage = Field(
        default=PipelineStage.SOURCE, description="Discrete stage currently active"
    )
    current_serial: int = Field(default=0, description="Serial number of active task unit")
    total_units: int = Field(default=1, description="Total units/chunks in job")
    completed_units: int = Field(default=0, description="Successfully committed units")
    failed_units: int = Field(default=0, description="Units with errors")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[str] = Field(default=None, description="Current or last error message")
    retry_count: int = Field(default=0, description="Total retries attempted")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom job metadata")
