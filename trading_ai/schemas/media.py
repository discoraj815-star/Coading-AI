"""Schemas for media ingestion and job metadata."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from trading_ai.schemas.enums import JobStatus, PipelineStage


class JobMetadata(BaseModel):
    """Metadata and processing state for a lecture media job."""

    job_id: str = Field(description="Unique identifier for the lecture job")
    file_hash: str = Field(description="SHA-256 hash of the media file")
    filename: str = Field(description="Original filename of the lecture media")
    file_path: str = Field(description="Path to local stored file")
    duration: float = Field(default=0.0, description="Duration in seconds")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted media streams and codec metadata"
    )
    status: JobStatus = Field(
        default=JobStatus.PENDING, description="Current lifecycle state"
    )
    current_stage: PipelineStage = Field(
        default=PipelineStage.INGESTION, description="Active pipeline stage"
    )
    serial_number: int = Field(
        default=1, description="Latest processing serial number"
    )
    retry_count: int = Field(
        default=0, description="Total retry attempts executed"
    )
    last_error: Optional[str] = Field(
        default=None, description="Details of latest error if failed/retrying"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestionRequest(BaseModel):
    """Request payload to initiate lecture media ingestion."""

    filename: str
    media_type: str = Field(
        default="video", description="Media format category: 'video' or 'audio'"
    )
