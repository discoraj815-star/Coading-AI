"""Step 18: Real Lecture Performance Benchmarking Foundation Schemas."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BenchmarkStage(str, Enum):
    QUEUED = "QUEUED"
    INGESTION = "INGESTION"
    AUDIO = "AUDIO"
    VISUAL = "VISUAL"
    ALIGNMENT = "ALIGNMENT"
    KNOWLEDGE = "KNOWLEDGE"
    MEMORY_COMMIT = "MEMORY_COMMIT"
    FINALIZING = "FINALIZING"
    COMPLETE = "COMPLETE"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class TimelineCheckpoint(BaseModel):
    """Immutable persistent processing timeline milestone."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    stage: str
    description: str
    elapsed_seconds: float = 0.0


class LectureBenchmarkRecord(BaseModel):
    """Comprehensive performance instrumentation record for real user lectures."""

    lecture_id: str = Field(..., description="Unique lecture job identifier")
    file_name: str = Field(default="", description="Original video filename")
    file_hash: str = Field(default="", description="SHA-256 hash of media file")
    file_size_bytes: int = Field(default=0, description="Size in bytes")
    duration_seconds: float = Field(default=0.0, description="Media runtime in seconds")
    resolution: str = Field(default="1280x720", description="Video dimensions")
    fps: float = Field(default=30.0, description="Video framerate")
    audio_duration_seconds: float = Field(default=0.0, description="Extracted audio duration")

    # Processing Stage Timestamps (ISO formatted or seconds)
    timings: Dict[str, Optional[float]] = Field(
        default_factory=lambda: {
            "ingestion_start": None,
            "ingestion_end": None,
            "audio_start": None,
            "audio_end": None,
            "visual_start": None,
            "visual_end": None,
            "alignment_start": None,
            "alignment_end": None,
            "knowledge_start": None,
            "knowledge_end": None,
            "finalization_start": None,
            "finalization_end": None,
        }
    )

    # Detailed Duration Metrics
    metrics: Dict[str, float] = Field(
        default_factory=lambda: {
            "total_processing_time": 0.0,
            "audio_processing_time": 0.0,
            "visual_processing_time": 0.0,
            "alignment_time": 0.0,
            "knowledge_time": 0.0,
            "gemini_time": 0.0,
            "queue_wait_time": 0.0,
            "retry_time": 0.0,
            "database_time": 0.0,
        }
    )

    # Granular Processing Counts
    counts: Dict[str, int] = Field(
        default_factory=lambda: {
            "total_frames": 0,
            "sampled_frames": 0,
            "processed_frames": 0,
            "skipped_frames": 0,
            "audio_segments": 0,
            "visual_segments": 0,
            "aligned_segments": 0,
            "gemini_calls": 0,
            "failed_calls": 0,
            "retry_count": 0,
            "tokens_input": 0,
            "tokens_output": 0,
        }
    )

    # Estimated Cost Tracking
    estimated_cost_usd: float = Field(default=0.0, description="Calculated API cost based on token usage")

    # Extracted Knowledge Metrics
    knowledge_metrics: Dict[str, int] = Field(
        default_factory=lambda: {
            "concepts_extracted": 0,
            "rules_extracted": 0,
            "evidence_items": 0,
            "exceptions": 0,
            "examples": 0,
            "uncertainties": 0,
            "needs_review_items": 0,
        }
    )

    # Epistemic Quality Metrics
    quality_metrics: Dict[str, Any] = Field(
        default_factory=lambda: {
            "transcript_completeness": 1.0,
            "visual_evidence_count": 0,
            "cross_modal_alignment_count": 0,
            "supporting_relationships": 0,
            "contradicting_relationships": 0,
            "ambiguous_relationships": 0,
        }
    )

    # Persistent Timeline
    timeline: List[TimelineCheckpoint] = Field(default_factory=list)

    # Bottleneck Breakdown (Percentages summing to 100%)
    bottlenecks: Dict[str, float] = Field(default_factory=dict)

    current_stage: BenchmarkStage = Field(default=BenchmarkStage.QUEUED)
    is_complete: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
