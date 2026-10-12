"""Schemas for explicit Audio/Visual alignment layer."""

from typing import List, Optional
from pydantic import BaseModel, Field
from trading_ai.schemas.enums import (
    AlignmentRelationship,
    CrossModalConflictType,
    ReferenceResolutionStatus,
)
from trading_ai.schemas.transcript import TranscriptSegment
from trading_ai.schemas.visual import VisualEvent, VisualRegion


class CrossModalConflict(BaseModel):
    """Detailed discrepancy between audio statement and visual chart reality."""

    conflict_type: str = Field(
        default="NONE",
        description="Type of conflict: DIRECTIONAL_CONTRADICTION, TIMEFRAME_MISMATCH, LEVEL_DISCREPANCY, UNRESOLVED_REFERENCE, etc.",
    )
    audio_claim: str = Field(description="What was claimed in speech")
    visual_evidence: str = Field(description="What was actually observed on screen")
    severity: str = Field(default="MEDIUM", description="HIGH, MEDIUM, LOW")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Conflict assessment confidence")


class DeicticResolution(BaseModel):
    """Grounding resolution for deictic/pointing words ('this candle', 'here', 'that level')."""

    phrase: str = Field(description="The deictic phrase in speech")
    status: str = Field(
        default="NOT_APPLICABLE",
        description="Status: RESOLVED, PARTIALLY_RESOLVED, REFERENCE_UNRESOLVED, NOT_APPLICABLE",
    )
    target_region: Optional[VisualRegion] = Field(
        default=None, description="The identified visual chart region corresponding to the phrase"
    )
    target_description: Optional[str] = Field(
        default=None, description="Verbal description of grounded target"
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Resolution confidence")


class AudioVisualAlignment(BaseModel):
    """Correlates what was said and what was shown over a temporal window."""

    alignment_id: str = Field(description="Unique alignment identifier")
    job_id: str = Field(description="Associated lecture job ID")
    start_time: float = Field(description="Window start time in seconds")
    end_time: float = Field(description="Window end time in seconds")
    audio: Optional[TranscriptSegment] = Field(
        default=None, description="What was spoken during this window"
    )
    visual: List[VisualEvent] = Field(
        default_factory=list, description="What was shown on screen during this window"
    )
    relationship: AlignmentRelationship = Field(
        description="Relationship: supporting, contradicting, unrelated, partially_supporting, ambiguous"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in alignment classification"
    )
    reasoning: str = Field(
        default="", description="Explanatory justification for the assigned relationship"
    )
    # Step 1 Hardening Extensions
    conflicts: List[CrossModalConflict] = Field(
        default_factory=list, description="Cross-modal discrepancies between audio claim and visual evidence"
    )
    deictic_resolutions: List[DeicticResolution] = Field(
        default_factory=list, description="Resolved deictic references linking phrases to visual objects"
    )
    temporal_sequence: Optional[str] = Field(
        default=None, description="Temporal relationship: BEFORE, CHANGE, AFTER, or PROGRESSIVE"
    )

