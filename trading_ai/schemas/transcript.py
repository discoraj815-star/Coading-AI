"""Schemas for audio processing and timestamped transcripts."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """Timestamped audio segment preserving speech and trading nuance."""

    segment_id: str = Field(description="Unique segment identifier")
    job_id: str = Field(description="Associated lecture job ID")
    start_time: float = Field(description="Segment start time in seconds")
    end_time: float = Field(description="Segment end time in seconds")
    transcript: str = Field(description="Spoken text content")
    language: str = Field(default="en", description="Detected language code")
    speaker: Optional[str] = Field(
        default=None, description="Speaker name or ID if diarization is available"
    )
    important_statements: List[str] = Field(
        default_factory=list, description="Key statements spoken in this segment"
    )
    terminology: List[str] = Field(
        default_factory=list, description="Domain terms uttered (e.g. order block, FVG, liquidity)"
    )
    rules: List[str] = Field(
        default_factory=list, description="Spoken trading rules or conditions"
    )
    examples: List[str] = Field(
        default_factory=list, description="Spoken examples or scenarios"
    )
    uncertainty: List[str] = Field(
        default_factory=list, description="Ambiguities or hedges in spoken words"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Speech recognition confidence"
    )
    # Step 1 Hardening Extensions
    statement_types: Dict[str, str] = Field(
        default_factory=dict,
        description="Epistemic categorization mapping each statement to: RULE, EXAMPLE, EXCEPTION, etc.",
    )
    deictic_phrases: List[str] = Field(
        default_factory=list,
        description="Detected deictic gestures in speech ('this candle', 'look here', 'that zone', 'above this')",
    )

