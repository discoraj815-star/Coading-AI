"""Schemas for structured multimodal lecture notes."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TimestampedNoteEntry(BaseModel):
    """Note item with explicit evidence timestamps."""

    title: str
    content: str
    start_time: float
    end_time: float
    evidence_ids: List[str] = Field(default_factory=list)


class LectureNotes(BaseModel):
    """Comprehensive structured notes synthesized from multimodal evidence."""

    job_id: str = Field(description="Associated lecture job ID")
    title: str = Field(default="Multimodal Lecture Analysis Notes")
    overview: str = Field(description="1. High-level synthesis of lecture contents and primary thesis")
    concepts: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="2. Core theoretical concepts introduced"
    )
    definitions: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="3. Formal trading terminology definitions"
    )
    rules: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="4. Spoken and visually demonstrated operational rules"
    )
    examples: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="5. Concrete trade scenarios and historical examples"
    )
    exceptions: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="6. Market edge-cases, invalidations, and counterexamples"
    )
    visual_explanations: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="7. Visual breakdown of charts, drawings, zones, and candles"
    )
    important_statements: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="8. High-conviction teacher quotes and principles"
    )
    relationships: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="9. Cause-and-effect relationships between indicators and price action"
    )
    uncertainty: List[TimestampedNoteEntry] = Field(
        default_factory=list, description="10. Caveats, discretionary elements, and unresolved questions"
    )
    evidence_timestamps: List[Dict[str, Any]] = Field(
        default_factory=list, description="11. Complete chronological index of evidence timestamps"
    )
