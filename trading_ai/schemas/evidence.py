"""Schemas for multi-modal evidence linking and traceability."""

from typing import List, Optional
from pydantic import BaseModel, Field


class EvidenceTrace(BaseModel):
    """Grounding evidence linking statements back to exact audio and visual sources."""

    evidence_id: str = Field(description="Unique evidence identifier")
    job_id: str = Field(description="Associated lecture job ID")
    start_time: float = Field(description="Evidence start timestamp in seconds")
    end_time: float = Field(description="Evidence end timestamp in seconds")
    audio_quote: Optional[str] = Field(
        default=None, description="Exact spoken quote or transcript excerpt"
    )
    visual_reference: Optional[str] = Field(
        default=None, description="Frame reference, visual event ID, or visual description"
    )
    extracted_statement: str = Field(
        description="The statement directly backed by this evidence"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in evidence association"
    )
    notes: Optional[str] = Field(
        default=None, description="Supplementary context or observer notes"
    )
