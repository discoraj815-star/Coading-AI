"""Universal learning source schemas for Step 3."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.enums import SourceStatus, SourceType


class SourceMetadata(BaseModel):
    """Metadata and acquisition status for any universal learning source."""

    source_id: str = Field(description="Unique source identifier")
    source_type: SourceType = Field(description="Category of source")
    source_location: str = Field(description="URL, local path, or external identifier")
    title: Optional[str] = Field(default=None, description="Human readable title or document name")
    language: str = Field(default="en", description="Detected or declared source language")
    normalized_language: str = Field(default="en", description="Normalized working language representation")
    permissions: Dict[str, Any] = Field(
        default_factory=dict, description="Authorization, robots compliance, and access limitations"
    )
    acquisition: Dict[str, Any] = Field(
        default_factory=dict, description="Technical acquisition metadata (content-type, size, headers, etc.)"
    )
    processing_status: SourceStatus = Field(
        default=SourceStatus.PENDING, description="Availability and ingestion lifecycle state"
    )
    error_reason: Optional[str] = Field(default=None, description="Reason if source is inaccessible")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SourceResolutionRequest(BaseModel):
    """Request payload to inspect and resolve a source URL or location."""

    source_location: str = Field(description="URL or path to resolve")
    source_type: Optional[SourceType] = Field(default=None, description="Explicit source type override if known")
    auth_token: Optional[str] = Field(default=None, description="Optional authentication credential")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplemental metadata")


class SourceResolutionResult(BaseModel):
    """Result from resolving and validating an external or local source."""

    source_id: str
    source_type: SourceType
    source_location: str
    is_accessible: bool
    status: SourceStatus
    title: Optional[str] = None
    language: str = "en"
    normalized_language: str = "en"
    details: Dict[str, Any] = Field(default_factory=dict)
    error_reason: Optional[str] = None
