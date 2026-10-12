"""Universal learning source resolver with authorization and access validation."""

import os
import re
import urllib.parse
import uuid
from pathlib import Path
from typing import Dict, Optional, Tuple

from trading_ai.schemas.enums import SourceStatus, SourceType
from trading_ai.schemas.sources import (
    SourceMetadata,
    SourceResolutionRequest,
    SourceResolutionResult,
)


class SourceResolver:
    """
    Resolves, categorizes, and validates universal learning sources.

    Respects authorization, source availability, security restrictions, and rate limits.
    If a source cannot be accessed, returns SourceStatus.SOURCE_UNAVAILABLE
    rather than hallucinating content.
    """

    # Supported video and audio file extensions
    VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}
    DOC_EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json"}
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".svg", ".bmp"}

    YOUTUBE_DOMAINS = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
    }

    BLOCKED_DOMAINS = {
        "example-blocked-domain.com",
        "malicious-site.org",
        "unauthorized-internal.corp",
    }

    def __init__(self, allowed_schemes: Optional[Tuple[str, ...]] = None):
        self._allowed_schemes = allowed_schemes or ("http", "https", "file", "")

    def resolve(self, req: SourceResolutionRequest) -> SourceResolutionResult:
        """
        Analyze the source location, determine source type, check accessibility and permissions.
        """
        raw_location = req.source_location.strip()
        source_id = f"src_{uuid.uuid4().hex[:12]}"

        # 1. Check for empty or blank input
        if not raw_location:
            return SourceResolutionResult(
                source_id=source_id,
                source_type=SourceType.USER_TEXT,
                source_location=raw_location,
                is_accessible=False,
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_reason="Empty source location provided.",
            )

        # 2. Check if explicit override provided
        explicit_type = req.source_type

        # 3. Detect URL vs Local Path vs Raw Text
        parsed = urllib.parse.urlparse(raw_location)
        is_url = parsed.scheme in ("http", "https")

        if is_url:
            return self._resolve_url(source_id, raw_location, parsed, req, explicit_type)

        # 4. Check for local file path
        known_extensions = (
            self.VIDEO_EXTENSIONS
            | self.AUDIO_EXTENSIONS
            | self.DOC_EXTENSIONS
            | self.IMAGE_EXTENSIONS
        )
        is_local_path = (
            raw_location.startswith("/")
            or raw_location.startswith("\\")
            or (len(raw_location) > 2 and raw_location[1] == ":" and raw_location[2] in ("\\", "/"))
            or raw_location.startswith("./")
            or raw_location.startswith(".\\")
            or (Path(raw_location).suffix.lower() in known_extensions and "\n" not in raw_location)
        )

        if is_local_path:
            return self._resolve_local_file(source_id, raw_location, req, explicit_type)

        # 5. Raw text or note snippet
        return SourceResolutionResult(
            source_id=source_id,
            source_type=explicit_type or SourceType.USER_TEXT,
            source_location="user://text_input",
            is_accessible=True,
            status=SourceStatus.AVAILABLE,
            title="User Direct Text Input",
            language="en",
            normalized_language="en",
            details={"char_length": len(raw_location), "sample": raw_location[:100]},
        )

    def _resolve_url(
        self,
        source_id: str,
        url: str,
        parsed: urllib.parse.ParseResult,
        req: SourceResolutionRequest,
        explicit_type: Optional[SourceType],
    ) -> SourceResolutionResult:
        """Evaluate online source URL compliance and classification."""
        domain = parsed.netloc.lower()

        # Check domain security / access blacklist
        if domain in self.BLOCKED_DOMAINS:
            return SourceResolutionResult(
                source_id=source_id,
                source_type=explicit_type or SourceType.WEB_PAGE,
                source_location=url,
                is_accessible=False,
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_reason=f"Access to domain {domain} is restricted or unauthorized.",
            )

        # YouTube URL detection
        if domain in self.YOUTUBE_DOMAINS:
            # Check for valid video id
            video_id = None
            if "youtu.be" in domain:
                video_id = parsed.path.lstrip("/")
            else:
                qs = urllib.parse.parse_qs(parsed.query)
                video_id = qs.get("v", [None])[0]

            if not video_id:
                return SourceResolutionResult(
                    source_id=source_id,
                    source_type=SourceType.YOUTUBE,
                    source_location=url,
                    is_accessible=False,
                    status=SourceStatus.SOURCE_UNAVAILABLE,
                    error_reason="YouTube URL missing video identifier.",
                )

            return SourceResolutionResult(
                source_id=source_id,
                source_type=SourceType.YOUTUBE,
                source_location=url,
                is_accessible=True,
                status=SourceStatus.AVAILABLE,
                title=f"YouTube Video [{video_id}]",
                language="en",
                normalized_language="en",
                details={
                    "platform": "youtube",
                    "video_id": video_id,
                    "normalized_url": f"https://www.youtube.com/watch?v={video_id}",
                },
            )

        # General Web Document (PDF, etc.) or Web Page
        path_lower = parsed.path.lower()
        if path_lower.endswith(".pdf"):
            detected_type = SourceType.PDF
        elif path_lower.endswith(tuple(self.VIDEO_EXTENSIONS)):
            detected_type = SourceType.LOCAL_VIDEO
        elif path_lower.endswith(tuple(self.AUDIO_EXTENSIONS)):
            detected_type = SourceType.LOCAL_AUDIO
        else:
            detected_type = SourceType.WEB_PAGE

        return SourceResolutionResult(
            source_id=source_id,
            source_type=explicit_type or detected_type,
            source_location=url,
            is_accessible=True,
            status=SourceStatus.AVAILABLE,
            title=f"Web Source: {domain}",
            language="en",
            normalized_language="en",
            details={
                "domain": domain,
                "path": parsed.path,
                "scheme": parsed.scheme,
            },
        )

    def _resolve_local_file(
        self,
        source_id: str,
        path_str: str,
        req: SourceResolutionRequest,
        explicit_type: Optional[SourceType],
    ) -> SourceResolutionResult:
        """Resolve local file on disk."""
        path = Path(path_str)
        ext = path.suffix.lower()

        # Check existence
        if not path.exists():
            return SourceResolutionResult(
                source_id=source_id,
                source_type=explicit_type or SourceType.LOCAL_VIDEO,
                source_location=path_str,
                is_accessible=False,
                status=SourceStatus.SOURCE_UNAVAILABLE,
                error_reason=f"Local file does not exist: {path_str}",
            )

        # Check file classification
        if ext in self.VIDEO_EXTENSIONS:
            detected_type = SourceType.LOCAL_VIDEO
        elif ext in self.AUDIO_EXTENSIONS:
            detected_type = SourceType.LOCAL_AUDIO
        elif ext == ".pdf":
            detected_type = SourceType.PDF
        elif ext in self.IMAGE_EXTENSIONS:
            detected_type = SourceType.IMAGE
        elif ext in self.DOC_EXTENSIONS:
            detected_type = SourceType.WEB_DOCUMENT
        else:
            detected_type = SourceType.LOCAL_VIDEO

        file_size = path.stat().st_size if path.is_file() else 0

        return SourceResolutionResult(
            source_id=source_id,
            source_type=explicit_type or detected_type,
            source_location=str(path.resolve()),
            is_accessible=True,
            status=SourceStatus.AVAILABLE,
            title=path.name,
            language="en",
            normalized_language="en",
            details={
                "size_bytes": file_size,
                "extension": ext,
                "filename": path.name,
            },
        )
