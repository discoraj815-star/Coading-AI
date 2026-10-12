"""Unified AI Brain request, response, and session schemas across modalities (Text, Voice, Multimodal)."""

from enum import Enum
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.schemas.chat import ChatNavigationAction


class InputModality(str, Enum):
    """Input modality through which the user prompt arrived."""

    TEXT = "text"
    VOICE = "voice"
    MULTIMODAL = "multimodal"


class OutputModality(str, Enum):
    """Requested output delivery modality."""

    TEXT = "text"
    VOICE = "voice"
    BOTH = "both"


class UnifiedAudioMetadata(BaseModel):
    """Audio metadata captured from voice gateway or speech-to-text pipeline."""

    sample_rate: Optional[int] = 16000
    duration_seconds: Optional[float] = None
    stt_confidence: float = 1.0
    confidence: Optional[float] = None
    language: str = "en"
    speaker: Optional[str] = None
    device_id: Optional[str] = None
    source: Optional[str] = None
    is_user_voice: bool = True

    def model_post_init(self, __context: Any) -> None:
        if self.confidence is not None:
            self.stt_confidence = self.confidence
        else:
            self.confidence = self.stt_confidence


class UnifiedAIRequest(BaseModel):
    """
    Normalized internal request contract accepted by the Unified AI Brain.
    Both Text Chat and Voice Gateway inputs are converted into this common representation.
    """

    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:10]}")
    session_id: str = Field(default="default_session", description="Conversation session identifier")
    user_id: str = Field(default="default_user", description="Authenticated user ID")
    input_modality: InputModality = Field(default=InputModality.TEXT, description="Input modality (text/voice)")
    output_modality: OutputModality = Field(default=OutputModality.TEXT, description="Desired response modality")
    input_text: str = Field(..., description="Normalized prompt string or transcribed voice text")
    audio_base64: Optional[str] = Field(default=None, description="Optional raw audio payload")
    audio_metadata: Optional[UnifiedAudioMetadata] = Field(default=None, description="Metadata from STT")
    symbol: Optional[str] = Field(default=None, description="Target financial symbol (e.g. EUR/USD, BTC/USD)")
    timeframe: Optional[str] = Field(default=None, description="Chart timeframe (e.g. 15m, 1H)")
    current_tab: Optional[str] = Field(default=None, description="Active UI tab identifier")
    screen_context: Optional[Dict[str, Any]] = Field(default=None, description="Optional active screen metadata")
    market_context: Optional[Dict[str, Any]] = Field(default=None, description="Optional caller-supplied market state")
    requested_action: Optional[str] = Field(default=None, description="Optional explicit action requested")
    timestamp: float = Field(default_factory=lambda: time.time())


class UnifiedAIResponse(BaseModel):
    """
    Canonical response emitted by the Unified AI Brain.
    Can be consumed directly by Text Chat or synthesized into speech for Voice / JARVIS.
    """

    request_id: str
    session_id: str
    response_text: str = Field(..., description="Canonical grounded reasoning text")
    speech_text: str = Field(..., description="Speech-friendly, natural phrasing for TTS synthesis")
    input_modality: InputModality
    output_modality: OutputModality
    intent: str = Field(default="GENERAL", description="Detected intent or command category")
    evidence: List[str] = Field(default_factory=list, description="Grounding data points and evidence traces")
    citations: List[str] = Field(default_factory=list, description="Grounding citations and references")
    navigation_action: Optional[ChatNavigationAction] = None
    navigation_actions: List[ChatNavigationAction] = Field(default_factory=list)
    audio_output_url: Optional[str] = Field(default=None, description="Synthesized TTS speech URL if requested")
    context_summary: str = Field(default="", description="Summary of market and screen context utilized")
    market_context_used: Optional[str] = Field(default=None, description="Market context utilized for reasoning")
    memory_updates: List[str] = Field(default_factory=list, description="Rules, corrections, or memory records updated")
    token_diagnostics: Optional[Dict[str, Any]] = Field(default=None, description="Safe token usage diagnostics")
    action_intent: Optional[str] = Field(default=None, description="Proposed trading or system action if any")
    requires_confirmation: bool = Field(default=False, description="Whether action requires explicit user confirmation")
    safety_status: str = Field(default="SAFE", description="Safety gate status (SAFE, BLOCKED, CONFIRMATION_REQUIRED)")
    timestamp: float = Field(default_factory=lambda: time.time())


class UnifiedSessionState(BaseModel):
    """Tracks cross-modality session state, active symbols, timeframes, and conversational topic."""

    session_id: str
    user_id: str = "default_user"
    active_symbol: str = "EUR/USD"
    active_timeframe: str = "15m"
    active_tab: str = "tab-notes"
    last_topic: Optional[str] = None
    last_intent: Optional[str] = None
    last_action_proposed: Optional[str] = None
    pending_action_intent: Optional[str] = None
    pending_action_symbol: Optional[str] = None
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: float = Field(default_factory=lambda: time.time())
