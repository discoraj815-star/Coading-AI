"""Schemas for AI Chat, Multi-Turn Conversation, and Application Guide."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatNavigationAction(BaseModel):
    """Action emitted by AI Chat to direct user to a specific UI tab or panel."""

    tab_id: str = Field(..., description="ID of destination tab (e.g. 'tab-market', 'tab-step17')")
    label: str = Field(..., description="Human-readable button text (e.g. 'Go to Tab 9: Live Market')")
    element_selector: Optional[str] = Field(None, description="CSS selector of element to highlight/scroll to")


class ChatMessageRequest(BaseModel):
    """User prompt to the AI Assistant."""

    message: str = Field(..., description="User question, command, or inquiry")
    session_id: str = Field(default="default_session", description="Conversation session identifier")
    symbol: str = Field(default="EUR/USD", description="Currently active or queried financial symbol")
    current_tab: str = Field(default="tab-notes", description="Active tab ID user is currently viewing")
    include_audio: bool = Field(default=False, description="Whether to generate TTS audio mock")


class ChatMessageResponse(BaseModel):
    """Response from AI Assistant with evidence and optional navigation trigger."""

    session_id: str
    message_id: str
    user_message: str
    ai_message: str
    answer: str = ""
    role: str = "assistant"
    intent: str = "GENERAL"
    evidence: List[str] = Field(default_factory=list)
    navigation_action: Optional[ChatNavigationAction] = None
    navigation_actions: List[ChatNavigationAction] = Field(default_factory=list)
    context_summary: str = ""
    timestamp: float

    @property
    def reply(self) -> str:
        return self.ai_message or self.answer

    @property
    def response_text(self) -> str:
        return self.reply

    @property
    def speech_text(self) -> str:
        return self.reply


class ChatHistoryTurn(BaseModel):
    """Single turn in a conversation history thread."""

    message_id: str
    role: str  # "user" or "assistant"
    text: str
    timestamp: float
    evidence: List[str] = Field(default_factory=list)
    navigation_action: Optional[ChatNavigationAction] = None


class ChatHistoryResponse(BaseModel):
    """Full message thread history for a session."""

    session_id: str
    turns: List[ChatHistoryTurn] = Field(default_factory=list)


class AppFeatureItem(BaseModel):
    """Authoritative metadata describing an application feature or UI location."""

    feature_id: str
    name: str
    description: str
    tab_id: str
    tab_name: str
    panel_id: Optional[str] = None
    status: str = "AVAILABLE"  # "AVAILABLE", "NOT_IMPLEMENTED", "CONFIGURED", etc.
    how_to_access: str
    related_features: List[str] = Field(default_factory=list)


class AppGuideResponse(BaseModel):
    """Structured catalog of application layout, features, and navigation targets."""

    total_tabs: int = 18
    overview: str = ""
    tour_summary: str = ""
    tabs: List[Dict[str, Any]] = Field(default_factory=list)
    features: List[AppFeatureItem] = Field(default_factory=list)

