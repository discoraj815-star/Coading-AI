"""Configurable ModelRouter mapping logical AI roles to actual Gemini model IDs."""

from enum import Enum
from typing import Dict, Optional
from trading_ai.core.config import Settings, get_settings


class ModelRole(str, Enum):
    """Logical model roles defined across the architecture."""

    FAST_MULTIMODAL = "fast_multimodal"
    VISUAL_ANALYSIS = "visual_analysis"
    AUDIO_ANALYSIS = "audio_analysis"
    REASONING = "reasoning"
    KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
    EMBEDDING = "embedding"
    FUTURE_LIVE = "future_live"


class ModelRouter:
    """Routes logical model roles to configured Gemini model identifiers.

    Prevents scattering model names in business logic and enables changing
    models dynamically via configuration/environment without modifying
    pipeline code.
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._role_mapping: Dict[ModelRole, str] = {
            ModelRole.FAST_MULTIMODAL: self._settings.gemini_model_fast_multimodal,
            ModelRole.VISUAL_ANALYSIS: self._settings.gemini_model_visual_analysis,
            ModelRole.AUDIO_ANALYSIS: self._settings.gemini_model_audio_analysis,
            ModelRole.REASONING: self._settings.gemini_model_reasoning,
            ModelRole.KNOWLEDGE_SYNTHESIS: self._settings.gemini_model_knowledge_synthesis,
            ModelRole.EMBEDDING: self._settings.gemini_model_embedding,
            ModelRole.FUTURE_LIVE: self._settings.gemini_model_future_live,
        }

    def get_model(self, role: ModelRole) -> str:
        """Resolve a logical ModelRole to its actual Gemini model identifier."""
        model_id = self._role_mapping.get(role)
        if not model_id:
            raise ValueError(f"No configured model mapping found for role: {role}")
        return model_id

    def override_role(self, role: ModelRole, model_id: str) -> None:
        """Dynamically override a model mapping for experimentation or testing."""
        if not model_id.strip():
            raise ValueError("Model ID cannot be empty.")
        self._role_mapping[role] = model_id

    def list_mappings(self) -> Dict[str, str]:
        """Return a dictionary of all current role-to-model mappings."""
        return {role.value: model_id for role, model_id in self._role_mapping.items()}
