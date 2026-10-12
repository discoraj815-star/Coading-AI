"""Unit tests for Settings and ModelRouter."""

import pytest
from trading_ai.core.config import Settings
from trading_ai.core.model_router import ModelRole, ModelRouter


def test_settings_defaults():
    settings = Settings()
    assert settings.app_env in ["development", "production", "test"]
    assert settings.token_safety_margin_percent == 0.20
    assert settings.gemini_model_fast_multimodal == "gemini-3.8-flash"
    assert settings.gemini_model_reasoning == "gemini-3.1-pro-preview"


def test_model_router_resolves_all_roles():
    router = ModelRouter()
    roles = [
        ModelRole.FAST_MULTIMODAL,
        ModelRole.VISUAL_ANALYSIS,
        ModelRole.AUDIO_ANALYSIS,
        ModelRole.REASONING,
        ModelRole.KNOWLEDGE_SYNTHESIS,
        ModelRole.EMBEDDING,
        ModelRole.FUTURE_LIVE,
    ]
    for role in roles:
        model_id = router.get_model(role)
        assert isinstance(model_id, str)
        assert len(model_id) > 0


def test_model_router_override():
    router = ModelRouter()
    original = router.get_model(ModelRole.FAST_MULTIMODAL)
    router.override_role(ModelRole.FAST_MULTIMODAL, "custom-gemini-test")
    assert router.get_model(ModelRole.FAST_MULTIMODAL) == "custom-gemini-test"

    # Rest of roles should be untouched
    assert router.get_model(ModelRole.REASONING) == "gemini-3.1-pro-preview"

    # Empty override should raise ValueError
    with pytest.raises(ValueError):
        router.override_role(ModelRole.FAST_MULTIMODAL, "")


def test_model_router_list_mappings():
    router = ModelRouter()
    mappings = router.list_mappings()
    assert "fast_multimodal" in mappings
    assert "reasoning" in mappings
    assert mappings["reasoning"] == "gemini-3.1-pro-preview"
