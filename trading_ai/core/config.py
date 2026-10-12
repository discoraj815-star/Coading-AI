"""Application configuration loaded from environment and .env."""

import os
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Trading AI Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini API Configuration
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")

    # Gemini Model Router Mappings (Logical Roles)
    # Updated to current 2026 production model IDs.
    # Flash = gemini-3.8-flash (fastest, cost-efficient multimodal)
    # Reasoning = gemini-3.1-pro-preview (deep reasoning, complex conflicts only)
    # Embedding = gemini-embedding-2 (current Gemini Embedding 2 flagship)
    gemini_model_fast_multimodal: str = Field(
        default="gemini-3.8-flash", alias="GEMINI_MODEL_FAST_MULTIMODAL"
    )
    gemini_model_visual_analysis: str = Field(
        default="gemini-3.8-flash", alias="GEMINI_MODEL_VISUAL_ANALYSIS"
    )
    gemini_model_audio_analysis: str = Field(
        default="gemini-3.8-flash", alias="GEMINI_MODEL_AUDIO_ANALYSIS"
    )
    gemini_model_reasoning: str = Field(
        default="gemini-3.1-pro-preview", alias="GEMINI_MODEL_REASONING"
    )
    gemini_model_knowledge_synthesis: str = Field(
        # Default Flash for cost efficiency; orchestrator escalates to Pro only when needed
        default="gemini-3.8-flash", alias="GEMINI_MODEL_KNOWLEDGE_SYNTHESIS"
    )
    gemini_model_embedding: str = Field(
        default="gemini-embedding-2", alias="GEMINI_MODEL_EMBEDDING"
    )
    gemini_model_future_live: str = Field(
        # Leave configurable — no hardcoded live model until verified
        default="gemini-3.8-flash", alias="GEMINI_MODEL_FUTURE_LIVE"
    )

    # System Configuration
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    db_path: str = Field(default="data/trading_ai.db", alias="DB_PATH")
    upload_dir: str = Field(default="data/uploads", alias="UPLOAD_DIR")
    frames_dir: str = Field(default="data/frames", alias="FRAMES_DIR")

    # Private Security & Authentication Configuration
    auth_enabled: bool = Field(default=False, alias="AUTH_ENABLED")
    auth_token: str = Field(default="", alias="AUTH_TOKEN")
    session_secret: str = Field(default="", alias="SESSION_SECRET")

    # Token Management
    token_default_input_budget: int = Field(
        default=30000, alias="TOKEN_DEFAULT_INPUT_BUDGET"
    )
    token_default_output_reserve: int = Field(
        default=4000, alias="TOKEN_DEFAULT_OUTPUT_RESERVE"
    )
    token_safety_margin_percent: float = Field(
        default=0.20, alias="TOKEN_SAFETY_MARGIN_PERCENT"
    )
    max_chunk_duration_seconds: int = Field(
        default=60, alias="MAX_CHUNK_DURATION_SECONDS"
    )

    def ensure_directories(self) -> None:
        """Ensure necessary storage directories exist."""
        for path_str in [self.data_dir, self.upload_dir, self.frames_dir]:
            Path(path_str).mkdir(parents=True, exist_ok=True)
        db_parent = Path(self.db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
