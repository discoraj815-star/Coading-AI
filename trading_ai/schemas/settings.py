"""Unified Settings Schemas with Masking and Validation for Trading AI Assistant."""

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class AISettings(BaseModel):
    """Centralized AI model and token configuration."""
    default_model: str = Field(default="gemini-3.8-flash", description="Primary fast multimodal model")
    reasoning_model: str = Field(default="gemini-3.1-pro-preview", description="Deep reasoning model")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0, description="Sampling temperature")
    max_input_tokens: int = Field(default=30000, ge=1000, le=100000, description="Input token budget")
    safety_headroom_pct: float = Field(default=0.20, ge=0.05, le=0.50, description="Token safety headroom (20%)")


class VoiceSettings(BaseModel):
    """JARVIS voice interaction and audio synthesis configuration."""
    auto_tts: bool = Field(default=True, description="Enable automatic spoken responses")
    speech_rate: float = Field(default=1.0, ge=0.5, le=2.0, description="TTS playback speed")
    continuous_mode: bool = Field(default=False, description="Continuous conversational listening")
    wake_word_enabled: bool = Field(default=False, description="Wake-word activation")
    voice_name: str = Field(default="default", description="Browser or backend TTS voice profile")


class TradingSettings(BaseModel):
    """Risk guardrails and execution limits."""
    execution_mode: str = Field(default="MANUAL", description="MANUAL or AUTOMATIC execution gate")
    execution_environment: str = Field(default="PAPER", description="PAPER or LIVE trading environment")
    max_daily_loss_pct: float = Field(default=3.0, ge=0.5, le=10.0, description="Max daily loss stop")
    risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=5.0, description="Risk allocation per trade")
    min_rr_ratio: float = Field(default=2.0, ge=1.0, le=10.0, description="Minimum acceptable risk/reward ratio")
    max_slippage_pips: float = Field(default=1.5, ge=0.1, le=10.0, description="Max allowable slippage in pips")
    require_stop_loss: bool = Field(default=True, description="Strict stop-loss requirement on every trade")
    max_consecutive_losses: int = Field(default=3, ge=1, le=10, description="Max consecutive losses before mandatory cooldown")
    cooldown_minutes: int = Field(default=30, ge=5, le=240, description="Trading cooldown duration in minutes")


class BrokerSettings(BaseModel):
    """Broker routing and credential status."""
    active_broker: str = Field(default="PaperBrokerAdapter", description="Active broker adapter")
    api_key_masked: str = Field(default="***configured***", description="Masked API credential")
    account_id: str = Field(default="ACC-PAPER-001", description="Trading account identifier")
    environment: str = Field(default="sandbox", description="Broker network environment")


class NotificationSettings(BaseModel):
    """Multi-channel dispatch preferences."""
    desktop_alerts: bool = Field(default=True, description="Desktop system notifications")
    sound_alerts: bool = Field(default=True, description="Audio chimes for trade alerts")
    critical_risk_voice_alerts: bool = Field(default=True, description="JARVIS voice alerts on high risk")
    telegram_enabled: bool = Field(default=False, description="Telegram bot integration")
    webhook_url_masked: str = Field(default="", description="Masked webhook endpoint")


class SystemSettings(BaseModel):
    """System runtime and diagnostics configuration."""
    log_level: str = Field(default="INFO", description="Application log level")
    app_env: str = Field(default="development", description="Environment stage")
    wal_mode: bool = Field(default=True, description="SQLite Write-Ahead Logging active")
    version: str = Field(default="1.0.0", description="Trading AI Assistant core version")


class UnifiedSettings(BaseModel):
    """Centralized Settings Hub payload unifying all subsystem configurations."""
    ai: AISettings = Field(default_factory=AISettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    system: SystemSettings = Field(default_factory=SystemSettings)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def mask_sensitive_settings(settings: UnifiedSettings) -> UnifiedSettings:
    """Return a deep copy of settings with secret tokens safely masked."""
    masked = settings.model_copy(deep=True)
    if masked.broker.api_key_masked and masked.broker.api_key_masked != "***configured***":
        masked.broker.api_key_masked = "***configured***"
    if masked.notifications.webhook_url_masked:
        masked.notifications.webhook_url_masked = "***configured***"
    return masked
