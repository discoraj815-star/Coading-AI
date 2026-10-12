"""Step 18: Universal Provider Registry and Lifecycle Schemas."""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ProviderCategory(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    BACKTEST = "BACKTEST"
    BROKER = "BROKER"
    TRADING_TERMINAL = "TRADING_TERMINAL"
    ORDERFLOW = "ORDERFLOW"
    FOOTPRINT = "FOOTPRINT"
    VOLUME_PROFILE = "VOLUME_PROFILE"
    NEWS = "NEWS"
    ECONOMIC_CALENDAR = "ECONOMIC_CALENDAR"
    NOTIFICATION = "NOTIFICATION"
    VOICE = "VOICE"
    OTHER = "OTHER"


class ProviderLifecycleState(str, Enum):
    DISCOVERED = "DISCOVERED"
    CONFIGURED = "CONFIGURED"
    AUTHENTICATING = "AUTHENTICATING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    DISCONNECTED = "DISCONNECTED"
    RATE_LIMITED = "RATE_LIMITED"
    ERROR = "ERROR"
    DISABLED = "DISABLED"
    REVOKED = "REVOKED"


class TradingProviderExecutionState(str, Enum):
    PAPER_ONLY = "PAPER_ONLY"
    LIVE_PENDING_APPROVAL = "LIVE_PENDING_APPROVAL"
    LIVE_ENABLED = "LIVE_ENABLED"
    LIVE_LOCKED = "LIVE_LOCKED"


class ProviderMetadata(BaseModel):
    """Normalized metadata for all registered universal providers."""

    provider_id: str = Field(..., description="Unique alphanumeric identifier (e.g. delta_exchange, mt5, forex_factory)")
    provider_name: str = Field(..., description="Human-readable provider title")
    provider_type: ProviderCategory = Field(..., description="Provider functional category")
    version: str = Field(default="1.0.0", description="Adapter version string")
    status: ProviderLifecycleState = Field(default=ProviderLifecycleState.DISCOVERED, description="Operational lifecycle state")
    trading_state: TradingProviderExecutionState = Field(
        default=TradingProviderExecutionState.PAPER_ONLY,
        description="Paper vs live execution authorization level",
    )
    capabilities: List[str] = Field(default_factory=list, description="List of declared supported capabilities")
    supported_symbols: List[str] = Field(default_factory=list, description="List of tradable/monitored symbols (or ['*'] for all)")
    supported_markets: List[str] = Field(default_factory=list, description="Crypto, Forex, Metals, Equities, etc.")
    supported_timeframes: List[str] = Field(default_factory=list, description="Supported timeframes (e.g. 1m, 5m, 15m, 1h, 1d)")
    authentication_type: str = Field(default="NONE", description="NONE, API_KEY, HMAC_SHA256, OAUTH, IPC, MT5_TERMINAL")
    configuration_schema: Dict[str, Any] = Field(default_factory=dict, description="Expected configuration keys & types (NO SECRETS)")
    health_status: str = Field(default="HEALTHY", description="HEALTHY, DEGRADED, FAILING, UNKNOWN")
    last_success: Optional[datetime] = Field(default=None, description="Timestamp of last successful query or ping")
    last_error: Optional[str] = Field(default=None, description="Diagnostic message from last failure")
    rate_limit: Optional[str] = Field(default=None, description="Rate limit description (e.g. 10 req/sec)")
    cost_class: str = Field(default="FREE", description="FREE, FREEMIUM, PAID_TIER, SUBSCRIPTION")
    data_quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Empirical quality score (0.0 to 1.0)")
    latency_class: str = Field(default="LOW", description="ULTRA_LOW, LOW, MEDIUM, HIGH, BATCH")
    sandbox_available: bool = Field(default=True, description="Whether testnet/demo environment is available")
    paper_supported: bool = Field(default=True, description="Whether paper simulation is supported")
    live_supported: bool = Field(default=False, description="Whether real live trading or feeds are supported")
    enabled: bool = Field(default=True, description="User toggle to enable or disable provider")
    approved: bool = Field(default=False, description="User approval flag for production participation")
    # Dynamic Broker & Secure Credential Status
    api_key_status: str = Field(default="NOT_CONFIGURED", description="CONFIGURED, MASKED, NOT_CONFIGURED")
    secret_status: str = Field(default="NOT_CONFIGURED", description="CONFIGURED, MASKED, NOT_CONFIGURED")
    masked_api_key: Optional[str] = Field(default=None, description="Masked key display string")
    masked_api_secret: Optional[str] = Field(default=None, description="Masked secret display string")
    last_validation_timestamp: Optional[datetime] = Field(default=None, description="Timestamp of last validation")
    connection_latency_ms: Optional[float] = Field(default=None, description="Measured ping or test latency in ms")
    is_active_broker: bool = Field(default=False, description="Whether currently selected as active broker context")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def execution_state(self) -> str:
        return self.trading_state.value if hasattr(self.trading_state, "value") else str(self.trading_state)


class ConnectionTestResult(BaseModel):
    """Result of a safe, non-destructive connection test operation."""

    success: bool
    provider_id: str
    provider_name: str
    latency_ms: float
    message: str
    capabilities_verified: List[str] = Field(default_factory=list)
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)
