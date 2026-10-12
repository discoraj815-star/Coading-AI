"""Base plugin interface and contracts for Step 15."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from trading_ai.plugins.manifest import PluginHealthStatus, PluginManifest, PluginStatus

logger = logging.getLogger(__name__)


class BasePlugin(ABC):
    """Abstract base class for all Trading AI plugins."""

    def __init__(self, manifest: PluginManifest):
        self.manifest = manifest
        self.status: PluginStatus = PluginStatus.LOADED
        self.health_status: PluginHealthStatus = PluginHealthStatus.HEALTHY
        self.consecutive_failures: int = 0
        self.last_error: Optional[str] = None
        self._config: Dict[str, Any] = dict(manifest.config)

    @property
    def plugin_id(self) -> str:
        return self.manifest.plugin_id

    @property
    def name(self) -> str:
        return self.manifest.name

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize plugin resources and validate configuration."""
        pass

    @abstractmethod
    def start(self) -> bool:
        """Activate plugin."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Gracefully deactivate plugin and release resources."""
        pass

    def health_check(self) -> Dict[str, Any]:
        """Inspect and return current plugin health metrics."""
        return {
            "plugin_id": self.plugin_id,
            "status": self.status.value,
            "health_status": self.health_status.value,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
        }

    @abstractmethod
    def execute_action(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Safely execute a declared plugin action within isolation boundaries."""
        pass

    def record_success(self) -> None:
        """Reset consecutive failure counter on successful execution."""
        self.consecutive_failures = 0
        if self.health_status in (PluginHealthStatus.DEGRADED, PluginHealthStatus.FAILING):
            self.health_status = PluginHealthStatus.HEALTHY
        self.last_error = None

    def record_failure(self, error_message: str) -> None:
        """Increment failure counter and transition health state on failure."""
        self.consecutive_failures += 1
        self.last_error = error_message
        if self.consecutive_failures >= 5:
            self.health_status = PluginHealthStatus.UNHEALTHY
            self.status = PluginStatus.ERROR
        elif self.consecutive_failures >= 3:
            self.health_status = PluginHealthStatus.FAILING
        else:
            self.health_status = PluginHealthStatus.DEGRADED


class MarketDataPlugin(BasePlugin):
    """Specialized base class for market data feeder plugins."""

    @abstractmethod
    def fetch_snapshot(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        pass


class NewsPlugin(BasePlugin):
    """Specialized base class for news and economic calendar plugins."""

    @abstractmethod
    def fetch_upcoming_events(self, currency: Optional[str] = None) -> Dict[str, Any]:
        pass


class TechnicalAnalysisPlugin(BasePlugin):
    """Specialized base class for technical indicators and quantitative algorithms."""

    @abstractmethod
    def compute_indicators(self, symbol: str, candles: list) -> Dict[str, Any]:
        pass


class BrokerReadPlugin(BasePlugin):
    """Specialized base class for read-only broker data plugins.

    HARD TRADING SAFETY INVARIANT:
    A Broker plugin must NEVER execute trades directly.
    Execution must strictly flow through CoreTradeLogic -> RiskEngine -> ExecutionGate.
    """

    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_open_positions(self) -> list:
        pass
