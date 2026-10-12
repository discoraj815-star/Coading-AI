"""Step 15 Plugin Architecture subsystem."""

from trading_ai.plugins.manifest import (
    FORBIDDEN_CAPABILITIES,
    PluginCapability,
    PluginCategory,
    PluginHealthStatus,
    PluginManifest,
    PluginStatus,
)
from trading_ai.plugins.base import (
    BasePlugin,
    BrokerReadPlugin,
    MarketDataPlugin,
    NewsPlugin,
    TechnicalAnalysisPlugin,
)
from trading_ai.plugins.registry import (
    PluginPermissionDenied,
    PluginRegistry,
    TradingSafetyViolation,
)
from trading_ai.plugins.samples import (
    MockBrokerReadPlugin,
    MockForexFactoryNewsPlugin,
    MockTechnicalAnalysisPlugin,
)

__all__ = [
    "FORBIDDEN_CAPABILITIES",
    "PluginCapability",
    "PluginCategory",
    "PluginHealthStatus",
    "PluginManifest",
    "PluginStatus",
    "BasePlugin",
    "BrokerReadPlugin",
    "MarketDataPlugin",
    "NewsPlugin",
    "TechnicalAnalysisPlugin",
    "PluginRegistry",
    "PluginPermissionDenied",
    "TradingSafetyViolation",
    "MockBrokerReadPlugin",
    "MockForexFactoryNewsPlugin",
    "MockTechnicalAnalysisPlugin",
]
