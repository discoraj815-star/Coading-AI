"""Plugin architecture manifests, capabilities, categories, and lifecycle schemas for Step 15."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class PluginCategory(str, Enum):
    MARKET_DATA = "MARKET_DATA"
    BROKER = "BROKER"
    NEWS = "NEWS"
    FOREX_FACTORY = "FOREX_FACTORY"
    ECONOMIC_CALENDAR = "ECONOMIC_CALENDAR"
    TECHNICAL_ANALYSIS = "TECHNICAL_ANALYSIS"
    FUNDAMENTAL_ANALYSIS = "FUNDAMENTAL_ANALYSIS"
    STORAGE = "STORAGE"
    VOICE = "VOICE"
    SCREEN_ACCESS = "SCREEN_ACCESS"
    NOTIFICATION = "NOTIFICATION"
    EXTERNAL_DATA = "EXTERNAL_DATA"
    RESEARCH = "RESEARCH"
    FUTURE_AI_PROVIDERS = "FUTURE_AI_PROVIDERS"


class PluginCapability(str, Enum):
    READ_MARKET_DATA = "READ_MARKET_DATA"
    READ_NEWS = "READ_NEWS"
    PROVIDE_TECHNICAL_ANALYSIS = "PROVIDE_TECHNICAL_ANALYSIS"
    PROVIDE_FUNDAMENTAL_ANALYSIS = "PROVIDE_FUNDAMENTAL_ANALYSIS"
    SEND_NOTIFICATION = "SEND_NOTIFICATION"
    READ_STORAGE = "READ_STORAGE"
    WRITE_STORAGE = "WRITE_STORAGE"
    VOICE_STREAM = "VOICE_STREAM"
    SCREEN_CAPTURE = "SCREEN_CAPTURE"
    EXTERNAL_API = "EXTERNAL_API"
    BROKER_READ = "BROKER_READ"
    BROKER_INTENT = "BROKER_INTENT"


# Explicitly forbidden capabilities that will always be rejected with DENY
FORBIDDEN_CAPABILITIES = {
    "DIRECT_BROKER_EXECUTE",
    "SHELL_EXECUTION",
    "TERMINAL_EXECUTION",
    "UNRESTRICTED_FILESYSTEM",
    "UNRESTRICTED_NETWORK",
    "SECRETS_ACCESS",
    "ROOT_ACCESS",
}


class PluginStatus(str, Enum):
    UNLOADED = "UNLOADED"
    LOADED = "LOADED"
    ACTIVE = "ACTIVE"
    DISABLED = "DISABLED"
    ERROR = "ERROR"


class PluginHealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILING = "FAILING"
    UNHEALTHY = "UNHEALTHY"


class PluginManifest(BaseModel):
    """Declarative manifest defining plugin metadata, capabilities, and boundaries."""

    plugin_id: str = Field(..., description="Unique alphanumeric identifier for plugin")
    name: str = Field(..., description="Human-readable plugin name")
    version: str = Field(default="1.0.0", description="SemVer plugin version")
    category: PluginCategory = Field(..., description="Classification category")
    description: str = Field(default="", description="Plugin purpose and description")
    author: str = Field(default="TradingAI", description="Plugin author or maintainer")
    capabilities: List[Union[PluginCapability, str]] = Field(default_factory=list, description="Declared capabilities")
    permissions: List[str] = Field(default_factory=list, description="Required system permissions")
    entrypoint: str = Field(default="", description="Python module or class path")
    min_core_version: str = Field(default="1.0.0", description="Minimum Trading AI version")
    config: Dict[str, Any] = Field(default_factory=dict, description="Default or provided configuration")
