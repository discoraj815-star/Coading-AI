"""Step 11 High-Performance & Adaptive AI Architecture Package."""

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.adaptive.memory_tier import TieredMemoryManager
from trading_ai.adaptive.incremental_engine import IncrementalAnalysisEngine
from trading_ai.adaptive.priority_scheduler import AdaptiveComputeScheduler
from trading_ai.adaptive.upgrade_manager import ComponentUpgradeManager
from trading_ai.adaptive.core import AdaptiveAICore

__all__ = [
    "ReactiveEventBus",
    "TieredMemoryManager",
    "IncrementalAnalysisEngine",
    "AdaptiveComputeScheduler",
    "ComponentUpgradeManager",
    "AdaptiveAICore",
]
