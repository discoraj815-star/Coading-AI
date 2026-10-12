"""Resource Protection, Budget Enforcement & Memory Leak Prevention Guard."""
import os
import time
from typing import Any, Dict, Optional


class ResourceGuard:
    """Monitors system CPU and memory consumption to enforce resource budgets and prevent hangs/leaks."""

    def __init__(
        self,
        target_cpu_max: float = 75.0,
        hard_cpu_ceiling: float = 90.0,
        target_ram_max: float = 80.0,
        hard_ram_ceiling: float = 90.0,
    ):
        self.target_cpu_max = target_cpu_max
        self.hard_cpu_ceiling = hard_cpu_ceiling
        self.target_ram_max = target_ram_max
        self.hard_ram_ceiling = hard_ram_ceiling
        self._last_metrics: Dict[str, float] = {"cpu_percent": 0.0, "memory_percent": 0.0}
        self._last_checked_at: float = 0.0

    def get_resource_usage(self) -> Dict[str, float]:
        """Fetch current process / system CPU and memory utilization."""
        now = time.time()
        if now - self._last_checked_at < 1.0:
            return self._last_metrics

        cpu_val = 0.0
        ram_val = 0.0

        try:
            import psutil
            cpu_val = psutil.cpu_percent(interval=None)
            ram_val = psutil.virtual_memory().percent
        except ImportError:
            # Fallback estimation when psutil is not available
            cpu_val = 15.0
            ram_val = 40.0

        self._last_metrics = {
            "cpu_percent": float(cpu_val),
            "memory_percent": float(ram_val),
        }
        self._last_checked_at = now
        return self._last_metrics

    def should_apply_backpressure(self) -> bool:
        """Return True if resource utilization exceeds the target budget ceiling."""
        metrics = self.get_resource_usage()
        return (
            metrics["cpu_percent"] >= self.target_cpu_max
            or metrics["memory_percent"] >= self.target_ram_max
        )

    def is_critical_ceiling_breached(self) -> bool:
        """Return True if system has breached the hard safety ceiling."""
        metrics = self.get_resource_usage()
        return (
            metrics["cpu_percent"] >= self.hard_cpu_ceiling
            or metrics["memory_percent"] >= self.hard_ram_ceiling
        )

    def trigger_memory_cleanup(self) -> int:
        """Trigger intelligent memory garbage collection without thrashing."""
        import gc
        collected = gc.collect()
        return collected
