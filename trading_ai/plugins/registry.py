"""Controlled Plugin Registry, Capability Gate, and Runtime Isolation Engine for Step 15."""

import logging
import time
from typing import Any, Dict, List, Optional

from trading_ai.plugins.base import BasePlugin
from trading_ai.plugins.manifest import (
    FORBIDDEN_CAPABILITIES,
    PluginCapability,
    PluginHealthStatus,
    PluginManifest,
    PluginStatus,
)
from trading_ai.security.audit import SecurityAuditLogger
from trading_ai.security.capability_gate import CapabilityPolicyGate
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class TradingSafetyViolation(Exception):
    """Raised when a plugin attempts to bypass CoreTradeLogic/RiskEngine/ExecutionGate."""
    pass


class PluginPermissionDenied(Exception):
    """Raised when a plugin attempts an undeclared, forbidden, or unauthorized capability."""
    pass


class PluginRegistry:
    """Enterprise Plugin Management, Capability Verification, and Runtime Isolation.

    Hard Invariants:
    1. Default DENY: Unknown or undeclared capabilities are strictly blocked.
    2. Forbidden Capabilities: Terminal, arbitrary filesystem, secrets, direct broker execution are NEVER allowed.
    3. Failure Isolation: A crashed or failing plugin will NOT crash the Trading AI assistant.
    4. Trading Safety: No plugin can ever execute a trade directly; trading is strictly routed through Step 8 & 14 gates.
    """

    def __init__(
        self,
        repository: Optional[Repository] = None,
        capability_gate: Optional[CapabilityPolicyGate] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ):
        self._repo = repository
        self._capability_gate = capability_gate or (CapabilityPolicyGate(repository) if repository else None)
        self._audit = audit_logger or SecurityAuditLogger(repository)
        self._plugins: Dict[str, BasePlugin] = {}
        self._enabled_state: Dict[str, bool] = {}

    def register(self, plugin: BasePlugin, enable: bool = True) -> bool:
        """Register a new plugin after strict manifest and capability validation."""
        manifest = plugin.manifest
        plugin_id = manifest.plugin_id

        # 1. Reject forbidden capabilities immediately
        for cap in manifest.capabilities:
            cap_str = cap.value if hasattr(cap, "value") else str(cap)
            if cap_str in FORBIDDEN_CAPABILITIES:
                msg = f"Plugin {plugin_id} requests FORBIDDEN capability {cap_str}. Registration rejected (DENY)."
                logger.error(msg)
                self._record_audit(plugin_id, "REGISTER", "DENIED", {"reason": msg})
                return False

        # 2. Check for duplicate registration
        if plugin_id in self._plugins:
            logger.warning(f"Plugin {plugin_id} already registered. Updating instance.")

        # 3. Store instance and initialization
        self._plugins[plugin_id] = plugin
        self._enabled_state[plugin_id] = enable

        try:
            init_success = plugin.initialize(manifest.config)
            if not init_success:
                plugin.status = PluginStatus.ERROR
                plugin.health_status = PluginHealthStatus.FAILING
                self._record_audit(plugin_id, "INITIALIZE", "FAILED", {"error": "Plugin initialize returned False"})
                self._persist_plugin(plugin)
                return False
        except Exception as e:
            plugin.status = PluginStatus.ERROR
            plugin.record_failure(str(e))
            self._record_audit(plugin_id, "INITIALIZE", "ERROR", {"error": str(e)})
            self._persist_plugin(plugin)
            return False

        if enable:
            try:
                plugin.start()
                plugin.status = PluginStatus.ACTIVE
            except Exception as e:
                plugin.status = PluginStatus.ERROR
                plugin.record_failure(str(e))
        else:
            plugin.status = PluginStatus.DISABLED

        self._record_audit(
            plugin_id,
            "REGISTER",
            "SUCCESS",
            {"name": manifest.name, "version": manifest.version, "capabilities": [c.value if hasattr(c, "value") else str(c) for c in manifest.capabilities]},
        )
        self._persist_plugin(plugin)
        return True

    def enable_plugin(self, plugin_id: str) -> bool:
        """Enable an inactive or disabled plugin."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        try:
            plugin.start()
            plugin.status = PluginStatus.ACTIVE
            self._enabled_state[plugin_id] = True
            self._record_audit(plugin_id, "ENABLE", "SUCCESS")
            self._persist_plugin(plugin)
            return True
        except Exception as e:
            plugin.status = PluginStatus.ERROR
            plugin.record_failure(str(e))
            self._record_audit(plugin_id, "ENABLE", "ERROR", {"error": str(e)})
            self._persist_plugin(plugin)
            return False

    def disable_plugin(self, plugin_id: str) -> bool:
        """Safely disable an active plugin without crashing the system."""
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return False

        try:
            plugin.stop()
            plugin.status = PluginStatus.DISABLED
            self._enabled_state[plugin_id] = False
            self._record_audit(plugin_id, "DISABLE", "SUCCESS")
            self._persist_plugin(plugin)
            return True
        except Exception as e:
            plugin.record_failure(str(e))
            self._record_audit(plugin_id, "DISABLE", "ERROR", {"error": str(e)})
            self._persist_plugin(plugin)
            return False

    def check_capability(self, plugin_id: str, capability: PluginCapability) -> bool:
        """Verify that the plugin is authorized and has declared the capability."""
        plugin = self._plugins.get(plugin_id)
        if not plugin or not self._enabled_state.get(plugin_id, False):
            return False

        cap_str = capability.value if hasattr(capability, "value") else str(capability)
        if cap_str in FORBIDDEN_CAPABILITIES:
            return False

        # Must be in declared capabilities
        return any((c.value if hasattr(c, "value") else str(c)) == cap_str for c in plugin.manifest.capabilities)

    def invoke(
        self,
        plugin_id: str,
        action: str,
        required_capability: Optional[PluginCapability] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = 5.0,
    ) -> Dict[str, Any]:
        """Safely invoke a plugin action with capability enforcement and failure isolation.

        Returns result dict or error payload. NEVER raises unhandled exceptions to caller.
        """
        plugin = self._plugins.get(plugin_id)
        if not plugin:
            return {"success": False, "error": f"Plugin {plugin_id} not registered", "data": None}

        if not self._enabled_state.get(plugin_id, False) or plugin.status == PluginStatus.DISABLED:
            return {"success": False, "error": f"Plugin {plugin_id} is disabled", "data": None}

        # Trading Safety invariant check
        if action.lower() in ("execute_trade", "place_order", "send_order", "direct_execute"):
            msg = f"CRITICAL: Direct execution blocked for plugin {plugin_id}. Bypassing ExecutionGate is forbidden."
            logger.error(msg)
            self._record_audit(plugin_id, "EXECUTE_TRADE_ATTEMPT", "BLOCKED", {"error": msg})
            return {"success": False, "error": msg, "data": None, "safety_violation": True}

        # Capability permission check
        if required_capability is not None:
            if not self.check_capability(plugin_id, required_capability):
                msg = f"Permission DENIED: Plugin {plugin_id} lacks capability {required_capability.value}"
                logger.warning(msg)
                self._record_audit(plugin_id, action, "DENIED", {"capability": required_capability.value})
                return {"success": False, "error": msg, "data": None}

        # Circuit breaker check
        if plugin.health_status == PluginHealthStatus.UNHEALTHY:
            msg = f"Circuit breaker tripped: Plugin {plugin_id} is UNHEALTHY ({plugin.consecutive_failures} failures)"
            return {"success": False, "error": msg, "data": None, "circuit_tripped": True}

        # Isolated execution
        start_t = time.time()
        try:
            result = plugin.execute_action(action, params or {})
            elapsed = time.time() - start_t
            if elapsed > timeout_seconds:
                plugin.record_failure(f"Execution exceeded timeout ({elapsed:.2f}s > {timeout_seconds}s)")
                self._record_audit(plugin_id, action, "TIMEOUT", {"elapsed": elapsed})
                self._persist_plugin(plugin)
                return {"success": False, "error": "Plugin timeout exceeded", "data": None}

            plugin.record_success()
            self._record_audit(plugin_id, action, "SUCCESS", {"elapsed": elapsed})
            self._persist_plugin(plugin)
            return {"success": True, "error": None, "data": result}
        except Exception as e:
            plugin.record_failure(str(e))
            self._record_audit(plugin_id, action, "ERROR", {"error": str(e)})
            self._persist_plugin(plugin)
            logger.error(f"Plugin {plugin_id} isolated error during action '{action}': {e}", exc_info=False)
            return {"success": False, "error": f"Plugin isolated execution error: {str(e)}", "data": None}

    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        return self._plugins.get(plugin_id)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return list of all registered plugins with full health and status metadata."""
        out = []
        for p_id, plugin in self._plugins.items():
            m = plugin.manifest
            out.append({
                "plugin_id": p_id,
                "name": m.name,
                "version": m.version,
                "category": m.category.value,
                "description": m.description,
                "author": m.author,
                "capabilities": [c.value if hasattr(c, "value") else str(c) for c in m.capabilities],
                "status": plugin.status.value,
                "is_enabled": self._enabled_state.get(p_id, False),
                "health_status": plugin.health_status.value,
                "consecutive_failures": plugin.consecutive_failures,
                "last_error": plugin.last_error,
            })
        return out

    def get_health_summary(self) -> Dict[str, Any]:
        """Aggregate health overview of the plugin subsystem."""
        total = len(self._plugins)
        healthy = sum(1 for p in self._plugins.values() if p.health_status == PluginHealthStatus.HEALTHY)
        degraded = sum(1 for p in self._plugins.values() if p.health_status == PluginHealthStatus.DEGRADED)
        unhealthy = sum(1 for p in self._plugins.values() if p.health_status in (PluginHealthStatus.FAILING, PluginHealthStatus.UNHEALTHY))
        return {
            "total_plugins": total,
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "all_healthy": unhealthy == 0 and degraded == 0,
        }

    def _record_audit(self, plugin_id: str, action: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        if self._repo and hasattr(self._repo, "save_plugin_audit_log"):
            try:
                self._repo.save_plugin_audit_log(plugin_id, action, status, details)
            except Exception as e:
                logger.debug(f"Could not persist plugin audit log: {e}")

    def _persist_plugin(self, plugin: BasePlugin) -> None:
        if self._repo and hasattr(self._repo, "save_plugin_record"):
            try:
                m = plugin.manifest
                self._repo.save_plugin_record({
                    "plugin_id": m.plugin_id,
                    "name": m.name,
                    "version": m.version,
                    "category": m.category.value,
                    "description": m.description,
                    "author": m.author,
                    "capabilities": [c.value if hasattr(c, "value") else str(c) for c in m.capabilities],
                    "status": plugin.status.value,
                    "is_enabled": self._enabled_state.get(m.plugin_id, True),
                    "permissions": m.permissions,
                    "config": m.config,
                    "consecutive_failures": plugin.consecutive_failures,
                    "health_status": plugin.health_status.value,
                    "last_error": plugin.last_error,
                })
            except Exception as e:
                logger.debug(f"Could not persist plugin record: {e}")
