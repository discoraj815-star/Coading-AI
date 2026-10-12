"""Unit tests for Step 15 Controlled Plugin Architecture."""

import pytest
from trading_ai.plugins.base import BasePlugin, BrokerReadPlugin
from trading_ai.plugins.manifest import (
    FORBIDDEN_CAPABILITIES,
    PluginCapability,
    PluginCategory,
    PluginHealthStatus,
    PluginManifest,
    PluginStatus,
)
from trading_ai.plugins.registry import PluginRegistry
from trading_ai.plugins.samples import (
    MockBrokerReadPlugin,
    MockForexFactoryNewsPlugin,
    MockTechnicalAnalysisPlugin,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo():
    return Repository(DatabaseManager(":memory:"))


@pytest.fixture
def registry(repo):
    return PluginRegistry(repository=repo)


def test_plugin_registration_and_discovery(registry):
    """Verify clean plugin registration, lifecycle, and discovery."""
    plugin = MockForexFactoryNewsPlugin()
    assert registry.register(plugin, enable=True) is True

    plugins = registry.list_plugins()
    assert len(plugins) == 1
    p_meta = plugins[0]
    assert p_meta["plugin_id"] == "plugin_forex_factory"
    assert p_meta["category"] == "FOREX_FACTORY"
    assert p_meta["status"] == "ACTIVE"
    assert p_meta["is_enabled"] is True
    assert p_meta["health_status"] == "HEALTHY"


def test_plugin_capability_enforcement(registry):
    """Verify declared capabilities pass while undeclared capabilities are DENIED."""
    plugin = MockForexFactoryNewsPlugin()
    registry.register(plugin, enable=True)

    # Declared capability passes
    assert registry.check_capability("plugin_forex_factory", PluginCapability.READ_NEWS) is True
    # Undeclared capability denied
    assert registry.check_capability("plugin_forex_factory", PluginCapability.PROVIDE_TECHNICAL_ANALYSIS) is False

    res = registry.invoke(
        "plugin_forex_factory",
        "fetch_news",
        required_capability=PluginCapability.READ_NEWS,
        params={"currency": "USD"},
    )
    assert res["success"] is True
    assert res["data"] is not None

    # Invocation with undeclared capability is DENIED
    res_denied = registry.invoke(
        "plugin_forex_factory",
        "fetch_news",
        required_capability=PluginCapability.PROVIDE_TECHNICAL_ANALYSIS,
    )
    assert res_denied["success"] is False
    assert "DENIED" in res_denied["error"]


def test_forbidden_capabilities_rejected():
    """Verify inherently dangerous capabilities (shell, root, terminal) are rejected at registration."""
    reg = PluginRegistry()

    class MaliciousPlugin(BasePlugin):
        def initialize(self, config=None):
            return True
        def start(self):
            return True
        def stop(self):
            return True
        def execute_action(self, action, params=None):
            return {}

    manifest = PluginManifest(
        plugin_id="plugin_malicious",
        name="Malicious Terminal Plugin",
        category=PluginCategory.RESEARCH,
        capabilities=["TERMINAL_EXECUTION"],  # In forbidden list
    )
    p = MaliciousPlugin(manifest)
    assert reg.register(p) is False
    assert reg.get_plugin("plugin_malicious") is None


def test_broker_safety_hard_invariant(registry):
    """Verify that a broker plugin can NEVER execute trades directly (Trading Safety)."""
    broker_plugin = MockBrokerReadPlugin()
    registry.register(broker_plugin, enable=True)

    # Read action allowed
    read_res = registry.invoke("plugin_broker_read", "account_summary")
    assert read_res["success"] is True
    assert read_res["data"]["currency"] == "USD"

    # Direct trade execution attempt is BLOCKED by trading safety gate
    exec_res = registry.invoke("plugin_broker_read", "execute_trade", params={"symbol": "EURUSD", "units": 10000})
    assert exec_res["success"] is False
    assert exec_res.get("safety_violation") is True
    assert "ExecutionGate is forbidden" in exec_res["error"]


def test_plugin_failure_isolation_and_circuit_breaker(registry):
    """Verify that plugin crashes are isolated and do not crash the core AI."""

    class CrashingPlugin(BasePlugin):
        def __init__(self):
            manifest = PluginManifest(
                plugin_id="plugin_crash_test",
                name="Crashing Plugin",
                category=PluginCategory.EXTERNAL_DATA,
                capabilities=[PluginCapability.READ_STORAGE],
            )
            super().__init__(manifest)

        def initialize(self, config=None):
            return True
        def start(self):
            return True
        def stop(self):
            return True
        def execute_action(self, action, params=None):
            raise RuntimeError("Fatal unhandled plugin crash!")

    crash_p = CrashingPlugin()
    registry.register(crash_p, enable=True)

    # Core AI does NOT crash when invoking crashing plugin
    res = registry.invoke("plugin_crash_test", "do_something")
    assert res["success"] is False
    assert "Fatal unhandled plugin crash!" in res["error"]
    assert crash_p.consecutive_failures == 1
    assert crash_p.health_status == PluginHealthStatus.DEGRADED

    # Trigger repeated failures to trip circuit breaker
    for _ in range(5):
        registry.invoke("plugin_crash_test", "do_something")

    assert crash_p.health_status == PluginHealthStatus.UNHEALTHY
    res_cb = registry.invoke("plugin_crash_test", "do_something")
    assert res_cb["success"] is False
    assert res_cb.get("circuit_tripped") is True


def test_enable_disable_lifecycle(registry):
    """Verify clean enable and disable state transitions."""
    plugin = MockTechnicalAnalysisPlugin()
    registry.register(plugin, enable=True)
    assert plugin.status == PluginStatus.ACTIVE

    # Disable
    assert registry.disable_plugin("plugin_ta_engine") is True
    assert plugin.status == PluginStatus.DISABLED

    # Action invocation while disabled is rejected
    res = registry.invoke("plugin_ta_engine", "compute")
    assert res["success"] is False
    assert "disabled" in res["error"]

    # Re-enable
    assert registry.enable_plugin("plugin_ta_engine") is True
    assert plugin.status == PluginStatus.ACTIVE
    res_enabled = registry.invoke("plugin_ta_engine", "compute")
    assert res_enabled["success"] is True
