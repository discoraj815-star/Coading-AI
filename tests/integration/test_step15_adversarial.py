"""Adversarial and Edge Case Tests for Step 15."""

import time
import pytest
from fastapi.testclient import TestClient

from trading_ai.ai_core.large_video_engine import LargeVideoStreamingProcessor
from trading_ai.plugins.base import BasePlugin
from trading_ai.plugins.manifest import PluginCapability, PluginCategory, PluginManifest
from trading_ai.plugins.registry import PluginRegistry
from trading_ai.server.app import app
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier


@pytest.fixture
def repo():
    return Repository(DatabaseManager(":memory:"))


@pytest.fixture
def client():
    return TestClient(app)


def test_adversarial_malicious_plugin_capabilities(repo):
    """Verify malicious plugin attempting root/terminal/unrestricted access is rejected."""
    reg = PluginRegistry(repository=repo)

    class HostilePlugin(BasePlugin):
        def initialize(self, config=None):
            return True
        def start(self):
            return True
        def stop(self):
            return True
        def execute_action(self, action, params=None):
            return {"pawned": True}

    manifest = PluginManifest(
        plugin_id="plugin_hostile_1",
        name="Hostile Shell Plugin",
        category=PluginCategory.RESEARCH,
        capabilities=["SHELL_EXECUTION"],  # Forbidden
    )
    assert reg.register(HostilePlugin(manifest)) is False


def test_adversarial_plugin_attempting_direct_trade_execution(repo):
    """Verify plugin cannot execute a trade or bypass the Step 8/14 gates."""
    reg = PluginRegistry(repository=repo)

    class SneakyBrokerPlugin(BasePlugin):
        def __init__(self):
            manifest = PluginManifest(
                plugin_id="plugin_sneaky_broker",
                name="Sneaky Broker Plugin",
                category=PluginCategory.BROKER,
                capabilities=[PluginCapability.BROKER_READ],
            )
            super().__init__(manifest)

        def initialize(self, config=None):
            return True
        def start(self):
            return True
        def stop(self):
            return True
        def execute_action(self, action, params=None):
            return {"action": action}

    p = SneakyBrokerPlugin()
    reg.register(p, enable=True)

    # Attempt direct trade placement
    res = reg.invoke("plugin_sneaky_broker", "execute_trade", params={"action": "BUY"})
    assert res["success"] is False
    assert res.get("safety_violation") is True
    assert "ExecutionGate is forbidden" in res["error"]


def test_adversarial_plugin_timeout_isolation(repo):
    """Verify slow or hung plugins are timed out without hanging the AI assistant."""
    reg = PluginRegistry(repository=repo)

    class HungPlugin(BasePlugin):
        def __init__(self):
            manifest = PluginManifest(
                plugin_id="plugin_hung",
                name="Hung Plugin",
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
            time.sleep(0.3)
            return {"done": True}

    reg.register(HungPlugin(), enable=True)

    # Invoke with strict 0.05s timeout
    res = reg.invoke("plugin_hung", "read", timeout_seconds=0.05)
    assert res["success"] is False
    assert "timeout" in res["error"].lower()


def test_adversarial_serial_commit_barrier_interruption_and_recovery(repo):
    """Verify that a worker crash on chunk 3 blocks chunks 4 & 5 until chunk 3 recovers."""
    barrier = CommitBarrier(repo)
    engine = LargeVideoStreamingProcessor(repository=repo, commit_barrier=barrier)

    job_id = "vjob_crash_test"
    engine.initialize_job(job_id, "data/large.mp4", 100 * (1024**3), duration_seconds=150.0)

    # 1. Chunks 1 and 2 succeed
    c1 = engine.process_chunk(job_id, chunk_index=1, start_time=0.0, end_time=30.0)
    c2 = engine.process_chunk(job_id, chunk_index=2, start_time=30.0, end_time=60.0)
    assert c1.status == "COMMITTED"
    assert c2.status == "COMMITTED"

    # 2. Chunk 3 fails (simulated crash)
    c3 = engine.process_chunk(job_id, chunk_index=3, start_time=60.0, end_time=90.0, simulate_failure=True)
    assert c3.status == "FAILED"

    # 3. Chunks 4 and 5 succeed out of order -> MUST NOT COMMIT
    c4 = engine.process_chunk(job_id, chunk_index=4, start_time=90.0, end_time=120.0)
    c5 = engine.process_chunk(job_id, chunk_index=5, start_time=120.0, end_time=150.0)
    assert c4.status == "BUFFERED"
    assert c5.status == "BUFFERED"

    # Verify barrier state
    b_state = barrier.get_barrier_state(job_id)
    assert b_state.highest_committed_serial == 2
    assert c4.serial_number in b_state.buffered_units
    assert c5.serial_number in b_state.buffered_units

    # 4. Recover Chunk 3
    committed = engine.recover_chunk(job_id, serial_number=c3.serial_number)
    # Barrier releases: chunks 3, 4, 5 are all committed in strict order
    assert c3.serial_number in committed
    assert c4.serial_number in committed
    assert c5.serial_number in committed

    b_state_after = barrier.get_barrier_state(job_id)
    assert b_state_after.highest_committed_serial == 5
    assert len(b_state_after.buffered_units) == 0


def test_adversarial_news_risk_blocks_live_decision(client):
    """Verify live decision engine blocks trading when news risk is high even if pattern is present."""
    res = client.post("/api/live-decision/evaluate", json={
        "symbol": "EURUSD",
        "direction": "BUY",
        "observed_sweep": True,
        "observed_displacement": True,
        "observed_fvg": True,
        "news_risk_level": "CRITICAL",  # High-impact news event pending
    })
    assert res.status_code == 200
    data = res.json()
    assert data["final_decision"] == "NO_TRADE"
    assert data["why_not_trade_result"]["has_blocking"] is True
    assert "High-impact news" in data["why_not_trade_result"]["blocking_reasons"][0]
