"""Unit tests for Step 16 24×7 Cloud Runtime Manager and Cloud Market Monitor."""
import time
import pytest
from trading_ai.cloud.market_monitor import CloudMarketMonitor
from trading_ai.cloud.runtime import CloudRuntimeManager
from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
from trading_ai.core.mtf_engine import MultiTimeframeEngine
from trading_ai.core.lecture_validator import LectureValidator
from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.situation_engine import CurrentSituationEngine
from trading_ai.events.gateway import WebSocketGateway
from trading_ai.events.router import EventRouter
from trading_ai.notifications.orchestrator import NotificationOrchestrator
from trading_ai.pipeline.watchlist_monitor import WatchlistMonitor
from trading_ai.schemas.enums import CloudRuntimeState, SetupStatus
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.storage.task_queue import LocalPersistentTaskQueue
from trading_ai.storage.vector_store import SQLiteVectorStore
from trading_ai.supervisor.commit_barrier import CommitBarrier


@pytest.fixture
def memory_repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


@pytest.fixture
def cloud_setup(memory_repo):
    repo = memory_repo
    data_provider = SimulatedMarketDataProvider(seed=42)
    mtf = MultiTimeframeEngine(data_provider)
    validator = LectureValidator(repo)
    marker = ChartMarker()
    barrier = CommitBarrier(repo)
    situation_engine = CurrentSituationEngine(repo, marker)

    watchlist_monitor = WatchlistMonitor(
        repository=repo,
        data_provider=data_provider,
        mtf_engine=mtf,
        lecture_validator=validator,
        chart_marker=marker,
        commit_barrier=barrier,
        watchlist=["BTC/USD", "EUR/USD"],
        situation_engine=situation_engine,
    )

    vector_store = SQLiteVectorStore(repo)
    task_queue = LocalPersistentTaskQueue(repo, barrier)
    gateway = WebSocketGateway(repo)
    router = EventRouter(
        event_bus=None,
        repository=repo,
        gateway=gateway,
    )
    # Mock event bus subscribe not needed for direct routing test
    cloud_monitor = CloudMarketMonitor(watchlist_monitor, repo, event_router=router)
    orchestrator = NotificationOrchestrator(repo, gateway)

    runtime_mgr = CloudRuntimeManager(
        repository=repo,
        market_monitor=cloud_monitor,
        event_router=router,
        gateway=gateway,
        orchestrator=orchestrator,
        task_queue=task_queue,
        vector_store=vector_store,
    )

    return {
        "repo": repo,
        "runtime": runtime_mgr,
        "monitor": cloud_monitor,
        "router": router,
        "data_provider": data_provider,
    }


def test_cloud_runtime_lifecycle(cloud_setup):
    runtime = cloud_setup["runtime"]
    assert runtime.state == CloudRuntimeState.ONLINE

    # Pause
    runtime.pause()
    assert runtime.state == CloudRuntimeState.PAUSED

    # Resume
    runtime.resume()
    assert runtime.state == CloudRuntimeState.ONLINE

    # Stop
    runtime.stop()
    assert runtime.state == CloudRuntimeState.OFFLINE


def test_cloud_runtime_step_execution_and_heartbeat(cloud_setup):
    runtime = cloud_setup["runtime"]
    repo = cloud_setup["repo"]
    runtime.start()

    res = runtime.step_execution(force=True)
    assert res["state"] == "ONLINE"
    assert "market_monitor" in res
    assert res["market_monitor"]["symbols_processed"] >= 2

    # Verify heartbeat in SQLite
    state_rec = repo.get_cloud_runtime_state("primary_cloud")
    assert state_rec is not None
    assert state_rec["state"] == "ONLINE"
    assert "last_heartbeat" in state_rec


def test_cloud_runtime_health_matrix(cloud_setup):
    runtime = cloud_setup["runtime"]
    status = runtime.get_status()
    assert status.runtime_state == CloudRuntimeState.ONLINE
    assert "market_data" in status.components
    assert "cloud_monitor" in status.components
    assert "vector_store" in status.components
    assert "task_queue" in status.components
    assert status.components["vector_store"].details["backend"] == "SQLiteVectorStore"


def test_cloud_market_monitor_stale_data_gating(cloud_setup):
    monitor = cloud_setup["monitor"]
    repo = cloud_setup["repo"]

    # In SimulatedMarketDataProvider, initial candle timestamp might be 0 or small
    # Process symbol directly
    res = monitor.process_symbol_cloud("EUR/USD", force=True)
    assert res["status"] in ("PROCESSED", "STALE_DATA")

    # If marked stale, ensure it emitted event and tracked state
    if res["status"] == "STALE_DATA":
        events = repo.get_stream_events(limit=5)
        stale_events = [e for e in events if e["event_type"] == "MARKET_DATA_STALE"]
        assert len(stale_events) >= 1
