"""Integration tests for Step 11 High-Performance & Adaptive AI REST API endpoints."""

from fastapi.testclient import TestClient
import pytest

from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_performance_status_endpoint(client):
    """GET /api/performance/status returns healthy system metrics."""
    res = client.get("/api/performance/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    assert "event_bus" in data
    assert "memory_tiers" in data
    assert "average_latency_ms" in data


def test_performance_latency_endpoint(client):
    """GET /api/performance/latency returns list of telemetry measurements."""
    res = client.get("/api/performance/latency?limit=5")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_performance_workers_and_queues(client):
    """GET /api/performance/workers and /api/performance/queues return queue breakdown."""
    res_w = client.get("/api/performance/workers")
    assert res_w.status_code == 200
    assert res_w.json()["status"] == "ACTIVE"

    res_q = client.get("/api/performance/queues")
    assert res_q.status_code == 200
    assert "priority_breakdown" in res_q.json()


def test_performance_cache_endpoint(client):
    """GET /api/performance/cache returns memory tier counts."""
    res = client.get("/api/performance/cache")
    assert res.status_code == 200
    data = res.json()
    assert "hot_symbols_count" in data
    assert "warm_candle_series_count" in data


def test_process_tick_endpoint(client):
    """POST /api/performance/tick in-memory hot tier ingestion and latency recording."""
    payload = {
        "symbol": "BTC/USD",
        "price": 64850.25,
        "source": "INTEGRATION_TEST_FEED",
    }
    res = client.post("/api/performance/tick", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PROCESSED"
    assert data["symbol"] == "BTC/USD"
    assert data["price"] == 64850.25
    assert data["event_accepted"] is True
    assert "latency_ms" in data


def test_component_upgrade_api_lifecycle_and_user_approval_gate(client):
    """Test full API upgrade ladder including strict enforcement of 403 on automated self-promotion."""
    # 1. Register candidate
    reg_payload = {
        "component_name": "TestVisionModel_v2",
        "current_version": "1.0.0",
        "candidate_version": "2.0.0",
        "notes": "Testing integration upgrade ladder",
    }
    res_reg = client.post("/api/performance/upgrades", json=reg_payload)
    assert res_reg.status_code == 200
    cand = res_reg.json()
    upgrade_id = cand["upgrade_id"]
    assert cand["status"] == "CANDIDATE"

    # 2. Benchmark
    bench_payload = {
        "accuracy": 0.94,
        "latency_ms": 11.5,
        "regression_pass_rate": 1.0,
    }
    res_b = client.post(f"/api/performance/upgrades/{upgrade_id}/benchmark", json=bench_payload)
    assert res_b.status_code == 200
    assert res_b.json()["status"] == "BENCHMARK_PASSED"

    # 3. Paper validation
    res_p = client.post(f"/api/performance/upgrades/{upgrade_id}/paper", json={"passed": True})
    assert res_p.status_code == 200
    assert res_p.json()["status"] == "AWAITING_USER_APPROVAL"

    # 4. Strict Safety Gate: Automated promotion MUST return 403 Forbidden!
    res_auto = client.post(
        f"/api/performance/upgrades/{upgrade_id}/approve",
        json={"approved_by": "", "is_user": False},
    )
    assert res_auto.status_code == 403

    # 5. User approval succeeds
    res_user = client.post(
        f"/api/performance/upgrades/{upgrade_id}/approve",
        json={"approved_by": "Senior_Portfolio_Manager", "is_user": True},
    )
    assert res_user.status_code == 200
    promoted = res_user.json()
    assert promoted["status"] == "APPROVED"
    assert promoted["user_approved"] is True
    assert promoted["approved_by"] == "Senior_Portfolio_Manager"

    # 6. Rollback endpoint
    res_rb = client.post(
        f"/api/performance/upgrades/{upgrade_id}/rollback",
        json={"reason": "Test completed, rolling back"},
    )
    assert res_rb.status_code == 200
    assert res_rb.json()["status"] == "ROLLED_BACK"
