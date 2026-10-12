"""Integration API tests for Step 16 Cloud Runtime and Event Stream Endpoints."""
from fastapi.testclient import TestClient
import pytest
from trading_ai.server.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_cloud_status_endpoint(client):
    res = client.get("/api/cloud/status")
    assert res.status_code == 200
    data = res.json()
    assert "runtime_state" in data
    assert "components" in data
    assert "market_data" in data["components"]
    assert "cloud_monitor" in data["components"]
    assert "vector_store" in data["components"]
    assert "task_queue" in data["components"]


def test_cloud_runtime_controls(client):
    # Pause
    r_pause = client.post("/api/cloud/pause")
    assert r_pause.status_code == 200
    assert r_pause.json()["status"] == "PAUSED"

    # Resume
    r_resume = client.post("/api/cloud/resume")
    assert r_resume.status_code == 200
    assert r_resume.json()["status"] == "ONLINE"

    # Start
    r_start = client.post("/api/cloud/start")
    assert r_start.status_code == 200


def test_cloud_cycle_execution(client):
    res = client.post("/api/cloud/cycle")
    assert res.status_code == 200
    data = res.json()
    assert data["state"] == "ONLINE"
    assert "market_monitor" in data


def test_events_and_replay_endpoints(client):
    res = client.get("/api/events?limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    r_replay = client.get("/api/events/replay?client_id=test_client&limit=5")
    assert r_replay.status_code == 200
    assert isinstance(r_replay.json(), list)


def test_notifications_and_test_dispatch(client):
    res = client.get("/api/notifications?limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)

    # Dispatch test notification
    r_test = client.post(
        "/api/notifications/test",
        json={
            "symbol": "EURUSD",
            "priority": "P3_CONFIRMED_SETUP",
            "pattern": "Bullish FVG",
            "direction": "BUY",
            "confidence": 0.95,
        },
    )
    assert r_test.status_code == 200
    data = r_test.json()
    assert data["status"] in ("DISPATCHED", "FILTERED")
    if data["status"] == "DISPATCHED":
        assert len(data["records"]) >= 1
        rec = data["records"][0]
        assert "Hello Alam" in rec["title"]
        assert "Hello Alam" in rec["message_hindi"] or "alert" in rec["message_hindi"] or "EURUSD" in rec["message_hindi"]


def test_notification_preferences_endpoints(client):
    # GET
    res_get = client.get("/api/notifications/preferences")
    assert res_get.status_code == 200
    current = res_get.json()
    assert "notification_enabled" in current
    assert "greeting_name" in current

    # PUT
    res_put = client.put(
        "/api/notifications/preferences",
        json={"greeting_name": "Alam", "notification_verbosity": "short"},
    )
    assert res_put.status_code == 200
    assert res_put.json()["notification_verbosity"] == "short"


def test_adapters_health_endpoints(client):
    # Vector store health
    r_vec = client.get("/api/vector_store/health")
    assert r_vec.status_code == 200
    assert r_vec.json()["backend"] == "SQLiteVectorStore"

    # Task queue health
    r_queue = client.get("/api/task_queue/health")
    assert r_queue.status_code == 200
    assert r_queue.json()["backend"] == "LocalPersistentTaskQueue"


def test_voice_preview_endpoint(client):
    res = client.post(
        "/api/notifications/voice/preview",
        json={"text": "Hello Alam, confirmation मिला है।", "speed": 1.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["format"] == "audio/mp3"
    assert "audio_base64" in data
    assert data["duration_estimated_seconds"] > 0
