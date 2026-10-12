import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_list_universal_providers(client):
    res = client.get("/api/providers/list")
    assert res.status_code == 200
    providers = res.json()
    assert isinstance(providers, list)
    assert len(providers) >= 8

    provider_ids = [p["provider_id"] for p in providers]
    assert "paper_broker" in provider_ids
    assert "delta_exchange" in provider_ids
    assert "metatrader5" in provider_ids
    assert "forex_factory" in provider_ids
    assert "simulated_market_data" in provider_ids


def test_api_get_single_provider(client):
    res = client.get("/api/providers/delta_exchange")
    assert res.status_code == 200
    data = res.json()
    assert data["provider_id"] == "delta_exchange"
    assert data["provider_type"] == "BROKER"
    assert data["trading_state"] == "LIVE_LOCKED"


def test_api_toggle_and_approve_provider(client):
    # Toggle off then on
    res = client.post("/api/providers/paper_broker/toggle")
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Approve provider
    res_app = client.post("/api/providers/paper_broker/approve?approved=true")
    assert res_app.status_code == 200
    assert res_app.json()["approved"] is True


def test_api_test_provider_connection(client):
    # Test paper broker link
    res = client.post("/api/providers/paper_broker/test")
    assert res.status_code == 200
    result = res.json()
    assert result["success"] is True
    assert result["provider_id"] == "paper_broker"


def test_api_switch_active_provider(client):
    res = client.post("/api/providers/active/BROKER/switch?provider_id=paper_broker")
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert res.json()["active_provider_id"] == "paper_broker"

    res_active = client.get("/api/providers/active/BROKER")
    assert res_active.status_code == 200
    assert res_active.json()["provider_id"] == "paper_broker"


def test_api_configure_provider_masks_secrets(client):
    config_payload = {
        "api_key": "my_api_key_12345",
        "api_secret": "my_secret_key_67890",
        "server": "MetaQuotes-Demo",
    }
    res = client.post("/api/providers/metatrader5/configure", json=config_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True

    # Check masked config output
    masked = data["masked_config"]
    assert "MASKED" in masked["api_key"]
    assert "MASKED" in masked["api_secret"]
    assert "my_secret_key_67890" not in str(masked)


def test_api_lecture_benchmark_endpoints(client):
    # Generate sample benchmark
    sample_res = client.post("/api/benchmark/lecture/sample")
    assert sample_res.status_code == 200
    benchmark = sample_res.json()
    assert "lecture_id" in benchmark
    assert benchmark["is_complete"] is True
    assert "bottlenecks" in benchmark
    assert len(benchmark["timeline"]) > 0

    lid = benchmark["lecture_id"]

    # Retrieve by id
    get_res = client.get(f"/api/benchmark/lecture/{lid}")
    assert get_res.status_code == 200
    assert get_res.json()["lecture_id"] == lid

    # List benchmarks
    list_res = client.get("/api/benchmark/lecture/list")
    assert list_res.status_code == 200
    benchmarks_list = list_res.json()["benchmarks"]
    assert len(benchmarks_list) >= 1
