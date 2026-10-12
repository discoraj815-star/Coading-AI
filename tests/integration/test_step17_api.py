"""Integration API tests for Step 17 Master Decision Engine REST Endpoints."""
import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_master_decision_get_endpoint(client):
    res = client.get("/api/decision/master/evaluate?symbol=EURUSD&timeframe=15m")
    assert res.status_code == 200
    data = res.json()
    assert "decision_id" in data
    assert "final_state" in data
    assert "confluence_score" in data
    assert "concept_results" in data
    assert "user_summary_en" in data
    assert "user_summary_hi" in data


def test_master_decision_post_endpoint_with_confluence(client):
    payload = {
        "symbol": "EURUSD",
        "timeframe": "15m",
        "direction": "BUY",
        "market_data": {
            "current_price": 1.0850,
            "candles": [
                {"timestamp": 100, "open": 1.080, "high": 1.082, "low": 1.079, "close": 1.081, "volume": 500},
                {"timestamp": 200, "open": 1.081, "high": 1.084, "low": 1.0805, "close": 1.0835, "volume": 600},
                {"timestamp": 300, "open": 1.0835, "high": 1.086, "low": 1.083, "close": 1.0855, "volume": 800},
            ],
            "structure_bias": "BULLISH",
            "liquidity_swept": True,
            "fvg_detected": True,
            "orderflow_data": {
                "delta": 380.0,
                "cumulative_delta": 950.0,
                "cvd_slope": "RISING",
                "absorption_detected": True,
                "absorption_side": "SELLERS_ABSORBED",
            },
            "volume_profile_data": {
                "poc": 1.0840,
                "vah": 1.0870,
                "val": 1.0810,
                "value_area_acceptance": "REJECTION_AT_VAL",
            },
        },
    }
    res = client.post("/api/decision/master/evaluate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["final_state"] == "CONFIRMED_SETUP"
    assert data["direction"] == "BUY"
    assert data["confluence_score"] >= 8.0
    assert len(data["participating_concepts"]) >= 2


def test_decision_audit_list_and_detail(client):
    # Trigger an evaluation to generate an audit record
    eval_res = client.get("/api/decision/master/evaluate?symbol=GBPUSD")
    assert eval_res.status_code == 200
    dec_id = eval_res.json()["decision_id"]

    # List audits
    list_res = client.get("/api/decision/audit/list?limit=10")
    assert list_res.status_code == 200
    audits = list_res.json()
    assert isinstance(audits, list)
    assert len(audits) >= 1

    # Get specific audit trace
    detail_res = client.get(f"/api/decision/audit/{dec_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["decision_id"] == dec_id
    assert detail["symbol"] == "GBPUSD"


def test_concept_modules_registry_endpoints(client):
    # List concepts
    res = client.get("/api/decision/concepts")
    assert res.status_code == 200
    concepts = res.json()
    assert len(concepts) >= 5
    ids = [c["module_id"] for c in concepts]
    assert "smc_ict" in ids
    assert "orderflow" in ids
    assert "footprint" in ids
    assert "volume_profile" in ids
    assert "wyckoff" in ids

    # Toggle concept module
    t_res = client.post("/api/decision/concepts/footprint/toggle", json={"enable": False})
    assert t_res.status_code == 200
    assert t_res.json()["is_enabled"] is False

    t_res2 = client.post("/api/decision/concepts/footprint/toggle", json={"enable": True})
    assert t_res2.status_code == 200
    assert t_res2.json()["is_enabled"] is True

    # Approve plugin concept module
    app_res = client.post("/api/decision/concepts/auction_market_theory/approve", json={"approve": True})
    assert app_res.status_code == 200
    assert app_res.json()["is_approved"] is True


def test_threshold_update_endpoint(client):
    res = client.post("/api/decision/threshold", json={"threshold": 8.5})
    assert res.status_code == 200
    assert res.json()["threshold"] == 8.5

    # Reset back to default
    client.post("/api/decision/threshold", json={"threshold": 8.0})
