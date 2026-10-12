"""Integration tests for Step 7 FastAPI Trading Companion endpoints."""

import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app, repo, companion_core
from trading_ai.schemas.companion import AlertEvent
from trading_ai.schemas.enums import AlertDecision, AlertLifecycleState, AlertPriority, AlertQuality, AlertType


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_step7(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["version"] in ("0.7.0", "0.8.0")
    assert "complete" in data["step"]


def test_api_companion_status(client):
    res = client.get("/api/companion/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert data["companion_engine"] == "ACTIVE"
    assert "MARKET_STATUS" in data["supported_intents"]


def test_api_companion_context(client):
    res = client.get("/api/companion/context/EUR/USD")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "EUR/USD"
    assert "bias" in data
    assert "discipline_status" in data


def test_api_alerts_and_acknowledge(client):
    # Save a test alert in repository
    alert = AlertEvent(
        alert_id="alt_api_test_1",
        symbol="EUR/USD",
        alert_type=AlertType.SETUP_HIGH_PROBABILITY,
        priority=AlertPriority.HIGH,
        lifecycle_state=AlertLifecycleState.ALERTED,
        decision=AlertDecision.ALERT,
        quality=AlertQuality.HIGH_QUALITY,
        title="Test Alert",
        message="Test alert message",
        dedup_key="api_test_dedup",
    )
    repo.save_alert_event(alert)

    # 1. Fetch unacknowledged alerts
    res_list = client.get("/api/alerts?unacknowledged_only=true")
    assert res_list.status_code == 200
    alerts = res_list.json()
    assert any(a["alert_id"] == "alt_api_test_1" for a in alerts)

    # 2. Acknowledge alert
    res_ack = client.post("/api/alerts/acknowledge", json={"alert_id": "alt_api_test_1"})
    assert res_ack.status_code == 200
    assert res_ack.json()["acknowledged"] is True

    # 3. Verify it is no longer unacknowledged
    res_after = client.get("/api/alerts?unacknowledged_only=true")
    assert not any(a["alert_id"] == "alt_api_test_1" for a in res_after.json())


def test_api_discipline_endpoints(client):
    # 1. Get status
    res = client.get("/api/discipline/status?user_id=test_api_user")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == "test_api_user"

    # 2. Update config
    cfg_payload = {
        "user_id": "test_api_user",
        "stop_loss_required": True,
        "daily_loss_limit": 500.0,
        "max_daily_trades": 4,
        "max_risk_per_trade_pct": 2.0,
    }
    res_cfg = client.post("/api/discipline/config", json=cfg_payload)
    assert res_cfg.status_code == 200
    assert res_cfg.json()["config"]["daily_loss_limit"] == 500.0

    # 3. Check trade intent (missing stop loss)
    trade_payload = {
        "symbol": "EUR/USD",
        "direction": "BUY",
        "entry_price": 1.0850,
        "stop_loss": None,
        "account_balance": 10000.0,
    }
    res_check = client.post("/api/discipline/check_trade?user_id=test_api_user", json=trade_payload)
    assert res_check.status_code == 200
    check_data = res_check.json()
    assert check_data["status"] == "USER_CONFIRMATION_REQUIRED"
    assert check_data["can_proceed"] is False
    assert check_data["checklist"]["stop_loss_defined"] is False


def test_api_scenarios_endpoint(client):
    res = client.get("/api/scenarios/EUR/USD?level=1.0920")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)


def test_api_companion_query_endpoint(client):
    query_payload = {
        "symbol": "EUR/USD",
        "query": "Market mein abhi kya chal raha hai?",
    }
    res = client.post("/api/companion/query", json=query_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "MARKET_STATUS"
    assert data["symbol"] == "EUR/USD"
    assert len(data["answer"]) > 10
