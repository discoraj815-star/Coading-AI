"""Integration tests for Step 8 REST API endpoints."""

import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health_step8(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["version"] == "0.8.0"
    assert "Step 8" in data["step"]


def test_api_execution_status(client):
    res = client.get("/api/execution/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "operational"
    assert data["engine"] == "HedgeFundExecutionCore"
    assert "policy" in data


def test_api_risk_status(client):
    res = client.get("/api/risk/status")
    assert res.status_code == 200
    data = res.json()
    assert "policy" in data
    assert "discipline" in data


def test_api_execution_policy_update(client):
    res = client.post(
        "/api/execution/policy",
        json={
            "user_id": "test_user_exec",
            "execution_mode": "MANUAL",
            "auto_execution_enabled": False,
            "min_rr_ratio": 2.5,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["min_rr_ratio"] == 2.5


def test_api_trade_proposal_lifecycle_and_execution(client):
    from trading_ai.server.app import repo
    from trading_ai.schemas.market import CurrentSituationSnapshot, MarketSetup, MarketLevel
    from trading_ai.schemas.enums import MarketBias, MarketDirection, SetupStatus, LevelType

    # Ensure a fresh, deterministic confirmed situation exists
    setup = MarketSetup(
        setup_id="setup_api_test",
        symbol="EUR/USD",
        timeframe="15m",
        setup_type="ORDER_BLOCK",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.CONFIRMED,
        applicable_rule_id="RULE_OB_01",
        conditions_satisfied=["HTF alignment", "Displacement confirmed"],
        confidence=0.92,
        supporting_evidence=["Lecture rule satisfied"],
    )
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="Strong bullish HTF/MTF structure",
        active_confirmed_setup=setup,
        nearest_major_liquidity=MarketLevel(
            symbol="EUR/USD",
            timeframe="1h",
            price=1.0920,
            level_type=LevelType.SWING_HIGH,
        ),
        technical_fundamental_relationship="TECHNICALLY_AND_FUNDAMENTALLY_ALIGNED",
    )
    repo.save_current_situation(sit)

    # 1. Generate Proposal (align with prevailing market context)
    res_gen = client.post(
        "/api/trades/proposals/generate",
        json={
            "symbol": "EUR/USD",
            "direction": "BUY",
            "engagement_mode": "DEFENSIVE",
            "account_equity": 10000.0,
        },
    )
    assert res_gen.status_code == 200
    prop = res_gen.json()
    assert "proposal_id" in prop
    prop_id = prop["proposal_id"]
    assert prop["symbol"] == "EUR/USD"
    assert prop["hard_stop_loss"] > 0
    assert prop["tp1"] > 0

    # 2. Get proposal by ID
    res_get = client.get(f"/api/trades/proposals/{prop_id}")
    assert res_get.status_code == 200
    assert res_get.json()["proposal_id"] == prop_id

    # 3. Approve and Execute via ExecutionGate
    res_app = client.post(
        f"/api/trades/proposals/{prop_id}/approve",
        json={"user_confirmed": True},
    )
    assert res_app.status_code == 200
    exec_data = res_app.json()
    assert exec_data["status"] == "executed"
    assert exec_data["order"]["status"] == "FILLED"
    assert exec_data["position"]["status"] == "OPEN"
    pos_id = exec_data["position"]["position_id"]

    # 4. Check Open Positions
    res_pos = client.get("/api/positions?open_only=true")
    assert res_pos.status_code == 200
    positions = res_pos.json()
    assert any(p["position_id"] == pos_id for p in positions)

    # 5. Update Stop Loss on Open Position
    is_buy = prop["direction"].upper() in ("BUY", "LONG", "BULLISH")
    new_sl_val = round(prop["entry_price"] * (0.999 if is_buy else 1.001), 5)
    res_sl = client.post(
        f"/api/positions/{pos_id}/update_sl",
        json={"new_sl": new_sl_val, "reason": "Trailing SL adjustment"},
    )
    assert res_sl.status_code == 200
    assert res_sl.json()["current_sl"] == new_sl_val

    # 6. Close Position
    res_close = client.post(
        f"/api/positions/{pos_id}/close",
        json={"reason": "Manual close test"},
    )
    assert res_close.status_code == 200
    assert res_close.json()["status"] == "CLOSED"


def test_api_orders_list(client):
    res = client.get("/api/orders?limit=10")
    assert res.status_code == 200
    assert isinstance(res.json(), list)
