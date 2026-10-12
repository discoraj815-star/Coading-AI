"""Integration tests for Step 9 Backtesting, Trading Brain, and Promotion REST APIs."""

import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_rule_versions_lifecycle(client):
    """Test creating and listing versioned trading rules via REST API."""
    # 1. Create a candidate rule version
    res = client.post(
        "/api/rules/versions",
        json={
            "title": "ICT Liquidity Sweep Integration Rule",
            "rule_text": "Enter long after liquidity sweep of Asian low followed by displacement into FVG.",
            "conditions": ["Asian low swept", "Displacement candle on 15m", "FVG created"],
            "exceptions": ["High-impact USD news within 15 minutes"],
            "tags": ["ICT", "Liquidity", "Forex"],
        },
    )
    assert res.status_code == 200
    rule = res.json()
    assert "rule_id" in rule
    assert rule["version"] == 1
    assert rule["state"] == "CANDIDATE"
    rule_id = rule["rule_id"]

    # 2. List rule versions
    list_res = client.get("/api/rules/versions")
    assert list_res.status_code == 200
    versions = list_res.json()
    assert any(v["rule_id"] == rule_id for v in versions)


def test_api_trading_brain_extraction_and_analysis(client):
    """Test qualitative rule extraction and backtest analysis via Gemini Trading Brain."""
    # 1. Rule extraction endpoint
    extract_res = client.post(
        "/api/trading-brain/extract-rule",
        json={
            "lecture_text": "When you see price aggressively sweep the London low, wait for an impulsive displacement back inside the range.",
        },
    )
    assert extract_res.status_code == 200
    extracted = extract_res.json()
    assert "rule_id" in extracted
    assert extracted["state"] == "CANDIDATE"

    # 2. Create and run a backtest
    bt_res = client.post(
        "/api/backtests",
        json={
            "mode": "AI_BACKTEST",
            "asset_class": "FOREX",
            "symbol": "EUR/USD",
            "timeframe": "15m",
            "max_candles": 35,
        },
    )
    assert bt_res.status_code == 200
    bt_id = bt_res.json()["backtest_id"]

    start_res = client.post(f"/api/backtests/{bt_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "COMPLETED"

    # 3. Analyze backtest performance qualitatively
    analyze_res = client.post(
        "/api/trading-brain/analyze-backtest",
        json={"backtest_id": bt_id},
    )
    assert analyze_res.status_code == 200
    analysis = analyze_res.json()
    assert "assessment" in analysis
    assert "recommendation" in analysis


def test_api_backtest_execution_and_controls(client):
    """Test creating, starting, pausing, resuming, and manual trade recording in backtests."""
    # 1. Create a backtest
    create_res = client.post(
        "/api/backtests",
        json={
            "mode": "AI_BACKTEST",
            "asset_class": "CRYPTO",
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "max_candles": 40,
        },
    )
    assert create_res.status_code == 200
    bt_id = create_res.json()["backtest_id"]

    # 2. Pause
    pause_res = client.post(f"/api/backtests/{bt_id}/pause")
    assert pause_res.status_code == 200

    # 3. Resume
    resume_res = client.post(f"/api/backtests/{bt_id}/resume")
    assert resume_res.status_code == 200
    assert resume_res.json()["status"] == "COMPLETED"

    # 4. Fetch trade history
    trades_res = client.get(f"/api/backtests/{bt_id}/trades")
    assert trades_res.status_code == 200

    # 5. Record manual replay trade
    manual_res = client.post(
        f"/api/backtests/{bt_id}/manual-trade",
        json={
            "symbol": "BTC/USD",
            "asset_class": "CRYPTO",
            "direction": "BUY",
            "entry_time": 1700000000.0,
            "entry_price": 50000.0,
            "exit_time": 1700003600.0,
            "exit_price": 51000.0,
            "stop_loss": 49500.0,
            "target_price": 51000.0,
            "quantity": 0.1,
            "confluence_summary": "Manual HTF support retest",
        },
    )
    assert manual_res.status_code == 200
    trade = manual_res.json()
    assert trade["symbol"] == "BTC/USD"
    assert trade["source"] == "USER"


def test_api_promotion_gate_ladder_and_safety(client):
    """Test the strict multi-stage promotion gate ladder and rejection of premature promotions."""
    # 1. Create fresh rule
    rule_res = client.post(
        "/api/rules/versions",
        json={
            "title": "Strict Gated Test Rule",
            "rule_text": "Only trade when all HTF and LTF criteria align.",
        },
    )
    rule_id = rule_res.json()["rule_id"]

    # 2. Premature paper promotion without completed backtest MUST fail with 400
    paper_fail = client.post(
        f"/api/rules/{rule_id}/approve-paper",
        json={"user_confirmed": True, "notes": "Premature attempt"},
    )
    assert paper_fail.status_code == 400
    assert "cannot be promoted to paper without a completed backtest" in paper_fail.json()["detail"]

    # 3. Premature live promotion without backtest & paper MUST fail with 400
    live_fail = client.post(
        f"/api/rules/{rule_id}/approve-live",
        json={"user_confirmed": True, "notes": "Premature attempt"},
    )
    assert live_fail.status_code == 400
    assert "Live promotion rejected" in live_fail.json()["detail"]

    # 4. Check promotion audit record
    prom_res = client.get(f"/api/rules/{rule_id}/promotion")
    assert prom_res.status_code == 200
    record = prom_res.json()
    assert record["rule_version_id"] == rule_id
    assert record["user_backtest_approved"] is False
    assert record["paper_validation_approved"] is False
    assert record["user_live_approved"] is False


def test_api_cross_asset_summary(client):
    """Verify cross-asset summary endpoint preserves Forex and Crypto separation."""
    res = client.get("/api/backtests/cross-asset-summary")
    assert res.status_code == 200
    data = res.json()
    assert "forex_results" in data
    assert "crypto_results" in data
    assert "cross_asset_summary" in data
