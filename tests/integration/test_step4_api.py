"""
Integration tests for Step 4 Live Market Intelligence FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app
from trading_ai.schemas.enums import SetupStatus, WatchlistMonitorStatus, ScreenAccessStatus


@pytest.fixture
def client():
    return TestClient(app)


def test_market_snapshot_endpoint(client):
    res = client.get("/api/market/snapshot/BTC/USD?timeframe=15m&limit=30")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "BTC/USD"
    assert data["timeframe"] == "15m"
    assert len(data["candles"]) == 30
    assert data["latest_price"] > 0


def test_market_analysis_endpoints(client):
    # GET analysis
    res = client.get("/api/market/analysis/ETH/USD")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "ETH/USD"
    assert "overall_bias" in data
    assert "htf_analysis" in data
    assert "mtf_analysis" in data
    assert "ltf_analysis" in data

    # POST fresh analysis
    post_res = client.post("/api/market/analysis/ETH/USD")
    assert post_res.status_code == 200
    assert post_res.json()["symbol"] == "ETH/USD"


def test_market_setups_and_markings_endpoints(client):
    # Ensure analysis has been run to populate setups and markings
    client.post("/api/market/analysis/SOL/USD")

    # Setups
    setups_res = client.get("/api/market/setups/SOL/USD")
    assert setups_res.status_code == 200
    setups_data = setups_res.json()
    assert setups_data["symbol"] == "SOL/USD"
    assert "setups" in setups_data

    # Markings
    markings_res = client.get("/api/market/markings/SOL/USD")
    assert markings_res.status_code == 200
    markings_data = markings_res.json()
    assert markings_data["symbol"] == "SOL/USD"
    assert "markings" in markings_data

    # Levels
    levels_res = client.get("/api/market/levels/SOL/USD")
    assert levels_res.status_code == 200
    assert levels_res.json()["symbol"] == "SOL/USD"


def test_watchlist_and_monitor_endpoints(client):
    # Watchlist overview
    wl_res = client.get("/api/market/watchlist")
    assert wl_res.status_code == 200
    wl_data = wl_res.json()
    assert "watchlist" in wl_data
    assert len(wl_data["watchlist"]) >= 3

    # Scan watchlist
    scan_res = client.post("/api/market/watchlist/scan", json={"force": True})
    assert scan_res.status_code == 200
    scan_data = scan_res.json()
    assert scan_data["status"] == "success"
    assert "BTC/USD" in scan_data["summary"]

    # Monitor status
    status_res = client.get("/api/market/monitor/status")
    assert status_res.status_code == 200

    # Stop monitor
    stop_res = client.post("/api/market/monitor/stop")
    assert stop_res.status_code == 200
    assert stop_res.json()["monitor_status"] == WatchlistMonitorStatus.STOPPED.value

    # Start monitor
    start_res = client.post("/api/market/monitor/start")
    assert start_res.status_code == 200
    assert start_res.json()["monitor_status"] == WatchlistMonitorStatus.MONITORING.value


def test_screen_permissions_lifecycle(client):
    # 1. Request permission
    req_res = client.post(
        "/api/screen/request",
        json={"app_name": "TradingView", "window_title": "Charts - Brave"},
    )
    assert req_res.status_code == 200
    perm = req_res.json()
    perm_id = perm["permission_id"]
    assert perm["status"] == ScreenAccessStatus.SCREEN_ACCESS_REQUESTED.value

    # 2. Grant permission
    grant_res = client.post(f"/api/screen/{perm_id}/grant")
    assert grant_res.status_code == 200
    assert grant_res.json()["status"] == ScreenAccessStatus.SCREEN_ACCESS_GRANTED.value

    # 3. List permissions
    list_res = client.get("/api/screen/permissions")
    assert list_res.status_code == 200
    assert list_res.json()["active_permission"]["permission_id"] == perm_id

    # 4. Revoke permission
    revoke_res = client.post(f"/api/screen/{perm_id}/revoke")
    assert revoke_res.status_code == 200
    assert revoke_res.json()["status"] == ScreenAccessStatus.SCREEN_ACCESS_REVOKED.value


def test_step4_commands_via_api(client):
    cmd_res = client.post(
        "/api/commands",
        json={"command_text": "Analyze BTC/USD"},
    )
    assert cmd_res.status_code == 200
    data = cmd_res.json()
    assert data["parsed_success"] is True
    assert data["result"]["status"] == "success"
    assert "overall_bias" in data["result"]
