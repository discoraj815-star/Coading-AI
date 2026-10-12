"""Integration tests for Step 18 Correction APIs: TradingView Webhook & Replay Bar Clock."""

import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_tradingview_status(client):
    """Verify TradingView status endpoint returns instructions and unconfigured state."""
    res = client.get("/api/providers/tradingview/status")
    assert res.status_code == 200
    data = res.json()
    assert data["webhook_endpoint"] == "/api/providers/tradingview/webhook"
    assert "instructions" in data


def test_api_tradingview_webhook_ingestion(client):
    """Verify posting a TradingView alert normalizes data into canonical schema."""
    payload = {
        "ticker": "BTCUSD",
        "close": 69200.5,
        "time": 1700050000.0,
        "timeframe": "15m",
        "exchange": "BINANCE",
        "strategy": "FVG_Sweep",
    }
    res = client.post("/api/providers/tradingview/webhook", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ACCEPTED"
    assert data["canonical_symbol"] == "BTC/USD"
    assert data["close"] == 69200.5
    assert data["is_live"] is True


def test_api_replay_clock_status_and_controls(client):
    """Verify Replay Bar Clock status, step, speed, play, pause, reset."""
    # 1. Get status
    res = client.get("/api/backtest/replay/clock/status")
    assert res.status_code == 200
    data = res.json()
    assert "current_timestamp" in data
    assert "replay_mode" in data

    # 2. Set speed to 5x
    res = client.post("/api/backtest/replay/clock/speed", json={"speed": 5.0})
    assert res.status_code == 200
    assert res.json()["speed"] == 5.0

    # 3. Play and Pause
    res = client.post("/api/backtest/replay/clock/play")
    assert res.status_code == 200
    assert res.json()["is_paused"] is False

    res = client.post("/api/backtest/replay/clock/pause")
    assert res.status_code == 200
    assert res.json()["is_paused"] is True

    # 4. Step forward 1 bar
    res = client.post("/api/backtest/replay/clock/step", json={"bars": 1})
    assert res.status_code == 200
    assert res.json()["status"] == "STEPPED"

    # 5. Reset
    res = client.post("/api/backtest/replay/clock/reset")
    assert res.status_code == 200
    assert res.json()["status"] == "RESET"
