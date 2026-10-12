"""
Integration tests for Step 6 Fundamental Market Intelligence API endpoints.
Tests endpoints:
- GET /api/fundamental/status
- GET /api/fundamental/events
- GET /api/fundamental/context/{symbol}
- GET /api/fundamental/analysis/{symbol}
- GET /api/market/situation/{symbol} (verifying presence of Step 6 fields)
- POST /api/fundamental/sync
"""

import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_fundamental_status_endpoint(client):
    res = client.get("/api/fundamental/status")
    assert res.status_code == 200
    data = res.json()
    assert "provider_name" in data
    assert "status" in data
    assert data["provider_name"] == "forex_factory"


def test_fundamental_events_endpoint(client):
    res = client.get("/api/fundamental/events?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)


def test_fundamental_context_endpoint(client):
    res = client.get("/api/fundamental/context/EURUSD")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "EURUSD"
    assert "relevant_currencies" in data
    assert "EUR" in data["relevant_currencies"]
    assert "USD" in data["relevant_currencies"]
    assert "category" in data
    assert "risk_level" in data


def test_fundamental_analysis_endpoint(client):
    res = client.get("/api/fundamental/analysis/EURUSD")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "EURUSD"
    assert "relationship" in data
    assert "action_context" in data
    assert "technical_bias" in data


def test_market_situation_step6_fields(client):
    res = client.get("/api/market/situation/EURUSD")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "EURUSD"
    assert "news_risk" in data
    assert "technical_fundamental_relationship" in data
    assert "recommended_action_context" in data
    assert "relevant_news" in data


def test_fundamental_sync_endpoint(client):
    res = client.post("/api/fundamental/sync")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "provider_status" in data
