"""Integration tests for Phase 1 AI Chat and Application Guide REST endpoints."""

import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_api_chat_guide_endpoint(client):
    """GET /api/chat/guide returns application component registry and tour."""
    res = client.get("/api/chat/guide")
    assert res.status_code == 200
    data = res.json()
    assert "features" in data
    assert "tabs" in data
    assert "tour_summary" in data
    assert len(data["tabs"]) >= 18
    assert "TRADING AI ASSISTANT" in data["tour_summary"]


def test_api_chat_message_tour(client):
    """POST /api/chat/message returns structured tour answer and navigation actions."""
    payload = {
        "session_id": "test_api_session_tour",
        "message": "Give me a complete tour of the application.",
        "symbol": "EUR/USD",
        "current_tab": "tab-notes",
    }
    res = client.post("/api/chat/message", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == "test_api_session_tour"
    assert "ai_message" in data
    assert "Tab 1: Notes" in data["ai_message"] or "TRADING AI ASSISTANT" in data["ai_message"]
    assert data["navigation_action"] is not None
    assert data["navigation_action"]["tab_id"] == "tab-market"


def test_api_chat_message_where_is_watchlist(client):
    """POST /api/chat/message returns exact location of Watchlist with tab-market navigation."""
    payload = {
        "session_id": "test_api_session_loc",
        "message": "Where is the Watchlist?",
        "symbol": "EUR/USD",
        "current_tab": "tab-notes",
    }
    res = client.post("/api/chat/message", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "Tab 9" in data["ai_message"] or "Live Market" in data["ai_message"]
    assert data["navigation_action"] is not None
    assert data["navigation_action"]["tab_id"] == "tab-market"


def test_api_chat_history_and_clear(client):
    """POST /api/chat/message followed by GET /api/chat/history and POST /api/chat/clear."""
    sess = "test_api_session_history"
    # Send message 1
    res1 = client.post(
        "/api/chat/message",
        json={"session_id": sess, "message": "What is the AI currently doing right now?", "symbol": "EUR/USD"},
    )
    assert res1.status_code == 200

    # Send message 2
    res2 = client.post(
        "/api/chat/message",
        json={"session_id": sess, "message": "Where is the Watchlist?", "symbol": "EUR/USD"},
    )
    assert res2.status_code == 200

    # Check history
    h_res = client.get(f"/api/chat/history?session_id={sess}")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert len(h_data["turns"]) >= 2

    # Clear history
    c_res = client.post("/api/chat/clear", json={"session_id": sess})
    assert c_res.status_code == 200
    assert c_res.json()["status"] == "CLEARED"

    # Verify history empty
    h_res2 = client.get(f"/api/chat/history?session_id={sess}")
    assert h_res2.status_code == 200
    assert len(h_res2.json()["turns"]) == 0
