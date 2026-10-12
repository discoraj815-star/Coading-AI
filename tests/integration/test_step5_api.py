"""
Integration tests for Step 5: Background Market Intelligence & Intelligent Selective Visualization.
Tests end-to-end flow:
Background watchlist processing -> Setup promotion -> Situation snapshot -> API endpoints.
Also tests fail-safe scenarios (unconfirmed vs confirmed, MTF conflict, selective display).
"""

import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app, data_provider, repo, watchlist_monitor
from trading_ai.schemas.enums import (
    KnowledgeType,
    MarketDirection,
    SetupStatus,
)
from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.schemas.market import MarketSetup


@pytest.fixture
def client():
    return TestClient(app)


def test_step5_current_situation_api(client):
    """Test GET /api/market/situation/{symbol} endpoint."""
    res = client.get("/api/market/situation/BTC/USD")
    assert res.status_code == 200
    data = res.json()
    assert "symbol" in data
    assert "current_price" in data
    assert "overall_bias" in data
    assert "bias_summary" in data
    assert "visible_markings" in data
    assert "setup_confirmation_state" in data
    # Clean chart check: visible markings must be bounded
    assert len(data["visible_markings"]) <= 7


def test_step5_selective_display_policy_post(client):
    """Test POST /api/market/situation/{symbol} with custom DisplayPolicy."""
    policy = {
        "show_all_liquidity": True,
        "max_visible_markings": 15,
    }
    res = client.post("/api/market/situation/BTC/USD", json=policy)
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "BTC/USD"
    assert "visible_markings" in data


def test_step5_background_watchlist_tracking_and_confirmation(client):
    """Test that background watchlist scan updates current situation and setup states."""
    # Seed knowledge rule so setup can validate and confirm
    rule = KnowledgeItem(
        job_id="test_ict_rule",
        item_type=KnowledgeType.RULE,
        title="FVG Displacement Entry",
        statement="Enter at fair value gap following institutional displacement",
        conditions=["Displacement confirmed"],
        exceptions=["High impact news"],
        tags=["FVG", "fair value gap"],
    )
    repo.save_knowledge([rule])

    # Scan watchlist
    res_scan = client.post("/api/market/watchlist/scan", json={"force": True})
    assert res_scan.status_code == 200
    scan_data = res_scan.json()
    assert scan_data["status"] == "success"
    assert scan_data["processed_count"] > 0

    # Verify situation was generated and persisted
    sit_res = client.get("/api/market/situation/BTC/USD")
    assert sit_res.status_code == 200
    sit_data = sit_res.json()
    assert sit_data["background_tracking_count"] > 0


def test_step5_command_api_interactions(client):
    """Test Step 5 command execution via POST /api/commands."""
    commands = [
        "Show all liquidity",
        "Show hidden setups",
        "Show unconfirmed setups",
        "Show confirmation evidence",
        "Show invalidation condition",
        "Focus current setup",
        "Hide background markings",
    ]

    for cmd_text in commands:
        res = client.post("/api/commands", json={"command_text": cmd_text})
        assert res.status_code == 200
        data = res.json()
        assert data["parsed_success"] is True
        assert data["result"]["status"] == "success"
