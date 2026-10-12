"""Unit tests for Step 5: Unified Settings Hub, Interactive Chart Canvas & Execution Mode Toggle."""

import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app, unified_settings_state, execution_core
from trading_ai.schemas.settings import UnifiedSettings, mask_sensitive_settings
from trading_ai.schemas.enums import ExecutionMode, ExecutionEnvironment
from trading_ai.chat.app_guide import ApplicationGuide


@pytest.fixture
def client():
    return TestClient(app)


def test_step5_settings_endpoint_retrieval_and_masking(client):
    """Verify GET /api/settings returns valid unified schema with sensitive credentials masked."""
    res = client.get("/api/settings")
    assert res.status_code == 200
    data = res.json()
    assert "ai" in data
    assert "voice" in data
    assert "trading" in data
    assert "broker" in data
    assert "notifications" in data
    assert "system" in data

    # Verify secret masking
    assert data["broker"]["api_key_masked"] == "***configured***"
    assert data["ai"]["default_model"] in ["gemini-3.8-flash", "gemini-1.5-flash"]
    assert data["trading"]["execution_mode"] in ["MANUAL", "AUTO"]


def test_step5_settings_update_propagation(client):
    """Verify POST /api/settings updates subsystem states and propagates to ExecutionPolicy."""
    payload = client.get("/api/settings").json()
    payload["trading"]["risk_per_trade_pct"] = 1.5
    payload["trading"]["min_rr_ratio"] = 2.5
    payload["ai"]["temperature"] = 0.35

    res = client.post("/api/settings", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["settings"]["trading"]["risk_per_trade_pct"] == 1.5
    assert data["settings"]["trading"]["min_rr_ratio"] == 2.5
    assert data["settings"]["ai"]["temperature"] == 0.35

    # Verify propagation to active ExecutionPolicy
    active_policy = execution_core.get_or_create_policy("default_user")
    assert active_policy.max_auto_risk_pct == 1.5
    assert active_policy.min_rr_ratio == 2.5


def test_step5_execution_mode_safety_handshake(client):
    """Verify switching to AUTOMATIC gate requires human confirmation, while switching to MANUAL is instantaneous."""
    # 1. Unconfirmed attempt to toggle AUTOMATIC
    res_unconfirmed = client.post("/api/execution/mode", json={"mode": "AUTO", "confirm": False})
    assert res_unconfirmed.status_code == 200
    data_unconfirmed = res_unconfirmed.json()
    assert data_unconfirmed["status"] == "CONFIRMATION_REQUIRED"
    assert data_unconfirmed["requires_confirmation"] is True

    # 2. Confirmed toggle to AUTOMATIC
    res_confirmed = client.post("/api/execution/mode", json={"mode": "AUTO", "confirm": True})
    assert res_confirmed.status_code == 200
    data_confirmed = res_confirmed.json()
    assert data_confirmed["status"] == "SUCCESS"
    assert data_confirmed["execution_mode"] == "AUTO"
    assert data_confirmed["auto_execution_enabled"] is True

    # 3. Toggle back to MANUAL (safe & immediate)
    res_manual = client.post("/api/execution/mode", json={"mode": "MANUAL", "confirm": False})
    assert res_manual.status_code == 200
    data_manual = res_manual.json()
    assert data_manual["status"] == "SUCCESS"
    assert data_manual["execution_mode"] == "MANUAL"
    assert data_manual["auto_execution_enabled"] is False


def test_step5_market_chart_endpoint_ohlcv_and_markings(client):
    """Verify GET /api/market/chart/{symbol} returns full candlestick series and SMC markings."""
    res = client.get("/api/market/chart/BTC/USD?timeframe=15m&limit=30")
    assert res.status_code == 200
    data = res.json()
    assert data["symbol"] == "BTC/USD"
    assert data["timeframe"] == "15m"
    assert data["candle_count"] > 0
    assert len(data["candles"]) == data["candle_count"]

    # Verify candle OHLCV schema
    sample_candle = data["candles"][0]
    assert "open" in sample_candle
    assert "high" in sample_candle
    assert "low" in sample_candle
    assert "close" in sample_candle
    assert "volume" in sample_candle
    assert "timestamp" in sample_candle
    assert sample_candle["high"] >= sample_candle["low"]

    # Verify markings array present
    assert "markings" in data
    assert isinstance(data["markings"], list)


def test_step5_app_guide_feature_promotion_and_grounding():
    """Verify ApplicationGuide authoritative catalog promotes Step 5 features to AVAILABLE."""
    guide = ApplicationGuide()

    # Settings Hub
    feat_settings = guide.find_feature("settings")
    assert feat_settings is not None
    assert feat_settings.status == "AVAILABLE"
    assert feat_settings.feature_id == "settings_page"

    # Interactive Chart
    feat_chart = guide.find_feature("chart")
    assert feat_chart is not None
    assert feat_chart.status == "AVAILABLE"
    assert feat_chart.feature_id == "interactive_chart"

    # Auto Execution Toggle
    feat_toggle = guide.find_feature("auto execution")
    assert feat_toggle is not None
    assert feat_toggle.status == "AVAILABLE"
    assert feat_toggle.feature_id == "auto_execution_toggle"

    # Verify prompt summary includes operational descriptions
    summary = guide.get_system_prompt_summary()
    assert "Dedicated Settings Hub: OPERATIONAL" in summary
    assert "Interactive Chart Canvas: OPERATIONAL" in summary
    assert "Execution Mode Toggle: OPERATIONAL" in summary
