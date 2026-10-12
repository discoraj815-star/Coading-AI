"""Tests for Live Broker AI Marking, Dynamic Broker Integration, MarketSupervisor, and JARVIS Grounding."""

import pytest
import sqlite3
from unittest.mock import MagicMock
from trading_ai.schemas.enums import (
    MarkingType,
    MarkingValidity,
    UnifiedExecutionTestingMode,
    ExecutionEnvironment,
    ProviderStatus,
)
from trading_ai.schemas.market import ChartMarking
from trading_ai.schemas.provider import ProviderMetadata
from trading_ai.providers.broker import DynamicBrokerProvider
from trading_ai.providers.registry import UniversalProviderRegistry
from trading_ai.supervisor.market_supervisor import MarketSupervisor
from trading_ai.chat.chat_service import AIChatService
from trading_ai.storage.repository import StorageRepository
from trading_ai.storage.db import init_db
from fastapi.testclient import TestClient
from trading_ai.server.app import app


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "test_trading.db")
    init_db(db_path)
    return StorageRepository(db_path)


def test_dynamic_broker_masked_credentials_and_isolation():
    """Verify dynamic broker initialization, secret masking, and strict trade isolation."""
    provider = DynamicBrokerProvider(
        provider_id="binance_test_live",
        name="Binance Test Live",
        api_key="ak_live_1234567890abcdef",
        api_secret="sk_secret_super_classified_9876",
        server="api.binance.com",
        execution_state="PAPER_ONLY",
    )

    meta = provider.get_metadata()
    assert meta.provider_id == "binance_test_live"
    assert meta.provider_name == "Binance Test Live"
    assert meta.provider_type == "BROKER"
    # Verify zero secret leakage in metadata
    assert "super_classified" not in str(meta)
    assert meta.masked_api_key == "ak_l************cdef"
    assert meta.masked_api_secret == "••••••••••••"
    assert meta.api_key_status == "CONFIGURED"
    assert meta.secret_status == "CONFIGURED"
    # Strict separation: Broker Connection != Trade Authorization
    assert meta.execution_state == "PAPER_ONLY"

    # Test non-destructive connection
    conn_result = provider.test_connection()
    assert conn_result["success"] is True
    assert conn_result["latency_ms"] >= 0
    assert provider.status == ProviderStatus.CONNECTED


def test_registry_add_and_switch_broker():
    """Verify dynamic broker addition and active broker switching in UniversalProviderRegistry."""
    registry = UniversalProviderRegistry()
    initial_count = len(registry.list_providers())

    broker = registry.add_broker(
        broker_id="bybit_spot",
        name="Bybit Spot Market",
        api_key="bybit_key_abc123",
        api_secret="bybit_secret_xyz789",
        execution_state="PAPER_ONLY",
    )
    assert broker.provider_id == "bybit_spot"
    assert len(registry.list_providers()) == initial_count + 1

    # Switch active broker
    success = registry.set_active_broker("bybit_spot")
    assert success is True
    active = registry.get_active_broker()
    assert active is not None
    assert active.provider_id == "bybit_spot"
    assert active.get_metadata().is_active_broker is True


def test_market_supervisor_generates_canonical_markings_and_clutter_filter(repo):
    """Verify MarketSupervisor evaluates market structure and applies clutter filtering."""
    supervisor = MarketSupervisor(storage_repo=repo)

    result = supervisor.evaluate_market(symbol="BTC/USD", timeframe="15m")
    assert result["success"] is True
    assert result["symbol"] == "BTC/USD"
    assert result["timeframe"] == "15m"

    markings = result["markings"]
    assert len(markings) > 0
    # Clutter filter enforces maximum limit
    assert len(markings) <= 15

    # Verify canonical attributes on markings
    marking_types = [m.marking_type for m in markings]
    assert any(m_type in [MarkingType.FAIR_VALUE_GAP, MarkingType.ORDER_BLOCK, MarkingType.BREAK_OF_STRUCTURE, MarkingType.SWING_HIGH, MarkingType.SWING_LOW] for m_type in marking_types)

    for m in markings:
        assert m.symbol == "BTC/USD"
        assert m.source in ["LIVE", "SUPERVISOR", "MANUAL"]
        assert m.confidence >= 0.5
        assert m.status in ["ACTIVE", "VALID"]
        assert m.visibility == "VISIBLE"


def test_canonical_marking_backtest_source_parity(repo):
    """Verify backtest and live use the exact same canonical ChartMarking schema."""
    live_marking = ChartMarking(
        symbol="BTC/USD",
        timeframe="15m",
        marking_type=MarkingType.ORDER_BLOCK,
        label="H1 Bullish Order Block",
        concept="ORDER_BLOCK",
        price=64250.0,
        price_range=[64200.0, 64300.0],
        direction="BULLISH",
        source="LIVE",
        rule_reference="ICT_OB_ENTRY_01",
        status="ACTIVE",
        confidence=0.88,
    )
    repo.save_chart_marking(live_marking)

    backtest_marking = ChartMarking(
        symbol="BTC/USD",
        timeframe="15m",
        marking_type=MarkingType.ORDER_BLOCK,
        label="Backtest Replay Order Block",
        concept="ORDER_BLOCK",
        price=64250.0,
        price_range=[64200.0, 64300.0],
        direction="BULLISH",
        source="BACKTEST",
        backtest_run_id="bt_run_9999",
        rule_reference="ICT_OB_ENTRY_01",
        status="ACTIVE",
        confidence=0.88,
    )
    repo.save_chart_marking(backtest_marking)

    saved_live = repo.get_chart_markings("BTC/USD", only_valid=True)
    assert any(m.source == "LIVE" for m in saved_live)

    saved_bt = repo.get_chart_markings_for_backtest("bt_run_9999")
    assert len(saved_bt) == 1
    assert saved_bt[0].source == "BACKTEST"
    assert saved_bt[0].backtest_run_id == "bt_run_9999"
    assert saved_bt[0].concept == live_marking.concept


def test_jarvis_grounding_active_broker_and_zero_hallucinations(repo):
    """Verify JARVIS answers accurately about active broker and does NOT hallucinate nonexistent markings."""
    chat_service = AIChatService(storage_repo=repo)
    supervisor = MarketSupervisor(storage_repo=repo)
    chat_service.set_market_supervisor(supervisor)

    # 1. Ask about active broker
    resp_broker = chat_service.process_message("What is my currently active broker and connection status?")
    assert "Active Broker" in resp_broker.reply
    assert "SIMULATED PAPER" in resp_broker.reply or "CONNECTED" in resp_broker.reply

    # 2. Ask about unknown asset with NO markings: MUST NOT hallucinate
    resp_empty = chat_service.process_message("What markings are plotted on XYZ/USD right now?")
    assert "No markings plotted" in resp_empty.reply or "No active AI chart markings" in resp_empty.reply
    # Ensure it did not claim to find anything
    assert "BOS found" not in resp_empty.reply
    assert "CHOCH found" not in resp_empty.reply


def test_server_broker_and_supervisor_endpoints():
    """Verify HTTP API endpoints for brokers, active broker, supervisor evaluation, and execution mode."""
    client = TestClient(app)

    # 1. Get brokers
    res = client.get("/api/brokers")
    assert res.status_code == 200
    brokers = res.json()
    assert len(brokers) > 0

    # 2. Add dynamic broker
    add_payload = {
        "provider_id": "test_broker_api",
        "provider_name": "API Test Broker",
        "provider_type": "BROKER",
        "api_key": "test_api_key_12345",
        "api_secret": "test_secret_98765",
        "execution_state": "PAPER_ONLY",
    }
    res_add = client.post("/api/brokers/add", json=add_payload)
    assert res_add.status_code == 200
    data_add = res_add.json()
    assert data_add["success"] is True
    assert data_add["broker"]["provider_id"] == "test_broker_api"
    # Verify masked credentials in API response (zero secrets exposed)
    assert "98765" not in str(data_add)

    # 3. Switch active broker
    res_switch = client.post("/api/brokers/test_broker_api/switch")
    assert res_switch.status_code == 200
    data_switch = res_switch.json()
    assert data_switch["success"] is True
    assert data_switch["active_broker"]["provider_id"] == "test_broker_api"

    # 4. Get active broker
    res_active = client.get("/api/brokers/active")
    assert res_active.status_code == 200
    data_active = res_active.json()
    assert data_active["active_broker"]["provider_id"] == "test_broker_api"

    # 5. Evaluate supervisor
    res_eval = client.post("/api/supervisor/evaluate", json={"symbol": "BTC/USD", "timeframe": "15m"})
    assert res_eval.status_code == 200
    data_eval = res_eval.json()
    assert data_eval["success"] is True
    assert len(data_eval["markings"]) > 0

    # 6. Change execution mode
    res_mode = client.post("/api/execution/environment-mode", json={"environment": "PAPER", "mode": "PAPER"})
    assert res_mode.status_code == 200
    data_mode = res_mode.json()
    assert data_mode["success"] is True
    assert data_mode["current_mode"] == "PAPER"


def test_ui_dom_contains_required_elements():
    """Verify that index.html contains all newly introduced buttons, modals, and elements."""
    with open("trading_ai/ui/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert 'id="btn-add-broker"' in html
    assert 'id="step18-add-broker-modal"' in html
    assert 'id="live-broker-marking-bar"' in html
    assert 'id="live-banner-broker"' in html
    assert 'id="live-banner-markings-count"' in html
    assert 'id="exec-btn-backtest"' in html
    assert 'id="exec-btn-paper"' in html
    assert 'id="exec-btn-manual"' in html
    assert 'id="exec-btn-auto"' in html
    assert 'id="btn-eval-supervisor"' in html
    assert "openAddBrokerModal()" in html
    assert "saveNewBrokerUI()" in html
    assert "switchActiveBrokerUI" in html
    assert "setExecutionModeUI" in html
    assert "evaluateSupervisorLiveUI" in html
