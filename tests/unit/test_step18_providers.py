import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from trading_ai.schemas.provider import (
    ProviderCategory,
    ProviderLifecycleState,
    TradingProviderExecutionState,
    ProviderMetadata,
)
from trading_ai.providers import (
    UniversalProviderRegistry,
    UniversalConnectionManager,
    StandardHistoricalBacktestProvider,
    TradingLeadsBacktestProvider,
    PaperBrokerProvider,
    DeltaExchangeBrokerProvider,
    XMBrokerProvider,
    MetaTrader5TerminalProvider,
    SimulatedMarketDataProviderAdapter,
    LiveFeedMarketDataProvider,
    OrderflowFeedProvider,
    FootprintFeedProvider,
    VolumeProfileFeedProvider,
    ForexFactoryNewsProvider,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def temp_repo(tmp_path):
    db_file = tmp_path / "test_providers.db"
    db = DatabaseManager(str(db_file))
    return Repository(db)


def test_registry_registration_and_lookup():
    reg = UniversalProviderRegistry()
    broker = PaperBrokerProvider()
    reg.register_provider(broker)

    meta = reg.get_provider_metadata("paper_broker")
    assert meta is not None
    assert meta.provider_id == "paper_broker"
    assert meta.provider_type == ProviderCategory.BROKER
    assert meta.trading_state == TradingProviderExecutionState.PAPER_ONLY

    active = reg.get_active_provider(ProviderCategory.BROKER)
    assert active is not None
    assert active.metadata.provider_id == "paper_broker"


def test_registry_active_provider_switching():
    reg = UniversalProviderRegistry()
    paper = PaperBrokerProvider()
    delta = DeltaExchangeBrokerProvider()
    reg.register_provider(paper)
    reg.register_provider(delta)

    # Initial active is paper
    assert reg.get_active_provider(ProviderCategory.BROKER).metadata.provider_id == "paper_broker"

    # Switch active broker
    success = reg.set_active_provider(ProviderCategory.BROKER, "delta_exchange")
    assert success is True
    assert reg.get_active_provider(ProviderCategory.BROKER).metadata.provider_id == "delta_exchange"


def test_backtest_provider_strict_time_slicing():
    provider = StandardHistoricalBacktestProvider()
    meta = provider.metadata
    assert meta.provider_type == ProviderCategory.BACKTEST
    assert meta.status == ProviderLifecycleState.CONNECTED

    # Verify no lookahead: asking for cutoff at 1000 should strictly return data <= 1000
    candles = provider.get_historical_candles("EURUSD", "15m", start_time=0.0, end_time=1000.0, limit=100)
    assert isinstance(candles, list)
    for c in candles:
        assert c["timestamp"] <= 1000.0


def test_trading_leads_reality_check():
    provider = TradingLeadsBacktestProvider()
    meta = provider.metadata
    assert meta.status in (ProviderLifecycleState.DISCOVERED, ProviderLifecycleState.CONFIGURED)
    assert "ORDERBOOK_HEATMAP_EXPORT" in meta.capabilities

    # Historical requests gracefully report unconfigured status without throwing synthetic fills
    data = provider.get_historical_candles("BTCUSD", "1h", start_time=0.0, end_time=100.0, limit=10)
    assert data == []


def test_delta_exchange_broker_safety():
    delta = DeltaExchangeBrokerProvider()
    meta = delta.metadata
    assert meta.provider_id == "delta_exchange"
    assert meta.trading_state == TradingProviderExecutionState.LIVE_LOCKED
    assert "API_KEY_AND_HMAC_SHA256" in meta.authentication_type

    # Connection test without credentials safely indicates not configured
    unconfigured_res = delta.test_connection()
    assert unconfigured_res.success is False
    assert "NOT_CONFIGURED" in unconfigured_res.message

    # Initializing with credentials allows safe connectivity check without placing orders
    delta.initialize({"api_key": "dummy", "api_secret": "dummy_secret"})
    test_res = delta.test_connection()
    assert test_res.provider_id == "delta_exchange"
    assert test_res.success is True


def test_xm_broker_routes_via_mt5():
    xm = XMBrokerProvider()
    meta = xm.metadata
    assert meta.provider_id == "xm_broker"
    assert meta.authentication_type == "MT5_TERMINAL_IPC"
    assert meta.trading_state == TradingProviderExecutionState.LIVE_LOCKED
    assert "MT5_TERMINAL_INTEGRATION" in meta.capabilities


def test_metatrader5_terminal_not_connected_when_offline():
    mt5 = MetaTrader5TerminalProvider()
    meta = mt5.metadata
    assert meta.provider_id == "metatrader5"
    assert meta.provider_type == ProviderCategory.TRADING_TERMINAL

    # Without a running MetaTrader terminal process, connection test marks unreachable
    res = mt5.test_connection()
    assert res.provider_id == "metatrader5"
    assert res.success is False
    assert "NOT_CONFIGURED" in res.message or "NOT_CONNECTED" in res.message


def test_market_data_simulated_vs_live_labeling():
    sim = SimulatedMarketDataProviderAdapter()
    live = LiveFeedMarketDataProvider()

    assert sim.metadata.status == ProviderLifecycleState.CONNECTED
    assert sim.metadata.trading_state == TradingProviderExecutionState.PAPER_ONLY
    res = sim.test_connection()
    assert res.success is True
    assert res.details.get("is_simulated") is True

    assert live.metadata.trading_state == TradingProviderExecutionState.LIVE_PENDING_APPROVAL


def test_orderflow_footprint_vp_gating():
    of = OrderflowFeedProvider()
    fp = FootprintFeedProvider()
    vp = VolumeProfileFeedProvider()

    # When unconfigured/disconnected, must return explicit UNAVAILABLE dict, zero synthetic fabrication
    of_data = of.get_delta_snapshot("BTC/USD", "15m")
    assert of_data["is_available"] is False
    assert of_data["reason"] == "ORDERFLOW_DATA_UNAVAILABLE"

    fp_data = fp.get_footprint_clusters("BTC/USD", "15m")
    assert fp_data["is_available"] is False
    assert fp_data["reason"] == "FOOTPRINT_DATA_UNAVAILABLE"
    assert fp_data["clusters"] == []

    vp_data = vp.get_volume_profile("EURUSD", "1h")
    assert vp_data["is_available"] is False
    assert vp_data["reason"] == "VOLUME_PROFILE_DATA_UNAVAILABLE"
    assert vp_data["poc"] is None


def test_forex_factory_news_provider():
    news = ForexFactoryNewsProvider()
    meta = news.metadata
    assert meta.provider_id == "forex_factory"
    assert meta.provider_type == ProviderCategory.NEWS
    assert meta.status == ProviderLifecycleState.CONNECTED

    res = news.test_connection()
    assert res.provider_id == "forex_factory"
    assert res.success is True

    # Calendar events return valid structure
    events = news.get_economic_calendar()
    assert isinstance(events, list)
    for ev in events:
        assert "title" in ev or "event" in ev
        assert "currency" in ev
        assert "impact" in ev


def test_connection_manager_secret_masking(temp_repo):
    reg = UniversalProviderRegistry()
    delta = DeltaExchangeBrokerProvider()
    reg.register(delta)

    mgr = UniversalConnectionManager(reg, temp_repo)

    raw_config = {
        "api_key": "actual_delta_key_999",
        "api_secret": "super_secret_private_key_abc",
        "account_id": "12345",
    }

    # Save config
    mgr.save_connection_config("delta_exchange", raw_config)

    # Fetch masked config for UI / API
    masked = mgr.get_masked_config("delta_exchange")
    assert "MASKED" in masked["api_key"]
    assert "MASKED" in masked["api_secret"]
    assert masked["account_id"] == "12345"  # Non-secret preserved

    # Verify secret is never printed in string representations
    assert "super_secret_private_key_abc" not in str(masked)
