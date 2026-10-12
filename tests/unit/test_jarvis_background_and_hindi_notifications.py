"""Unit tests for JARVIS Master Toggle, Hindi Grounded Voice Answers, Notification Pipeline, and Broker Integration."""

import os
import tempfile
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient

from trading_ai.chat.chat_service import AIChatService
from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
from trading_ai.core.mtf_engine import MultiTimeframeEngine
from trading_ai.core.situation_engine import CurrentSituationEngine
from trading_ai.events.gateway import WebSocketGateway
from trading_ai.notifications.orchestrator import NotificationOrchestrator
from trading_ai.notifications.voice import CloudVoiceSynthesizer
from trading_ai.providers.registry import UniversalProviderRegistry
from trading_ai.schemas.chat import ChatMessageRequest
from trading_ai.schemas.cloud import CloudStreamEvent
from trading_ai.schemas.enums import EventPriorityLevel, MarkingType, SetupStatus, UnifiedExecutionTestingMode
from trading_ai.schemas.market import Candle, ChartMarking, MarketSnapshot
from trading_ai.storage.db import DatabaseManager, init_db
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.market_supervisor import MarketSupervisor
from trading_ai.server.app import app


@pytest.fixture
def test_db_repo():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_jarvis.db")
    init_db(db_path)
    db_mgr = DatabaseManager(db_path)
    repo = Repository(db_mgr)
    return repo


@pytest.fixture
def notification_pipeline(test_db_repo):
    gateway = WebSocketGateway(test_db_repo)
    synth = CloudVoiceSynthesizer()
    orchestrator = NotificationOrchestrator(
        repository=test_db_repo,
        gateway=gateway,
        voice_synthesizer=synth,
    )
    return orchestrator


def test_jarvis_status_and_toggle_endpoints():
    """Verify Master JARVIS ON/OFF toggle and status endpoints."""
    client = TestClient(app)

    # 1. Check initial status
    res = client.get("/api/jarvis/status")
    assert res.status_code == 200
    data = res.json()
    assert "state" in data
    assert "label" in data
    assert data["backend_ready"] is True
    assert "active_broker" in data

    # 2. Toggle to OFF
    res_off = client.post("/api/jarvis/toggle", json={"state": "OFF", "spoken_alerts": False})
    assert res_off.status_code == 200
    data_off = res_off.json()
    assert data_off["state"] == "OFF"
    assert "OFF" in data_off["label"]
    assert data_off["spoken_alerts"] is False

    # 3. Toggle back to ON
    res_on = client.post("/api/jarvis/toggle", json={"state": "ON", "spoken_alerts": True})
    assert res_on.status_code == 200
    data_on = res_on.json()
    assert data_on["state"] == "ON"
    assert "READY" in data_on["label"] or "ON" in data_on["label"]
    assert data_on["spoken_alerts"] is True


def test_market_supervisor_dispatches_notification_events(test_db_repo, notification_pipeline):
    """Verify MarketSupervisor emits qualified events to NotificationOrchestrator."""
    provider = SimulatedMarketDataProvider(seed=42)
    reg = UniversalProviderRegistry(test_db_repo)
    mtf = MultiTimeframeEngine(data_provider=provider)
    sit = CurrentSituationEngine(test_db_repo, mtf)
    marker = ChartMarker()

    supervisor = MarketSupervisor(
        repository=test_db_repo,
        provider_registry=reg,
        mtf_engine=mtf,
        situation_engine=sit,
        chart_marker=marker,
        data_provider=provider,
        notification_orchestrator=notification_pipeline,
        active_symbol="EUR/USD",
        active_timeframe="15m",
    )

    now_ts = datetime.now(timezone.utc).timestamp()
    snapshot = MarketSnapshot(
        symbol="EUR/USD",
        timeframe="15m",
        latest_price=1.0950,
        timestamp=now_ts,
        candles=[
            Candle(timestamp=now_ts - 1800, open=1.0880, high=1.0900, low=1.0870, close=1.0890, volume=100),
            Candle(timestamp=now_ts - 900, open=1.0890, high=1.0960, low=1.0885, close=1.0950, volume=250),
            Candle(timestamp=now_ts, open=1.0950, high=1.0980, low=1.0940, close=1.0970, volume=300),
        ],
    )

    eval_result = supervisor.evaluate_market(symbol="EUR/USD", timeframe="15m", snapshot=snapshot)
    assert eval_result["success"] is True
    assert eval_result["count"] > 0

    # Verify notifications were generated and saved in repository
    records = test_db_repo.get_notification_records(limit=10)
    assert len(records) > 0
    latest = records[0]
    sym = latest.get("symbol") if isinstance(latest, dict) else latest.symbol
    msg_hindi = latest.get("message_hindi") if isinstance(latest, dict) else latest.message_hindi
    assert sym == "EUR/USD"
    assert msg_hindi is not None
    assert len(msg_hindi) > 0


def test_jarvis_grounded_hindi_responses(test_db_repo):
    """Verify JARVIS answers marking and broker questions with grounded Hindi explanations."""
    # Persist a canonical marking
    now_ts = datetime.now(timezone.utc).timestamp()
    test_db_repo.save_chart_marking(
        ChartMarking(
            marking_id="mark_eurusd_fvg",
            symbol="EUR/USD",
            timeframe="15m",
            marking_type=MarkingType.FVG,
            concept="Fair Value Gap",
            direction="BULLISH",
            price=1.0850,
            confidence=0.92,
            status="ACTIVE",
            rule_reference="RULE-ICT-FVG-01",
            source="SUPERVISOR",
            broker_id="paper_broker",
            evidence_json={
                "conditions": ["3-candle imbalance detected", "Displacement above liquidity pool"],
                "evidence": ["Candle 1 high: 1.0820", "Candle 3 low: 1.0850"],
            },
            timestamp=now_ts,
        )
    )

    chat_service = AIChatService(repository=test_db_repo)

    # 1. Why was marking created? (Hindi prompt)
    req1 = ChatMessageRequest(
        message="यह marking क्यों की गई थी?",
        symbol="EUR/USD",
        timeframe="15m",
        session_id="test_hindi_sess",
    )
    res1 = chat_service.process_message(req1)
    assert "FVG" in res1.response_text or "Fair Value Gap" in res1.response_text
    assert "EUR/USD" in res1.response_text
    assert "RULE-ICT-FVG-01" in res1.response_text
    assert "92%" in res1.response_text
    assert "यह marking" in res1.speech_text or "यह marking" in res1.response_text

    # 2. Active broker query (Hindi prompt)
    req2 = ChatMessageRequest(
        message="Active broker कौन सा है?",
        symbol="EUR/USD",
        session_id="test_hindi_sess",
    )
    res2 = chat_service.process_message(req2)
    assert "Broker" in res2.response_text
    assert "Broker Connection ≠ Trade Authorization" in res2.response_text or "स्मरण रहे" in res2.response_text

    # 3. Notification query (Hindi prompt)
    req3 = ChatMessageRequest(
        message="आखिरी message में क्या आया था?",
        symbol="EUR/USD",
        session_id="test_hindi_sess",
    )
    res3 = chat_service.process_message(req3)
    assert res3.response_text is not None
    assert len(res3.response_text) > 0


def test_notification_convenience_endpoints():
    """Verify /api/notifications/recent and /api/notifications/subscribe endpoints."""
    client = TestClient(app)

    # 1. Subscribe phone device
    sub_res = client.post("/api/notifications/subscribe", json={
        "device_id": "user_pixel_9",
        "platform": "android",
        "push_token": "fcm_token_sample_12345",
    })
    assert sub_res.status_code == 200
    sub_data = sub_res.json()
    assert sub_data["status"] == "SUBSCRIBED"
    assert sub_data["device_id"] == "user_pixel_9"

    # 2. Get recent notifications
    rec_res = client.get("/api/notifications/recent?limit=5")
    assert rec_res.status_code == 200
    assert isinstance(rec_res.json(), list)


def test_safety_invariants_preserved_when_jarvis_toggled():
    """Verify that toggling JARVIS OFF never compromises execution gating or broker isolation."""
    client = TestClient(app)

    # Toggle JARVIS to OFF
    client.post("/api/jarvis/toggle", json={"state": "OFF"})

    # Check execution environment remains paper/gated
    mode_res = client.post("/api/execution/environment-mode", json={"mode": "PAPER"})
    assert mode_res.status_code == 200

    # Ensure AUTO_LIVE without broker approval fails safely
    auto_res = client.post("/api/execution/environment-mode", json={"mode": "AUTO_LIVE"})
    # Either rejected by gate or rejected by broker approval check
    assert auto_res.status_code in (200, 400, 422)
    if auto_res.status_code == 200:
        data = auto_res.json()
        assert data.get("success") is False or "error" in data or "rejected" in str(data).lower()
