"""Integration tests for Step 15 REST API Endpoints."""

import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_plugins_endpoints(client):
    """Test listing plugins, health summary, enable, disable, and invoking actions."""
    # 1. List plugins
    res = client.get("/api/plugins")
    assert res.status_code == 200
    plugins = res.json()
    assert len(plugins) >= 3
    plugin_ids = [p["plugin_id"] for p in plugins]
    assert "plugin_forex_factory" in plugin_ids
    assert "plugin_ta_engine" in plugin_ids
    assert "plugin_broker_read" in plugin_ids

    # 2. Health summary
    h_res = client.get("/api/plugins/health")
    assert h_res.status_code == 200
    h_data = h_res.json()
    assert h_data["total_plugins"] >= 3
    assert h_data["all_healthy"] is True

    # 3. Invoke plugin
    inv_res = client.post("/api/plugins/plugin_ta_engine/invoke", json={"action": "compute", "params": {"symbol": "EURUSD"}})
    assert inv_res.status_code == 200
    inv_data = inv_res.json()
    assert inv_data["success"] is True
    assert inv_data["data"]["rsi_14"] == 52.4

    # 4. Disable and verify
    dis_res = client.post("/api/plugins/plugin_ta_engine/disable")
    assert dis_res.status_code == 200
    assert dis_res.json()["status"] == "DISABLED"

    # Invocation while disabled fails
    dis_inv = client.post("/api/plugins/plugin_ta_engine/invoke", json={"action": "compute"})
    assert dis_inv.status_code == 200
    assert dis_inv.json()["success"] is False
    assert "disabled" in dis_inv.json()["error"]

    # Re-enable
    en_res = client.post("/api/plugins/plugin_ta_engine/enable")
    assert en_res.status_code == 200
    assert en_res.json()["status"] == "ACTIVE"


def test_api_ai_core_comprehend(client):
    """Test multilingual comprehension API with Hindi synthesis."""
    res = client.post("/api/ai-core/comprehend", json={
        "text": "Humein liquidity sweep ke baad displacement ka wait karna hai."
    })
    assert res.status_code == 200
    data = res.json()
    assert data["detected_language"] == "Hinglish"
    assert "Displacement" in data["extracted_concepts"]
    assert "LiquiditySweep" in data["extracted_concepts"]
    assert "displacement" in data["hindi_explanation"].lower()


def test_api_ai_core_epistemic_state(client):
    """Test four-layer epistemic state endpoint."""
    res = client.get("/api/ai-core/epistemic-state?symbol=EURUSD")
    assert res.status_code == 200
    data = res.json()
    assert "what_i_learned" in data
    assert "what_i_observe" in data
    assert "what_i_infer" in data
    assert "what_i_decide" in data
    assert "overall_epistemic_state" in data


def test_api_ai_core_review_concept(client):
    """Test iterative comprehension review endpoint."""
    res = client.post("/api/ai-core/review-concept", json={
        "job_id": "job_api_test",
        "concept_title": "ICT Liquidity Sweep",
        "initial_confidence": 0.60,
        "audio_evidence": "Clear audio statement from mentor",
        "visual_evidence": "Chart shows clean sweep and rejection",
        "transcript_context": "Sweep previous low",
    })
    assert res.status_code == 200
    data = res.json()
    assert data["iterations_run"] >= 1
    assert data["final_confidence"] >= 0.70


def test_api_video_streaming_and_chunk_barrier(client):
    """Test large video streaming job creation and chunk processing."""
    # 1. Create 50GB job
    c_res = client.post("/api/video-processing/jobs/create", json={
        "job_id": "vjob_api_50gb",
        "file_path": "data/mentorship_lecture.mp4",
        "file_size_bytes": 50 * (1024**3),
        "duration_seconds": 60.0,
        "chunk_duration": 30.0,
    })
    assert c_res.status_code == 200
    c_data = c_res.json()
    assert c_data["total_chunks"] == 2
    assert c_data["status"] == "PROCESSING"

    # 2. Process chunk 1
    p1 = client.post("/api/video-processing/jobs/vjob_api_50gb/process-chunk", json={
        "chunk_index": 1,
        "start_time": 0.0,
        "end_time": 30.0,
        "simulate_failure": False,
    })
    assert p1.status_code == 200
    assert p1.json()["status"] == "COMMITTED"

    # 3. List video jobs
    list_res = client.get("/api/video-processing/jobs")
    assert list_res.status_code == 200
    assert any(j["job_id"] == "vjob_api_50gb" for j in list_res.json())


def test_api_live_decision_evaluate(client):
    """Test proactive live decision evaluation with concept mapping and Hindi explanation."""
    res = client.post("/api/live-decision/evaluate", json={
        "symbol": "EURUSD",
        "timeframe": "15m",
        "current_price": 1.0850,
        "direction": "BUY",
        "observed_sweep": True,
        "observed_displacement": False,
        "observed_fvg": False,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["concept_match_status"] == "PARTIAL_MATCH"
    assert data["final_decision"] == "WAIT"
    assert len(data["scenarios"]) == 2
    assert "प्रतीक्षा" in data["hindi_speech"] or "इंतज़ार" in data["hindi_speech"]
