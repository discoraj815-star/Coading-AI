"""Integration tests for FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "TRADING AI ASSISTANT" in response.text
    assert "Live Market" in response.text
    assert "tab-market" in response.text


def test_api_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "mappings" in data
    assert "fast_multimodal" in data["mappings"]
    assert "reasoning" in data["mappings"]


def test_sample_selection_and_analysis_flow(client):
    # 1. Select sample lecture
    sel_res = client.post(
        "/api/select_sample",
        json={"sample_name": "API_Test_Lecture_M15.mp4"},
    )
    assert sel_res.status_code == 200
    job_data = sel_res.json()["job"]
    job_id = job_data["job_id"]
    assert job_id.startswith("job_")

    # 2. Trigger analysis
    analyze_res = client.post(f"/api/jobs/{job_id}/analyze")
    assert analyze_res.status_code == 200
    res_job = analyze_res.json()["job"]
    assert res_job["status"] in ["processing", "completed"]

    # 3. Retrieve results
    results_res = client.get(f"/api/jobs/{job_id}/results")
    assert results_res.status_code == 200
    results = results_res.json()

    assert "job" in results
    assert "transcripts" in results
    assert len(results["transcripts"]) >= 1
    assert "visual_events" in results
    assert len(results["visual_events"]) >= 1
    assert "alignments" in results
    assert len(results["alignments"]) >= 1
    assert "knowledge" in results
    assert len(results["knowledge"]) >= 1
    assert "notes" in results
    assert results["notes"] is not None
    assert "overview" in results["notes"]
    assert "token_usage" in results
    assert results["token_usage"]["total_tokens"] > 0
    assert "barrier" in results
