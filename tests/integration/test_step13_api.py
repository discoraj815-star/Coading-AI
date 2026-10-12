"""Integration tests for Step 13 Auto-Recovery, Cyber Wall, and Resilience API endpoints."""
import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_resilience_status(client):
    res = client.get("/api/resilience/status")
    assert res.status_code == 200
    data = res.json()
    assert "overall_state" in data
    assert "cpu_percent" in data
    assert "memory_percent" in data
    assert "threat_level" in data
    assert "knowledge_integrity_status" in data


def test_api_threat_evaluation_and_listing(client):
    # 1. Normal payload
    res1 = client.post(
        "/api/resilience/threats/evaluate",
        json={"content": "Check market liquidity on EUR/USD", "source": "API_TEST"},
    )
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["threat_level"] == "NORMAL"

    # 2. Dangerous shell command payload
    res2 = client.post(
        "/api/resilience/threats/evaluate",
        json={"content": "powershell -enc malformed_base64_exploit", "source": "API_TEST"},
    )
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["threat_level"] == "CRITICAL"
    assert d2["action_taken"] == "BLOCKED"

    # 3. List threats
    res3 = client.get("/api/resilience/threats")
    assert res3.status_code == 200
    threats = res3.json()
    assert len(threats) >= 1


def test_api_backup_lifecycle(client):
    # 1. Create a backup
    res = client.post(
        "/api/resilience/backups/create",
        json={"notes": "API test backup snapshot"},
    )
    assert res.status_code == 200
    backup = res.json()
    assert "backup_id" in backup
    assert "sha256_checksum" in backup

    # 2. List backups
    res_list = client.get("/api/resilience/backups")
    assert res_list.status_code == 200
    backups = res_list.json()
    assert any(b["backup_id"] == backup["backup_id"] for b in backups)

    # 3. Restore backup
    res_restore = client.post(f"/api/resilience/backups/{backup['backup_id']}/restore")
    assert res_restore.status_code == 200
    restore_data = res_restore.json()
    assert restore_data["success"] is True
    assert restore_data["verified_checksum"] is True


def test_api_cyberwall_validate_path(client):
    # 1. Valid safe path
    res_safe = client.post(
        "/api/resilience/cyberwall/validate-path",
        json={"target_path": "data/lecture_notes.json", "action": "READ"},
    )
    assert res_safe.status_code == 200
    assert res_safe.json()["allowed"] is True

    # 2. Path traversal attack
    res_traversal = client.post(
        "/api/resilience/cyberwall/validate-path",
        json={"target_path": "../../windows/system32/cmd.exe", "action": "READ"},
    )
    assert res_traversal.status_code == 200
    d_trav = res_traversal.json()
    assert d_trav["allowed"] is False
    assert d_trav["threat_level"] in ("BLOCKED", "CRITICAL")


def test_api_watchdog_tasks_and_audit(client):
    # List active watchdog tasks
    res_tasks = client.get("/api/resilience/watchdog/tasks")
    assert res_tasks.status_code == 200
    assert isinstance(res_tasks.json(), list)

    # Trigger audit check
    res_check = client.post("/api/resilience/watchdog/check")
    assert res_check.status_code == 200
    assert "hung_tasks_count" in res_check.json()


def test_api_recovery_trigger(client):
    res = client.post(
        "/api/resilience/recovery/trigger",
        json={"job_id": "test_job_api_recovery"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["job_id"] == "test_job_api_recovery"
    assert "action" in data
