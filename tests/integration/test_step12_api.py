"""Integration tests for Step 12 Security, Privacy, and HCI REST APIs."""

import pytest
from fastapi.testclient import TestClient
from trading_ai.server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_security_status_and_capabilities(client):
    # 1. Status
    res_status = client.get("/api/security/status")
    assert res_status.status_code == 200
    data = res_status.json()
    assert data["status"] == "SECURE"
    assert data["default_policy"] == "DENY"
    assert data["privacy_mode"] == "LOCAL_FIRST"

    # 2. Capabilities list
    res_caps = client.get("/api/security/capabilities")
    assert res_caps.status_code == 200
    caps = res_caps.json()
    assert len(caps) >= 11
    assert any(c["capability"] == "TERMINAL_EXECUTE" and c["is_dangerous"] is True for c in caps)


def test_api_capability_request_authorize_revoke_lifecycle(client):
    # 1. Request capability
    req_payload = {
        "capability": "VOICE_OUTPUT",
        "requesting_component": "VoiceTestClient",
        "context": {"reason": "Speaking trade alert"},
    }
    res_req = client.post("/api/security/capabilities/request", json=req_payload)
    assert res_req.status_code == 200
    req_id = res_req.json()["request_id"]

    # 2. Authorize capability as user
    auth_payload = {
        "request_id": req_id,
        "capability": "VOICE_OUTPUT",
        "approved": True,
        "is_user": True,
        "approved_by": "TestUser",
        "reason": "Approved for live alerts",
    }
    res_auth = client.post("/api/security/capabilities/authorize", json=auth_payload)
    assert res_auth.status_code == 200
    assert res_auth.json()["state"] == "APPROVED_ALWAYS"

    # 3. Revoke capability
    res_rev = client.post("/api/security/capabilities/revoke", json={"capability": "VOICE_OUTPUT", "revoked_by": "TestUser"})
    assert res_rev.status_code == 200
    assert res_rev.json()["state"] == "REVOKED"


def test_api_device_pairing_and_revocation(client):
    # 1. Generate pairing PIN
    gen_res = client.post("/api/devices/pairing/generate", json={"device_name": "API Test Phone", "device_type": "phone"})
    assert gen_res.status_code == 200
    gen_data = gen_res.json()
    dev_id = gen_data["device_id"]
    pin = gen_data["pairing_code"]

    # 2. Submit pairing request
    pair_res = client.post("/api/devices/pairing/request", json={
        "device_id": dev_id,
        "device_name": "API Test Phone",
        "pairing_code": pin,
        "nonce": "nonce_api_123",
    })
    assert pair_res.status_code == 200
    assert pair_res.json()["status"] == "AWAITING_USER_APPROVAL"

    # 3. Approve pairing as user
    appr_res = client.post("/api/devices/pairing/approve", json={
        "device_id": dev_id,
        "approved": True,
        "is_user": True,
        "approved_by": "TraderAdmin",
    })
    assert appr_res.status_code == 200
    assert appr_res.json()["status"] == "ACTIVE"
    assert "shared_secret" in appr_res.json()

    # 4. List devices
    list_res = client.get("/api/devices")
    assert list_res.status_code == 200
    devices = list_res.json()
    assert any(d["device_id"] == dev_id and d["status"] == "ACTIVE" for d in devices)

    # 5. Revoke device
    rev_res = client.post(f"/api/devices/{dev_id}/revoke", json={"reason": "Test finished"})
    assert rev_res.status_code == 200
    assert rev_res.json()["status"] == "REVOKED"


def test_api_privacy_sanitize_endpoint(client):
    payload = {
        "input": "Password=TopSecretPassword123 with api_key=sk-12345678901234567890",
        "target_destination": "cloud",
        "user_cloud_consent": False,
    }
    res = client.post("/api/privacy/sanitize", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["allowed"] is False
    assert data["classification"] == "BLOCKED_EXTERNAL"


def test_api_screen_capture_scope_and_blocking(client):
    # Ensure SCREEN_READ is authorized
    client.post("/api/security/capabilities/authorize", json={
        "request_id": "req_screen",
        "capability": "SCREEN_READ",
        "approved": True,
        "is_user": True,
        "approved_by": "Admin",
    })

    # 1. Allowed App
    res_tv = client.post("/api/screen/capture", json={
        "app_name": "TradingView",
        "window_title": "EURUSD 15m Chart",
        "ocr_text": "Price 1.0850 | User Session Active",
    })
    assert res_tv.status_code == 200
    assert res_tv.json()["status"] in ["CAPTURED_AND_FILTERED", "SKIPPED_UNCHANGED"]

    # 2. Blocked Sensitive App
    res_blocked = client.post("/api/screen/capture", json={
        "app_name": "Bitwarden",
        "window_title": "Master Password Vault",
    })
    assert res_blocked.status_code == 403


def test_api_voice_command_and_cancellation(client):
    # 1. Voice Status
    res_status = client.get("/api/voice/status")
    assert res_status.status_code == 200
    assert "state" in res_status.json()

    # 2. Voice Command
    res_cmd = client.post("/api/voice/command", json={"command_text": "What is the current market status?"})
    assert res_cmd.status_code == 200
    assert res_cmd.json()["authorized"] is True

    # 3. Voice Cancel
    res_cancel = client.post("/api/voice/cancel")
    assert res_cancel.status_code == 200
    assert res_cancel.json()["status"] == "CANCELLED"


def test_api_notifications_dispatch_and_read(client):
    # Authorize NOTIFICATION_SEND
    client.post("/api/security/capabilities/authorize", json={
        "request_id": "req_notif",
        "capability": "NOTIFICATION_SEND",
        "approved": True,
        "is_user": True,
        "approved_by": "Admin",
    })

    # Dispatch notification
    disp_res = client.post("/api/notifications/dispatch", json={
        "title": "Trade Alert",
        "message": "EURUSD Liquidity Swept. Password=None",
        "priority": "HIGH",
        "category": "market",
    })
    assert disp_res.status_code == 200
    notif = disp_res.json()
    assert notif["priority"] == "HIGH"

    # List notifications
    list_res = client.get("/api/notifications")
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1

    # Mark all read
    read_res = client.post("/api/notifications/read-all")
    assert read_res.status_code == 200
    assert read_res.json()["status"] == "ok"
