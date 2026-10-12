"""Integration tests for Private-Only Authentication REST API & PWA Endpoints."""

import pytest
from fastapi.testclient import TestClient

from trading_ai.server.app import app, security_core


@pytest.fixture
def client():
    return TestClient(app)


def test_public_pwa_endpoints(client):
    # Manifest
    res_manifest = client.get("/manifest.webmanifest")
    assert res_manifest.status_code == 200
    assert "Trading AI Assistant" in res_manifest.text
    assert "standalone" in res_manifest.text

    # Service Worker
    res_sw = client.get("/sw.js")
    assert res_sw.status_code == 200
    assert "trading-ai-shell-v1" in res_sw.text
    assert "NEVER cache private API" in res_sw.text

    # Icon
    res_icon = client.get("/icons/icon-192.png")
    assert res_icon.status_code == 200
    assert res_icon.headers["content-type"].startswith("image/")


def test_auth_status_endpoint(client):
    # Default state
    res = client.get("/api/auth/status")
    assert res.status_code == 200
    data = res.json()
    assert "auth_required" in data


def test_auth_lifecycle_enforcement_and_sessions(client):
    # Configure auth_manager dynamically for this test
    orig_enabled = security_core.auth_manager.is_auth_enabled
    orig_token = security_core.auth_manager._auth_token

    try:
        security_core.auth_manager.set_auth_token("test_master_token_777")
        security_core.auth_manager.set_auth_enabled(True)

        # 1. Unauthenticated request to private API must return 401
        res_unauth = client.get("/api/jobs")
        assert res_unauth.status_code == 401
        assert "Unauthorized" in res_unauth.text

        # 2. Login with bad token must fail with 401
        res_bad_login = client.post("/api/auth/login", json={"token": "wrong_token"})
        assert res_bad_login.status_code == 401

        # 3. Login with correct token succeeds and sets cookie
        res_login = client.post("/api/auth/login", json={"token": "test_master_token_777"})
        assert res_login.status_code == 200
        assert res_login.json()["success"] is True
        assert "trading_ai_session" in res_login.cookies

        session_cookie = res_login.cookies["trading_ai_session"]

        # 4. Request with valid session cookie succeeds (HTTP 200)
        res_auth_jobs = client.get("/api/jobs", cookies={"trading_ai_session": session_cookie})
        assert res_auth_jobs.status_code == 200

        # 5. Request with Authorization: Bearer succeeds
        res_bearer_jobs = client.get("/api/jobs", headers={"Authorization": "Bearer test_master_token_777"})
        assert res_bearer_jobs.status_code == 200

        # 6. Request with X-API-Key succeeds
        res_apikey_jobs = client.get("/api/jobs", headers={"X-API-Key": "test_master_token_777"})
        assert res_apikey_jobs.status_code == 200

        # 7. Logout revokes the session
        res_logout = client.post("/api/auth/logout", cookies={"trading_ai_session": session_cookie})
        assert res_logout.status_code == 200

        # 8. Subsequent request with revoked cookie returns 401
        res_after_logout = client.get("/api/jobs", cookies={"trading_ai_session": session_cookie})
        assert res_after_logout.status_code == 401

    finally:
        # Restore original state to preserve other tests
        security_core.auth_manager.set_auth_token(orig_token)
        security_core.auth_manager.set_auth_enabled(orig_enabled)
