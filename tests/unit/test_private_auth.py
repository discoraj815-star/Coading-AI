"""Unit tests for Private-Only Authentication, Session Lifecycle, and Rate-Limiting."""

import time
import pytest

from trading_ai.security.auth import (
    MAX_FAILED_ATTEMPTS,
    RATE_LIMIT_WINDOW_SECONDS,
    PrivateAuthManager,
)


def test_auth_manager_disabled_by_default():
    mgr = PrivateAuthManager(auth_enabled=False)
    assert mgr.is_auth_enabled is False
    assert mgr.verify_token("any_token") is True
    assert mgr.verify_session("any_session") is True


def test_auth_manager_token_verification():
    mgr = PrivateAuthManager(auth_token="secret_personal_token_123", auth_enabled=True)
    assert mgr.is_auth_enabled is True
    # Correct token passes
    assert mgr.verify_token("secret_personal_token_123") is True
    # Incorrect token fails
    assert mgr.verify_token("wrong_token") is False
    # Empty token fails
    assert mgr.verify_token("") is False


def test_auth_manager_rate_limiting():
    mgr = PrivateAuthManager(auth_token="secure_token_abc", auth_enabled=True)
    client_ip = "192.168.1.50"

    # 5 failed attempts
    for _ in range(MAX_FAILED_ATTEMPTS):
        assert mgr.verify_token("bad_attempt", client_ip=client_ip) is False

    # 6th attempt should be blocked by rate limit
    assert mgr.is_rate_limited(client_ip) is True
    assert mgr.verify_token("secure_token_abc", client_ip=client_ip) is False

    # Different IP is not blocked
    assert mgr.verify_token("secure_token_abc", client_ip="192.168.1.51") is True


def test_auth_manager_session_lifecycle():
    mgr = PrivateAuthManager(
        auth_token="token_xyz",
        session_secret="session_secret_key_999",
        auth_enabled=True,
    )

    # Create session
    session_id = mgr.create_session(client_ip="127.0.0.1")
    assert session_id is not None
    assert len(session_id.split(":")) == 3

    # Verify session passes
    assert mgr.verify_session(session_id) is True

    # Tampered session fails
    tampered = session_id[:-4] + "dead"
    assert mgr.verify_session(tampered) is False

    # Revoke session
    assert mgr.revoke_session(session_id) is True
    assert mgr.verify_session(session_id) is False


def test_auth_manager_token_rotation():
    mgr = PrivateAuthManager(auth_token="old_token", auth_enabled=True)
    session_1 = mgr.create_session()
    assert mgr.verify_session(session_1) is True

    # Rotate secret
    mgr.rotate_secret("new_token_456")
    assert mgr.verify_token("old_token") is False
    assert mgr.verify_token("new_token_456") is True

    # Old session is revoked upon token rotation
    assert mgr.verify_session(session_1) is False
