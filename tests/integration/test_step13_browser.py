"""Targeted Real Google Chrome Interactive Verification Test for Step 13 (Resilience & Cyber Wall)."""
import socket
import threading
import time
import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from tests.helpers.browser_helper import launch_real_chrome, find_system_chrome
from trading_ai.server.app import app


def _is_server_listening(host="127.0.0.1", port=8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@pytest.fixture(scope="module", autouse=True)
def ensure_server():
    """Ensure server is active before running browser verification."""
    for _ in range(50):
        if _is_server_listening("127.0.0.1", 8000):
            break
        time.sleep(0.2)
    yield


def test_step13_real_chrome_resilience_and_cyberwall_interaction():
    """Verify Step 13 Auto-Recovery & Cyber Wall panel works interactively in real Google Chrome."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p, headless=True)
        assert browser is not None
        assert "chrome" in method.lower()

        page = browser.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigation
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200

        # 2. Switch to Tab 13
        res_btn = page.query_selector("#tab-resilience-btn")
        assert res_btn is not None
        res_btn.click()
        page.wait_for_timeout(800)

        res_pane_class = page.get_attribute("#tab-resilience", "class") or ""
        assert "active" in res_pane_class

        # 3. Verify Status Indicators
        sys_status = page.inner_text("#res-system-status")
        assert any(s in sys_status for s in ["HEALTHY", "RESILIENT", "WARNING", "DEGRADED", "CRITICAL"])

        cyber_mode = page.inner_text("#res-cyberwall-mode")
        assert "ACTIVE" in cyber_mode

        threat_level = page.inner_text("#res-threat-level")
        assert any(t in threat_level for t in ["NORMAL", "LOW", "SUSPICIOUS", "BLOCKED", "CRITICAL"])

        # 4. Interactive Test: Cyber Wall Path Validation (Blocked Traversal)
        page.fill("#res-path-input", "../../windows/system32/cmd.exe")
        page.click("#res-validate-path-btn")
        page.wait_for_timeout(800)

        path_result = page.inner_text("#res-path-result")
        assert "[BLOCKED]" in path_result

        # 5. Interactive Test: Create Knowledge Backup Snapshot
        page.click("#res-create-backup-btn")
        page.wait_for_timeout(1000)

        backup_msg = page.inner_text("#res-backup-status-msg")
        assert "created successfully" in backup_msg.lower()

        backups_html = page.inner_html("#res-backups-list")
        assert "backup_" in backups_html

        # 6. Interactive Test: Watchdog Audit & Memory Trim
        page.click("#res-check-hangs-btn")
        page.wait_for_timeout(800)

        page.click("#res-cleanup-mem-btn")
        page.wait_for_timeout(800)
        watchdog_result = page.inner_text("#res-watchdog-result")
        assert "memory" in watchdog_result.lower() or "audit" in watchdog_result.lower()

        # 7. Verify Subsystem Health & Incidents
        services_html = page.inner_html("#res-services-list")
        assert len(services_html) > 0

        incidents_html = page.inner_html("#res-incidents-tbody")
        assert len(incidents_html) > 0

        # 8. Assert zero console/page script errors
        filtered_console = [e for e in console_errors if "favicon.ico" not in e]
        assert len(page_errors) == 0, f"Page errors: {page_errors}"
        assert len(filtered_console) == 0, f"Console errors: {filtered_console}"

        browser.close()
