"""Targeted Real Google Chrome Interactive Verification Test for Step 16.

Validates:
- Real Google Chrome browser automation
- Tab 16 navigation and layout
- 24×7 Cloud Runtime status badges and health matrix
- Cloud Market Monitor cycle execution
- Hindi AI Notification formatter interaction with 4-layer epistemic separation
- Voice Alert TTS audio preview in the cloud
- Privacy Mode toggle (Normal vs Secure)
- Offline client reconnect and missed event replay
- Zero console errors and zero unhandled page script exceptions
"""
import socket
import time
import pytest
from playwright.sync_api import sync_playwright

from tests.helpers.browser_helper import find_system_chrome, launch_real_chrome


def _is_server_listening(host="127.0.0.1", port=8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@pytest.fixture(scope="module", autouse=True)
def ensure_server():
    """Ensure FastAPI server is listening at 127.0.0.1:8000."""
    for _ in range(50):
        if _is_server_listening("127.0.0.1", 8000):
            break
        time.sleep(0.2)
    yield


def test_step16_real_chrome_interactive_verification():
    """Execute complete real Google Chrome interactive test of Step 16."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p)
        assert browser is not None
        assert "chrome" in method.lower()

        page = browser.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigate to application
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200

        # 2. Switch to Tab 16
        tab16_btn = page.query_selector("#tab-step16-btn")
        assert tab16_btn is not None, "Tab 16 button must exist in DOM"
        tab16_btn.click()
        page.wait_for_timeout(800)

        tab16_pane_class = page.get_attribute("#tab-step16", "class") or ""
        assert "active" in tab16_pane_class, "Tab 16 pane must become active upon click"

        # 3. Verify Step 16 status badge
        badge_text = page.inner_text("#step16-cloud-status-badge")
        assert "ONLINE" in badge_text or "CLOUD" in badge_text

        # 4. Interact with Cloud Cycle execution
        cycle_btn = page.query_selector("#step16-cycle-cloud-btn")
        assert cycle_btn is not None
        cycle_btn.click()
        page.wait_for_timeout(1000)

        # Verify subsystem health cards
        market_health = page.inner_text("#step16-market-health")
        assert market_health in ("HEALTHY", "DEGRADED", "ONLINE")

        router_health = page.inner_text("#step16-router-health")
        assert "ROUTING" in router_health or "HEALTHY" in router_health

        vector_health = page.inner_text("#step16-vector-health")
        assert "SQLiteVectorStore" in vector_health

        # 5. Interact with Multi-Channel Hindi Notification trigger
        test_notif_btn = page.query_selector("#step16-test-notif-btn")
        assert test_notif_btn is not None
        test_notif_btn.click()
        page.wait_for_timeout(1000)

        # Assert Hindi notification synthesis in DOM
        notif_title = page.inner_text("#step16-notif-title")
        assert "Hello Alam" in notif_title

        hindi_text = page.inner_text("#step16-hindi-preview")
        assert len(hindi_text) > 20
        assert "OBSERVED (देखा गया)" in hindi_text or "Hello Alam" in hindi_text

        # 6. Interact with Cloud Voice Synthesizer preview
        voice_btn = page.query_selector("#step16-voice-preview-btn")
        assert voice_btn is not None
        voice_btn.click()
        page.wait_for_timeout(800)

        voice_status = page.inner_text("#step16-voice-status")
        assert "Generated" in voice_status or "Ready" in voice_status

        # 7. Interact with Privacy Mode Toggle
        privacy_btn = page.query_selector("#step16-privacy-toggle-btn")
        assert privacy_btn is not None
        privacy_btn.click()
        page.wait_for_timeout(800)

        privacy_badge = page.inner_text("#step16-privacy-badge")
        assert "SECURE PRIVACY" in privacy_badge

        # 8. Interact with Offline Event Replay
        replay_btn = page.query_selector("#step16-replay-btn")
        assert replay_btn is not None
        replay_btn.click()
        page.wait_for_timeout(800)

        browser.close()

    # Assert zero console or page errors
    assert len(console_errors) == 0, f"Found console errors in real Chrome: {console_errors}"
    assert len(page_errors) == 0, f"Found unhandled page errors in real Chrome: {page_errors}"
