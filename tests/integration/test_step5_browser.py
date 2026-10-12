"""Real Google Chrome Browser Verification Test for Step 5: Unified Settings, Interactive Chart & Execution Mode Toggle.

Mandatory Real Google Chrome verification per Repository Rules:
- Real Google Chrome browser automation (C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)
- Dashboard loads at http://127.0.0.1:8000
- Execution Mode header toggle & safety handshake modal
- Unified Settings Hub modal with category navigation
- Interactive Candlestick Chart canvas in Tab 9 (Live Market)
- Zero fatal console errors and zero unhandled page script exceptions
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
    """Ensure FastAPI server is active at 127.0.0.1:8000."""
    for _ in range(50):
        if _is_server_listening("127.0.0.1", 8000):
            break
        time.sleep(0.2)
    assert _is_server_listening("127.0.0.1", 8000), "Server not listening on 127.0.0.1:8000"
    yield


def test_step5_real_chrome_settings_chart_execution_verification():
    """Verify Step 5 Settings Hub, Interactive Chart Canvas & Execution Mode Toggle in Real Google Chrome."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p, headless=True)
        assert browser is not None, "Real Chrome browser instance must launch"
        assert "chrome" in method.lower(), f"Must use real Chrome, got: {method}"

        page = browser.new_page()
        page.on("console", lambda msg: console_errors.append(f"{msg.text} @ {msg.location}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigation
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200, "Dashboard must return 200 OK"
        page.wait_for_timeout(1000)

        # 2. Execution Mode Toggle & Safety Modal
        exec_btn = page.locator("#btn-toggle-execution-mode")
        assert exec_btn.is_visible(), "Header execution mode toggle button must be visible"
        initial_text = exec_btn.inner_text()
        assert "MANUAL" in initial_text, f"Initial mode must be MANUAL, got: {initial_text}"

        # Click to open safety confirmation modal
        exec_btn.click()
        page.wait_for_timeout(500)
        exec_modal = page.locator("#execution-mode-modal")
        assert exec_modal.is_visible(), "Execution mode confirmation modal must appear"

        # Confirm switch to AUTOMATIC
        confirm_auto_btn = page.locator("#btn-confirm-auto-mode")
        assert confirm_auto_btn.is_visible()
        confirm_auto_btn.click()
        page.wait_for_timeout(1000)

        # Verify button updated to AUTOMATIC GATE
        auto_text = exec_btn.inner_text()
        assert "AUTOMATIC" in auto_text, f"Mode should be updated to AUTOMATIC, got: {auto_text}"

        # Toggle back to MANUAL (immediate safe switch)
        exec_btn.click()
        page.wait_for_timeout(1000)
        back_text = exec_btn.inner_text()
        assert "MANUAL" in back_text, f"Mode should revert to MANUAL, got: {back_text}"

        # 3. Unified Settings Hub Modal
        settings_btn = page.locator("#btn-toggle-settings-modal")
        assert settings_btn.is_visible(), "Header Settings button must be visible"
        settings_btn.click()
        page.wait_for_timeout(800)

        settings_modal = page.locator("#settings-modal")
        assert settings_modal.is_visible(), "Settings modal must be visible after clicking header button"

        # Check default AI input value populated from API
        ai_model_input = page.locator("#setting-ai-default-model")
        assert ai_model_input.is_visible()
        val = ai_model_input.input_value()
        assert len(val) > 0, "Default AI model setting must be populated"

        # Switch to Voice tab
        page.locator("#settings-tab-btn-voice").click()
        page.wait_for_timeout(300)
        assert page.locator("#settings-pane-voice").is_visible(), "Voice settings pane must be visible"

        # Switch to Trading tab
        page.locator("#settings-tab-btn-trading").click()
        page.wait_for_timeout(300)
        assert page.locator("#settings-pane-trading").is_visible(), "Trading settings pane must be visible"

        # Close Settings modal
        page.locator("#btn-close-settings").click()
        page.wait_for_timeout(500)
        assert not settings_modal.is_visible(), "Settings modal must close on close button click"

        # 4. Tab 9: Interactive Candlestick Chart Canvas
        page.locator("#tab-market-btn").click()
        page.wait_for_timeout(1000)

        chart_container = page.locator("#interactive-chart-container")
        assert chart_container.is_visible(), "Interactive Candlestick Chart container must be visible in Tab 9"

        chart_canvas = page.locator("#market-candlestick-chart")
        assert chart_canvas.is_visible(), "Candlestick HTML5 canvas must be visible"

        # Timeframe selector switch
        page.locator("#tf-btn-1h").click()
        page.wait_for_timeout(1000)
        title_text = page.locator("#chart-display-symbol-title").inner_text()
        assert "1h" in title_text.lower(), f"Chart title must reflect selected timeframe 1h: {title_text}"

        # 5. Assert Zero Fatal Console Errors
        fatal_errors = [e for e in console_errors if "favicon" not in e and "429" not in e and "quota" not in e.lower()]
        assert len(fatal_errors) == 0, f"Fatal console errors detected: {fatal_errors}"
        assert len(page_errors) == 0, f"Unhandled page script exceptions: {page_errors}"

        browser.close()
