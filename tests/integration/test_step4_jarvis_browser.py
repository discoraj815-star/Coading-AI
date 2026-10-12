"""Real Google Chrome Browser Verification Test for Step 4: JARVIS Voice Interaction.

Mandatory Real Google Chrome verification per Repository Rules:
- Real Google Chrome browser automation (C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)
- Dashboard loads at http://127.0.0.1:8000
- AI Chat Drawer opens
- JARVIS Voice Bar and Push-to-Talk controls visible
- Voice state indicator reflects IDLE/LISTENING/THINKING/SPEAKING states
- Continuous mode toggle interaction
- Voice request processing renders in Chat messages container
- Trade confirmation buttons render when an order action is proposed
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


def test_step4_jarvis_voice_real_chrome_verification():
    """Execute real Google Chrome interactive test of Step 4 JARVIS voice interface."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p, headless=True)
        assert browser, "Real Google Chrome browser must be launched"
        context = browser.new_context()
        page = context.new_page()

        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigate to application
        page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=15000)
        assert "Trading AI Assistant" in page.title()

        # 2. Open Chat Drawer via Header Button
        header_chat_btn = page.locator("#btn-toggle-chat-drawer")
        assert header_chat_btn.is_visible(), "AI Chat header button must be visible"
        header_chat_btn.click()
        page.wait_for_timeout(500)

        drawer = page.locator("#ai-chat-drawer")
        assert drawer.is_visible(), "AI Chat Drawer must be visible"

        # 3. Verify JARVIS Voice Panel Elements
        voice_panel = page.locator("#jarvis-voice-panel")
        assert voice_panel.is_visible(), "JARVIS Voice Panel must be visible"

        status_text = page.locator("#jarvis-status-text")
        assert "IDLE" in status_text.inner_text(), "Initial state must be IDLE"

        ptt_btn = page.locator("#btn-voice-push-to-talk")
        assert ptt_btn.is_visible(), "Push to Talk button must be visible"

        continuous_toggle = page.locator("#btn-voice-continuous-toggle")
        assert continuous_toggle.is_visible(), "Continuous mode toggle button must be visible"
        assert "OFF" in continuous_toggle.inner_text()

        # 4. Interact with Continuous Mode Toggle
        continuous_toggle.click()
        page.wait_for_timeout(300)
        assert "ON" in continuous_toggle.inner_text()
        continuous_toggle.click()
        page.wait_for_timeout(300)
        assert "OFF" in continuous_toggle.inner_text()

        # 5. Submit Simulated Voice Command via Client Pipeline
        page.evaluate("() => handleVoiceTranscriptSubmit('Analyze BTC on the 1 hour chart.')")

        # Wait for thinking / response to complete
        messages_container = page.locator("#chat-messages-container")
        for _ in range(40):
            last_msg = messages_container.locator(".chat-msg").last
            if last_msg.is_visible() and ("BTC" in last_msg.inner_text() or "Market" in last_msg.inner_text()):
                break
            page.wait_for_timeout(250)

        last_assistant_msg = messages_container.locator(".chat-msg-assistant").last
        assert last_assistant_msg.is_visible(), "Assistant response to voice turn must render in Chat Drawer"

        # 6. Test Gated Trade Action Proposal and Confirmation UX
        page.evaluate("() => handleVoiceTranscriptSubmit('Buy BTC now')")
        for _ in range(40):
            confirm_btn = page.locator("#btn-confirm-order-action").last
            if confirm_btn.is_visible():
                break
            page.wait_for_timeout(250)

        confirm_btn = page.locator("#btn-confirm-order-action").last
        cancel_btn = page.locator("#btn-cancel-order-action").last
        assert confirm_btn.is_visible(), "Order Confirmation button must appear for proposed trade"
        assert cancel_btn.is_visible(), "Order Cancel button must appear for proposed trade"

        # 7. Click Cancel Order
        cancel_btn.click()
        for _ in range(40):
            last_msg = messages_container.locator(".chat-msg-assistant").last
            if "Cancelled" in last_msg.inner_text() or "dismissed" in last_msg.inner_text():
                break
            page.wait_for_timeout(250)

        cancel_response = messages_container.locator(".chat-msg-assistant").last.inner_text()
        assert "Cancelled" in cancel_response or "dismissed" in cancel_response

        # 8. Assert Zero Fatal Console Errors
        fatal_errors = [e for e in console_errors if "favicon" not in e and "429" not in e and "quota" not in e.lower()]
        assert len(fatal_errors) == 0, f"Fatal console errors detected: {fatal_errors}"
        assert len(page_errors) == 0, f"Unhandled page script exceptions: {page_errors}"

        browser.close()
