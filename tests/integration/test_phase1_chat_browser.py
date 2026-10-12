"""Real Google Chrome Browser Verification Test for Phase 1 AI Chat + Application Guide.

Mandatory Real Google Chrome verification per Repository Rules:
- Real Google Chrome browser automation (C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)
- Dashboard loads at http://127.0.0.1:8000
- AI Chat drawer button in header and floating launcher
- Opening and closing the Chat Drawer
- Multi-turn conversation interactions
- Application Tour request
- Feature location query ("Where is the Watchlist?")
- Clickable Navigation Actions that switch tabs (Tab 9 Live Market)
- Negative Hallucination check ("Where are the settings?")
- Quick action pills
- Clear history functionality
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


def test_phase1_chat_real_chrome_verification():
    """Execute real Google Chrome interactive test of Phase 1 AI Chat and Application Guide."""
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

        # 2. Verify AI Chat Header Button & Floating Launcher
        header_chat_btn = page.locator("#btn-toggle-chat-drawer")
        assert header_chat_btn.is_visible(), "AI Chat & Guide header button must be visible"

        floating_chat_btn = page.locator("#floating-chat-btn")
        assert floating_chat_btn.is_visible(), "Floating launcher button must be visible"

        # 3. Open Chat Drawer via Header Button
        header_chat_btn.click()
        page.wait_for_timeout(500)

        drawer = page.locator("#ai-chat-drawer")
        assert drawer.is_visible(), "AI Chat Drawer must be visible"

        chat_input = page.locator("#chat-user-input")
        send_btn = page.locator("#btn-chat-send")
        assert chat_input.is_visible(), "Chat input textarea must be visible"
        assert send_btn.is_visible(), "Send button must be visible"

        # 4. Turn 1: Ask for Application Tour
        chat_input.fill("Give me a complete tour of the application.")
        send_btn.click()
        page.wait_for_timeout(1500)

        # Verify AI Tour response rendered
        messages_container = page.locator("#chat-messages-container")
        tour_msg = messages_container.locator(".chat-msg-assistant").last
        tour_text = tour_msg.inner_text()
        assert "TOUR" in tour_text or "Notes" in tour_text or "Live Market" in tour_text, f"Expected tour content in response: {tour_text}"

        # 5. Turn 2: Ask for Feature Location ("Where is the Watchlist?")
        chat_input.fill("Where is the Watchlist?")
        send_btn.click()
        page.wait_for_timeout(1500)

        # Verify Watchlist response and navigation action
        watchlist_msg = messages_container.locator(".chat-msg-assistant").last
        watchlist_text = watchlist_msg.inner_text()
        assert "Tab 9" in watchlist_text or "Live Market" in watchlist_text or "Watchlist" in watchlist_text

        # Verify clickable navigation button inside the message
        nav_btn = watchlist_msg.locator("button:has-text('Live Market')")
        if not nav_btn.count():
            nav_btn = messages_container.locator("button:has-text('Live Market')").last

        assert nav_btn.count() > 0, "Navigation action button to Tab 9 must be present"
        nav_btn.click()
        page.wait_for_timeout(800)

        # Verify Tab 9 Live Market is active
        market_pane = page.locator("#tab-market")
        assert market_pane.is_visible(), "Clicking navigation action button must activate Tab 9 Live Market"

        # 6. Turn 3: Ask for Settings (Hallucination Control Test)
        chat_input.fill("Where are the broker settings?")
        send_btn.click()
        page.wait_for_timeout(1500)

        settings_msg = messages_container.locator(".chat-msg-assistant").last
        settings_text = settings_msg.inner_text()
        # Verify AI grounds settings accurately (Settings Hub operational or Tab 18)
        assert "settings" in settings_text.lower() or "no dedicated settings page" in settings_text.lower() or "not implemented" in settings_text.lower() or "tab 18" in settings_text.lower()

        # 7. Test Quick Action Pill ("Market Status")
        quick_market_pill = page.locator(".chat-quick-pill:has-text('Market Status')")
        assert quick_market_pill.is_visible(), "Market Status quick action pill must be visible"
        quick_market_pill.click()
        page.wait_for_timeout(1500)

        last_msg = messages_container.locator(".chat-msg-assistant").last
        last_text = last_msg.inner_text()
        assert len(last_text) > 20, "Quick action must produce an informative market answer"

        # 8. Test Clear Conversation
        clear_btn = page.locator("#btn-chat-clear-history")
        assert clear_btn.is_visible(), "Clear history button must be visible"
        clear_btn.click()
        page.wait_for_timeout(500)

        # Verify messages cleared
        cleared_text = messages_container.inner_text()
        assert "cleared" in cleared_text.lower()

        # 9. Test Close Drawer
        close_btn = page.locator("#btn-close-chat-drawer")
        assert close_btn.is_visible(), "Close drawer button must be visible"
        close_btn.click()
        page.wait_for_timeout(400)

        # 10. Re-open via floating launcher
        floating_chat_btn.click()
        page.wait_for_timeout(400)
        assert drawer.is_visible(), "Drawer must re-open via floating launcher"

        # 11. Assert zero unhandled JavaScript page errors
        assert len(page_errors) == 0, f"Page had script errors: {page_errors}"

        # Filter acceptable browser warnings (e.g. favicon 204 or expected network timeouts)
        critical_errors = [e for e in console_errors if "favicon" not in e and "429" not in e]
        assert len(critical_errors) == 0, f"Page had critical console errors: {critical_errors}"

        browser.close()
