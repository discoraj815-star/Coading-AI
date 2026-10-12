"""Real Google Chrome Browser Verification Test for Step 6: Fundamental & Economic Event Intelligence.

Mandatory Real Google Chrome verification per Repository Rules:
- Real Google Chrome browser automation (C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)
- Dashboard loads at http://127.0.0.1:8000
- Tab 9 (Live Market) contains visible Fundamental & Economic Event Intelligence card
- Card elements: #fund-risk-badge, #fund-provider-status, #fund-next-event, #fund-time-dist,
  #fund-relationship, #fund-action-context, #fund-explanation, #btn-sync-fundamentals
- Interacts with Sync Feed button (#btn-sync-fundamentals)
- Switches watchlist symbols (EUR/USD) and verifies live situation update
- Opens AI Chat and asks about economic calendar/news risk, verifying grounded fundamental answer
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


def test_step6_real_chrome_fundamental_intelligence_verification():
    """Verify Step 6 Fundamental Market Intelligence in Real Google Chrome."""
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

        # 2. Switch to Tab 9: Live Market
        page.locator("#tab-market-btn").click()
        page.wait_for_timeout(1000)

        # 3. Verify Step 6 DOM Elements
        risk_badge = page.locator("#fund-risk-badge")
        assert risk_badge.is_visible(), "Fundamental risk badge must be visible"

        prov_status = page.locator("#fund-provider-status")
        assert prov_status.is_visible(), "Provider status indicator must be visible"

        next_event = page.locator("#fund-next-event")
        assert next_event.is_visible(), "Next event element must be visible"

        time_dist = page.locator("#fund-time-dist")
        assert time_dist.is_visible(), "Time to event element must be visible"

        rel_el = page.locator("#fund-relationship")
        assert rel_el.is_visible(), "Tech+Fund relationship element must be visible"

        action_el = page.locator("#fund-action-context")
        assert action_el.is_visible(), "Action context element must be visible"

        explanation_el = page.locator("#fund-explanation")
        assert explanation_el.is_visible(), "Fundamental explanation must be visible"

        sync_btn = page.locator("#btn-sync-fundamentals")
        assert sync_btn.is_visible(), "Sync Feed button must be visible"

        # 4. Trigger Feed Sync
        sync_btn.click()
        page.wait_for_timeout(1500)
        status_text = prov_status.inner_text()
        assert any(k in status_text for k in ["OPERATIONAL", "SYNCED", "SYNCING"]), f"Status must reflect sync: {status_text}"

        # 5. Switch Symbol to EUR/USD in Watchlist
        eur_btn = page.locator("#market-watchlist-buttons button:has-text('EUR/USD')")
        assert eur_btn.is_visible(), "EUR/USD watchlist button must be visible"
        eur_btn.click()
        page.wait_for_timeout(1500)

        # Verify active symbol title updated
        title_text = page.locator("#market-active-sym-title").inner_text()
        assert "EUR/USD" in title_text or "EURUSD" in title_text, f"Title should reflect EUR/USD: {title_text}"

        # 6. Verify AI Chat Grounding on Fundamentals
        # Open Chat Drawer
        header_chat_btn = page.locator("#btn-toggle-chat-drawer")
        assert header_chat_btn.is_visible(), "AI Chat & Guide header button must be visible"
        header_chat_btn.click()
        page.wait_for_timeout(800)

        chat_drawer = page.locator("#ai-chat-drawer")
        assert chat_drawer.is_visible(), "AI Chat drawer must be visible"

        # Send fundamental intelligence query
        chat_input = page.locator("#chat-user-input")
        assert chat_input.is_visible(), "Chat input textarea must be visible"
        chat_input.fill("What is the fundamental news risk and economic events for EUR/USD?")
        page.locator("#btn-chat-send").click()
        page.wait_for_timeout(2500)

        # Verify response text appears in chat
        messages_container = page.locator("#chat-messages-container")
        fund_msg = messages_container.locator(".chat-msg-assistant").last
        chat_content = fund_msg.inner_text()
        assert any(w in chat_content for w in ["Fundamental", "Risk", "Event", "economic", "news"]), f"AI Chat should output grounded fundamental context, got: {chat_content}"

        # 7. Assert Zero Fatal Console Errors
        fatal_errors = [e for e in console_errors if "favicon" not in e and "429" not in e and "quota" not in e.lower()]
        assert len(fatal_errors) == 0, f"Fatal console errors detected: {fatal_errors}"
        assert len(page_errors) == 0, f"Unhandled page script exceptions: {page_errors}"

        browser.close()
