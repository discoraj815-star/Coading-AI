"""Real Google Chrome Browser Verification Test for Step 8: Hedge Fund Execution & JARVIS Grounding.

Mandatory Real Google Chrome verification per Repository Rules:
- Real Google Chrome browser automation (C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)
- Dashboard loads at http://127.0.0.1:8000
- Tab 9 (Live Market) contains Hedge Fund Execution Panel (#hedge-fund-execution-panel)
- Displays execution metrics: #execution-mode-badge, #execution-env-badge, #engagement-mode-select
- Clicking 'Generate Proposal' synthesizes proposal and renders in #execution-proposal-card
- Positions table (#positions-table-body) renders
- Interactive AI Chat drawer asks execution queries:
  1. "What open positions do I have right now?" -> verifies grounded positions response
  2. "What is my current execution mode and risk status?" -> verifies grounded execution policy response
  3. "What are my trade proposals?" -> verifies grounded proposal response
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


def test_step8_real_chrome_execution_and_jarvis_grounding():
    """Verify Step 8 Execution Panel & JARVIS Grounding in Real Google Chrome."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p, headless=True)
        assert browser is not None, "Real Chrome browser instance must launch"
        assert "chrome" in method.lower(), f"Must use real Chrome, got: {method}"

        context = browser.new_context()
        page = context.new_page()

        page.on("console", lambda msg: console_errors.append(f"{msg.text} @ {msg.location}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Load Application
        page.goto("http://127.0.0.1:8000", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(2000)

        # 2. Navigate to Tab 9 (Live Market)
        tab_market_btn = page.locator("#tab-market-btn")
        assert tab_market_btn.is_visible(), "Tab 9 Live Market button must be visible"
        tab_market_btn.click()
        page.wait_for_timeout(1000)

        # 3. Verify Hedge Fund Execution Panel DOM Structure
        exec_panel = page.locator("#hedge-fund-execution-panel")
        assert exec_panel.is_visible(), "Hedge Fund Execution Panel must be visible in Tab 9"

        mode_badge = page.locator("#execution-mode-badge")
        assert mode_badge.is_visible(), "Execution mode badge must be visible"
        mode_text = mode_badge.inner_text()
        assert "MANUAL" in mode_text or "AUTO" in mode_text

        env_badge = page.locator("#execution-env-badge")
        assert env_badge.is_visible(), "Execution environment badge must be visible"
        assert "PAPER" in env_badge.inner_text() or "LIVE" in env_badge.inner_text()

        engagement_select = page.locator("#engagement-mode-select")
        assert engagement_select.is_visible(), "Engagement mode dropdown must be visible"

        btn_generate = page.locator("#btn-generate-proposal")
        assert btn_generate.is_visible(), "Generate Proposal button must be visible"

        positions_tbody = page.locator("#positions-table-body")
        assert positions_tbody.is_visible(), "Positions table body must be visible"

        # 4. Interactive Proposal Generation in Execution Panel
        btn_generate.click()
        page.wait_for_timeout(1500)

        proposal_card = page.locator("#execution-proposal-card")
        assert proposal_card.is_visible(), "Proposal card must be visible"
        card_text = proposal_card.inner_text()
        assert len(card_text) > 0

        # 5. AI Chat Drawer: JARVIS Grounded Queries Verification
        header_chat_btn = page.locator("#btn-toggle-chat-drawer")
        assert header_chat_btn.is_visible(), "AI Chat header button must be visible"
        header_chat_btn.click()
        page.wait_for_timeout(800)

        chat_drawer = page.locator("#ai-chat-drawer")
        assert chat_drawer.is_visible(), "AI Chat drawer must be open"

        chat_input = page.locator("#chat-user-input")
        send_btn = page.locator("#btn-chat-send")
        assert chat_input.is_visible()
        assert send_btn.is_visible()

        # Turn 1: Ask for Open Positions
        chat_input.fill("What open positions do I have right now?")
        send_btn.click()
        page.wait_for_timeout(2000)

        messages_container = page.locator("#chat-messages-container")
        pos_msg = messages_container.locator(".chat-msg-assistant").last
        pos_text = pos_msg.inner_text()
        assert "position" in pos_text.lower(), f"Expected grounded position information, got: {pos_text}"

        # Turn 2: Ask for Execution Mode and Status
        chat_input.fill("What is my current execution mode and risk status?")
        send_btn.click()
        page.wait_for_timeout(2000)

        exec_msg = messages_container.locator(".chat-msg-assistant").last
        exec_text = exec_msg.inner_text()
        assert "execution" in exec_text.lower() or "mode" in exec_text.lower() or "manual" in exec_text.lower(), (
            f"Expected grounded execution policy in chat response, got: {exec_text}"
        )

        # Turn 3: Ask for Trade Proposals
        chat_input.fill("What are my trade proposals?")
        send_btn.click()
        page.wait_for_timeout(2000)

        prop_msg = messages_container.locator(".chat-msg-assistant").last
        prop_text = prop_msg.inner_text()
        assert "proposal" in prop_text.lower(), f"Expected grounded proposal information, got: {prop_text}"

        # 6. Assert Zero Fatal Console Errors
        fatal_errors = [e for e in console_errors if "favicon" not in e and "429" not in e and "quota" not in e.lower()]
        assert len(fatal_errors) == 0, f"Fatal console errors detected: {fatal_errors}"
        assert len(page_errors) == 0, f"Unhandled page script exceptions: {page_errors}"

        browser.close()
