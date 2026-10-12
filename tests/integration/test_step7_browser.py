"""Real Google Chrome Browser Verification Test for Step 7: AI Trading Companion & JARVIS Grounding.

Mandatory Real Google Chrome verification per Repository Rules:
- Real Google Chrome browser automation (C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)
- Dashboard loads at http://127.0.0.1:8000
- Tab 9 (Live Market) contains Trading Companion (#trading-companion-panel)
- Displays companion metrics: symbol badge, discipline ratio, bias text, setup text, alert guidance
- Scenario card and active alerts container render properly
- Settings Hub modal trading pane contains Step 7 discipline guardrails (#setting-trading-max-consec-losses, #setting-trading-cooldown-mins, #setting-trading-require-sl)
- Settings Hub updates discipline guardrails and saves successfully
- Interactive AI Chat drawer asks companion queries:
  1. "What are my current discipline limits?" -> verifies grounded discipline response
  2. "What is the forward scenario for EUR/USD?" -> verifies grounded conditional scenario response
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


def test_step7_real_chrome_trading_companion_and_jarvis_grounding():
    """Verify Step 7 Trading Companion & JARVIS Grounding in Real Google Chrome."""
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

        # 1. Navigation
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200, "Dashboard must return 200 OK"
        page.wait_for_timeout(1000)

        # 2. Switch to Tab 9: Live Market
        tab_market_btn = page.locator("#tab-market-btn")
        assert tab_market_btn.is_visible(), "Tab 9 Live Market button must be visible"
        tab_market_btn.click()
        page.wait_for_timeout(1000)

        # 3. Verify Trading Companion Panel in Tab 9
        companion_panel = page.locator("#trading-companion-panel")
        assert companion_panel.is_visible(), "Trading Companion panel must be visible in Tab 9"

        # Verify companion subcomponents
        symbol_badge = page.locator("#companion-symbol-badge")
        assert symbol_badge.is_visible(), "Companion symbol badge must be visible"

        discipline_ratio = page.locator("#companion-discipline-ratio")
        assert discipline_ratio.is_visible(), "Discipline ratio indicator must be visible"

        bias_text = page.locator("#companion-bias-text")
        assert bias_text.is_visible(), "Companion bias text must be visible"

        setup_text = page.locator("#companion-setup-text")
        assert setup_text.is_visible(), "Companion setup confirmation text must be visible"

        alerts_container = page.locator("#companion-alerts-container")
        assert alerts_container.is_visible(), "Companion alerts container must be visible"

        query_input = page.locator("#companion-query-input")
        assert query_input.is_visible(), "Companion quick query input must be visible"

        response_box = page.locator("#companion-response-box")

        # 4. Interactive Companion Query in Tab 9 Panel
        query_input.fill("What is the scenario?")
        page.locator("#btn-submit-companion-query").click()
        page.wait_for_timeout(1500)

        assert response_box.is_visible(), "Companion response box must be visible after query submission"
        companion_resp_text = response_box.inner_text()
        assert len(companion_resp_text) > 0, "Companion response box must receive grounded answer"
        assert "EUR/USD" in companion_resp_text or "BTC/USD" in companion_resp_text or "Scenario" in companion_resp_text

        # 5. Settings Hub: Discipline Guardrails Verification & Update
        settings_btn = page.locator("#btn-toggle-settings-modal")
        assert settings_btn.is_visible(), "Header Settings button must be visible"
        settings_btn.click()
        page.wait_for_timeout(800)

        settings_modal = page.locator("#settings-modal")
        assert settings_modal.is_visible(), "Settings modal must be visible"

        # Switch to Trading tab
        page.locator("#settings-tab-btn-trading").click()
        page.wait_for_timeout(500)
        assert page.locator("#settings-pane-trading").is_visible(), "Trading settings pane must be visible"

        # Verify Step 7 discipline guardrails controls exist
        max_consec_input = page.locator("#setting-trading-max-consec-losses")
        assert max_consec_input.is_visible(), "Max consecutive losses input must be visible"

        cooldown_input = page.locator("#setting-trading-cooldown-mins")
        assert cooldown_input.is_visible(), "Mandatory cooldown minutes input must be visible"

        sl_checkbox = page.locator("#setting-trading-require-sl")
        assert sl_checkbox.is_visible(), "Strict stop-loss requirement checkbox must be visible"

        # Modify values to verify save interaction
        max_consec_input.fill("4")
        cooldown_input.fill("45")

        # Save settings
        page.locator("#btn-save-settings").click()
        page.wait_for_timeout(1000)
        assert not settings_modal.is_visible(), "Settings modal must close upon saving"

        # 6. AI Chat Drawer: JARVIS Grounded Queries Verification
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

        # Turn 1: Ask for Discipline Limits
        chat_input.fill("What are my current discipline limits?")
        send_btn.click()
        page.wait_for_timeout(2000)

        messages_container = page.locator("#chat-messages-container")
        discipline_msg = messages_container.locator(".chat-msg-assistant").last
        discipline_text = discipline_msg.inner_text()
        assert "discipline" in discipline_text.lower() or "stop-loss" in discipline_text.lower() or "cooldown" in discipline_text.lower(), (
            f"Expected grounded discipline limits in chat response, got: {discipline_text}"
        )

        # Turn 2: Ask for Forward Scenario
        chat_input.fill("What is the forward scenario for EUR/USD?")
        send_btn.click()
        page.wait_for_timeout(2000)

        scenario_msg = messages_container.locator(".chat-msg-assistant").last
        scenario_text = scenario_msg.inner_text()
        assert "EUR/USD" in scenario_text, f"Response must be grounded in EUR/USD, got: {scenario_text}"
        assert ("scenario" in scenario_text.lower() or "if" in scenario_text.lower() or "poi" in scenario_text.lower()), (
            f"Expected grounded forward scenario in chat response, got: {scenario_text}"
        )

        # 7. Assert Zero Fatal Console Errors
        fatal_errors = [e for e in console_errors if "favicon" not in e and "429" not in e and "quota" not in e.lower()]
        assert len(fatal_errors) == 0, f"Fatal console errors detected: {fatal_errors}"
        assert len(page_errors) == 0, f"Unhandled page script exceptions: {page_errors}"

        browser.close()
