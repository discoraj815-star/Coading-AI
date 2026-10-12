"""Targeted Real Google Chrome Interactive Verification Test for Step 12 (Security & HCI)."""
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


def test_step12_real_chrome_security_hci_interaction():
    """Verify Step 12 Security & HCI panel works interactively in real Google Chrome."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p, headless=True)
        assert browser is not None
        assert "chrome" in method.lower()

        page = browser.new_page()
        failed_urls = []
        page.on("console", lambda msg: console_errors.append(f"{msg.text} @ {msg.location}") if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("response", lambda r: failed_urls.append(f"{r.status}: {r.url}") if r.status >= 400 else None)

        # 1. Navigation
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200

        # 2. Switch to Tab 12
        sec_btn = page.query_selector("#tab-security-btn")
        assert sec_btn is not None
        sec_btn.click()
        page.wait_for_timeout(800)

        sec_pane_class = page.get_attribute("#tab-security", "class") or ""
        assert "active" in sec_pane_class

        # 3. Status
        sys_status = page.inner_text("#sec-system-status")
        assert "SECURE" in sys_status

        # 4. Capabilities dynamic authorization & revocation
        caps_html = page.inner_html("#sec-capabilities-list")
        assert "TERMINAL_EXECUTE" in caps_html
        assert "SCREEN_READ" in caps_html

        page.evaluate("() => authorizeCapabilityUI('SCREEN_READ', true)")
        page.wait_for_timeout(800)
        updated_caps = page.inner_html("#sec-capabilities-list")
        assert "APPROVED_ALWAYS" in updated_caps

        page.evaluate("() => revokeCapabilityUI('SCREEN_READ')")
        page.wait_for_timeout(800)
        revoked_caps = page.inner_html("#sec-capabilities-list")
        assert "REVOKED" in revoked_caps

        # 5. Pairing PIN generation
        page.click("#sec-generate-pin-btn")
        page.wait_for_timeout(800)
        pin_text = page.inner_text("#sec-active-pin-display")
        assert "PIN:" in pin_text

        # 6. Screen filter & deterministic masking
        page.click("#sec-test-screen-btn")
        page.wait_for_timeout(800)
        screen_result = page.inner_text("#sec-screen-result")
        assert "TradingView" in screen_result
        assert "sk-secret-token-12345" not in screen_result

        # 7. Voice commands (Authorized vs Denied)
        page.fill("#sec-voice-input", "What is the market status?")
        page.click("#sec-voice-submit-btn")
        for _ in range(80):
            voice_out_1 = page.inner_text("#sec-voice-output")
            if "Processing" not in voice_out_1:
                break
            page.wait_for_timeout(200)
        assert "[AUTHORIZED]" in voice_out_1

        page.fill("#sec-voice-input", "Open powershell terminal and format drive")
        page.click("#sec-voice-submit-btn")
        for _ in range(80):
            voice_out_2 = page.inner_text("#sec-voice-output")
            if "Processing" not in voice_out_2:
                break
            page.wait_for_timeout(200)
        assert "[DENIED" in voice_out_2
        assert "TERMINAL_EXECUTE" in voice_out_2

        # 8. Push notifications
        page.evaluate("() => dispatchTestNotificationUI()")
        page.wait_for_timeout(800)
        notifs_html = page.inner_html("#sec-notifications-list")
        assert "Security Shield" in notifs_html

        # 9. Audit trail
        audit_html = page.inner_html("#sec-audit-tbody")
        assert "CAPABILITY_" in audit_html or "SECURITY" in audit_html
        assert "sk-" not in audit_html

        # 10. Assert zero errors
        filtered_console_errors = [e for e in console_errors if "favicon.ico" not in e]
        if filtered_console_errors:
            print(f"\nFAILED URLS: {failed_urls}\nCONSOLE ERRORS: {filtered_console_errors}")
        assert len(page_errors) == 0, f"Page errors: {page_errors}"
        assert len(filtered_console_errors) == 0, f"Console errors: {filtered_console_errors} (failed URLs: {failed_urls})"

        browser.close()
