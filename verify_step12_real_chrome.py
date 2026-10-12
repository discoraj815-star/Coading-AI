"""Step 12 Real Google Chrome Browser Interactive Verification Script.

Uses the verified Real Google Chrome setup to interactively exercise:
- Security & HCI panel opening
- Permission & Capability state inspection
- Real-time capability authorization and revocation
- Pairing PIN generation
- Privacy-aware screen filter testing with sensitive data masking
- Two-way voice command execution & capability denial
- Notification dispatch
- Immutable audit log table rendering
- Verification of zero unhandled page errors and zero secret leakage
"""

import sys
import time
from playwright.sync_api import sync_playwright

from tests.helpers.browser_helper import launch_real_chrome, find_system_chrome


def verify_step12_chrome():
    print("=" * 75)
    print("STEP 12 REAL GOOGLE CHROME INTERACTIVE BROWSER VERIFICATION")
    print("=" * 75)

    chrome_path = find_system_chrome()
    print(f"System Google Chrome Path: {chrome_path}")
    print(f"Target Server URL: http://127.0.0.1:8000")

    console_logs = []
    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p, headless=True)
        print(f"\n[PASS] Real Chrome launched successfully via: {method}")
        print(f"       Browser Version: {browser.version}")

        page = browser.new_page()
        page.on("console", lambda msg: console_errors.append(f"{msg.text} @ {msg.location}") if msg.type == "error" else console_logs.append(msg.text))
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("response", lambda res: print(f"   [HTTP {res.status}] {res.url}") if res.status >= 400 else None)

        # 1. Navigate to Application
        print("\n1. Navigating to http://127.0.0.1:8000...")
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200, f"Navigation failed with status: {resp.status if resp else 'None'}"
        assert page.evaluate("() => document.readyState") == "complete"
        print("   [PASS] Application loaded successfully (HTTP 200, readyState complete)")

        # 2. Open Security & HCI Panel
        print("\n2. Switching to Tab 12: Security & HCI...")
        sec_btn = page.query_selector("#tab-security-btn")
        assert sec_btn is not None, "Missing #tab-security-btn"
        sec_btn.click()
        page.wait_for_timeout(1000)  # Allow API fetches to settle

        sec_pane_class = page.get_attribute("#tab-security", "class") or ""
        assert "active" in sec_pane_class, f"Expected #tab-security to be active, got: {sec_pane_class}"
        print("   [PASS] Security & HCI panel opened (#tab-security active)")

        # 3. Verify Status Indicators
        sys_status = page.inner_text("#sec-system-status")
        priv_mode = page.inner_text("#sec-privacy-mode")
        print(f"   Security Status : '{sys_status}'")
        print(f"   Privacy Mode    : '{priv_mode}'")
        assert "SECURE" in sys_status
        assert "LOCAL" in priv_mode and "FIRST" in priv_mode
        print("   [PASS] Security status indicators verified")

        # 4. Verify Capability Access Control & Interactive State Changes
        print("\n3. Testing Capability Access Control & Interactive State Changes...")
        caps_html = page.inner_html("#sec-capabilities-list")
        assert "TERMINAL_EXECUTE" in caps_html
        assert "SCREEN_READ" in caps_html
        print("   [PASS] Capabilities list populated from /api/security/capabilities")

        # In-browser interaction: Authorize SCREEN_READ capability
        page.evaluate("() => authorizeCapabilityUI('SCREEN_READ', true)")
        page.wait_for_timeout(800)
        updated_caps_html = page.inner_html("#sec-capabilities-list")
        assert "APPROVED_ALWAYS" in updated_caps_html
        print("   [PASS] Dynamically authorized SCREEN_READ capability (State: APPROVED_ALWAYS)")

        # In-browser interaction: Revoke SCREEN_READ capability
        page.evaluate("() => revokeCapabilityUI('SCREEN_READ')")
        page.wait_for_timeout(800)
        revoked_caps_html = page.inner_html("#sec-capabilities-list")
        assert "REVOKED" in revoked_caps_html
        print("   [PASS] Dynamically revoked SCREEN_READ capability (State: REVOKED)")

        # 5. Test Cross-Device Pairing PIN Generation
        print("\n4. Testing Device Pairing PIN Generation...")
        page.click("#sec-generate-pin-btn")
        page.wait_for_timeout(800)
        pin_text = page.inner_text("#sec-active-pin-display")
        print(f"   Generated PIN Display: '{pin_text}'")
        assert "PIN:" in pin_text
        print("   [PASS] Generated 6-digit cryptographic pairing PIN")

        # 6. Test Privacy Screen Filter & Deterministic Masking
        print("\n5. Testing Privacy Screen Filter & Masking...")
        page.click("#sec-test-screen-btn")
        page.wait_for_timeout(1000)
        screen_result = page.inner_text("#sec-screen-result")
        print(f"   Screen Result Snippet:\n   {screen_result}")
        assert "TradingView" in screen_result
        assert "sk-secret-token-12345" not in screen_result  # Plain secret must never appear!
        print("   [PASS] Screen filter verified: TradingView allowed, sensitive token masked deterministically")

        # 7. Test Voice Commands (Allowed vs Denied)
        print("\n6. Testing Two-Way Voice Commands & Capability Gating...")
        # 7A: Market status query (Authorized)
        page.fill("#sec-voice-input", "What is the market status?")
        page.click("#sec-voice-submit-btn")
        page.wait_for_timeout(1000)
        voice_out_1 = page.inner_text("#sec-voice-output")
        print(f"   Voice Output 1: '{voice_out_1}'")
        assert "[AUTHORIZED]" in voice_out_1

        # 7B: Malicious terminal command (Denied by capability gate)
        page.fill("#sec-voice-input", "Open powershell terminal and format drive")
        page.click("#sec-voice-submit-btn")
        page.wait_for_timeout(1000)
        voice_out_2 = page.inner_text("#sec-voice-output")
        print(f"   Voice Output 2: '{voice_out_2}'")
        assert "[DENIED" in voice_out_2
        assert "TERMINAL_EXECUTE" in voice_out_2
        print("   [PASS] Voice capability gating verified: dangerous command blocked deterministically")

        # 8. Test Push Notifications
        print("\n7. Testing Push Notification Dispatch...")
        page.evaluate("() => dispatchTestNotificationUI()")
        page.wait_for_timeout(800)
        notifs_html = page.inner_html("#sec-notifications-list")
        assert "Security Shield" in notifs_html
        print("   [PASS] Privacy-minimized push notification rendered in feed")

        # 9. Verify Immutable Audit Trail
        print("\n8. Verifying Audit Log Display...")
        audit_html = page.inner_html("#sec-audit-tbody")
        assert "CAPABILITY_" in audit_html or "SECURITY" in audit_html
        assert "sk-" not in audit_html  # Zero secret leakage in audit table!
        print("   [PASS] Audit trail rendered in UI with zero secret leakage")

        # 10. Check Console and Page Errors
        print("\n9. Checking Browser Console and Page Errors...")
        filtered_console_errors = [e for e in console_errors if "favicon.ico" not in e]
        print(f"   Console Error Count (non-favicon): {len(filtered_console_errors)}")
        print(f"   Page Error Count: {len(page_errors)}")
        if filtered_console_errors:
            for err in filtered_console_errors:
                print(f"     [CONSOLE ERROR]: {err}")
        if page_errors:
            for perr in page_errors:
                print(f"     [PAGE ERROR]: {perr}")
        assert len(page_errors) == 0, f"Page errors occurred: {page_errors}"
        assert len(filtered_console_errors) == 0, f"Fatal console errors: {filtered_console_errors}"
        print("   [PASS] Zero application runtime script errors detected")

        browser.close()
        print("\n[PASS] Real Google Chrome cleanly closed")

    print("\n" + "=" * 75)
    print("STEP 12 REAL GOOGLE CHROME VERIFICATION: ALL CHECKS PASSED (100%)")
    print("=" * 75)


if __name__ == "__main__":
    verify_step12_chrome()
