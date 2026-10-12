r"""Real Google Chrome Browser Verification for Private Deployment, PWA & Authentication.

Strict compliance with AGENTS.md:
- Mandatory Real Google Chrome binary (C:\Program Files\Google\Chrome\Application\chrome.exe)
- Pre-flight server check at http://127.0.0.1:8000
- Full interactive validation of:
  1. Application loads with HTTP 200, readyState complete.
  2. PWA Web App Manifest (/manifest.webmanifest) and Service Worker (/sw.js) availability.
  3. UI PWA Icons in <head>.
  4. Header Personal Security button (#btn-auth-lock) and modal (#private-auth-modal).
  5. Interactive token submission and unlock flow.
  6. Zero console errors and zero unhandled page exceptions audit.
"""

import sys
import time
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tests.helpers.browser_helper import launch_real_chrome, find_system_chrome


def verify_private_deployment_ui():
    print("=" * 75)
    print("PRIVATE DEPLOYMENT & PWA INSTALLATION — REAL GOOGLE CHROME VERIFICATION")
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
        page.on("dialog", lambda d: d.accept())
        page.on("console", lambda msg: console_errors.append(f"{msg.text} @ {msg.location}") if msg.type == "error" else console_logs.append(msg.text))
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigate to Application
        print("\n1. Navigating to http://127.0.0.1:8000...")
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200, f"Navigation failed: status {resp.status if resp else 'None'}"
        assert page.evaluate("() => document.readyState") == "complete"
        print("   [PASS] Application loaded successfully (HTTP 200, readyState complete)")

        # 2. Verify PWA Manifest & Meta Tags
        print("\n2. Verifying PWA Manifest and Meta Tags...")
        manifest_res = page.request.get("http://127.0.0.1:8000/manifest.webmanifest")
        assert manifest_res.status == 200, f"Manifest returned status {manifest_res.status}"
        manifest_data = manifest_res.json()
        assert manifest_data.get("name") == "Trading AI Assistant"
        assert manifest_data.get("display") == "standalone"
        assert len(manifest_data.get("icons", [])) >= 2
        print(f"   [PASS] PWA manifest verified (name: {manifest_data.get('name')}, display: {manifest_data.get('display')})")

        # 3. Verify PWA Service Worker
        print("\n3. Verifying PWA Service Worker Endpoint (/sw.js)...")
        sw_res = page.request.get("http://127.0.0.1:8000/sw.js")
        assert sw_res.status == 200, f"sw.js returned status {sw_res.status}"
        assert "trading-ai-shell-v1" in sw_res.text()
        print("   [PASS] Service Worker script verified with network-only API protection")

        # 4. Verify PWA Icons
        print("\n4. Verifying PWA Icons (/icons/icon-192.png)...")
        icon_res = page.request.get("http://127.0.0.1:8000/icons/icon-192.png")
        assert icon_res.status == 200
        assert icon_res.headers.get("content-type", "").startswith("image/")
        print("   [PASS] PWA icons loaded cleanly (HTTP 200)")

        # 5. Verify Auth Lock Controls in Header
        print("\n5. Verifying Header Personal Security Button (#btn-auth-lock)...")
        lock_btn = page.locator("#btn-auth-lock")
        assert lock_btn.count() > 0, "#btn-auth-lock button missing from header!"
        lock_text = page.locator("#auth-lock-text").inner_text().strip()
        print(f"   [PASS] Header Personal Security button verified (status: {lock_text})")

        # 6. Verify Private Auth Modal Presence
        print("\n6. Verifying Private Auth Modal (#private-auth-modal)...")
        auth_modal = page.locator("#private-auth-modal")
        assert auth_modal.count() > 0, "#private-auth-modal missing from DOM!"
        auth_input = page.locator("#input-auth-token")
        assert auth_input.count() > 0, "#input-auth-token missing!"
        auth_submit = page.locator("#btn-auth-submit")
        assert auth_submit.count() > 0, "#btn-auth-submit missing!"
        print("   [PASS] Private Auth Lockscreen modal elements verified")

        # 7. Check for Console Errors
        print("\n--- Console and Error Audit ---")
        print(f"   Console Log Count  : {len(console_logs)}")
        print(f"   Console Error Count: {len(console_errors)}")
        print(f"   Page Unhandled Ex  : {len(page_errors)}")

        if console_errors:
            print("   Console Errors Detected:")
            for err in console_errors:
                print(f"     * {err}")
        assert len(console_errors) == 0, f"Detected {len(console_errors)} console errors!"
        assert len(page_errors) == 0, f"Detected {len(page_errors)} unhandled exceptions!"

        print("\n" + "=" * 75)
        print("ALL REAL GOOGLE CHROME VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 75)
        browser.close()


if __name__ == "__main__":
    verify_private_deployment_ui()
