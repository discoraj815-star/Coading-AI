"""DOM and UI Verification for Step 12: System Requirements, Core Security & HCI."""

from pathlib import Path


def test_index_html_step12_dom_elements():
    """Verify that all Step 12 Security and HCI panel elements exist in index.html."""
    html_path = Path("trading_ai/ui/index.html")
    assert html_path.exists(), "index.html must exist"
    content = html_path.read_text(encoding="utf-8")

    required_ids = [
        "tab-security",
        "tab-security-btn",
        "sec-system-status",
        "sec-system-status-indicator",
        "sec-privacy-mode",
        "sec-active-capabilities-count",
        "sec-paired-devices-count",
        "sec-capabilities-list",
        "sec-devices-list",
        "sec-generate-pin-btn",
        "sec-active-pin-display",
        "sec-screen-status",
        "sec-test-screen-btn",
        "sec-screen-result",
        "sec-voice-status",
        "sec-voice-input",
        "sec-voice-submit-btn",
        "sec-voice-output",
        "sec-notifications-list",
        "sec-audit-tbody",
    ]

    for elem_id in required_ids:
        assert f'id="{elem_id}"' in content, f"Missing required Step 12 DOM id: {elem_id}"

    # Verify functions are registered in script
    required_js_fns = [
        "refreshSecurityUI",
        "authorizeCapabilityUI",
        "revokeCapabilityUI",
        "generatePairingCodeUI",
        "revokeDeviceUI",
        "testScreenCaptureUI",
        "submitVoiceCommandUI",
        "dispatchTestNotificationUI",
    ]
    for fn in required_js_fns:
        assert f"function {fn}" in content, f"Missing required Step 12 JS function: {fn}"
