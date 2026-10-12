"""Unit test verifying that Step 13 Auto-Recovery & Cyber Wall DOM elements exist in index.html."""
from pathlib import Path


def test_index_html_step13_dom_elements():
    index_path = Path("trading_ai/ui/index.html")
    assert index_path.exists(), "trading_ai/ui/index.html must exist"
    content = index_path.read_text(encoding="utf-8")

    expected_ids = [
        "tab-resilience-btn",
        "tab-resilience",
        "res-system-status",
        "res-system-status-indicator",
        "res-cyberwall-mode",
        "res-threat-level",
        "res-watchdog-status",
        "res-resource-budget",
        "res-path-input",
        "res-validate-path-btn",
        "res-path-result",
        "res-threats-list",
        "res-integrity-badge",
        "res-create-backup-btn",
        "res-backup-status-msg",
        "res-backups-list",
        "res-watchdog-badge",
        "res-check-hangs-btn",
        "res-cleanup-mem-btn",
        "res-watchdog-result",
        "res-tasks-list",
        "res-recovery-badge",
        "res-recovery-job-id",
        "res-trigger-recovery-btn",
        "res-recovery-output",
        "res-services-list",
        "res-incidents-tbody",
    ]

    for element_id in expected_ids:
        assert f'id="{element_id}"' in content, f"Expected id='{element_id}' in index.html"

    # Verify Step 13 JavaScript functions are defined
    expected_js_functions = [
        "refreshResilienceUI",
        "validateCyberWallPathUI",
        "createBackupUI",
        "restoreBackupUI",
        "checkWatchdogHangsUI",
        "triggerMemoryCleanupUI",
        "triggerRecoveryUI",
    ]
    for fn in expected_js_functions:
        assert f"function {fn}(" in content or f"async function {fn}(" in content, f"Expected JS function '{fn}' in index.html"
