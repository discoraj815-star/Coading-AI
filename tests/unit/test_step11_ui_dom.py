"""DOM and UI Verification for Step 11: High-Performance & Adaptive AI Architecture."""

from pathlib import Path


def test_index_html_step11_dom_elements():
    """Verify that all Step 11 performance panel elements exist in index.html."""
    html_path = Path("trading_ai/ui/index.html")
    assert html_path.exists(), "index.html must exist"
    content = html_path.read_text(encoding="utf-8")

    required_ids = [
        "tab-performance",
        "tab-performance-btn",
        "perf-system-status",
        "perf-avg-latency",
        "perf-queue-status",
        "perf-hot-memory",
        "perf-gemini-gating",
        "perf-priority-breakdown",
        "prio-critical-cnt",
        "prio-high-cnt",
        "prio-medium-cnt",
        "prio-low-cnt",
        "prio-bg-cnt",
        "perf-upgrades-list",
        "perf-telemetry-tbody",
    ]

    for elem_id in required_ids:
        assert f'id="{elem_id}"' in content, f"Missing required Step 11 DOM id: {elem_id}"

    # Verify functions are registered in script
    assert "refreshPerformanceUI" in content
    assert "approveUpgradeUI" in content
