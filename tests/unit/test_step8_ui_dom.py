"""Unit tests verifying Step 8 UI DOM elements and JavaScript functions in index.html."""

from pathlib import Path


def test_index_html_step8_dom_elements():
    html_path = Path("trading_ai/ui/index.html")
    assert html_path.exists(), "index.html must exist"
    html = html_path.read_text(encoding="utf-8")

    expected_elements = [
        "HEDGE FUND EXECUTION & RISK",
        'id="hedge-fund-execution-panel"',
        'id="execution-mode-badge"',
        'id="engagement-mode-select"',
        'id="btn-generate-proposal"',
        'id="execution-proposal-card"',
        'id="proposal-state-badge"',
        'id="execution-positions-container"',
        'id="positions-table-body"',
        'id="execution-orders-container"',
        "loadExecutionPanelUI(",
        "renderActiveProposalUI(",
        "generateTradeProposalUI(",
        "approveTradeProposalUI(",
        "rejectTradeProposalUI(",
        "loadPositionsUI(",
        "loadOrdersUI(",
        "manualClosePositionUI(",
        "manualUpdateSLUI(",
    ]

    for element in expected_elements:
        assert element in html, f"Missing Step 8 element: {element}"
