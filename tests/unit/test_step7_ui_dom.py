"""Unit test verifying Step 7 UI DOM elements and JavaScript functions in index.html."""

from pathlib import Path


def test_index_html_step7_dom_elements():
    html_path = Path("trading_ai/ui/index.html")
    assert html_path.exists(), "index.html must exist"
    html = html_path.read_text(encoding="utf-8")

    expected_elements = [
        "TRADING COMPANION",
        'id="trading-companion-panel"',
        'id="companion-symbol-badge"',
        'id="companion-discipline-ratio"',
        'id="companion-bias-text"',
        'id="companion-setup-text"',
        'id="companion-news-text"',
        'id="companion-alert-guidance"',
        'id="companion-alerts-container"',
        'id="companion-query-input"',
        'id="companion-response-box"',
        "loadTradingCompanionUI(",
        "submitCompanionQuery()",
        "quickCompanionQuery(",
        "acknowledgeAlertUI(",
    ]

    for element in expected_elements:
        assert element in html, f"Missing Step 7 element: {element}"
