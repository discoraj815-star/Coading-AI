"""
Unit test verifying Step 6 UI DOM elements and functions in index.html.
"""

from pathlib import Path


def test_index_html_step6_dom_elements():
    html_path = Path("trading_ai/ui/index.html")
    assert html_path.exists(), "index.html must exist"
    html = html_path.read_text(encoding="utf-8")

    # Step 6 HTML elements
    expected_elements = [
        "FUNDAMENTAL & ECONOMIC EVENT INTELLIGENCE",
        'id="fund-risk-badge"',
        'id="fund-provider-status"',
        'id="fund-next-event"',
        'id="fund-time-dist"',
        'id="fund-relationship"',
        'id="fund-action-context"',
        'id="fund-conflict-banner"',
        'id="fund-conflict-text"',
        'id="fund-explanation"',
        "syncFundamentalsUI()",
    ]

    for element in expected_elements:
        assert element in html, f"Missing element: {element}"
