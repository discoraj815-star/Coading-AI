import os
import pytest

def test_step14_ui_dom_elements():
    """Verify that trading_ai/ui/index.html contains all Step 14 UI DOM elements."""
    html_path = os.path.join(os.getcwd(), "trading_ai", "ui", "index.html")
    assert os.path.exists(html_path), "index.html must exist"

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Tab navigation button
    assert 'id="tab-trade-logic-btn"' in content
    assert "switchTab('tab-trade-logic')" in content

    # Tab pane
    assert 'id="tab-trade-logic"' in content
    assert 'id="tl-isolation-badge"' in content
    assert 'id="tl-refresh-btn"' in content

    # Domain counters
    assert 'id="tl-domain-stats-list"' in content
    assert 'id="tl-domain-count-yt"' in content
    assert 'id="tl-domain-count-mentor"' in content
    assert 'id="tl-domain-count-live"' in content
    assert 'id="tl-domain-count-tech"' in content

    # Knowledge retrieval
    assert 'id="tl-search-query"' in content
    assert 'id="tl-domain-select"' in content
    assert 'id="tl-search-btn"' in content
    assert 'id="tl-knowledge-list"' in content

    # Confluence evaluation
    assert 'id="tl-symbol-input"' in content
    assert 'id="tl-tf-input"' in content
    assert 'id="tl-dir-select"' in content
    assert 'id="tl-evaluate-btn"' in content
    assert 'id="tl-confluence-state"' in content
    assert 'id="tl-confluence-score"' in content
    assert 'id="tl-matched-count"' in content
    assert 'id="tl-conflicts-count"' in content
    assert 'id="tl-missing-count"' in content
    assert 'id="tl-confluence-insights"' in content

    # Why Not Trade
    assert 'id="tl-why-not-badge"' in content
    assert 'id="tl-why-not-list"' in content

    # Core Trade Decision
    assert 'id="tl-decision-badge"' in content
    assert 'id="tl-entry-price"' in content
    assert 'id="tl-sl-price"' in content
    assert 'id="tl-tp1-price"' in content
    assert 'id="tl-rr-ratio"' in content
    assert 'id="tl-invalidation-text"' in content
    assert 'id="tl-confirmation-text"' in content
    assert 'id="tl-evidence-count"' in content

    # Journal feedback
    assert 'id="tl-journal-feedback-btn"' in content
    assert 'id="tl-journal-winrate"' in content
    assert 'id="tl-journal-warnings"' in content
    assert 'id="tl-journal-lessons-list"' in content

    # JavaScript Handlers
    assert "refreshTradeLogicUI" in content
    assert "searchTradeLogicKnowledgeUI" in content
    assert "evaluateTradeLogicUI" in content
    assert "fetchJournalFeedbackUI" in content
    assert 'tabId === "tab-trade-logic"' in content
