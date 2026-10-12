"""Unit test for Step 15 UI DOM elements in index.html."""

import os
import pytest


def test_step15_ui_dom_elements():
    """Verify that trading_ai/ui/index.html contains all Step 15 DOM elements."""
    html_path = os.path.join(os.getcwd(), "trading_ai", "ui", "index.html")
    assert os.path.exists(html_path), "index.html must exist"

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Tab navigation button
    assert 'id="tab-step15-btn"' in content
    assert "switchTab('tab-step15')" in content

    # Tab pane & Header
    assert 'id="tab-step15"' in content
    assert 'id="step15-status-badge"' in content
    assert 'id="step15-refresh-btn"' in content

    # Section 1: AI Core Multilingual & 4-Layer Epistemic Model
    assert 'id="step15-input-text"' in content
    assert 'id="step15-comprehend-btn"' in content
    assert 'id="step15-detected-lang"' in content
    assert 'id="step15-normalized-meaning"' in content
    assert 'id="step15-hindi-explanation"' in content
    assert 'id="step15-what-i-learned"' in content
    assert 'id="step15-what-i-observe"' in content
    assert 'id="step15-what-i-infer"' in content
    assert 'id="step15-what-i-decide"' in content

    # Section 2: Large Video Processing & Serial Barrier
    assert 'id="step15-create-vjob-btn"' in content
    assert 'id="step15-process-chunk-btn"' in content
    assert 'id="step15-fail-chunk-btn"' in content
    assert 'id="step15-vjob-id"' in content
    assert 'id="step15-barrier-badge"' in content
    assert 'id="step15-chunks-progress"' in content
    assert 'id="step15-buffered-list"' in content

    # Section 3: Live Market Decision & Forecasting
    assert 'id="step15-live-symbol"' in content
    assert 'id="step15-live-dir"' in content
    assert 'id="step15-live-eval-btn"' in content
    assert 'id="step15-check-sweep"' in content
    assert 'id="step15-check-disp"' in content
    assert 'id="step15-check-fvg"' in content
    assert 'id="step15-concept-match-badge"' in content
    assert 'id="step15-why-not-badge"' in content
    assert 'id="step15-missing-confirmations"' in content
    assert 'id="step15-live-hindi-advice"' in content
    assert 'id="step15-decision-badge"' in content

    # Section 4: Controlled Plugins
    assert 'id="step15-plugins-health-badge"' in content
    assert 'id="step15-refresh-plugins-btn"' in content
    assert 'id="step15-plugins-list"' in content

    # JavaScript Functions
    assert "function refreshStep15UI" in content
    assert "function comprehendInputUI" in content
    assert "function initVideoJobUI" in content
    assert "function streamNextChunkUI" in content
    assert "function evaluateLiveMarketUI" in content
    assert "function fetchPluginsUI" in content
    assert "function togglePluginUI" in content
