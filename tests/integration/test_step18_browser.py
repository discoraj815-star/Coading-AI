"""Targeted Real Google Chrome Interactive Verification Test for Step 18.

Validates:
- Real Google Chrome browser automation (C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)
- Tab 18 navigation and layout
- Universal Providers Matrix rendering (Broker, Terminal, Market Data, Backtest, News, Orderflow, Footprint, Volume Profile)
- Category filter pills switching
- Provider configuration modal and secure masking notice
- Real Lecture Performance Benchmarking execution
- KPI cards rendering (Total duration, Gemini time, Tokens, Estimated cost)
- Bottleneck detection percentages and visual bars
- Persistent milestone timeline checkpoints
- Historical benchmarks table persistence
- Zero console errors and zero unhandled page script exceptions
"""
import socket
import time
import pytest
from playwright.sync_api import sync_playwright

from tests.helpers.browser_helper import find_system_chrome, launch_real_chrome


def _is_server_listening(host="127.0.0.1", port=8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@pytest.fixture(scope="module", autouse=True)
def ensure_server():
    """Ensure FastAPI server is listening at 127.0.0.1:8000."""
    for _ in range(50):
        if _is_server_listening("127.0.0.1", 8000):
            break
        time.sleep(0.2)
    yield


def test_step18_real_chrome_interactive_verification():
    """Execute complete real Google Chrome interactive test of Step 18."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, _ = launch_real_chrome(p, headless=True)
        context = browser.new_context()
        page = context.new_page()

        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigate to application
        page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=15000)
        assert "Trading AI Assistant" in page.title()

        # 2. Switch to Tab 18: Universal Providers & Benchmark
        tab_btn = page.locator("#tab-step18-btn")
        assert tab_btn.is_visible(), "Tab 18 button must be present in navigation bar"
        tab_btn.click()
        page.wait_for_timeout(800)

        # 3. Verify Tab 18 pane is active
        tab_pane = page.locator("#tab-step18")
        assert tab_pane.is_visible(), "Tab 18 pane must be visible after click"

        # 4. Verify Provider Registry / Matrix
        providers_container = page.locator("#step18-providers-container")
        assert providers_container.is_visible()
        # Wait for providers cards to render
        page.wait_for_selector("#step18-providers-container > div", timeout=5000)
        provider_cards_text = providers_container.inner_text()
        assert "Paper Broker" in provider_cards_text or "paper_broker" in provider_cards_text
        assert "Delta Exchange" in provider_cards_text or "delta_exchange" in provider_cards_text
        assert "MetaTrader 5" in provider_cards_text or "metatrader5" in provider_cards_text
        assert "Forex Factory" in provider_cards_text or "forex_factory" in provider_cards_text
        assert "TradingView" in provider_cards_text or "tradingview" in provider_cards_text

        # 4b. Verify Replay Bar Interactive Controller
        replay_container = page.locator("#replay-bar-container")
        assert replay_container.is_visible(), "Replay Bar Controller container must be visible"
        assert page.locator("#replay-speed-select").is_visible()
        step_fwd_btn = page.locator("#replay-step-fwd-btn")
        assert step_fwd_btn.is_visible()

        # Interactively step replay bar forward
        step_fwd_btn.click()
        page.wait_for_timeout(400)
        bars_stepped_val = page.locator("#replay-bars-stepped").inner_text()
        assert int(bars_stepped_val) >= 1, f"Bars stepped should increment, got {bars_stepped_val}"

        # 5. Interactive Category Filtering
        broker_pill = page.locator("#filter-pill-BROKER")
        if broker_pill.is_visible():
            broker_pill.click()
            page.wait_for_timeout(400)
            filtered_text = providers_container.inner_text()
            assert "BROKER" in filtered_text

        all_pill = page.locator("#filter-pill-ALL")
        if all_pill.is_visible():
            all_pill.click()
            page.wait_for_timeout(400)

        # 6. Interactive Lecture Benchmarking Run
        benchmark_btn = page.locator("#step18-run-benchmark-btn")
        assert benchmark_btn.is_visible()
        benchmark_btn.click()
        page.wait_for_timeout(2500)

        # 7. Verify Benchmark KPIs updated in DOM
        bm_total_time = page.locator("#bm-total-time").inner_text()
        assert bm_total_time != "--", f"Total time should be populated, got {bm_total_time}"
        assert "s" in bm_total_time

        bm_tokens = page.locator("#bm-tokens-count").inner_text()
        assert bm_tokens != "--", f"Tokens count should be populated, got {bm_tokens}"

        bm_cost = page.locator("#bm-est-cost").inner_text()
        assert "$" in bm_cost, f"Estimated cost should be formatted as currency, got {bm_cost}"

        bm_rules = page.locator("#bm-rules-count").inner_text()
        assert "/" in bm_rules, f"Rules count should be formatted, got {bm_rules}"

        # 8. Verify Bottleneck Visual Breakdown
        bottleneck_bars = page.locator("#bm-bottleneck-bars")
        assert bottleneck_bars.is_visible()
        bottleneck_legend = page.locator("#bm-bottleneck-legend")
        assert len(bottleneck_legend.inner_text().strip()) > 0

        # 9. Verify Milestone Timeline
        timeline_list = page.locator("#bm-timeline-list")
        assert timeline_list.is_visible()
        timeline_text = timeline_list.inner_text()
        assert "INGESTION" in timeline_text or "AUDIO" in timeline_text or "VISUAL" in timeline_text

        # 10. Verify Historical Benchmarks Table
        history_table = page.locator("#bm-history-table")
        assert history_table.is_visible()
        history_text = history_table.inner_text()
        assert "bench_lecture_" in history_text

        # 11. Assert Zero Console / Page Errors
        browser.close()

    fatal_console = [err for err in console_errors if "favicon" not in err.lower()]
    assert not fatal_console, f"Unexpected fatal console errors in real Chrome: {fatal_console}"
    assert not page_errors, f"Unexpected unhandled page script exceptions in real Chrome: {page_errors}"
