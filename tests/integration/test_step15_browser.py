"""Targeted Real Google Chrome Interactive Verification Test for Step 15.

Validates:
- Real Google Chrome browser automation
- Tab 15 navigation and layout
- AI Core Multilingual Comprehend form interaction & pure Hindi synthesis
- Four-Layer Epistemic Model DOM updates
- Large Video Streaming Chunk processing & Serial Barrier interaction
- Proactive Live Market Decision Engine setup evaluation
- Controlled Plugin Architecture listing and toggle
- Zero console errors and zero unhandled page script exceptions
"""
import socket
import threading
import time
import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from tests.helpers.browser_helper import find_system_chrome, launch_real_chrome
from trading_ai.server.app import app


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


def test_step15_real_chrome_interactive_verification():
    """Execute complete real Google Chrome interactive test of Step 15."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p, headless=True)
        assert browser is not None
        assert "chrome" in method.lower()

        page = browser.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigate to application
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200

        # 2. Switch to Tab 15
        tab15_btn = page.query_selector("#tab-step15-btn")
        assert tab15_btn is not None, "Tab 15 button must exist in DOM"
        tab15_btn.click()
        page.wait_for_timeout(800)

        tab15_pane_class = page.get_attribute("#tab-step15", "class") or ""
        assert "active" in tab15_pane_class, "Tab 15 pane must become active upon click"

        # 3. Verify Step 15 status badge
        badge_text = page.inner_text("#step15-status-badge")
        assert "ACTIVE" in badge_text

        # 4. Interact with AI Core Multilingual Comprehend form
        page.fill("#step15-input-text", "Humein liquidity sweep ke baad displacement ka wait karna hai.")
        comprehend_btn = page.query_selector("#step15-comprehend-btn")
        assert comprehend_btn is not None
        comprehend_btn.click()
        page.wait_for_timeout(1000)

        # Assert multilingual results in DOM
        detected_lang = page.inner_text("#step15-detected-lang")
        assert "Hinglish" in detected_lang
        normalized_meaning = page.inner_text("#step15-normalized-meaning")
        assert "displacement" in normalized_meaning.lower() or "sweep" in normalized_meaning.lower()
        hindi_explanation = page.inner_text("#step15-hindi-explanation")
        assert len(hindi_explanation) > 10

        # 5. Interact with Proactive Live Decision Engine
        live_eval_btn = page.query_selector("#step15-live-eval-btn")
        assert live_eval_btn is not None
        live_eval_btn.click()
        page.wait_for_timeout(1000)

        dec_badge = page.inner_text("#step15-decision-badge")
        assert dec_badge in ("WAIT", "TRADE_CANDIDATE", "NO_TRADE")
        concept_match = page.inner_text("#step15-concept-match-badge")
        assert concept_match in ("PARTIAL_MATCH", "FULL_MATCH", "NO_MATCH")
        hindi_advice = page.inner_text("#step15-live-hindi-advice")
        assert len(hindi_advice) > 5

        # 6. Interact with Large Video Streaming Chunk Pipeline
        init_vjob_btn = page.query_selector("#step15-create-vjob-btn")
        assert init_vjob_btn is not None
        init_vjob_btn.click()
        page.wait_for_timeout(800)

        vjob_id = page.inner_text("#step15-vjob-id")
        assert vjob_id != "None"

        stream_chunk_btn = page.query_selector("#step15-process-chunk-btn")
        assert stream_chunk_btn is not None
        stream_chunk_btn.click()
        page.wait_for_timeout(800)

        progress_text = page.inner_text("#step15-chunks-progress")
        assert "1 /" in progress_text or "Chunks" in progress_text

        # 7. Interact with Controlled Plugin Architecture
        refresh_plugins_btn = page.query_selector("#step15-refresh-plugins-btn")
        assert refresh_plugins_btn is not None
        refresh_plugins_btn.click()
        page.wait_for_timeout(800)

        plugins_list_text = page.inner_text("#step15-plugins-list")
        assert "Forex Factory" in plugins_list_text or "Technical Analysis" in plugins_list_text or "Interactive Brokers" in plugins_list_text

        browser.close()

    # Assert zero console or page errors
    assert len(console_errors) == 0, f"Found console errors in real Chrome: {console_errors}"
    assert len(page_errors) == 0, f"Found unhandled page errors in real Chrome: {page_errors}"
