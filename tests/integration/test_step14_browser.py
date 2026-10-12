"""Targeted Real Google Chrome Interactive Verification Test for Step 14 (Multi-Source & Core Trade Logic)."""
import socket
import threading
import time
import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from tests.helpers.browser_helper import launch_real_chrome, find_system_chrome
from trading_ai.server.app import app


def _is_server_listening(host="127.0.0.1", port=8000) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


@pytest.fixture(scope="module", autouse=True)
def ensure_server():
    """Ensure server is active before running browser verification."""
    for _ in range(50):
        if _is_server_listening("127.0.0.1", 8000):
            break
        time.sleep(0.2)
    yield


def test_step14_real_chrome_multi_source_and_core_logic_interaction():
    """Verify Step 14 Multi-Source & Core Trade Logic panel works interactively in real Google Chrome."""
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

        # 1. Navigation
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200

        # 2. Switch to Tab 14
        tl_btn = page.query_selector("#tab-trade-logic-btn")
        assert tl_btn is not None, "Tab 14 button must exist in DOM"
        tl_btn.click()
        page.wait_for_timeout(800)

        tl_pane_class = page.get_attribute("#tab-trade-logic", "class") or ""
        assert "active" in tl_pane_class, "Tab 14 pane must become active upon click"

        # 3. Verify Isolation & Domain stats
        isolation_badge = page.inner_text("#tl-isolation-badge")
        assert "ENFORCED" in isolation_badge

        stats_list = page.query_selector("#tl-domain-stats-list")
        assert stats_list is not None
        assert page.query_selector("#tl-domain-count-yt") is not None
        assert page.query_selector("#tl-domain-count-mentor") is not None
        assert page.query_selector("#tl-domain-count-live") is not None
        assert page.query_selector("#tl-domain-count-tech") is not None

        # 4. Search Knowledge
        search_btn = page.query_selector("#tl-search-btn")
        assert search_btn is not None
        search_btn.click()
        page.wait_for_timeout(800)

        knowledge_list = page.query_selector("#tl-knowledge-list")
        assert knowledge_list is not None

        # 5. Evaluate Setup (Cross-Source Confluence + Why-Not-Trade + Core Decision)
        eval_btn = page.query_selector("#tl-evaluate-btn")
        assert eval_btn is not None
        eval_btn.click()
        for _ in range(30):
            why_not_badge = page.inner_text("#tl-why-not-badge")
            if why_not_badge in ["PASS", "BLOCKED"]:
                break
            page.wait_for_timeout(200)

        conf_state = page.inner_text("#tl-confluence-state")
        assert conf_state != ""
        assert why_not_badge in ["PASS", "BLOCKED"]
        decision_badge = page.inner_text("#tl-decision-badge")
        assert decision_badge != ""

        # 6. Journal Feedback Loop
        jf_btn = page.query_selector("#tl-journal-feedback-btn")
        assert jf_btn is not None
        jf_btn.click()
        page.wait_for_timeout(800)

        jf_list = page.query_selector("#tl-journal-lessons-list")
        assert jf_list is not None

        browser.close()

    # Assert no console/page exceptions
    assert len(console_errors) == 0, f"Found console errors: {console_errors}"
    assert len(page_errors) == 0, f"Found unhandled page errors: {page_errors}"
