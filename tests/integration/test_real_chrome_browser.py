"""Targeted Real Google Chrome Browser Automation Integration Test."""
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
    """Ensure a server is running on http://127.0.0.1:8000 before running browser tests."""
    for _ in range(50):
        if _is_server_listening("127.0.0.1", 8000):
            break
        time.sleep(0.2)
    yield


def test_real_chrome_browser_automation():
    """Verify real Google Chrome launches, navigates to localhost, and performs UI interactions."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be found on the system"

    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p, headless=True)
        assert browser is not None
        assert "chrome" in method.lower()

        page = browser.new_page()
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # Navigate to local server
        response = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert response is not None
        assert response.status == 200

        # Title & Document state
        title = page.title()
        assert "Trading AI Assistant" in title
        ready_state = page.evaluate("() => document.readyState")
        assert ready_state == "complete"

        # Verify key UI components
        assert page.query_selector(".dashboard-grid") is not None
        assert page.query_selector("#tab-performance-btn") is not None

        # Tab interaction: Click Performance tab
        page.click("#tab-performance-btn")
        page.wait_for_timeout(1000)

        perf_classes = page.get_attribute("#tab-performance", "class") or ""
        assert "active" in perf_classes

        # Verify dynamic DOM populated
        latency = page.inner_text("#perf-avg-latency")
        assert "ms" in latency

        # In-browser JS execution
        res = page.evaluate("() => ({ sum: 20 + 22, url: window.location.href })")
        assert res["sum"] == 42
        assert "8000" in res["url"]

        # Verify no fatal page script errors
        assert len(page_errors) == 0, f"Page errors occurred: {page_errors}"

        browser.close()
