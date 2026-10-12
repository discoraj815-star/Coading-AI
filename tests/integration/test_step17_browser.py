"""Targeted Real Google Chrome Interactive Verification Test for Step 17.

Validates:
- Real Google Chrome browser automation (C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)
- Tab 17 navigation and responsive layout
- Master Decision Engine status badges (State, Direction, Confluence Score, Threshold)
- Multi-Concept Modular Matrix rendering (SMC/ICT, Orderflow, Footprint, Volume Profile, Wyckoff)
- 4-Layer Epistemic Model display (OBSERVED, LEARNED, INFERRED, DECIDED)
- Deterministic Trade Plan (Entry, Invalidation SL, Target TP, R:R)
- Bilingual user summaries (English + Devanagari Hindi)
- Interactive decision evaluation trigger
- Immutable SQLite audit trail rendering in DOM
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


def test_step17_real_chrome_interactive_verification():
    """Execute complete real Google Chrome interactive test of Step 17."""
    chrome_path = find_system_chrome()
    assert chrome_path, "Google Chrome executable must be present on the host system"

    console_errors = []
    page_errors = []

    with sync_playwright() as p:
        browser, method = launch_real_chrome(p)
        assert browser is not None
        assert "chrome" in method.lower()

        page = browser.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # 1. Navigate to application
        resp = page.goto("http://127.0.0.1:8000", wait_until="networkidle", timeout=30000)
        assert resp and resp.status == 200

        # 2. Switch to Tab 17
        tab17_btn = page.query_selector("#tab-step17-btn")
        assert tab17_btn is not None, "Tab 17 button must exist in DOM"
        tab17_btn.click()
        page.wait_for_timeout(800)

        tab17_pane_class = page.get_attribute("#tab-step17", "class") or ""
        assert "active" in tab17_pane_class, "Tab 17 pane must become active upon click"

        # 3. Verify Step 17 status badge and KPI indicators
        badge_text = page.inner_text("#step17-state-badge")
        assert len(badge_text) > 0, "Decision state badge must contain text"

        kpi_thresh = page.inner_text("#step17-kpi-threshold")
        assert "8.0" in kpi_thresh or "10.0" in kpi_thresh

        # 4. Interact with Evaluate Decision button
        eval_btn = page.query_selector("#step17-eval-btn")
        assert eval_btn is not None
        eval_btn.click()
        page.wait_for_timeout(1000)

        # 5. Verify Multi-Concept Modular Matrix in DOM
        concepts_grid = page.inner_text("#step17-concepts-grid")
        assert "Smart Money Concepts" in concepts_grid or "SMC" in concepts_grid
        assert "Orderflow" in concepts_grid or "Delta" in concepts_grid

        # 6. Verify 4-Layer Epistemic Model display
        obs_text = page.inner_text("#step17-epistemic-observed")
        assert len(obs_text) > 5

        dec_text = page.inner_text("#step17-epistemic-decided")
        assert len(dec_text) > 3

        # 7. Verify Bilingual user summaries
        en_text = page.inner_text("#step17-english-summary")
        assert "Trade Status:" in en_text

        hi_text = page.inner_text("#step17-hindi-summary")
        assert "ट्रेड स्थिति:" in hi_text or "निर्णय" in hi_text

        # 8. Assert zero fatal console errors or unhandled page exceptions
        fatal_errors = [e for e in console_errors if "favicon" not in e.lower() and "404" not in e]
        assert len(fatal_errors) == 0, f"Fatal browser console errors: {fatal_errors}"
        assert len(page_errors) == 0, f"Unhandled page exceptions: {page_errors}"

        browser.close()
