"""Browser automation helper for Trading AI tests using installed Google Chrome."""
from pathlib import Path
from typing import Tuple, Any
from playwright.sync_api import sync_playwright, Browser, Playwright

DEFAULT_CHROME_PATHS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

def find_system_chrome() -> str:
    """Find installed Google Chrome executable path."""
    for p in DEFAULT_CHROME_PATHS:
        if Path(p).exists():
            return p
    return ""

def launch_real_chrome(p: Playwright, headless: bool = True, **kwargs) -> Tuple[Browser, str]:
    """Launch real installed Google Chrome via channel='chrome' or fallback to executable_path.
    
    Returns:
        Tuple[Browser, str]: The launched browser and the method used ("channel:chrome" or "executable_path:<path>").
    """
    # Attempt 1: channel="chrome"
    try:
        browser = p.chromium.launch(channel="chrome", headless=headless, **kwargs)
        return browser, "channel:chrome"
    except Exception as channel_err:
        pass

    # Attempt 2: explicit executable_path
    chrome_path = find_system_chrome()
    if chrome_path:
        browser = p.chromium.launch(executable_path=chrome_path, headless=headless, **kwargs)
        return browser, f"executable_path:{chrome_path}"

    raise RuntimeError("Could not launch Google Chrome via channel='chrome' or executable_path.")
