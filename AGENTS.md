# Repository Rules & Standards — Trading AI Assistant

## Real Chrome Browser Verification Policy

Whenever browser or UI verification is relevant:
1. **Mandatory Real Google Chrome**: Always use the REAL installed Google Chrome browser (`C:\Program Files\Google\Chrome\Application\chrome.exe`) via the verified Playwright setup (`tests.helpers.browser_helper.launch_real_chrome`).
2. **Pre-flight Server Check**: Ensure the Trading AI Assistant server is active at `http://127.0.0.1:8000`.
3. **No Substitutes**: Never use mocked browsers, DOM-only simulations, API-only verification, or Playwright-managed Chromium downloads as substitutes for real browser automation.
4. **Interactive Validation Requirements**:
   - Verify application loads correctly at `http://127.0.0.1:8000`.
   - Verify expected UI and components are visible.
   - Actively interact with controls (clicks, tab switches, form inputs).
   - Verify dynamic DOM state changes and real API-backed updates.
   - Inspect and assert no fatal console errors or unhandled page script exceptions.
5. **Real Chrome First**: Static checks (HTML selector greps, regexes) are supplementary only. Real Chrome interaction is required before declaring any UI-related task complete.
6. **Non-destructive Diagnostics**: If a browser test fails, diagnose the real issue and apply the minimal necessary fix without altering working business logic or downgrading tests.
