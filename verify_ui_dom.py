"""Script to verify HTML DOM and JavaScript syntax for Step 10 Journal."""
import sys

with open("trading_ai/ui/index.html", "r", encoding="utf-8") as f:
    html = f.read()

required_ids = [
    "tab-journal-btn",
    "tab-journal",
    "pill-env-backtest",
    "pill-env-paper",
    "pill-env-live",
    "jnl-total-trades",
    "jnl-win-rate",
    "jnl-profit-factor",
    "jnl-avg-r",
    "jnl-total-pnl",
    "jnl-max-dd",
    "jnl-rule-adherence",
    "pretrade-feedback-box",
    "pretrade-status-badge",
    "jnl-entries-container",
    "jnl-lessons-container",
]

for rid in required_ids:
    assert f'id="{rid}"' in html, f"Missing element with id: {rid}"

required_js = [
    "selectJournalEnv",
    "selectJournalBacktestType",
    "filterJournalAsset",
    "refreshJournalUI",
    "fetchJournalAnalyticsUI",
    "fetchJournalTradesUI",
    "fetchJournalLessonsUI",
    "reviewJournalLessonUI",
    "applyValidatedLessonUI",
]

for js in required_js:
    assert f"function {js}" in html, f"Missing JS function: {js}"

print("All Step 10 Journal UI DOM and JavaScript assertions passed successfully!")
