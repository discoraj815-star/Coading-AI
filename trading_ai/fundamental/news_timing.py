"""
Step 6 News Timing Engine.
Analyzes economic calendar event timing relative to current market time,
maps instrument currencies, detects event clustering, and determines market event risk.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

from trading_ai.schemas.enums import (
    EventImpact,
    EventStatus,
    NewsRiskLevel,
    NewsTimingCategory,
)
from trading_ai.schemas.fundamental import FundamentalEvent, NewsTimingContext

logger = logging.getLogger(__name__)


# Default standard FX and commodity currency mappings
KNOWN_SYMBOL_CURRENCIES: Dict[str, List[str]] = {
    "EURUSD": ["EUR", "USD"],
    "EUR/USD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "GBP/USD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "USD/JPY": ["USD", "JPY"],
    "AUDUSD": ["AUD", "USD"],
    "AUD/USD": ["AUD", "USD"],
    "NZDUSD": ["NZD", "USD"],
    "NZD/USD": ["NZD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "USD/CAD": ["USD", "CAD"],
    "USDCHF": ["USD", "CHF"],
    "USD/CHF": ["USD", "CHF"],
    "EURGBP": ["EUR", "GBP"],
    "EUR/GBP": ["EUR", "GBP"],
    "EURJPY": ["EUR", "JPY"],
    "EUR/JPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
    "GBP/JPY": ["GBP", "JPY"],
    "XAUUSD": ["USD"],
    "XAU/USD": ["USD"],
    "BTCUSD": ["USD"],
    "BTC/USD": ["USD"],
    "ETHUSD": ["USD"],
    "ETH/USD": ["USD"],
    "SOLUSD": ["USD"],
    "SOL/USD": ["USD"],
}


class NewsTimingEngine:
    """
    Evaluates temporal proximity of economic news to current market time,
    identifying upcoming, active, and recently released events.
    """

    def __init__(
        self,
        pre_event_window_seconds: float = 1800.0,   # 30 min
        immediate_window_seconds: float = 300.0,    # 5 min
        post_event_window_seconds: float = 3600.0,  # 60 min
        stale_window_seconds: float = 86400.0,      # 24 hr
    ):
        self.pre_event_window = pre_event_window_seconds
        self.immediate_window = immediate_window_seconds
        self.post_event_window = post_event_window_seconds
        self.stale_window = stale_window_seconds

    def resolve_symbol_currencies(self, symbol: str) -> List[str]:
        """Extract constituent trading currencies from symbol string."""
        clean = symbol.strip().upper()
        if clean in KNOWN_SYMBOL_CURRENCIES:
            return KNOWN_SYMBOL_CURRENCIES[clean]

        # Handle slashed pairs (e.g. AAA/BBB)
        if "/" in clean:
            parts = clean.split("/")
            return [p.strip() for p in parts if len(p.strip()) == 3]

        # Handle 6-character forex pairs (e.g. CADCHF)
        if len(clean) == 6 and clean.isalpha():
            return [clean[:3], clean[3:]]

        # Default fallback to USD for crypto or generic assets
        return ["USD"]

    def evaluate_timing(
        self,
        symbol: str,
        events: List[FundamentalEvent],
        current_time: Optional[datetime] = None,
    ) -> NewsTimingContext:
        """
        Produce a structured NewsTimingContext for a symbol given all available events.
        """
        now = current_time or datetime.now(timezone.utc)
        currencies = self.resolve_symbol_currencies(symbol)
        curr_set = {c.upper() for c in currencies}

        # Filter events for relevant currencies
        relevant_events = [
            e for e in events if e.currency.upper() in curr_set
        ]

        if not relevant_events:
            return NewsTimingContext(
                symbol=symbol,
                relevant_currencies=currencies,
                category=NewsTimingCategory.NO_EVENT,
                risk_level=NewsRiskLevel.NO_RISK,
                next_high_impact_event=None,
                time_to_next_seconds=None,
                recent_released_event=None,
                time_since_last_seconds=None,
                has_event_clustering=False,
                active_events=[],
                upcoming_events=[],
            )

        upcoming: List[FundamentalEvent] = []
        recent: List[FundamentalEvent] = []
        active_now: List[FundamentalEvent] = []

        next_high_impact: Optional[FundamentalEvent] = None
        min_high_impact_diff = float("inf")

        for e in relevant_events:
            diff_seconds = (e.event_time - now).total_seconds()

            # Active now window: happening right now (-60s to +60s)
            if abs(diff_seconds) <= 60.0:
                active_now.append(e)

            if diff_seconds > 0:
                upcoming.append(e)
                if e.impact == EventImpact.HIGH and diff_seconds < min_high_impact_diff:
                    min_high_impact_diff = diff_seconds
                    next_high_impact = e
            else:
                recent.append(e)

        # Sort upcoming by soonest, recent by most recent
        upcoming.sort(key=lambda x: x.event_time)
        recent.sort(key=lambda x: x.event_time, reverse=True)

        nearest_upcoming = upcoming[0] if upcoming else None
        nearest_recent = recent[0] if recent else None

        time_to_next = (nearest_upcoming.event_time - now).total_seconds() if nearest_upcoming else None
        time_since_last = (now - nearest_recent.event_time).total_seconds() if nearest_recent else None

        # Check for event clustering: >= 2 high/medium events within 30 min of each other
        has_clustering = self._check_clustering(relevant_events)

        # Determine Category
        category = NewsTimingCategory.NO_EVENT
        if active_now:
            category = NewsTimingCategory.EVENT_HAPPENING_NOW
        elif time_to_next is not None and time_to_next <= self.immediate_window:
            category = NewsTimingCategory.EVENT_IMMINENT
        elif time_to_next is not None and time_to_next <= self.pre_event_window:
            category = NewsTimingCategory.UPCOMING_EVENT
        elif time_since_last is not None and time_since_last <= self.post_event_window:
            category = NewsTimingCategory.RECENTLY_RELEASED
        elif upcoming:
            category = NewsTimingCategory.UPCOMING_EVENT
        elif recent:
            category = NewsTimingCategory.HISTORICAL

        # Determine Risk Level
        risk_level = self._assess_risk(
            category=category,
            active_now=active_now,
            nearest_upcoming=nearest_upcoming,
            time_to_next=time_to_next,
            next_high_impact=next_high_impact,
            min_high_impact_diff=min_high_impact_diff,
            time_since_last=time_since_last,
            nearest_recent=nearest_recent,
            has_clustering=has_clustering,
        )

        return NewsTimingContext(
            symbol=symbol,
            relevant_currencies=currencies,
            category=category,
            risk_level=risk_level,
            next_high_impact_event=next_high_impact,
            time_to_next_seconds=time_to_next,
            recent_released_event=nearest_recent,
            time_since_last_seconds=time_since_last,
            has_event_clustering=has_clustering,
            active_events=active_now,
            upcoming_events=upcoming[:5],  # top 5 upcoming
        )

    def _check_clustering(self, events: List[FundamentalEvent], window_seconds: float = 1800.0) -> bool:
        """Return True if 2 or more HIGH/MEDIUM events occur within 30 minutes."""
        important = [
            e for e in events if e.impact in (EventImpact.HIGH, EventImpact.MEDIUM)
        ]
        if len(important) < 2:
            return False

        important.sort(key=lambda x: x.event_time)
        for i in range(len(important) - 1):
            gap = (important[i + 1].event_time - important[i].event_time).total_seconds()
            if 0 <= gap <= window_seconds:
                return True
        return False

    def _assess_risk(
        self,
        category: NewsTimingCategory,
        active_now: List[FundamentalEvent],
        nearest_upcoming: Optional[FundamentalEvent],
        time_to_next: Optional[float],
        next_high_impact: Optional[FundamentalEvent],
        min_high_impact_diff: float,
        time_since_last: Optional[float],
        nearest_recent: Optional[FundamentalEvent],
        has_clustering: bool,
    ) -> NewsRiskLevel:
        """Deterministic rule-based risk level assessment."""
        # 1. Extreme risk if high impact is happening right now or within immediate window
        if any(e.impact == EventImpact.HIGH for e in active_now):
            return NewsRiskLevel.EXTREME_RISK

        if next_high_impact and min_high_impact_diff <= self.immediate_window:
            return NewsRiskLevel.EXTREME_RISK

        # 2. High risk if high impact is upcoming within pre-event window (30 min)
        if next_high_impact and min_high_impact_diff <= self.pre_event_window:
            return NewsRiskLevel.HIGH_RISK

        if has_clustering and time_to_next is not None and time_to_next <= self.pre_event_window:
            return NewsRiskLevel.HIGH_RISK

        # 3. Medium risk if medium impact upcoming within 30 min or high impact just released (< 15 min)
        if nearest_upcoming and nearest_upcoming.impact == EventImpact.MEDIUM and time_to_next and time_to_next <= self.pre_event_window:
            return NewsRiskLevel.MEDIUM_RISK

        if nearest_recent and nearest_recent.impact == EventImpact.HIGH and time_since_last and time_since_last <= 900.0:
            return NewsRiskLevel.MEDIUM_RISK

        # 4. Low risk for low impact upcoming or historical events
        if nearest_upcoming:
            return NewsRiskLevel.LOW_RISK

        return NewsRiskLevel.NO_RISK
