"""
Step 6 Forex Factory Economic Calendar Adapter.
Fetches, parses, and normalizes public economic calendar events from Forex Factory.
Fails gracefully without fabricating data if access is blocked, restricted, or rate-limited.
"""

import json
import logging
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.fundamental.provider import FundamentalDataProvider
from trading_ai.schemas.enums import EventImpact, EventStatus, ProviderStatus
from trading_ai.schemas.fundamental import FundamentalEvent, FundamentalProviderState

logger = logging.getLogger(__name__)

# Standard public JSON calendar endpoint for Forex Factory
DEFAULT_FOREX_FACTORY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


class ForexFactoryProvider(FundamentalDataProvider):
    """
    Production-grade adapter for Forex Factory economic calendar.
    Normalizes raw calendar events into unified FundamentalEvent models.
    """

    def __init__(
        self,
        calendar_url: str = DEFAULT_FOREX_FACTORY_URL,
        timeout_seconds: float = 5.0,
        stale_threshold_seconds: int = 86400,
    ):
        self._url = calendar_url
        self._timeout = timeout_seconds
        self._stale_threshold = stale_threshold_seconds
        self._cached_events: List[FundamentalEvent] = []
        self._state = FundamentalProviderState(
            provider_name="forex_factory",
            status=ProviderStatus.OPERATIONAL,
            stale_after_seconds=stale_threshold_seconds,
        )

    def get_provider_status(self) -> FundamentalProviderState:
        """Return current provider health and staleness."""
        if self._state.last_successful_fetch:
            elapsed = (datetime.now(timezone.utc) - self._state.last_successful_fetch).total_seconds()
            self._state.is_stale = elapsed > self._stale_threshold
            if self._state.is_stale and self._state.status == ProviderStatus.OPERATIONAL:
                self._state.status = ProviderStatus.STALE
        return self._state

    def fetch_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        currencies: Optional[List[str]] = None,
    ) -> List[FundamentalEvent]:
        """
        Fetch events from Forex Factory feed or fallback to cache.
        Returns filtered list of normalized events.
        """
        try:
            req = urllib.request.Request(
                self._url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TradingAIAssistant/1.0",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                if resp.status == 200:
                    raw_data = json.loads(resp.read().decode("utf-8"))
                    parsed = self._normalize_feed(raw_data)
                    self._cached_events = parsed
                    self._state.status = ProviderStatus.OPERATIONAL
                    self._state.last_successful_fetch = datetime.now(timezone.utc)
                    self._state.error_message = None
                    self._state.unavailable_reason = None
                    self._state.is_stale = False
                else:
                    self._handle_fetch_error(f"HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self._state.status = ProviderStatus.RATE_LIMITED
            elif e.code in (401, 403):
                self._state.status = ProviderStatus.BLOCKED
            else:
                self._state.status = ProviderStatus.ERROR
            self._handle_fetch_error(f"HTTP {e.code}: {e.reason}")
        except Exception as e:
            self._state.status = ProviderStatus.UNAVAILABLE
            self._handle_fetch_error(f"Connection failed: {str(e)}")

        # Filter cached/retrieved events
        events = self._cached_events
        if currencies:
            upper_curr = {c.upper() for c in currencies}
            events = [e for e in events if e.currency.upper() in upper_curr]

        if start_time:
            events = [e for e in events if e.event_time >= start_time]

        if end_time:
            events = [e for e in events if e.event_time <= end_time]

        return events

    def load_mock_events(self, mock_events: List[Dict[str, Any]]) -> List[FundamentalEvent]:
        """Explicitly load normalized events from pre-defined mock data for testing."""
        parsed = self._normalize_feed(mock_events)
        self._cached_events = parsed
        self._state.status = ProviderStatus.OPERATIONAL
        self._state.last_successful_fetch = datetime.now(timezone.utc)
        self._state.is_stale = False
        return parsed

    def _normalize_feed(self, raw_items: List[Dict[str, Any]]) -> List[FundamentalEvent]:
        """Convert raw Forex Factory objects into normalized FundamentalEvent models."""
        normalized: List[FundamentalEvent] = []
        now = datetime.now(timezone.utc)

        for item in raw_items:
            title = str(item.get("title", "Unknown Event")).strip()
            currency = str(item.get("country", item.get("currency", "USD"))).strip().upper()
            date_str = str(item.get("date", ""))

            event_dt = self._parse_event_datetime(date_str)
            if not event_dt:
                continue

            impact_raw = str(item.get("impact", "Low")).strip().lower()
            if "high" in impact_raw:
                impact = EventImpact.HIGH
            elif "medium" in impact_raw or "med" in impact_raw:
                impact = EventImpact.MEDIUM
            elif "holiday" in impact_raw or "non" in impact_raw:
                impact = EventImpact.NON_ECONOMIC
            else:
                impact = EventImpact.LOW

            actual = item.get("actual")
            forecast = item.get("forecast")
            previous = item.get("previous")

            actual_str = str(actual).strip() if actual is not None else None
            forecast_str = str(forecast).strip() if forecast is not None else None
            previous_str = str(previous).strip() if previous is not None else None

            # If actual is published, event is RELEASED
            status = EventStatus.RELEASED if actual_str else EventStatus.SCHEDULED
            if event_dt > now and status != EventStatus.RELEASED:
                status = EventStatus.SCHEDULED

            event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ff_{currency}_{date_str}_{title}"))

            evidence = [
                f"Source: Forex Factory ({self._url})",
                f"Currency: {currency} | Title: {title} | Impact: {impact.value}",
                f"Schedule: {event_dt.isoformat()}",
            ]
            if actual_str:
                evidence.append(f"Actual: {actual_str} (Forecast: {forecast_str or 'N/A'}, Previous: {previous_str or 'N/A'})")

            event = FundamentalEvent(
                event_id=event_id,
                source="forex_factory",
                title=title,
                currency=currency,
                event_time=event_dt,
                impact=impact,
                previous=previous_str,
                forecast=forecast_str,
                actual=actual_str,
                status=status,
                retrieved_at=now,
                source_url=self._url,
                confidence=1.0,
                evidence=evidence,
            )
            normalized.append(event)

        return normalized

    def _parse_event_datetime(self, date_str: str) -> Optional[datetime]:
        """Robustly parse ISO8601 timestamps from calendar feeds."""
        if not date_str:
            return None
        try:
            # Handle ISO with offset, e.g., '2026-10-02T14:30:00-04:00'
            dt = datetime.fromisoformat(date_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            try:
                # Fallback for simple date without 'T'
                dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
                return dt.replace(tzinfo=timezone.utc)
            except Exception:
                logger.warning(f"Could not parse event date: {date_str}")
                return None

    def _handle_fetch_error(self, reason: str):
        """Update provider state upon failure while preserving cached events."""
        logger.warning(f"Forex Factory fetch issue: {reason}")
        self._state.error_message = reason
        self._state.unavailable_reason = (
            f"Forex Factory feed unreachable: {reason}. Using cached fundamental data if present."
        )
