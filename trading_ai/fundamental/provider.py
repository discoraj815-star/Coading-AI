"""
Step 6 Fundamental Data Provider Interface.
Abstract base class decoupling the trading assistant from specific economic calendar sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from trading_ai.schemas.fundamental import FundamentalEvent, FundamentalProviderState


class FundamentalDataProvider(ABC):
    """Abstract interface for retrieving economic calendar events."""

    @abstractmethod
    def fetch_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        currencies: Optional[List[str]] = None,
    ) -> List[FundamentalEvent]:
        """Fetch and return normalized economic events."""
        pass

    @abstractmethod
    def get_provider_status(self) -> FundamentalProviderState:
        """Return provider connection health, staleness, and rate limit status."""
        pass
