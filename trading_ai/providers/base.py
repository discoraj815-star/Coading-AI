"""Step 18: Universal Provider Base Interfaces and Abstract Contracts."""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging

from trading_ai.schemas.provider import (
    ConnectionTestResult,
    ProviderCategory,
    ProviderLifecycleState,
    ProviderMetadata,
    TradingProviderExecutionState,
)

logger = logging.getLogger(__name__)


class BaseUniversalProvider(ABC):
    """Abstract base class for all Trading AI external providers."""

    def __init__(self, metadata: ProviderMetadata):
        self.metadata = metadata

    @property
    def provider_id(self) -> str:
        return self.metadata.provider_id

    @property
    def provider_name(self) -> str:
        return self.metadata.provider_name

    @property
    def category(self) -> ProviderCategory:
        return self.metadata.provider_type

    @property
    def status(self) -> ProviderLifecycleState:
        return self.metadata.status

    @property
    def is_enabled(self) -> bool:
        return self.metadata.enabled

    @property
    def is_approved(self) -> bool:
        return self.metadata.approved

    def get_metadata(self) -> ProviderMetadata:
        """Return provider metadata."""
        return self.metadata

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Dump underlying metadata model."""
        return self.metadata.model_dump(*args, **kwargs)

    @abstractmethod
    def test_connection(self) -> ConnectionTestResult:
        """
        Execute a safe, non-destructive connection test.
        Never place real orders or leak secrets.
        """
        pass

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize provider resources and validate non-secret config."""
        pass


class BacktestProvider(BaseUniversalProvider):
    """Abstract interface for historical backtesting and replay data providers."""

    @abstractmethod
    def get_historical_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: float,
        end_time: float,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Fetch historical candle bars strictly sliced up to end_time to prevent lookahead."""
        pass

    @abstractmethod
    def get_historical_news(
        self,
        symbol: str,
        start_time: float,
        end_time: float,
    ) -> List[Dict[str, Any]]:
        """Fetch historical economic events aligned to backtest timeframe."""
        pass


class BrokerProvider(BaseUniversalProvider):
    """
    Abstract interface for order execution and broker connectivity.
    Strictly isolated behind RiskEngine and ExecutionGate.
    """

    @abstractmethod
    def get_account_summary(self) -> Dict[str, Any]:
        """Fetch account balance, equity, and margin."""
        pass

    @abstractmethod
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Fetch currently held open positions."""
        pass

    @abstractmethod
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Fetch pending/active working orders."""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        direction: str,
        quantity: float,
        order_type: str = "LIMIT",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        client_order_id: Optional[str] = None,
        is_paper: bool = True,
    ) -> Tuple[bool, Optional[str], Dict[str, Any], Optional[str]]:
        """
        Submit validated order to broker.
        Returns: (success, broker_order_id, metadata, error_message)
        """
        pass

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel working order on broker."""
        pass


class TradingTerminalProvider(BaseUniversalProvider):
    """Abstract interface for professional desktop trading terminals (e.g. MetaTrader 5)."""

    @abstractmethod
    def check_terminal_status(self) -> Dict[str, Any]:
        """Verify terminal process reachability and account connectivity."""
        pass

    @abstractmethod
    def get_terminal_symbols(self) -> List[str]:
        """Fetch available tradable instruments in terminal."""
        pass

    @abstractmethod
    def execute_terminal_order(
        self,
        symbol: str,
        action: str,
        volume: float,
        price: Optional[float] = None,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
        comment: str = "TradingAI",
    ) -> Dict[str, Any]:
        """Route order through terminal IPC bridge."""
        pass


class MarketDataProvider(BaseUniversalProvider):
    """Abstract interface for market price and OHLCV feed providers."""

    @abstractmethod
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Return latest traded price."""
        pass

    @abstractmethod
    def get_candles(self, symbol: str, timeframe: str = "15m", limit: int = 100) -> List[Dict[str, Any]]:
        """Fetch recent candles."""
        pass

    @abstractmethod
    def check_data_freshness(self, symbol: str, max_age_seconds: float = 300.0) -> Tuple[bool, float]:
        """Verify data timestamp freshness to prevent stale trading."""
        pass


class OrderflowProvider(BaseUniversalProvider):
    """Abstract interface for delta, CVD, and execution pressure feeds."""

    @abstractmethod
    def get_delta_snapshot(self, symbol: str, timeframe: str = "15m") -> Dict[str, Any]:
        """
        Return delta, CVD, and aggressive volume stats.
        Must report is_available=False when feed is disconnected.
        """
        pass


class FootprintProvider(BaseUniversalProvider):
    """Abstract interface for bid/ask clusters and stacked imbalance feeds."""

    @abstractmethod
    def get_footprint_clusters(self, symbol: str, timeframe: str = "15m") -> Dict[str, Any]:
        """
        Return footprint volume clusters and imbalances.
        Must report is_available=False when feed is disconnected.
        """
        pass


class VolumeProfileProvider(BaseUniversalProvider):
    """Abstract interface for session and composite volume profiles."""

    @abstractmethod
    def get_volume_profile(self, symbol: str, timeframe: str = "15m") -> Dict[str, Any]:
        """
        Return POC, VAH, and VAL levels.
        Must report is_available=False when feed is disconnected.
        """
        pass


class NewsProvider(BaseUniversalProvider):
    """Abstract interface for economic calendar and high-impact news feeds."""

    @abstractmethod
    def get_upcoming_events(
        self,
        currencies: Optional[List[str]] = None,
        lookahead_hours: float = 24.0,
    ) -> List[Dict[str, Any]]:
        """Fetch upcoming economic events."""
        pass

    @abstractmethod
    def check_news_risk(self, symbol: str, window_minutes: float = 30.0) -> Dict[str, Any]:
        """Assess immediate event risk for a trading symbol."""
        pass
