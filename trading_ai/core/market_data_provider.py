"""Market Data Provider abstraction and realistic simulated provider for Trading AI."""

import math
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from trading_ai.schemas.enums import MarketDataStatus
from trading_ai.schemas.market import Candle, MarketSnapshot


class BaseMarketDataProvider(ABC):
    """Abstract interface for retrieving live and historical market OHLCV data."""

    @abstractmethod
    def get_snapshot(
        self, symbol: str, timeframe: str = "15m", limit: int = 100
    ) -> MarketSnapshot:
        """Fetch historical candles and latest price for a symbol and timeframe."""
        pass

    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """Fetch the most recent traded price for a symbol."""
        pass

    @abstractmethod
    def check_status(self, symbol: str) -> MarketDataStatus:
        """Check the operational health and freshness of data for a symbol."""
        pass

    def get_historical_series(
        self, symbol: str, timeframe: str = "15m", limit: int = 100, end_time: Optional[float] = None
    ) -> List[Candle]:
        """Fetch deterministic historical candles up to end_time strictly to prevent look-ahead bias."""
        snapshot = self.get_snapshot(symbol=symbol, timeframe=timeframe, limit=limit)
        if end_time is None:
            return snapshot.candles
        return [c for c in snapshot.candles if c.timestamp <= end_time]

    def is_stale(self, snapshot: MarketSnapshot, max_age_seconds: float = 300.0) -> bool:
        """Determine if a snapshot's data timestamp exceeds freshness limits."""
        if snapshot.status in (MarketDataStatus.DATA_STALE, MarketDataStatus.DATA_MISSING):
            return True
        current_time = time.time()
        return (current_time - snapshot.timestamp) > max_age_seconds


class SimulatedMarketDataProvider(BaseMarketDataProvider):
    """
    High-fidelity simulated market data provider.

    Generates realistic, deterministic market structures (swings, FVGs, Order Blocks)
    for reliable testing, verification, and live demonstration without external API keys.
    Supports forced error/stale injection for failure-path testing.
    """

    BASE_PRICES = {
        "BTC/USD": 65000.0,
        "ETH/USD": 3500.0,
        "SOL/USD": 145.0,
        "EUR/USD": 1.0850,
        "GBP/USD": 1.2750,
        "USD/JPY": 155.0,
        "XAU/USD": 2350.0,
    }

    TIMEFRAME_SECONDS = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }

    def __init__(self, seed: Optional[int] = 42):
        self._seed = seed
        self._cache: Dict[str, MarketSnapshot] = {}
        self._forced_status: Dict[str, MarketDataStatus] = {}
        self._stale_offsets: Dict[str, float] = {}

    def force_status(self, symbol: str, status: MarketDataStatus) -> None:
        """For testing: force a specific data status (e.g. DATA_STALE, DATA_MISSING, PROVIDER_ERROR)."""
        self._forced_status[symbol] = status

    def clear_forced_status(self, symbol: str) -> None:
        """Clear forced test status."""
        self._forced_status.pop(symbol, None)

    def set_stale_offset(self, symbol: str, offset_seconds: float) -> None:
        """For testing: backdate timestamps by offset_seconds to simulate staleness."""
        self._stale_offsets[symbol] = offset_seconds

    def check_status(self, symbol: str) -> MarketDataStatus:
        """Return operational status for a symbol."""
        if symbol in self._forced_status:
            return self._forced_status[symbol]
        return MarketDataStatus.DATA_AVAILABLE

    def get_latest_price(self, symbol: str) -> float:
        """Return the latest price for a symbol."""
        snapshot = self.get_snapshot(symbol=symbol, timeframe="15m", limit=10)
        return snapshot.latest_price

    def get_snapshot(
        self, symbol: str, timeframe: str = "15m", limit: int = 100
    ) -> MarketSnapshot:
        """
        Generate or retrieve a snapshot of OHLCV candles.

        Constructs realistic price movements:
          - Initial range / consolidation
          - Displacement creating Fair Value Gaps (FVG)
          - Structural swing points and breaks of structure (BOS)
        """
        status = self.check_status(symbol)
        cache_key = f"{symbol}_{timeframe}_{limit}"

        if status == MarketDataStatus.DATA_MISSING:
            return MarketSnapshot(
                symbol=symbol,
                timeframe=timeframe,
                candles=[],
                latest_price=0.0,
                timestamp=time.time(),
                status=MarketDataStatus.DATA_MISSING,
                error=f"No market data available for {symbol}",
            )

        if status == MarketDataStatus.PROVIDER_ERROR:
            return MarketSnapshot(
                symbol=symbol,
                timeframe=timeframe,
                candles=[],
                latest_price=0.0,
                timestamp=time.time(),
                status=MarketDataStatus.PROVIDER_ERROR,
                error=f"Provider connection error for {symbol}",
            )

        interval = self.TIMEFRAME_SECONDS.get(timeframe, 900)
        now = time.time()
        if symbol in self._stale_offsets:
            now -= self._stale_offsets[symbol]
            status = MarketDataStatus.DATA_STALE

        bar_base = float((int(now) // interval) * interval)
        base_price = self.BASE_PRICES.get(symbol, 100.0)
        volatility = base_price * 0.002

        # Deterministic generation based on symbol and seed
        rng = random.Random((hash(symbol) + (self._seed or 0)) & 0xFFFFFFFF)

        candles: List[Candle] = []
        current_open = base_price
        start_time = bar_base - (limit * interval)

        for i in range(limit):
            candle_time = start_time + (i * interval)

            # Inject a clear structure pattern:
            # - periodic impulsive pushes (displacement) to form FVGs
            # - swings for BOS / CHOCH
            cycle = i % 20
            if cycle == 10:  # Impulsive Bullish Displacement
                drift = volatility * 3.5
            elif cycle == 11:  # Follow-through
                drift = volatility * 2.0
            elif cycle == 18:  # Pullback / Swing high rejection
                drift = -volatility * 2.5
            else:
                drift = rng.uniform(-volatility, volatility)

            current_close = current_open + drift
            candle_high = max(current_open, current_close) + abs(rng.uniform(0.1, 0.8) * volatility)
            candle_low = min(current_open, current_close) - abs(rng.uniform(0.1, 0.8) * volatility)
            volume = rng.uniform(100.0, 5000.0) * (2.5 if cycle in (10, 11) else 1.0)

            candles.append(
                Candle(
                    timestamp=candle_time,
                    open=round(current_open, 4),
                    high=round(candle_high, 4),
                    low=round(candle_low, 4),
                    close=round(current_close, 4),
                    volume=round(volume, 2),
                )
            )
            current_open = current_close

        latest_price = candles[-1].close if candles else base_price

        snapshot = MarketSnapshot(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            latest_price=latest_price,
            timestamp=now,
            status=status,
            error=None,
        )
        self._cache[cache_key] = snapshot
        return snapshot
