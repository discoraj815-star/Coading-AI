"""Tiered Memory Architecture: Hot (<1ms in-memory), Warm (recent sliding cache), Cold (Repository on-demand)."""

from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.schemas import (
    AlertEvent,
    Candle,
    MarketBias,
    MarketSetup,
    Position,
    TradeJournalEntry,
)
from trading_ai.schemas.performance import HotStateMemory
from trading_ai.storage.repository import Repository


class TieredMemoryManager:
    """
    Manages market state across Hot, Warm, and Cold memory tiers.
    Strict rule: Cold memory is NEVER loaded wholesale into live decision paths.
    """

    def __init__(self, repository: Optional[Repository] = None, max_warm_candles: int = 150):
        self._repo = repository
        self._max_warm_candles = max_warm_candles

        # HOT tier: in-memory state dictionary keyed by symbol
        self._hot_states: Dict[str, HotStateMemory] = {}

        # WARM tier: sliding window deque structures
        # Key: (symbol, timeframe) -> deque[Candle]
        self._warm_candles: Dict[str, deque] = {}
        # Key: symbol -> deque[AlertEvent]
        self._warm_alerts: Dict[str, deque] = {}
        # Key: symbol -> deque[TradeJournalEntry]
        self._warm_journal: Dict[str, deque] = {}

    # --- HOT TIER METHODS (<0.1ms access) ---

    def get_hot_state(self, symbol: str) -> HotStateMemory:
        """Fetch or initialize hot state for a symbol."""
        if symbol not in self._hot_states:
            self._hot_states[symbol] = HotStateMemory(
                symbol=symbol,
                current_price=0.0,
                last_updated=datetime.now(timezone.utc),
            )
        return self._hot_states[symbol]

    def update_hot_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> HotStateMemory:
        """Update current price in hot state."""
        hot = self.get_hot_state(symbol)
        hot.current_price = price
        hot.latest_price = price
        hot.last_updated = timestamp or datetime.now(timezone.utc)
        hot.last_updated_at = hot.last_updated
        return hot

    def update_hot_candle(self, symbol: str, candle: Candle) -> HotStateMemory:
        """Update active candle and price in hot state."""
        hot = self.get_hot_state(symbol)
        hot.active_candle = candle
        hot.current_price = candle.close
        hot.latest_price = candle.close
        if isinstance(candle.timestamp, (int, float)):
            hot.last_updated = datetime.fromtimestamp(candle.timestamp, tz=timezone.utc)
        else:
            hot.last_updated = candle.timestamp
        hot.last_updated_at = hot.last_updated
        return hot

    def update_hot_setup(self, symbol: str, setup: Optional[MarketSetup]) -> HotStateMemory:
        """Update active setup in hot state."""
        hot = self.get_hot_state(symbol)
        hot.active_setup = setup
        hot.last_updated = datetime.now(timezone.utc)
        return hot

    def update_hot_bias(self, symbol: str, bias: MarketBias) -> HotStateMemory:
        """Update market bias in hot state."""
        hot = self.get_hot_state(symbol)
        hot.current_bias = bias
        hot.last_updated = datetime.now(timezone.utc)
        return hot

    def update_hot_liquidity(self, symbol: str, liquidity_level: Optional[float]) -> HotStateMemory:
        """Update nearest liquidity in hot state."""
        hot = self.get_hot_state(symbol)
        hot.nearest_liquidity = liquidity_level
        hot.last_updated = datetime.now(timezone.utc)
        return hot

    def update_hot_position(self, symbol: str, position: Optional[Position]) -> HotStateMemory:
        """Update active position in hot state."""
        hot = self.get_hot_state(symbol)
        hot.active_position = position
        hot.last_updated = datetime.now(timezone.utc)
        return hot

    def update_hot_risk(self, symbol: str, risk_score: float) -> HotStateMemory:
        """Update active risk score in hot state."""
        hot = self.get_hot_state(symbol)
        hot.active_risk_score = risk_score
        hot.last_updated = datetime.now(timezone.utc)
        return hot

    def update_hot_journal_state(self, symbol: str, last_entry_id: Optional[str]) -> HotStateMemory:
        """Update last active journal entry in hot state."""
        hot = self.get_hot_state(symbol)
        hot.last_journal_entry_id = last_entry_id
        hot.last_updated = datetime.now(timezone.utc)
        return hot

    # --- WARM TIER METHODS (Sliding window cache) ---

    def append_warm_candle(self, symbol: str, timeframe: str, candle: Candle) -> None:
        """Append a candle to the warm sliding window for symbol/timeframe."""
        key = f"{symbol}:{timeframe}"
        if key not in self._warm_candles:
            self._warm_candles[key] = deque(maxlen=self._max_warm_candles)
        self._warm_candles[key].append(candle)

    def get_warm_candles(self, symbol: str, timeframe: str, limit: int = 50) -> List[Candle]:
        """Get recent candles from the warm window."""
        key = f"{symbol}:{timeframe}"
        if key not in self._warm_candles:
            return []
        items = list(self._warm_candles[key])
        return items[-limit:]

    def append_warm_alert(self, symbol: str, alert: AlertEvent) -> None:
        """Cache recent alert in warm window."""
        if symbol not in self._warm_alerts:
            self._warm_alerts[symbol] = deque(maxlen=50)
        self._warm_alerts[symbol].append(alert)

    def get_warm_alerts(self, symbol: str, limit: int = 10) -> List[AlertEvent]:
        """Fetch recent alerts from warm window."""
        if symbol not in self._warm_alerts:
            return []
        items = list(self._warm_alerts[symbol])
        return items[-limit:]

    def append_warm_journal_entry(self, entry: TradeJournalEntry) -> None:
        """Cache recent journal entry in warm tier."""
        symbol = entry.symbol
        if symbol not in self._warm_journal:
            self._warm_journal[symbol] = deque(maxlen=20)
        self._warm_journal[symbol].append(entry)

    def get_warm_journal_entries(self, symbol: str, limit: int = 5) -> List[TradeJournalEntry]:
        """Fetch recent journal entries from warm tier."""
        if symbol not in self._warm_journal:
            return []
        items = list(self._warm_journal[symbol])
        return items[-limit:]

    # --- COLD TIER RETRIEVAL (On-demand via repository) ---

    def query_cold_journal_entries(self, symbol: Optional[str] = None, limit: int = 20) -> List[TradeJournalEntry]:
        """Cold tier query: fetch historical journal records via repository."""
        if self._repo is None:
            return []
        return self._repo.list_journal_entries(symbol=symbol, limit=limit)

    def query_cold_knowledge(self, query: str, limit: int = 5) -> List[Any]:
        """Cold tier query: retrieve specific historical knowledge items."""
        if self._repo is None:
            return []
        return self._repo.search_knowledge(job_id="", query=query, limit=limit)

    def get_tier_stats(self) -> Dict[str, Any]:
        """Return memory footprint and item counts across tiers."""
        return {
            "hot_symbols_count": len(self._hot_states),
            "warm_candle_series_count": len(self._warm_candles),
            "warm_alerts_count": sum(len(q) for q in self._warm_alerts.values()),
            "warm_journal_count": sum(len(q) for q in self._warm_journal.values()),
            "has_cold_repo": self._repo is not None,
        }
