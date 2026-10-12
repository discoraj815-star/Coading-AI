"""Performance Decision Cache for Step 17 Master Decision Engine.

Provides token-efficient, zero-overhead caching for deterministic concept analysis.
Invalidates cache entries when market candle timestamps change or when explicit re-evaluation is requested.
"""
import time
from typing import Any, Dict, Optional, Tuple

from trading_ai.schemas.master_decision import MasterDecisionResult


class MasterDecisionCache:
    """In-memory incremental evaluation cache for MasterDecisionEngine."""

    def __init__(self, ttl_seconds: float = 60.0):
        self._ttl = ttl_seconds
        self._cache: Dict[str, Tuple[float, float, MasterDecisionResult]] = {}  # key -> (insert_time, candle_ts, result)

    def _make_key(self, symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}:{timeframe.lower()}"

    def get(
        self, symbol: str, timeframe: str, current_candle_ts: float
    ) -> Optional[MasterDecisionResult]:
        key = self._make_key(symbol, timeframe)
        entry = self._cache.get(key)
        if not entry:
            return None

        insert_time, cached_candle_ts, result = entry
        now = time.time()

        # Invalidate if candle progressed or TTL expired
        if (now - insert_time) > self._ttl or current_candle_ts > cached_candle_ts:
            del self._cache[key]
            return None

        return result

    def put(
        self, symbol: str, timeframe: str, current_candle_ts: float, result: MasterDecisionResult
    ) -> None:
        key = self._make_key(symbol, timeframe)
        self._cache[key] = (time.time(), current_candle_ts, result)

    def clear(self) -> None:
        self._cache.clear()
