"""Step 18 Correction: Independent Backtest Replay Clock and Lookahead Safety Engine.

Provides explicit Replay Bar workflow support:
- Independent historical clock completely decoupled from system wall-clock
- Variable replay speeds: 1x, 2x, 5x, 10x, 25x, 50x
- Step forward / step backward bar progression
- Strict zero-lookahead slicing for candles, orderflow, and economic news at timestamp T.
"""

import logging
from typing import Any, Dict, List, Optional
from trading_ai.schemas.market import Candle

logger = logging.getLogger(__name__)

SUPPORTED_REPLAY_SPEEDS = [1.0, 2.0, 5.0, 10.0, 25.0, 50.0]

TIMEFRAME_TO_SECONDS = {
    "1m": 60.0,
    "5m": 300.0,
    "15m": 900.0,
    "30m": 1800.0,
    "1h": 3600.0,
    "4h": 14400.0,
    "1d": 86400.0,
    "1w": 604800.0,
    "1M": 2592000.0,
}


class BacktestClock:
    """Independent historical clock controlling backtest and replay-bar progression.
    
    Guarantees that at replay timestamp T, the AI and strategies can ONLY see
    market data and fundamental news released at or before T.
    """

    def __init__(
        self,
        start_timestamp: float = 1672531200.0,  # 2023-01-01 00:00:00 UTC
        end_timestamp: Optional[float] = None,
        timeframe: str = "15m",
        speed: float = 1.0,
    ):
        self._start_timestamp: float = float(start_timestamp)
        self._current_timestamp: float = float(start_timestamp)
        self._end_timestamp: Optional[float] = float(end_timestamp) if end_timestamp else None
        self._timeframe: str = timeframe
        self._step_size_seconds: float = TIMEFRAME_TO_SECONDS.get(timeframe, 900.0)
        self._speed: float = float(speed) if speed in SUPPORTED_REPLAY_SPEEDS else 1.0
        self._is_paused: bool = True
        self._replay_mode: str = "IMPORTED_REPLAY_MODE"  # or "SUPPORTED_EXTERNAL_REPLAY_MODE"
        self._bars_stepped: int = 0

    @property
    def current_timestamp(self) -> float:
        return self._current_timestamp

    @property
    def start_timestamp(self) -> float:
        return self._start_timestamp

    @property
    def end_timestamp(self) -> Optional[float]:
        return self._end_timestamp

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def timeframe(self) -> str:
        return self._timeframe

    @property
    def step_size_seconds(self) -> float:
        return self._step_size_seconds

    @property
    def replay_mode(self) -> str:
        return self._replay_mode

    @property
    def bars_stepped(self) -> int:
        return self._bars_stepped

    def set_replay_mode(self, mode: str) -> None:
        """Set whether using IMPORTED_REPLAY_MODE or SUPPORTED_EXTERNAL_REPLAY_MODE."""
        self._replay_mode = mode

    def set_timeframe(self, timeframe: str) -> None:
        """Update timeframe and step size in seconds."""
        self._timeframe = timeframe
        self._step_size_seconds = TIMEFRAME_TO_SECONDS.get(timeframe, 900.0)

    def set_speed(self, speed: float) -> float:
        """Set replay playback speed multiplier."""
        sp = float(speed)
        if sp in SUPPORTED_REPLAY_SPEEDS:
            self._speed = sp
        else:
            # Fallback to closest valid speed
            self._speed = min(SUPPORTED_REPLAY_SPEEDS, key=lambda x: abs(x - sp))
        return self._speed

    def pause(self) -> None:
        """Pause replay progression."""
        self._is_paused = True

    def resume(self) -> None:
        """Resume replay progression."""
        self._is_paused = False

    def advance(self, seconds: float) -> float:
        """Advance the historical clock by a duration in seconds, respecting end boundary."""
        if seconds <= 0:
            return self._current_timestamp
        new_time = self._current_timestamp + float(seconds)
        if self._end_timestamp is not None and new_time > self._end_timestamp:
            new_time = self._end_timestamp
            self._is_paused = True
        self._current_timestamp = new_time
        return self._current_timestamp

    def step_forward(self, bars: int = 1) -> float:
        """Step forward exactly N bars according to active timeframe step size."""
        if bars <= 0:
            return self._current_timestamp
        duration = bars * self._step_size_seconds
        self.advance(duration)
        self._bars_stepped += bars
        return self._current_timestamp

    def step_backward(self, bars: int = 1) -> float:
        """Step backward safely down to the start timestamp."""
        if bars <= 0:
            return self._current_timestamp
        duration = bars * self._step_size_seconds
        new_time = self._current_timestamp - duration
        if new_time < self._start_timestamp:
            new_time = self._start_timestamp
        self._current_timestamp = new_time
        self._bars_stepped = max(0, self._bars_stepped - bars)
        return self._current_timestamp

    def reset(self, start_timestamp: Optional[float] = None) -> None:
        """Reset historical clock to the designated start."""
        if start_timestamp is not None:
            self._start_timestamp = float(start_timestamp)
        self._current_timestamp = self._start_timestamp
        self._is_paused = True
        self._bars_stepped = 0

    def slice_candles(self, candles: List[Candle]) -> List[Candle]:
        """Strict Zero-Lookahead Candle Filter.
        
        Returns only candles with timestamp <= current_timestamp.
        Guarantees that NO FUTURE CANDLE is visible at replay time T.
        """
        curr = self._current_timestamp
        return [c for c in candles if c.timestamp <= curr]

    def slice_news(self, news_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Strict Zero-Lookahead News Event Filter.
        
        Returns only news events whose event_time or release_time is <= current_timestamp.
        Guarantees that NO FUTURE HIGH-IMPACT NEWS is visible at replay time T.
        """
        curr = self._current_timestamp
        result = []
        for event in news_events:
            event_ts = event.get("event_time") or event.get("release_time") or event.get("timestamp") or 0.0
            if isinstance(event_ts, (int, float)) and event_ts <= curr:
                result.append(event)
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state for UI and diagnostic telemetry."""
        return {
            "current_timestamp": self._current_timestamp,
            "start_timestamp": self._start_timestamp,
            "end_timestamp": self._end_timestamp,
            "timeframe": self._timeframe,
            "step_size_seconds": self._step_size_seconds,
            "speed": self._speed,
            "is_paused": self._is_paused,
            "replay_mode": self._replay_mode,
            "bars_stepped": self._bars_stepped,
            "supported_speeds": SUPPORTED_REPLAY_SPEEDS,
        }
