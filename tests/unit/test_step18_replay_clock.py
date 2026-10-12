"""Unit tests for Independent BacktestClock and Lookahead Bias Protection.

Verifies:
- BacktestClock advance, step, speed control (1x-50x), pause, resume, reset
- Zero-lookahead slicing on candles and news events at timestamp T
- TradingLeadsBacktestProvider Replay-Bar workflow
"""

import pytest

from trading_ai.backtest.replay_clock import BacktestClock, SUPPORTED_REPLAY_SPEEDS
from trading_ai.providers.backtest import TradingLeadsBacktestProvider
from trading_ai.schemas.market import Candle


def test_replay_clock_basic_progression():
    """Verify replay clock advance, pause, resume, reset."""
    clock = BacktestClock(start_timestamp=1000.0, end_timestamp=2000.0, timeframe="15m")
    assert clock.current_timestamp == 1000.0
    assert clock.is_paused is True

    # Resume and advance 900 seconds
    clock.resume()
    assert clock.is_paused is False
    clock.advance(900.0)
    assert clock.current_timestamp == 1900.0

    # Advance past end timestamp clamps to end
    clock.advance(500.0)
    assert clock.current_timestamp == 2000.0
    assert clock.is_paused is True

    # Reset
    clock.reset()
    assert clock.current_timestamp == 1000.0
    assert clock.is_paused is True


def test_replay_clock_speed_settings():
    """Verify supported speeds 1x, 2x, 5x, 10x, 25x, 50x."""
    clock = BacktestClock()
    for sp in SUPPORTED_REPLAY_SPEEDS:
        assert clock.set_speed(sp) == sp
        assert clock.speed == sp

    # Unsupported speed picks nearest valid
    assert clock.set_speed(3.0) in SUPPORTED_REPLAY_SPEEDS


def test_replay_clock_step_controls():
    """Verify step_forward and step_backward by bars."""
    clock = BacktestClock(start_timestamp=1000.0, timeframe="1m")
    assert clock.step_size_seconds == 60.0

    clock.step_forward(2)
    assert clock.current_timestamp == 1120.0
    assert clock.bars_stepped == 2

    clock.step_backward(1)
    assert clock.current_timestamp == 1060.0
    assert clock.bars_stepped == 1

    # Cannot step backward before start
    clock.step_backward(10)
    assert clock.current_timestamp == 1000.0
    assert clock.bars_stepped == 0


def test_zero_lookahead_candle_and_news_slicing():
    """Verify strict zero-lookahead slicing at replay timestamp T."""
    clock = BacktestClock(start_timestamp=1500.0)

    candles = [
        Candle(timestamp=1000.0, open=100, high=105, low=98, close=102, volume=10),
        Candle(timestamp=1400.0, open=102, high=108, low=101, close=107, volume=15),
        Candle(timestamp=1500.0, open=107, high=110, low=106, close=109, volume=20),
        Candle(timestamp=1600.0, open=109, high=112, low=108, close=111, volume=25),  # FUTURE
        Candle(timestamp=2000.0, open=111, high=115, low=110, close=114, volume=30),  # FUTURE
    ]

    visible_candles = clock.slice_candles(candles)
    assert len(visible_candles) == 3
    assert all(c.timestamp <= 1500.0 for c in visible_candles)

    # News events
    news = [
        {"event_time": 1200.0, "title": "Past CPI", "impact": "HIGH"},
        {"event_time": 1500.0, "title": "Present Rate Decision", "impact": "HIGH"},
        {"event_time": 1700.0, "title": "Future NFP", "impact": "HIGH"},  # FUTURE
    ]

    visible_news = clock.slice_news(news)
    assert len(visible_news) == 2
    assert all(n["event_time"] <= 1500.0 for n in visible_news)


def test_trading_leads_replay_bar_workflow():
    """Verify TradingLeadsBacktestProvider replay bar loading and step progression."""
    tl = TradingLeadsBacktestProvider()
    bars = [
        {"timestamp": 1000.0, "open": 50000, "high": 50500, "low": 49800, "close": 50200, "volume": 100},
        {"timestamp": 1900.0, "open": 50200, "high": 50800, "low": 50100, "close": 50700, "volume": 120},
        {"timestamp": 2800.0, "open": 50700, "high": 51000, "low": 50600, "close": 50900, "volume": 150},
    ]

    loaded = tl.load_replay_bars("BTC/USD", "15m", bars)
    assert loaded == 3

    # Initial bar at start timestamp (1000.0)
    current_bar = tl.get_current_replay_bar()
    assert current_bar is not None
    assert current_bar["timestamp"] == 1000.0

    # Step forward 1 bar (step size 900 -> 1900.0)
    next_bar = tl.step_replay(1)
    assert next_bar is not None
    assert next_bar["timestamp"] == 1900.0
    assert next_bar["close"] == 50700

    # Visible bars count is now 2
    assert len(tl.slice_visible_bars()) == 2
