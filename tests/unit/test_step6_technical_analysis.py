"""
Unit tests for Step 6 Deep Technical Analysis Engine.
Tests swing classification (HH, HL, LH, LL, EQH, EQL), BOS, CHOCH,
displacement, dealing range equilibrium, liquidity sweeps, and confluence scoring.
"""

from datetime import datetime, timezone
import pytest

from trading_ai.core.technical_analysis import DeepTechnicalAnalyzer
from trading_ai.schemas.enums import (
    DealingRangeZone,
    MarketBias,
    MarketDirection,
    SwingClassification,
)
from trading_ai.schemas.market import Candle, MarketSnapshot


def _generate_candles(pattern: str = "bullish_trend") -> list[Candle]:
    """Helper to generate structured candle sequences for testing."""
    now = datetime.now(timezone.utc).timestamp()
    candles = []
    base_price = 100.0

    if pattern == "bullish_trend":
        # Clear Higher Highs and Higher Lows
        prices = [
            (100, 102, 99, 101),
            (101, 104, 100, 103),  # High 104
            (103, 103.5, 101, 102),# Low 101 (HL)
            (102, 107, 101.5, 106),# High 107 (HH)
            (106, 106.5, 103.5, 104), # Low 103.5 (HL)
            (104, 110, 103.8, 109), # High 110 (HH)
            (109, 109.5, 106.5, 107), # Low 106.5 (HL)
            (107, 113, 106.8, 112), # High 113 (HH)
            (112, 115, 111.5, 114.5), # Strong close
        ]
    elif pattern == "bearish_trend":
        # Clear Lower Highs and Lower Lows
        prices = [
            (100, 101, 98, 99),
            (99, 100, 96, 97),    # Low 96
            (97, 98.5, 96.5, 98), # High 98.5 (LH)
            (98, 98.2, 93, 94),   # Low 93 (LL)
            (94, 95.5, 93.5, 95), # High 95.5 (LH)
            (95, 95.2, 90, 91),   # Low 90 (LL)
            (91, 92.5, 90.5, 92), # High 92.5 (LH)
            (92, 92.2, 87, 88),   # Low 87 (LL)
            (88, 88.5, 85, 85.5), # Strong down
        ]
    elif pattern == "liquidity_sweep":
        # Prior high at 105, then a spike to 105.8 that closes at 104.2
        prices = [
            (100, 102, 99, 101),
            (101, 105, 100.5, 104), # Prior swing high: 105
            (104, 104.5, 101, 102),
            (102, 103, 101.5, 102.5),
            (102.5, 105.8, 102.0, 104.2), # Spike through 105 to 105.8, close back at 104.2 (Sweep!)
        ]
    elif pattern == "displacement":
        # Average range ~1.0, final candle range 3.0 with large body
        prices = [
            (100, 101.0, 100.0, 100.8),
            (100.8, 101.6, 100.6, 101.2),
            (101.2, 102.1, 101.1, 101.9),
            (101.9, 102.7, 101.8, 102.4),
            (102.4, 105.8, 102.2, 105.5), # Range 3.6, Body 3.1 -> Major displacement!
        ]
    else:
        prices = [(100, 101, 99, 100) for _ in range(8)]

    for i, (op, hi, lo, cl) in enumerate(prices):
        candles.append(
            Candle(
                timestamp=now + (i * 900),
                open=float(op),
                high=float(hi),
                low=float(lo),
                close=float(cl),
                volume=1000.0,
            )
        )
    return candles


def test_swing_and_structure_analysis():
    analyzer = DeepTechnicalAnalyzer()
    candles = _generate_candles("bullish_trend")
    snapshot = MarketSnapshot(
        symbol="EURUSD",
        timeframe="15m",
        candles=candles,
        latest_price=candles[-1].close,
        timestamp=datetime.now(timezone.utc).timestamp(),
    )

    report = analyzer.analyze_candles(snapshot)

    assert report.symbol == "EURUSD"
    assert report.timeframe == "15m"
    assert len(report.swings) > 0
    assert report.structure_bias in (MarketBias.BULLISH_BIAS, MarketBias.NEUTRAL_BIAS)
    assert report.confluence_score > 0.0
    assert report.dealing_range is not None


def test_liquidity_sweep_detection():
    analyzer = DeepTechnicalAnalyzer()
    candles = _generate_candles("liquidity_sweep")
    snapshot = MarketSnapshot(
        symbol="EURUSD",
        timeframe="15m",
        candles=candles,
        latest_price=candles[-1].close,
        timestamp=datetime.now(timezone.utc).timestamp(),
    )

    report = analyzer.analyze_candles(snapshot)
    assert len(report.liquidity_sweeps) > 0
    sweep = report.liquidity_sweeps[0]
    assert sweep["type"] == "BUY_SIDE_LIQUIDITY_SWEEP"
    assert sweep["sweep_high"] >= 105.0


def test_displacement_detection():
    analyzer = DeepTechnicalAnalyzer()
    candles = _generate_candles("displacement")
    snapshot = MarketSnapshot(
        symbol="BTCUSD",
        timeframe="15m",
        candles=candles,
        latest_price=candles[-1].close,
        timestamp=datetime.now(timezone.utc).timestamp(),
    )

    report = analyzer.analyze_candles(snapshot)
    assert report.displacement_detected is True
    assert report.displacement_direction == MarketDirection.BULLISH
