"""Unit tests for Step 8 Confluence Engine."""

import pytest
from trading_ai.execution.confluence_engine import ConfluenceEngine
from trading_ai.schemas.enums import (
    LevelType,
    MarketBias,
    MarketDirection,
    SetupStatus,
)
from trading_ai.schemas.market import (
    CurrentSituationSnapshot,
    MarketLevel,
    MarketSetup,
)


@pytest.fixture
def confluence_engine():
    return ConfluenceEngine()


def test_confluence_engine_independent_confirmations(confluence_engine):
    poi = MarketLevel(
        symbol="EUR/USD",
        timeframe="15m",
        price=1.0851,
        level_type=LevelType.OB_HIGH,
        evidence_json={"name": "Bullish OB"},
    )
    liq = MarketLevel(
        symbol="EUR/USD",
        timeframe="1h",
        price=1.0920,
        level_type=LevelType.LIQUIDITY_BUY_SIDE,
    )
    setup = MarketSetup(
        setup_id="stp_conf_1",
        symbol="EUR/USD",
        timeframe="15m",
        setup_type="OB_REJECTION",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.CONFIRMED,
        applicable_rule_id="RULE_OB_01",
        conditions_satisfied=["SWEEP", "DISPLACEMENT"],
        confidence=0.9,
    )

    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.BULLISH_BIAS,
        bias_summary="HTF bullish structure intact.",
        current_poi=poi,
        nearest_major_liquidity=liq,
        active_confirmed_setup=setup,
        technical_fundamental_relationship="TECHNICALLY_AND_FUNDAMENTALLY_ALIGNED",
    )

    confluences, is_sufficient, warnings = confluence_engine.evaluate_confluences(
        symbol="EUR/USD",
        situation=sit,
        direction="BUY",
    )

    assert is_sufficient is True
    assert len(confluences) >= 3
    # Check that categories and independence groups are unique
    groups = [c.independence_group for c in confluences]
    assert len(groups) == len(set(groups)), "Confluences must not double-count the same underlying group"


def test_confluence_engine_insufficient_confluence(confluence_engine):
    sit = CurrentSituationSnapshot(
        symbol="EUR/USD",
        current_price=1.0850,
        overall_bias=MarketBias.NEUTRAL_BIAS,
        bias_summary="Ranging market without clean structure.",
    )

    confluences, is_sufficient, warnings = confluence_engine.evaluate_confluences(
        symbol="EUR/USD",
        situation=sit,
        direction="BUY",
    )

    assert is_sufficient is False
    assert any("INSUFFICIENT_CONFLUENCE" in w for w in warnings)
