"""Adversarial security and edge-case tests for Step 17 Master Decision Engine."""
import pytest
from trading_ai.master_decision.engine import MasterDecisionEngine
from trading_ai.schemas.enums import TradeDecisionState
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    db_file = str(tmp_path / "test_adv.db")
    db = DatabaseManager(db_file)
    return Repository(db)


def test_adversarial_score_alone_never_authorizes_trade(repo):
    """Even if confluence score is high, if RiskEngine or daily loss limit is hit, must produce BLOCKED."""
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    # High score inputs
    m_data = {
        "candles": [
            {"timestamp": 100, "open": 1.080, "high": 1.082, "low": 1.079, "close": 1.081, "volume": 500},
            {"timestamp": 200, "open": 1.081, "high": 1.084, "low": 1.0805, "close": 1.0835, "volume": 600},
            {"timestamp": 300, "open": 1.0835, "high": 1.086, "low": 1.083, "close": 1.0855, "volume": 800},
        ],
        "structure_bias": "BULLISH",
        "liquidity_swept": True,
        "fvg_detected": True,
        "orderflow_data": {
            "delta": 500.0,
            "cumulative_delta": 1500.0,
            "cvd_slope": "RISING",
            "absorption_detected": True,
            "absorption_side": "SELLERS_ABSORBED",
        },
        "volume_profile_data": {
            "poc": 1.0840,
            "vah": 1.0870,
            "val": 1.0810,
            "value_area_acceptance": "REJECTION_AT_VAL",
        },
        # ADVERSARIAL INJECTION: Daily loss limit hit + Circuit breaker active
        "daily_loss_limit_hit": True,
        "circuit_breaker_active": True,
    }

    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)

    # Must NOT produce CONFIRMED_SETUP
    assert res.final_state in (TradeDecisionState.BLOCKED, TradeDecisionState.NO_TRADE)
    assert res.final_state != TradeDecisionState.CONFIRMED_SETUP
    assert any("RISK_ENGINE_REJECTION" in r for r in res.blocking_reasons)
    assert any("EXECUTION_GATE_REJECTION" in r for r in res.blocking_reasons)


def test_adversarial_conflicting_signals_produces_conflicted(repo):
    """When concepts actively contradict each other, engine must NOT average them out."""
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = {
        "candles": [
            {"timestamp": 100, "open": 1.080, "high": 1.082, "low": 1.079, "close": 1.081, "volume": 500},
            {"timestamp": 200, "open": 1.081, "high": 1.084, "low": 1.0805, "close": 1.0835, "volume": 600},
            {"timestamp": 300, "open": 1.0835, "high": 1.086, "low": 1.083, "close": 1.0855, "volume": 800},
        ],
        "structure_bias": "BULLISH",
        "orderflow_data": {
            "delta": -850.0,  # Extreme seller aggression
            "cumulative_delta": -2500.0,
            "cvd_slope": "FALLING",
            "aggressive_sell_volume": 1200.0,
        },
    }

    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)

    assert res.final_state in (TradeDecisionState.CONFLICTED, TradeDecisionState.NO_TRADE)
    assert len(res.conflicts) >= 1


def test_adversarial_unapproved_plugin_blocked(repo):
    """An unapproved third-party concept module cannot contribute positive score."""
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    # AMT plugin is unapproved by default
    amt_mod = engine.get_registered_modules()["auction_market_theory"]
    amt_mod.is_approved = False

    res = engine.evaluate_decision(
        "EURUSD",
        "15m",
        market_data={"auction_data": {"state": "BUY_INITIATIVE"}},
    )

    amt_res = res.concept_results.get("auction_market_theory")
    assert amt_res is not None
    assert amt_res.is_data_available is False
    assert "CONCEPT_UNAPPROVED" in amt_res.notes
    assert "auction_market_theory" not in res.participating_concepts


def test_adversarial_stale_data_hard_block(repo):
    """Stale market data must immediately block confirmation even with good technical patterns."""
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = {
        "candles": [
            {"timestamp": 100, "open": 1.080, "high": 1.082, "low": 1.079, "close": 1.081, "volume": 500},
            {"timestamp": 200, "open": 1.081, "high": 1.084, "low": 1.0805, "close": 1.0835, "volume": 600},
        ],
        "structure_bias": "BULLISH",
        "is_stale": True,
    }

    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)
    assert res.final_state in (TradeDecisionState.CONFLICTED, TradeDecisionState.BLOCKED, TradeDecisionState.NO_TRADE)
    assert any("STALE" in c for c in res.blocking_reasons or res.conflicts)
