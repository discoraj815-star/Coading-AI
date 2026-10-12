"""Unit tests for Step 17 Master Decision Engine Central Architecture."""
import pytest
from trading_ai.master_decision.engine import MasterDecisionEngine
from trading_ai.schemas.enums import TradeDecisionState
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    db_file = str(tmp_path / "test_mde.db")
    db = DatabaseManager(db_file)
    return Repository(db)


def get_aligned_bullish_market_data():
    candles = [
        {"timestamp": 100, "open": 1.0800, "high": 1.0820, "low": 1.0790, "close": 1.0810, "volume": 500},
        {"timestamp": 200, "open": 1.0810, "high": 1.0840, "low": 1.0805, "close": 1.0835, "volume": 600},
        {"timestamp": 300, "open": 1.0835, "high": 1.0860, "low": 1.0830, "close": 1.0855, "volume": 800},
    ]
    return {
        "candles": candles,
        "current_price": 1.0855,
        "structure_bias": "BULLISH",
        "liquidity_swept": True,
        "fvg_detected": True,
        "orderflow_data": {
            "delta": 420.0,
            "cumulative_delta": 1300.0,
            "cvd_slope": "RISING",
            "absorption_detected": True,
            "absorption_side": "SELLERS_ABSORBED",
            "aggressive_buy_volume": 600.0,
            "aggressive_sell_volume": 180.0,
        },
        "volume_profile_data": {
            "poc": 1.0840,
            "vah": 1.0870,
            "val": 1.0810,
            "value_area_acceptance": "REJECTION_AT_VAL",
        },
    }


def test_master_decision_confirmed_setup(repo):
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = get_aligned_bullish_market_data()

    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)

    assert res.final_state == TradeDecisionState.CONFIRMED_SETUP
    assert res.direction == "BUY"
    assert res.confluence_score >= 8.0
    assert len(res.participating_concepts) >= 2
    assert res.invalidation_level is not None
    assert res.target_level is not None
    assert "CONFIRMED SETUP" in res.user_summary_en
    assert "CONFIRMED SETUP" in res.user_summary_hi or "पुष्टि" in res.user_summary_hi


def test_master_decision_macro_micro_conflict(repo):
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = get_aligned_bullish_market_data()
    # Invert orderflow to heavy aggressive selling
    m_data["orderflow_data"]["delta"] = -650.0
    m_data["orderflow_data"]["cvd_slope"] = "FALLING"
    m_data["orderflow_data"]["absorption_detected"] = False
    m_data["orderflow_data"]["aggressive_buy_volume"] = 100.0
    m_data["orderflow_data"]["aggressive_sell_volume"] = 750.0

    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)

    assert res.final_state in (TradeDecisionState.CONFLICTED, TradeDecisionState.NO_TRADE)
    assert any("MACRO_MICRO_CONFLICT" in c or "MISALIGNMENT" in c for c in res.conflicts)


def test_master_decision_insufficient_confluence_single_concept(repo):
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    # Market data with ONLY SMC candles; orderflow and VP missing
    m_data = {
        "candles": [
            {"timestamp": 100, "open": 1.0800, "high": 1.0820, "low": 1.0790, "close": 1.0810, "volume": 0},
            {"timestamp": 200, "open": 1.0810, "high": 1.0840, "low": 1.0805, "close": 1.0835, "volume": 0},
            {"timestamp": 300, "open": 1.0835, "high": 1.0860, "low": 1.0830, "close": 1.0855, "volume": 0},
        ],
        "structure_bias": "BULLISH",
    }
    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)

    assert res.final_state in (TradeDecisionState.INSUFFICIENT_EVIDENCE, TradeDecisionState.NO_TRADE)
    assert len(res.participating_concepts) < 2
    assert len(res.missing_evidence) >= 2


def test_master_decision_risk_engine_blocking(repo):
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = get_aligned_bullish_market_data()
    # Account hit daily loss limit
    m_data["daily_loss_limit_hit"] = True

    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)

    assert res.final_state in (TradeDecisionState.BLOCKED, TradeDecisionState.NO_TRADE)
    assert any("RISK_ENGINE_REJECTION" in r for r in res.blocking_reasons)
    assert res.risk_engine_status != "APPROVED"


def test_master_decision_execution_gate_circuit_breaker(repo):
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = get_aligned_bullish_market_data()
    m_data["circuit_breaker_active"] = True

    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)

    assert res.final_state in (TradeDecisionState.BLOCKED, TradeDecisionState.NO_TRADE)
    assert any("EXECUTION_GATE_REJECTION" in r for r in res.blocking_reasons)


def test_master_decision_backtest_no_lookahead(repo):
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = get_aligned_bullish_market_data()
    # Add future candles
    m_data["candles"].append({"timestamp": 500, "open": 1.0900, "high": 1.0950, "low": 1.0890, "close": 1.0940, "volume": 900})

    # Historical evaluation at timestamp 300
    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data, as_of_timestamp=300)
    assert res.timestamp == 300
    # Verified no lookahead beyond 300
    smc_res = res.concept_results.get("smc_ict")
    assert smc_res is not None
    assert smc_res.timestamp == 300


def test_deterministic_reproducibility(repo):
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = get_aligned_bullish_market_data()

    res1 = engine.evaluate_decision("EURUSD", "15m", market_data=m_data, force_recompute=True)
    res2 = engine.evaluate_decision("EURUSD", "15m", market_data=m_data, force_recompute=True)

    assert res1.final_state == res2.final_state
    assert res1.confluence_score == res2.confluence_score
    assert res1.direction == res2.direction
    assert res1.participating_concepts == res2.participating_concepts


def test_sqlite_audit_trail_saved(repo):
    engine = MasterDecisionEngine(repository=repo, threshold=8.0)
    m_data = get_aligned_bullish_market_data()

    res = engine.evaluate_decision("EURUSD", "15m", market_data=m_data)

    audit = repo.get_decision_audit(res.decision_id)
    assert audit is not None
    assert audit["symbol"] == "EURUSD"
    assert audit["final_state"] == res.final_state.value
    assert float(audit["confluence_score"]) == res.confluence_score
