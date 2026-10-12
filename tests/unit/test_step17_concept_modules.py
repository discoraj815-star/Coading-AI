"""Unit tests for Step 17 Modular Trading Concepts and Epistemic Separation."""
import pytest
from trading_ai.master_decision.modules.smc_ict import SMCICTModule
from trading_ai.master_decision.modules.orderflow import OrderflowModule
from trading_ai.master_decision.modules.footprint import FootprintModule
from trading_ai.master_decision.modules.volume_profile import VolumeProfileModule
from trading_ai.master_decision.modules.wyckoff import WyckoffModule
from trading_ai.master_decision.modules.auction_market import AuctionMarketTheoryModule
from trading_ai.schemas.enums import ConceptDirection


def test_smc_ict_module_evaluation():
    mod = SMCICTModule()
    candles = [
        {"timestamp": 100, "open": 1.0800, "high": 1.0820, "low": 1.0790, "close": 1.0810, "volume": 100},
        {"timestamp": 200, "open": 1.0810, "high": 1.0840, "low": 1.0805, "close": 1.0835, "volume": 150},
        {"timestamp": 300, "open": 1.0835, "high": 1.0860, "low": 1.0830, "close": 1.0855, "volume": 200},
    ]
    res = mod.evaluate("EURUSD", "15m", {"candles": candles, "structure_bias": "BULLISH", "liquidity_swept": True})

    assert res.is_data_available is True
    assert res.direction == ConceptDirection.BULLISH
    assert res.confidence > 0.70
    assert len(res.evidence) >= 2
    # Verify 4-layer epistemic separation
    assert len(res.epistemic_state.observed) >= 1
    assert len(res.epistemic_state.learned) >= 1
    assert len(res.epistemic_state.inferred) >= 1
    assert "BULLISH" in res.epistemic_state.decided


def test_smc_ict_missing_data_gating():
    mod = SMCICTModule()
    res = mod.evaluate("EURUSD", "15m", {})  # No candles provided
    assert res.is_data_available is False
    assert res.direction == ConceptDirection.UNKNOWN
    assert res.confidence == 0.0
    assert "candles" in res.missing_data


def test_orderflow_module_real_delta():
    mod = OrderflowModule()
    of_data = {
        "delta": 320.0,
        "cumulative_delta": 1150.0,
        "cvd_slope": "RISING",
        "absorption_detected": True,
        "absorption_side": "SELLERS_ABSORBED",
        "aggressive_buy_volume": 450.0,
        "aggressive_sell_volume": 130.0,
    }
    res = mod.evaluate("EURUSD", "15m", {"orderflow_data": of_data})
    assert res.is_data_available is True
    assert res.direction == ConceptDirection.BULLISH
    assert res.bias_score > 0.50
    assert "Delta: +320.0" in res.evidence[0]


def test_orderflow_missing_data_gating_never_fabricates():
    mod = OrderflowModule()
    # Market data with candles but NO orderflow
    res = mod.evaluate("EURUSD", "15m", {"candles": [{"close": 1.0850}]})
    assert res.is_data_available is False
    assert res.direction == ConceptDirection.UNKNOWN
    assert res.confidence == 0.0
    assert "orderflow_data" in res.missing_data
    assert "ORDERFLOW_DATA_UNAVAILABLE" in res.notes


def test_footprint_module_stacked_imbalance():
    mod = FootprintModule()
    fp_data = {
        "stacked_imbalance_side": "BUY",
        "stacked_imbalance_count": 4,
        "point_of_control_volume": 850.0,
        "unfinished_auction": False,
        "exhaustion_at_extreme": False,
    }
    res = mod.evaluate("EURUSD", "15m", {"footprint_data": fp_data})
    assert res.is_data_available is True
    assert res.direction == ConceptDirection.BULLISH
    assert res.confidence >= 0.80


def test_footprint_missing_data_gating_never_fabricates():
    mod = FootprintModule()
    res = mod.evaluate("EURUSD", "15m", {})
    assert res.is_data_available is False
    assert res.direction == ConceptDirection.UNKNOWN
    assert res.confidence == 0.0
    assert "FOOTPRINT_DATA_UNAVAILABLE" in res.notes


def test_volume_profile_module():
    mod = VolumeProfileModule()
    vp_data = {
        "poc": 1.0850,
        "vah": 1.0880,
        "val": 1.0820,
        "value_area_acceptance": "REJECTION_AT_VAL",
    }
    res = mod.evaluate("EURUSD", "15m", {"volume_profile_data": vp_data, "current_price": 1.0815})
    assert res.is_data_available is True
    assert res.direction == ConceptDirection.BULLISH
    assert "1.08500" in res.evidence[0]


def test_volume_profile_missing_data_gating():
    mod = VolumeProfileModule()
    # No volume profile and no candle volume
    res = mod.evaluate("EURUSD", "15m", {"candles": [{"close": 1.0850, "volume": 0}]})
    assert res.is_data_available is False
    assert res.direction == ConceptDirection.UNKNOWN
    assert "VOLUME_PROFILE_DATA_UNAVAILABLE" in res.notes


def test_wyckoff_unlearned_rule_gating():
    mod = WyckoffModule()
    # Without user taught Wyckoff rules in candidate_rules, module must not fabricate Wyckoff confirmation
    res = mod.evaluate("EURUSD", "15m", {"candles": [{"close": 1.0850}]}, candidate_rules=[])
    assert res.is_data_available is False
    assert res.direction == ConceptDirection.UNKNOWN
    assert "WYCKOFF_UNLEARNED_DATA_UNAVAILABLE" in res.notes


def test_wyckoff_with_learned_rules():
    mod = WyckoffModule()
    dummy_rule = type("DummyRule", (), {
        "item_id": "WYCKOFF_R1",
        "title": "Wyckoff Phase C Spring at Support",
        "statement": "Spring below support in accumulation phase confirms buying test.",
        "version": "1.0",
    })()
    res = mod.evaluate("EURUSD", "15m", {"candles": [{"close": 1.0850}]}, candidate_rules=[dummy_rule])
    assert res.is_data_available is True
    assert res.direction == ConceptDirection.BULLISH
    assert "SPRING" in res.evidence[0]


def test_plugin_module_approval_gating():
    # Unapproved plugin by default
    mod = AuctionMarketTheoryModule(is_approved=False)
    res = mod.evaluate("EURUSD", "15m", {"auction_data": {"state": "BUY_INITIATIVE"}})
    assert res.is_data_available is False
    assert res.direction == ConceptDirection.UNKNOWN
    assert "CONCEPT_UNAPPROVED" in res.notes

    # Approved plugin
    mod.is_approved = True
    res_approved = mod.evaluate("EURUSD", "15m", {"auction_data": {"state": "BUY_INITIATIVE"}})
    assert res_approved.is_data_available is True
    assert res_approved.direction == ConceptDirection.BULLISH
