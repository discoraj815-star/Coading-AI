"""Unit tests for Step 17 Confluence Scoring and Decision Conflict Engine."""
import pytest
from trading_ai.master_decision.conflict_engine import DecisionConflictEngine
from trading_ai.schemas.enums import ConceptDirection
from trading_ai.schemas.master_decision import ConceptResult, EpistemicConceptState


def make_dummy_concept(cid, name, direction, confidence=0.8, is_available=True):
    return ConceptResult(
        concept_id=cid,
        concept_name=name,
        symbol="EURUSD",
        timeframe="15m",
        timestamp=100.0,
        direction=direction,
        bias_score=0.8 if direction == ConceptDirection.BULLISH else (-0.8 if direction == ConceptDirection.BEARISH else 0.0),
        confidence=confidence,
        evidence=["Factual evidence 1"],
        evidence_quality=0.85,
        is_data_available=is_available,
        epistemic_state=EpistemicConceptState(decided=f"{name} {direction.value}"),
    )


def test_conflict_engine_macro_micro_conflict():
    engine = DecisionConflictEngine()
    # Macro SMC is BULLISH, but Micro Orderflow has strong aggressive BEARISH sellers
    concepts = {
        "smc_ict": make_dummy_concept("smc_ict", "SMC/ICT", ConceptDirection.BULLISH),
        "orderflow": make_dummy_concept("orderflow", "Orderflow", ConceptDirection.BEARISH),
    }

    has_conflict, conflicts, warnings = engine.evaluate_conflicts(
        proposed_direction="BUY",
        concept_results=concepts,
        market_data={},
    )
    assert has_conflict is True
    assert any("MACRO_MICRO_CONFLICT" in c for c in conflicts)


def test_conflict_engine_concept_polarity_conflict():
    engine = DecisionConflictEngine()
    concepts = {
        "smc_ict": make_dummy_concept("smc_ict", "SMC/ICT", ConceptDirection.BULLISH),
        "volume_profile": make_dummy_concept("volume_profile", "Volume Profile", ConceptDirection.BEARISH),
    }

    has_conflict, conflicts, warnings = engine.evaluate_conflicts(
        proposed_direction="BUY",
        concept_results=concepts,
        market_data={},
    )
    assert has_conflict is True
    assert any("CONCEPT_POLARITY_CONFLICT" in c for c in conflicts)


def test_conflict_engine_stale_data_blocks():
    engine = DecisionConflictEngine()
    concepts = {
        "smc_ict": make_dummy_concept("smc_ict", "SMC/ICT", ConceptDirection.BULLISH),
        "orderflow": make_dummy_concept("orderflow", "Orderflow", ConceptDirection.BULLISH),
    }

    has_conflict, conflicts, warnings = engine.evaluate_conflicts(
        proposed_direction="BUY",
        concept_results=concepts,
        market_data={"is_stale": True},
        is_stale_data=True,
    )
    assert has_conflict is True
    assert any("STALE_MARKET_DATA" in c for c in conflicts)


def test_conflict_engine_high_impact_news_blocks():
    engine = DecisionConflictEngine()
    concepts = {
        "smc_ict": make_dummy_concept("smc_ict", "SMC/ICT", ConceptDirection.BULLISH),
        "orderflow": make_dummy_concept("orderflow", "Orderflow", ConceptDirection.BULLISH),
    }

    has_conflict, conflicts, warnings = engine.evaluate_conflicts(
        proposed_direction="BUY",
        concept_results=concepts,
        market_data={},
        news_risk_level="HIGH_RISK",
    )
    assert has_conflict is True
    assert any("FUNDAMENTAL_NEWS_CONFLICT" in c for c in conflicts)


def test_conflict_engine_aligned_concepts_no_conflict():
    engine = DecisionConflictEngine()
    concepts = {
        "smc_ict": make_dummy_concept("smc_ict", "SMC/ICT", ConceptDirection.BULLISH),
        "orderflow": make_dummy_concept("orderflow", "Orderflow", ConceptDirection.BULLISH),
        "volume_profile": make_dummy_concept("volume_profile", "Volume Profile", ConceptDirection.BULLISH),
    }

    has_conflict, conflicts, warnings = engine.evaluate_conflicts(
        proposed_direction="BUY",
        concept_results=concepts,
        market_data={},
        news_risk_level="NO_RISK",
    )
    assert has_conflict is False
    assert len(conflicts) == 0
