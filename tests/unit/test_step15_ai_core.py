"""Unit tests for Step 15 AI Core Intelligence, Large Video, and Live Market Decision Engine."""

import pytest
from trading_ai.ai_core.epistemic import EpistemicSeparationEngine, EpistemicState
from trading_ai.ai_core.iterative_comprehension import IterativeComprehensionLoop
from trading_ai.ai_core.large_video_engine import LargeVideoStreamingProcessor
from trading_ai.ai_core.live_decision_engine import LiveMarketDecisionEngine
from trading_ai.ai_core.multilingual import MultilingualNormalizer
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier


@pytest.fixture
def repo():
    return Repository(DatabaseManager(":memory:"))


def test_multilingual_comprehension_and_evidence_preservation():
    """Verify omni-language inputs are parsed accurately with evidence preserved."""
    # Hinglish input
    hinglish_text = "Humein liquidity sweep ke baad displacement ka wait karna hai."
    res = MultilingualNormalizer.comprehend(hinglish_text, timestamp=45.5)

    assert res.detected_language == "Hinglish"
    assert res.original_text == hinglish_text
    assert res.timestamp == 45.5
    assert "Displacement" in res.extracted_concepts
    assert "LiquiditySweep" in res.extracted_concepts
    assert "Wait for displacement" in res.normalized_meaning
    # Verify Hindi explanation contains simple Hindi words
    assert "displacement" in res.hindi_explanation.lower()
    assert "इंतज़ार" in res.hindi_explanation or "प्रतीक्षा" in res.hindi_explanation

    # English input
    eng_res = MultilingualNormalizer.comprehend("FVG formed after break of structure.")
    assert eng_res.detected_language == "English"
    assert "FVG" in eng_res.extracted_concepts


def test_four_layer_epistemic_separation():
    """Verify separation of Learned, Observed, Inferred, and Decided layers."""
    learned = [
        {"rule_id": "r1", "title": "Sweep & Displacement", "conditions": ["Sweep", "Displacement"]}
    ]
    observations = [
        {"type": "PRICE", "value": 1.0850},
        {"type": "SWEEP", "detected": True},
        {"type": "DISPLACEMENT", "detected": False},
    ]
    inferences = [
        {"concept": "Sweep & Displacement", "status": "pending_displacement", "confidence": 0.65}
    ]

    state = EpistemicSeparationEngine.evaluate(
        learned_rules=learned,
        market_observations=observations,
        inferences=inferences,
        news_risk="NO_RISK",
    )

    assert len(state.what_i_learned) == 1
    assert len(state.what_i_observe) == 3
    assert len(state.what_i_infer) == 1
    assert state.overall_epistemic_state == EpistemicState.PARTIALLY_UNDERSTOOD
    assert state.what_i_decide["action"] == "WAIT"
    assert "अधूरा" in state.hindi_summary or "पुष्टि" in state.hindi_summary


def test_epistemic_insufficient_knowledge_and_evidence():
    """Verify system does NOT invent knowledge or hallucinate trades without backing."""
    # No learned rules -> INSUFFICIENT_KNOWLEDGE
    state_no_knowledge = EpistemicSeparationEngine.evaluate(
        learned_rules=[],
        market_observations=[{"type": "PRICE", "value": 1.0850}],
    )
    assert state_no_knowledge.what_i_decide["action"] == "NO_TRADE"
    assert "INSUFFICIENT_KNOWLEDGE" in state_no_knowledge.what_i_decide["reason"]

    # No observations -> INSUFFICIENT_EVIDENCE
    state_no_obs = EpistemicSeparationEngine.evaluate(
        learned_rules=[{"rule_id": "r1"}],
        market_observations=[],
    )
    assert state_no_obs.what_i_decide["action"] == "NO_TRADE"
    assert "INSUFFICIENT_EVIDENCE" in state_no_obs.what_i_decide["reason"]


def test_iterative_comprehension_loop(repo):
    """Verify iterative review bounds: max iterations, confidence threshold, honest ambiguity."""
    loop = IterativeComprehensionLoop(repository=repo, max_iterations=3, min_confidence_threshold=0.85)

    # Highly clear concept
    clear_res = loop.review_concept(
        job_id="job_test_1",
        concept_title="Clear ICT Liquidity Sweep",
        initial_confidence=0.90,
        audio_evidence="Clear audio transcript",
        visual_evidence="Valid chart frame",
        transcript_context="Context matches",
    )
    assert clear_res.is_resolved is True
    assert clear_res.epistemic_state == EpistemicState.VALIDATED
    assert clear_res.iterations_run == 1

    # Ambiguous concept missing audio & visual evidence -> Honest AMBIGUOUS or NEEDS_REVIEW
    ambig_res = loop.review_concept(
        job_id="job_test_2",
        concept_title="Muffled ambiguous whisper",
        initial_confidence=0.40,
        audio_evidence="",
        visual_evidence="",
        transcript_context="",
    )
    assert ambig_res.is_resolved is False
    assert ambig_res.epistemic_state in (EpistemicState.AMBIGUOUS, EpistemicState.NEEDS_REVIEW)
    assert ambig_res.iterations_run <= 3


def test_large_video_streaming_processor_and_serial_barrier(repo):
    """Verify chunk streaming, bounded RAM, and strict serial commit barrier behavior."""
    barrier = CommitBarrier(repo)
    engine = LargeVideoStreamingProcessor(repository=repo, commit_barrier=barrier)

    job_id = "vjob_large_50gb_test"
    summary = engine.initialize_job(
        job_id=job_id,
        file_path="data/mock_50gb_video.mp4",
        file_size_bytes=50 * (1024**3),
        duration_seconds=120.0,
        chunk_duration=30.0,
    )
    assert summary.total_chunks == 4
    assert summary.file_size_gb == 50.0

    # Chunk 1 succeeds -> COMMITTED
    c1 = engine.process_chunk(job_id, chunk_index=1, start_time=0.0, end_time=30.0)
    assert c1.status == "COMMITTED"

    # Chunk 2 FAILS -> Barrier blocks
    c2 = engine.process_chunk(job_id, chunk_index=2, start_time=30.0, end_time=60.0, simulate_failure=True)
    assert c2.status == "FAILED"

    # Chunk 3 succeeds -> MUST BE BUFFERED behind failed Chunk 2
    c3 = engine.process_chunk(job_id, chunk_index=3, start_time=60.0, end_time=90.0)
    assert c3.status == "BUFFERED"

    # Chunk 4 succeeds -> MUST ALSO BE BUFFERED
    c4 = engine.process_chunk(job_id, chunk_index=4, start_time=90.0, end_time=120.0)
    assert c4.status == "BUFFERED"

    # Recover Chunk 2 -> Unblocks serial barrier; chunks 2, 3, 4 now commit!
    recovered = engine.recover_chunk(job_id, serial_number=c2.serial_number)
    assert c2.serial_number in recovered
    assert c3.serial_number in recovered
    assert c4.serial_number in recovered


def test_live_market_decision_engine_and_forecasting(repo):
    """Verify live concept mapping, conditional forecasting, and why-not-trade checks."""
    engine = LiveMarketDecisionEngine(repository=repo)

    # 1. Partial match: Sweep detected, displacement pending -> WAIT
    res_partial = engine.evaluate_live_market(
        symbol="EURUSD",
        timeframe="15m",
        current_price=1.0850,
        direction="BUY",
        observed_sweep=True,
        observed_displacement=False,
        observed_fvg=False,
    )
    assert res_partial.concept_match_status == "PARTIAL_MATCH"
    assert res_partial.final_decision == "WAIT"
    assert len(res_partial.missing_confirmations) > 0
    assert len(res_partial.scenarios) == 2
    assert "Scenario A:" in res_partial.scenarios[0].scenario_name
    assert "Scenario B:" in res_partial.scenarios[1].scenario_name
    assert "प्रतीक्षा" in res_partial.hindi_speech or "इंतज़ार" in res_partial.hindi_speech

    # 2. High-impact news risk -> NO_TRADE via Why-Not-Trade
    res_news = engine.evaluate_live_market(
        symbol="EURUSD",
        timeframe="15m",
        current_price=1.0850,
        direction="BUY",
        observed_sweep=True,
        observed_displacement=True,
        observed_fvg=True,
        news_risk_level="HIGH_RISK",
    )
    assert res_news.final_decision == "NO_TRADE"
    assert res_news.why_not_trade_result["has_blocking"] is True

    # 3. Fully confirmed setup without blocking news -> TRADE_CANDIDATE
    res_confirmed = engine.evaluate_live_market(
        symbol="EURUSD",
        timeframe="15m",
        current_price=1.0850,
        direction="BUY",
        observed_sweep=True,
        observed_displacement=True,
        observed_fvg=True,
        news_risk_level="NO_RISK",
    )
    assert res_confirmed.concept_match_status == "FULL_MATCH"
    assert res_confirmed.final_decision == "TRADE_CANDIDATE"
    assert len(res_confirmed.missing_confirmations) == 0
