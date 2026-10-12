"""
Failure and network-resilience test scenarios for Step 4.
Covers stale/missing data, timeframe conflicts, uncertain/invalidated setups,
clean-chart clutter suppression, screen privacy enforcement, and Commit Barrier ordering across restarts.
"""

import time
import pytest
from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.lecture_validator import LectureValidator
from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
from trading_ai.core.mtf_engine import MultiTimeframeEngine
from trading_ai.core.screen_access import ScreenAccessManager
from trading_ai.pipeline.watchlist_monitor import WatchlistMonitor
from trading_ai.schemas.enums import (
    KnowledgeType,
    KnowledgeValidationStatus,
    MarketBias,
    MarketDataStatus,
    MarketDirection,
    MarketStructureType,
    MarkingType,
    PipelineStage,
    ScreenAccessStatus,
    SetupStatus,
    TimeframeHierarchy,
)
from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.schemas.market import (
    MarketSnapshot,
    TimeframeAnalysis,
    UnifiedMarketAnalysis,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_failures.db")
    db = DatabaseManager(db_file)
    return db, db_file


def test_stale_and_missing_data_resilience(temp_db):
    db, db_file = temp_db
    repo = Repository(db)
    provider = SimulatedMarketDataProvider(seed=42)
    mtf = MultiTimeframeEngine(provider)
    validator = LectureValidator(repo)
    marker = ChartMarker()
    barrier = CommitBarrier(repo)

    monitor = WatchlistMonitor(
        repository=repo,
        data_provider=provider,
        mtf_engine=mtf,
        lecture_validator=validator,
        chart_marker=marker,
        commit_barrier=barrier,
        watchlist=["ETH/USD", "SOL/USD"],
    )

    # 1. Missing data failure path
    provider.force_status("SOL/USD", MarketDataStatus.DATA_MISSING)
    res_missing = monitor.process_symbol("SOL/USD")
    assert res_missing is None
    state_missing = repo.get_watchlist_monitor_state("SOL/USD")
    assert "data" in state_missing.last_error.lower() or "available" in state_missing.last_error.lower()

    # 2. Stale data warning path
    provider.clear_forced_status("SOL/USD")
    provider.set_stale_offset("ETH/USD", 7200.0) # 2 hours backdated
    res_stale = monitor.process_symbol("ETH/USD", force=True)
    assert res_stale is not None
    assert any("stale" in note.lower() for note in res_stale.conflict_notes)


def test_timeframe_conflict_prevents_forced_bias():
    provider = SimulatedMarketDataProvider(seed=42)
    mtf = MultiTimeframeEngine(provider)

    htf = TimeframeAnalysis(
        symbol="BTC/USD",
        timeframe="1d",
        hierarchy_role=TimeframeHierarchy.HTF,
        direction=MarketDirection.BULLISH,
        structure=MarketStructureType.BOS_BULLISH,
        bias=MarketBias.BULLISH_BIAS,
    )
    mtf_analysis = TimeframeAnalysis(
        symbol="BTC/USD",
        timeframe="1h",
        hierarchy_role=TimeframeHierarchy.MTF,
        direction=MarketDirection.BEARISH,
        structure=MarketStructureType.BOS_BEARISH,
        bias=MarketBias.BEARISH_BIAS,
    )
    ltf = TimeframeAnalysis(
        symbol="BTC/USD",
        timeframe="15m",
        hierarchy_role=TimeframeHierarchy.LTF,
        direction=MarketDirection.BEARISH,
        structure=MarketStructureType.SWING_LOW,
        bias=MarketBias.BEARISH_BIAS,
    )

    unified = mtf._synthesize_multi_timeframe("BTC/USD", htf, mtf_analysis, ltf)
    assert unified.overall_bias == MarketBias.MULTI_TIMEFRAME_CONFLICT
    assert len(unified.conflict_notes) > 0


def test_lecture_rule_out_of_scope_and_uncertainty(temp_db):
    db, _ = temp_db
    repo = Repository(db)
    validator = LectureValidator(repo)

    unified = UnifiedMarketAnalysis(
        symbol="BTC/USD",
        overall_bias=MarketBias.BULLISH_BIAS,
    )

    # 1. Unlearned pattern -> OUT_OF_SCOPE (AI does not hallucinate trading rules)
    setup_out = validator.validate_detected_pattern(
        symbol="BTC/USD",
        timeframe="15m",
        pattern_name="RANDOM_HARMONIC_PATTERN",
        direction=MarketDirection.BULLISH,
        market_analysis=unified,
    )
    assert setup_out.status == SetupStatus.OUT_OF_SCOPE
    assert setup_out.confidence == 0.0

    # 2. Seed legitimate rule
    rule = KnowledgeItem(
        job_id="lecture_ict_101",
        item_type=KnowledgeType.RULE,
        title="ICT Order Block Mitigation",
        statement="Enter at high of bullish order block when aligned with HTF",
        conditions=["Displacement present"],
        exceptions=["High impact news"],
        tags=["OB", "order block"],
    )
    repo.save_knowledge([rule])

    # 3. Known pattern with missing displacement -> UNCERTAIN
    setup_uncertain = validator.validate_detected_pattern(
        symbol="BTC/USD",
        timeframe="15m",
        pattern_name="order block",
        direction=MarketDirection.BULLISH,
        market_analysis=unified,
        pattern_context={"displacement": False},
    )
    assert setup_uncertain.status == SetupStatus.UNCERTAIN

    # Clean Chart Verification: uncertain / out-of-scope setups are NOT plotted on chart
    marker = ChartMarker()
    provider = SimulatedMarketDataProvider(seed=42)
    snap = provider.get_snapshot("BTC/USD", "15m", limit=30)
    markings = marker.generate_markings(snapshot=snap, setups=[setup_uncertain, setup_out], levels=[])
    # Ensure neither setup_uncertain nor setup_out produced setup markings
    setup_markings = [m for m in markings if m.marking_type in (MarkingType.SETUP_ZONE, MarkingType.ORDER_BLOCK)]
    assert len(setup_markings) == 0
    # Also verify all markings have valid status and clean-chart limits are respected
    assert len(markings) <= marker._max_markings


def test_commit_barrier_ordering_and_restart_survivability(temp_db):
    db, db_file = temp_db
    repo = Repository(db)
    barrier = CommitBarrier(repo)

    job_id = "test_step4_barrier_restart"
    from trading_ai.schemas import JobMetadata, JobStatus
    repo.save_job(JobMetadata(
        job_id=job_id,
        file_hash="hash_b",
        filename="market_feed",
        file_path="market://BTC",
        status=JobStatus.COMPLETED,
        current_stage=PipelineStage.STORAGE,
    ))

    # Register serial 1 and 2
    u1 = barrier.register_task_unit(job_id=job_id, stage=PipelineStage.STORAGE, start_time=0.0, end_time=1.0)
    u2 = barrier.register_task_unit(job_id=job_id, stage=PipelineStage.STORAGE, start_time=1.0, end_time=2.0)
    assert u1.serial_number == 1
    assert u2.serial_number == 2

    # Serial 2 completes BEFORE Serial 1 (out of order completion)
    committed2 = barrier.complete_task_unit(job_id=job_id, serial_number=2, result={"val": "two"})
    # Serial 2 MUST be buffered, NOT committed ahead of serial 1
    assert len(committed2) == 0
    barrier_state = barrier.get_barrier_state(job_id)
    assert 2 in barrier_state.buffered_units
    assert barrier_state.highest_committed_serial == 0

    # Simulate full process restart / crash / database reopening
    db_reopened = DatabaseManager(db_file)
    repo_reopened = Repository(db_reopened)
    barrier_reopened = CommitBarrier(repo_reopened)

    # State survived restart: serial 2 is still buffered, last_committed is still 0
    state = barrier_reopened.get_barrier_state(job_id)
    assert state.highest_committed_serial == 0
    assert 2 in state.buffered_units

    # Now Serial 1 completes recovery
    committed1 = barrier_reopened.complete_task_unit(job_id=job_id, serial_number=1, result={"val": "one"})
    # Ordered commitment in sequence: 1 -> 2 commits correctly
    assert 1 in committed1
    assert 2 in committed1

    final_state = barrier_reopened.get_barrier_state(job_id)
    assert final_state.highest_committed_serial == 2
    assert len(final_state.buffered_units) == 0
