"""
Unit tests for Step 4: Live Market Intelligence, Multi-Timeframe Engine,
Lecture-Constrained Setup Validation, Clean Chart Markings, Screen Privacy,
and Watchlist Monitoring.
"""

import time
import pytest
from datetime import datetime, timezone

from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.lecture_validator import LectureValidator
from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
from trading_ai.core.mtf_engine import MultiTimeframeEngine
from trading_ai.core.screen_access import ScreenAccessManager
from trading_ai.pipeline.watchlist_monitor import WatchlistMonitor
from trading_ai.schemas.enums import (
    KnowledgeType,
    KnowledgeValidationStatus,
    LevelType,
    MarketBias,
    MarketDataStatus,
    MarketDirection,
    MarketStructureType,
    MarkingType,
    MarkingValidity,
    ScreenAccessStatus,
    SetupStatus,
    TimeframeHierarchy,
    WatchlistMonitorStatus,
)
from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.schemas.market import (
    Candle,
    ChartMarking,
    MarketLevel,
    MarketSetup,
    MarketSnapshot,
    ScreenPermission,
    TimeframeAnalysis,
    UnifiedMarketAnalysis,
    WatchlistMonitorState,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier


@pytest.fixture
def test_repo():
    """In-memory SQLite repository with initialized schema."""
    db = DatabaseManager(":memory:")
    return Repository(db)


@pytest.fixture
def sim_provider():
    """Simulated market data provider with reproducible seed."""
    return SimulatedMarketDataProvider(seed=42)


# =====================================================================
# 1. Market Data Provider & Staleness Tests
# =====================================================================

def test_market_data_provider_snapshot_generation(sim_provider):
    snapshot = sim_provider.get_snapshot("BTC/USD", "15m", limit=50)
    assert snapshot.symbol == "BTC/USD"
    assert snapshot.timeframe == "15m"
    assert len(snapshot.candles) == 50
    assert snapshot.latest_price > 0
    assert snapshot.status == MarketDataStatus.DATA_AVAILABLE
    assert not sim_provider.is_stale(snapshot, max_age_seconds=300)


def test_market_data_provider_forced_status_and_staleness(sim_provider):
    sim_provider.force_status("ETH/USD", MarketDataStatus.DATA_STALE)
    assert sim_provider.check_status("ETH/USD") == MarketDataStatus.DATA_STALE

    snapshot = sim_provider.get_snapshot("ETH/USD", "15m", limit=10)
    assert snapshot.status == MarketDataStatus.DATA_STALE
    assert sim_provider.is_stale(snapshot)

    # Missing status
    sim_provider.force_status("SOL/USD", MarketDataStatus.DATA_MISSING)
    snap_missing = sim_provider.get_snapshot("SOL/USD", "15m", limit=10)
    assert snap_missing.status == MarketDataStatus.DATA_MISSING
    assert sim_provider.is_stale(snap_missing)

    sim_provider.clear_forced_status("ETH/USD")
    assert sim_provider.check_status("ETH/USD") == MarketDataStatus.DATA_AVAILABLE


# =====================================================================
# 2. Multi-Timeframe Engine & Conflict Detection
# =====================================================================

def test_mtf_engine_single_timeframe_analysis(sim_provider):
    engine = MultiTimeframeEngine(sim_provider)
    snapshot = sim_provider.get_snapshot("BTC/USD", "15m", limit=60)
    tf_analysis = engine.analyze_single_timeframe(snapshot, role=TimeframeHierarchy.LTF)

    assert tf_analysis.symbol == "BTC/USD"
    assert tf_analysis.timeframe == "15m"
    assert tf_analysis.direction in (MarketDirection.BULLISH, MarketDirection.BEARISH, MarketDirection.RANGING)
    assert len(tf_analysis.key_levels) > 0
    # Levels include Session High/Low and Swings
    types = [lvl.level_type for lvl in tf_analysis.key_levels]
    assert LevelType.SESSION_HIGH in types
    assert LevelType.SESSION_LOW in types


def test_mtf_engine_unified_analysis_and_conflicts(sim_provider):
    engine = MultiTimeframeEngine(sim_provider)
    analysis = engine.analyze_symbol("BTC/USD")

    assert analysis.symbol == "BTC/USD"
    assert analysis.htf_analysis is not None
    assert analysis.mtf_analysis is not None
    assert analysis.ltf_analysis is not None
    assert analysis.overall_bias in (
        MarketBias.BULLISH_BIAS,
        MarketBias.BEARISH_BIAS,
        MarketBias.NEUTRAL_BIAS,
        MarketBias.MULTI_TIMEFRAME_CONFLICT,
    )


def test_mtf_engine_timeframe_conflict_detection():
    """Verify that opposing HTF and LTF directions result in MULTI_TIMEFRAME_CONFLICT."""
    mock_provider = SimulatedMarketDataProvider(seed=123)
    engine = MultiTimeframeEngine(mock_provider)

    htf = TimeframeAnalysis(
        symbol="EUR/USD",
        timeframe="1d",
        hierarchy_role=TimeframeHierarchy.HTF,
        direction=MarketDirection.BULLISH,
        structure=MarketStructureType.BOS_BULLISH,
        bias=MarketBias.BULLISH_BIAS,
    )
    ltf = TimeframeAnalysis(
        symbol="EUR/USD",
        timeframe="15m",
        hierarchy_role=TimeframeHierarchy.LTF,
        direction=MarketDirection.BEARISH,
        structure=MarketStructureType.BOS_BEARISH,
        bias=MarketBias.BEARISH_BIAS,
    )
    mtf = TimeframeAnalysis(
        symbol="EUR/USD",
        timeframe="1h",
        hierarchy_role=TimeframeHierarchy.MTF,
        direction=MarketDirection.BEARISH,
        structure=MarketStructureType.RANGE,
        bias=MarketBias.BEARISH_BIAS,
    )

    unified = engine._synthesize_multi_timeframe("EUR/USD", htf, mtf, ltf)
    assert unified.overall_bias == MarketBias.MULTI_TIMEFRAME_CONFLICT
    assert len(unified.conflict_notes) > 0
    assert "conflict" in unified.conflict_notes[0].lower()


# =====================================================================
# 3. Lecture-Constrained Setup Validation (DETECTION != VALIDATION)
# =====================================================================

def test_lecture_validator_out_of_scope_when_no_lecture_rule(test_repo, sim_provider):
    validator = LectureValidator(test_repo)
    engine = MultiTimeframeEngine(sim_provider)
    analysis = engine.analyze_symbol("BTC/USD")

    setup = validator.validate_detected_pattern(
        symbol="BTC/USD",
        timeframe="15m",
        pattern_name="UNKNOWN_HEAD_AND_SHOULDERS",
        direction=MarketDirection.BEARISH,
        market_analysis=analysis,
        pattern_context={"displacement": True},
    )

    assert setup.status == SetupStatus.OUT_OF_SCOPE
    assert setup.confidence == 0.0
    assert setup.applicable_rule_id is None
    assert "No lecture rule found" in setup.contradicting_evidence[0]


def test_lecture_validator_valid_and_invalidated(test_repo, sim_provider):
    # Seed a persistent lecture rule in repository
    rule_item = KnowledgeItem(
        job_id="lecture_001",
        item_type=KnowledgeType.RULE,
        title="ICT Fair Value Gap (FVG) Entry Rule",
        statement="Enter on 50% retrace of 15m FVG aligned with higher timeframe direction.",
        conditions=["Displacement present", "No opposing swing broken"],
        exceptions=["High impact news in next 15 minutes"],
        tags=["FVG", "fair value gap", "entry"],
        validation_status=KnowledgeValidationStatus.VERIFIED,
    )
    test_repo.save_knowledge([rule_item])

    validator = LectureValidator(test_repo)

    # Construct bullish aligned market analysis
    htf = TimeframeAnalysis(
        symbol="BTC/USD",
        timeframe="1d",
        hierarchy_role=TimeframeHierarchy.HTF,
        direction=MarketDirection.BULLISH,
        structure=MarketStructureType.BOS_BULLISH,
        bias=MarketBias.BULLISH_BIAS,
    )
    analysis = UnifiedMarketAnalysis(
        symbol="BTC/USD",
        htf_analysis=htf,
        overall_bias=MarketBias.BULLISH_BIAS,
    )

    # 1. Valid setup test
    valid_setup = validator.validate_detected_pattern(
        symbol="BTC/USD",
        timeframe="15m",
        pattern_name="FVG",
        direction=MarketDirection.BULLISH,
        market_analysis=analysis,
        pattern_context={"displacement": True, "is_invalidated": False},
    )
    assert valid_setup.status == SetupStatus.VALID
    assert valid_setup.applicable_rule_id == rule_item.item_id
    assert valid_setup.confidence > 0.85
    assert len(valid_setup.conditions_satisfied) > 0

    # 2. Invalidation triggered test
    invalidated_setup = validator.validate_detected_pattern(
        symbol="BTC/USD",
        timeframe="15m",
        pattern_name="FVG",
        direction=MarketDirection.BULLISH,
        market_analysis=analysis,
        pattern_context={"displacement": True, "is_invalidated": True, "invalidation_reason": "Price closed below low of candle 1"},
    )
    assert invalidated_setup.status == SetupStatus.INVALIDATED
    assert invalidated_setup.confidence == 0.0
    assert any("Invalidation triggered" in ev for ev in invalidated_setup.contradicting_evidence)

    # 3. Uncertain setup test (lacks displacement)
    uncertain_setup = validator.validate_detected_pattern(
        symbol="BTC/USD",
        timeframe="15m",
        pattern_name="FVG",
        direction=MarketDirection.BULLISH,
        market_analysis=analysis,
        pattern_context={"displacement": False, "is_invalidated": False},
    )
    assert uncertain_setup.status == SetupStatus.UNCERTAIN
    assert any("displacement" in ev.lower() for ev in uncertain_setup.contradicting_evidence)


# =====================================================================
# 4. ChartMarker Clean-Chart & Clutter Suppression
# =====================================================================

def test_chart_marker_clean_principles(sim_provider):
    marker = ChartMarker(max_markings=5, price_duplicate_tolerance=0.005)
    snapshot = sim_provider.get_snapshot("EUR/USD", "15m", limit=30)

    # Create dummy setups: 1 VALID, 1 UNCERTAIN (should be suppressed from chart)
    setups = [
        MarketSetup(
            symbol="EUR/USD",
            timeframe="15m",
            setup_type="FVG",
            direction=MarketDirection.BULLISH,
            status=SetupStatus.VALID,
            confidence=0.9,
        ),
        MarketSetup(
            symbol="EUR/USD",
            timeframe="15m",
            setup_type="BOS",
            direction=MarketDirection.BULLISH,
            status=SetupStatus.UNCERTAIN,
            confidence=0.4,
        ),
    ]

    levels = [
        MarketLevel(symbol="EUR/USD", timeframe="15m", price=1.0850, level_type=LevelType.SWING_HIGH),
        # Near-duplicate price level within tolerance
        MarketLevel(symbol="EUR/USD", timeframe="15m", price=1.0851, level_type=LevelType.SWING_HIGH),
        MarketLevel(symbol="EUR/USD", timeframe="15m", price=1.0800, level_type=LevelType.SWING_LOW),
    ]

    markings = marker.generate_markings(snapshot, setups, levels)

    # Clean chart rule 1: UNCERTAIN setup must not appear on chart
    for m in markings:
        if m.evidence_json and "setup_id" in m.evidence_json:
            assert m.validity == MarkingValidity.VALID

    # Clean chart rule 2: Clutter limit enforced
    assert len(markings) <= 5

    # Clean chart rule 3: Near-duplicate level suppressed
    high_levels = [m for m in markings if m.marking_type == MarkingType.IMPORTANT_LEVEL and abs(m.coordinates_json.get("price", 0) - 1.0850) < 0.002]
    assert len(high_levels) == 1


# =====================================================================
# 5. Screen Access Privacy Boundary
# =====================================================================

def test_screen_access_lifecycle_and_enforcement(test_repo):
    screen_mgr = ScreenAccessManager(test_repo)

    # Initial state: access disabled
    assert screen_mgr.check_access("TradingView") == ScreenAccessStatus.SCREEN_ACCESS_DISABLED

    # Cannot capture unauthorized frame
    with pytest.raises(PermissionError) as exc_info:
        screen_mgr.capture_authorized_frame("TradingView")
    assert "Explicit user consent required" in str(exc_info.value)

    # 1. Request access
    req = screen_mgr.request_access("TradingView", window_title="BTCUSD Chart - TradingView")
    assert req.status == ScreenAccessStatus.SCREEN_ACCESS_REQUESTED
    assert test_repo.get_screen_permission(req.permission_id) is not None

    # 2. Grant access
    granted = screen_mgr.grant_access(req.permission_id)
    assert granted.status == ScreenAccessStatus.SCREEN_ACCESS_GRANTED
    assert screen_mgr.check_access("TradingView") == ScreenAccessStatus.SCREEN_ACCESS_GRANTED

    # Authorized capture succeeds
    frame = screen_mgr.capture_authorized_frame("TradingView")
    assert frame["status"] == "CAPTURED"
    assert frame["source_type"] == "SCREEN_CAPTURE"

    # 3. Revoke access
    revoked = screen_mgr.revoke_access(req.permission_id)
    assert revoked.status == ScreenAccessStatus.SCREEN_ACCESS_REVOKED
    assert screen_mgr.check_access("TradingView") == ScreenAccessStatus.SCREEN_ACCESS_DISABLED

    # Re-capture now fails
    with pytest.raises(PermissionError):
        screen_mgr.capture_authorized_frame("TradingView")


# =====================================================================
# 6. Watchlist Background Monitor & Incremental Cache
# =====================================================================

def test_watchlist_monitor_processing_and_idempotency(test_repo, sim_provider):
    mtf = MultiTimeframeEngine(sim_provider)
    validator = LectureValidator(test_repo)
    marker = ChartMarker()
    barrier = CommitBarrier(test_repo)

    monitor = WatchlistMonitor(
        repository=test_repo,
        data_provider=sim_provider,
        mtf_engine=mtf,
        lecture_validator=validator,
        chart_marker=marker,
        commit_barrier=barrier,
        watchlist=["BTC/USD", "ETH/USD"],
    )

    # Start monitor
    monitor.start()
    assert monitor.status == WatchlistMonitorStatus.MONITORING

    # Process all symbols
    results = monitor.process_all_watchlist(force=True)
    assert len(results) == 2
    assert results["BTC/USD"] is not None
    assert results["ETH/USD"] is not None

    # Verify states persisted in DB
    btc_state = test_repo.get_watchlist_monitor_state("BTC/USD")
    assert btc_state is not None
    assert btc_state.last_processed_timestamp > 0

    # Incremental optimization: second run with unchanged data skips re-analysis
    first_processed_ts = btc_state.last_processed_timestamp
    cached_result = monitor.process_symbol("BTC/USD", force=False)
    assert cached_result is not None
    # Timestamp remains unchanged (didn't rerun pipeline unnecessarily)
    assert test_repo.get_watchlist_monitor_state("BTC/USD").last_processed_timestamp == first_processed_ts

    # Test pause and resume
    monitor.pause()
    assert monitor.status == WatchlistMonitorStatus.PAUSED
    assert test_repo.get_watchlist_monitor_state("BTC/USD").latest_status == WatchlistMonitorStatus.PAUSED

    monitor.resume()
    assert monitor.status == WatchlistMonitorStatus.MONITORING

    monitor.stop()
    assert monitor.status == WatchlistMonitorStatus.STOPPED


# =====================================================================
# 7. Repository Step 4 Persistence CRUD
# =====================================================================

def test_repository_step4_crud(test_repo):
    # Snapshot
    snap = MarketSnapshot(
        symbol="SOL/USD",
        timeframe="15m",
        latest_price=145.5,
        timestamp=time.time(),
        candles=[Candle(timestamp=time.time(), open=145, high=146, low=144, close=145.5)],
    )
    test_repo.save_market_snapshot(snap)
    loaded_snap = test_repo.get_latest_market_snapshot("SOL/USD", "15m")
    assert loaded_snap is not None
    assert loaded_snap.latest_price == 145.5

    # Level
    lvl = MarketLevel(
        symbol="SOL/USD",
        timeframe="15m",
        price=146.0,
        level_type=LevelType.SESSION_HIGH,
    )
    test_repo.save_market_level(lvl)
    levels = test_repo.get_market_levels("SOL/USD")
    assert len(levels) == 1
    assert levels[0].price == 146.0

    # Marking
    marking = ChartMarking(
        symbol="SOL/USD",
        timeframe="15m",
        marking_type=MarkingType.FVG,
        coordinates_json={"top": 146.0, "bottom": 145.0},
        validity=MarkingValidity.VALID,
    )
    test_repo.save_chart_marking(marking)
    markings = test_repo.get_chart_markings("SOL/USD")
    assert len(markings) == 1
    assert markings[0].marking_type == MarkingType.FVG

    # Update validity
    test_repo.update_chart_marking_validity(marking.marking_id, MarkingValidity.INVALIDATED)
    valid_markings = test_repo.get_chart_markings("SOL/USD", only_valid=True)
    assert len(valid_markings) == 0

    # Setup
    setup = MarketSetup(
        symbol="SOL/USD",
        timeframe="15m",
        setup_type="FVG",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.VALID,
    )
    test_repo.save_market_setup(setup)
    setups = test_repo.get_market_setups("SOL/USD")
    assert len(setups) == 1
    assert setups[0].status == SetupStatus.VALID
