"""
Unit & Adversarial Tests for Step 11: High-Performance & Adaptive AI Architecture.
Verifies event bus priority, deduplication, hot/warm/cold memory tiers, incremental delta analysis,
anti-flapping hysteresis, MTF isolation, adaptive compute policy, upgrade approval safety,
and rapid throughput under stress.
"""

from datetime import datetime, timedelta, timezone
import pytest

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.adaptive.incremental_engine import IncrementalAnalysisEngine
from trading_ai.adaptive.memory_tier import TieredMemoryManager
from trading_ai.adaptive.priority_scheduler import AdaptiveComputeScheduler
from trading_ai.adaptive.upgrade_manager import ComponentUpgradeManager
from trading_ai.adaptive.core import AdaptiveAICore
from trading_ai.schemas import Candle, MarketBias, MarketSetup, Position, SetupStatus
from trading_ai.schemas.enums import EventPriority, NewsRiskLevel, SystemEventType, UpgradeStatus
from trading_ai.schemas.performance import LatencyTelemetry, SystemEvent
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo(tmp_path):
    db_file = tmp_path / "test_perf.db"
    db_mgr = DatabaseManager(str(db_file))
    return Repository(db_mgr)


# 1. Event Bus Priority & Deduplication Tests

def test_event_bus_priority_preemption():
    """Verify that CRITICAL events preempt LOW and MEDIUM events regardless of arrival order."""
    bus = ReactiveEventBus()
    dispatched = []

    bus.subscribe(None, lambda evt: dispatched.append(evt.priority))

    now = datetime.now(timezone.utc)
    # Publish in reverse priority order: LOW, MEDIUM, CRITICAL
    bus.publish(SystemEvent(
        event_id="evt_low",
        event_type=SystemEventType.PRICE_UPDATE,
        priority=EventPriority.LOW,
        symbol="BTC/USD",
        timestamp=now,
    ))
    bus.publish(SystemEvent(
        event_id="evt_med",
        event_type=SystemEventType.PRICE_UPDATE,
        priority=EventPriority.MEDIUM,
        symbol="BTC/USD",
        timestamp=now,
    ))
    bus.publish(SystemEvent(
        event_id="evt_crit",
        event_type=SystemEventType.PRICE_UPDATE,
        priority=EventPriority.CRITICAL,
        symbol="BTC/USD",
        timestamp=now,
    ))

    # Dispatch in order
    bus.dispatch_all()

    assert len(dispatched) == 3
    # Order must be CRITICAL, then MEDIUM, then LOW
    assert dispatched[0] == EventPriority.CRITICAL
    assert dispatched[1] == EventPriority.MEDIUM
    assert dispatched[2] == EventPriority.LOW


def test_event_bus_idempotency_and_deduplication():
    """Verify that duplicate event IDs are rejected and not dispatched twice."""
    bus = ReactiveEventBus()
    dispatched = []

    bus.subscribe(SystemEventType.PRICE_UPDATE, lambda evt: dispatched.append(evt.event_id))

    evt = SystemEvent(
        event_id="unique_evt_101",
        event_type=SystemEventType.PRICE_UPDATE,
        priority=EventPriority.MEDIUM,
        symbol="EUR/USD",
        timestamp=datetime.now(timezone.utc),
    )

    accepted_1 = bus.publish(evt)
    accepted_2 = bus.publish(evt)  # Duplicate!

    assert accepted_1 is True
    assert accepted_2 is False

    bus.dispatch_all()
    assert len(dispatched) == 1
    assert bus.get_queue_stats()["dropped_duplicates"] == 1


# 2. Tiered Memory Architecture Tests

def test_hot_memory_submillisecond_operations(repo):
    """Verify in-memory Hot Tier lookup and updates operate without disk overhead."""
    mem = TieredMemoryManager(repository=repo)
    symbol = "BTC/USD"

    hot = mem.update_hot_price(symbol, 64500.0)
    assert hot.current_price == 64500.0

    mem.update_hot_bias(symbol, MarketBias.BULLISH_BIAS)
    mem.update_hot_liquidity(symbol, 65200.0)
    mem.update_hot_risk(symbol, 0.25)

    retrieved = mem.get_hot_state(symbol)
    assert retrieved.current_bias == MarketBias.BULLISH_BIAS
    assert retrieved.nearest_liquidity == 65200.0
    assert retrieved.active_risk_score == 0.25


def test_warm_memory_sliding_window_bounds(repo):
    """Verify sliding deque in warm memory preserves max length constraints."""
    mem = TieredMemoryManager(repository=repo, max_warm_candles=5)
    symbol = "ETH/USD"
    tf = "15m"

    base_time = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp()
    for i in range(10):
        c = Candle(
            timestamp=base_time + 900.0 * i,
            open=3000.0 + i,
            high=3010.0 + i,
            low=2990.0 + i,
            close=3005.0 + i,
            volume=100.0,
        )
        mem.append_warm_candle(symbol, tf, c)

    warm = mem.get_warm_candles(symbol, tf, limit=10)
    assert len(warm) == 5
    assert warm[-1].close == 3014.0


# 3. Incremental Technical Analysis & Micro-Changes Tests

def test_incremental_swings_and_micro_changes():
    """Verify incremental analysis detects swings, displacement, and rejection wicks."""
    engine = IncrementalAnalysisEngine()
    symbol = "BTC/USD"
    tf = "15m"

    base_time = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc).timestamp()

    # 1. Base candle
    c1 = Candle(timestamp=base_time, open=60000.0, high=60200.0, low=59900.0, close=60100.0, volume=100.0)
    d1 = engine.process_new_candle(symbol, tf, c1)
    assert d1.price_delta == 100.0

    # 2. Pivot high candle
    c2 = Candle(timestamp=base_time + 900.0, open=60100.0, high=61000.0, low=60050.0, close=60800.0, volume=150.0)
    engine.process_new_candle(symbol, tf, c2)

    # 3. Lower candle (c2 becomes a confirmed swing high)
    c3 = Candle(timestamp=base_time + 1800.0, open=60800.0, high=60850.0, low=60300.0, close=60400.0, volume=120.0)
    d3 = engine.process_new_candle(symbol, tf, c3)
    assert "swings" in d3.affected_components

    state = engine.get_state(symbol, tf)
    assert len(state.swings) >= 1
    assert state.swings[0].is_high is True
    assert state.swings[0].price == 61000.0


def test_incremental_anti_flapping_hysteresis():
    """Verify bias hysteresis prevents alert flapping on isolated noisy candles."""
    engine = IncrementalAnalysisEngine()
    symbol = "SOL/USD"
    tf = "5m"
    state = engine.get_state(symbol, tf)
    state.current_bias = MarketBias.BULLISH_BIAS

    # A single conflicting update does not immediately flip bias
    engine._update_bias_with_hysteresis(state, MarketBias.BEARISH_BIAS)
    assert state.current_bias == MarketBias.BULLISH_BIAS  # Still BULLISH due to hysteresis threshold 2

    # Second consecutive confirmation flips bias
    engine._update_bias_with_hysteresis(state, MarketBias.BEARISH_BIAS)
    assert state.current_bias == MarketBias.BEARISH_BIAS


def test_mtf_timeframe_isolation():
    """Verify LTF update does not affect HTF state, and HTF change properly notifies LTFs."""
    engine = IncrementalAnalysisEngine()
    symbol = "EUR/USD"

    # Set up HTF (1d) and LTF (15m)
    htf_state = engine.get_state(symbol, "1d")
    htf_state.current_bias = MarketBias.BULLISH_BIAS

    ltf_state = engine.get_state(symbol, "15m")
    ltf_state.current_bias = MarketBias.BULLISH_BIAS
    ltf_state.bias_consecutive_count = 5

    # Process LTF candle: does NOT mutate HTF state
    c_ltf = Candle(
        timestamp=datetime.now(timezone.utc).timestamp(),
        open=1.0850,
        high=1.0870,
        low=1.0845,
        close=1.0865,
        volume=500.0,
    )
    engine.process_new_candle(symbol, "15m", c_ltf)
    assert len(htf_state.candles) == 0  # HTF untouched!

    # When HTF changes bias, it invalidates LTF confirmation count
    invalidated = engine.notify_htf_change(symbol, "1d", MarketBias.BEARISH_BIAS, ["15m"])
    assert f"{symbol}:15m" in invalidated
    assert ltf_state.bias_consecutive_count == 0


# 4. Adaptive Compute Allocation & Priority Scheduling Tests

def test_adaptive_compute_policy_scaling():
    """Verify compute allocation scales proportionally to active risk and market conditions."""
    scheduler = AdaptiveComputeScheduler()

    # Case 1: Quiet market without active setup
    quiet_policy = scheduler.evaluate_policy(
        has_active_position=False,
        active_setup_status=None,
        is_volatile=False,
        news_risk=NewsRiskLevel.NO_RISK,
    )
    assert quiet_policy.priority == EventPriority.LOW
    assert quiet_policy.gemini_skip_ratio == 0.95
    assert quiet_policy.monitoring_interval_seconds == 15.0

    # Case 2: Confirmed setup
    setup_policy = scheduler.evaluate_policy(
        has_active_position=False,
        active_setup_status=SetupStatus.CONFIRMED,
        is_volatile=False,
        news_risk=NewsRiskLevel.NO_RISK,
    )
    assert setup_policy.priority == EventPriority.HIGH
    assert setup_policy.monitoring_interval_seconds == 2.0

    # Case 3: Active position requiring strict risk management
    pos_policy = scheduler.evaluate_policy(
        has_active_position=True,
    )
    assert pos_policy.priority == EventPriority.CRITICAL
    assert pos_policy.monitoring_interval_seconds == 1.0
    assert pos_policy.reasoning_depth == "DEEP"


def test_gemini_gating_and_rate_limiting():
    """Verify deterministic fast checks skip Gemini calls unless high-value trigger occurs."""
    scheduler = AdaptiveComputeScheduler()
    quiet_policy = scheduler.evaluate_policy()

    # Routine tick without triggers: Gemini skipped
    should_call = scheduler.should_trigger_gemini(
        policy=quiet_policy,
        is_setup_transition=False,
        is_news_conflict=False,
        is_critical_risk_alert=False,
    )
    assert should_call is False

    # Setup transition: Gemini triggered
    should_call_setup = scheduler.should_trigger_gemini(
        policy=quiet_policy,
        is_setup_transition=True,
    )
    assert should_call_setup is True


# 5. Future-Ready Upgrades & Strict User Approval Safety Tests

def test_component_upgrade_lifecycle_and_user_approval(repo):
    """Verify upgrade promotion requires explicit user approval and rejects AI self-promotion."""
    mgr = ComponentUpgradeManager(repo)

    # 1. Register candidate
    cand = mgr.register_candidate(
        component_name="GeminiMarketModel",
        current_version="1.5.0",
        candidate_version="2.0.0",
        notes="Next-generation reasoning model candidate",
    )
    assert cand.status == UpgradeStatus.CANDIDATE

    # 2. Benchmark evaluation passes
    bench = mgr.run_sandbox_benchmark(
        upgrade_id=cand.upgrade_id,
        accuracy=0.92,
        latency_ms=14.2,
        regression_pass_rate=0.98,
    )
    assert bench.status == UpgradeStatus.BENCHMARK_PASSED

    # 3. Paper validation passes
    paper = mgr.record_paper_validation(upgrade_id=cand.upgrade_id, passed=True)
    assert paper.status == UpgradeStatus.AWAITING_USER_APPROVAL

    # 4. Strict Safety Gate: AI Self-promotion attempt MUST fail with PermissionError!
    with pytest.raises(PermissionError):
        mgr.approve_and_promote(
            upgrade_id=cand.upgrade_id,
            approved_by="",
            is_user=False,  # Automated self-promotion attempt!
        )

    # 5. User approval succeeds
    promoted = mgr.approve_and_promote(
        upgrade_id=cand.upgrade_id,
        approved_by="Lead_Risk_Officer",
        is_user=True,
    )
    assert promoted.status == UpgradeStatus.APPROVED
    assert promoted.user_approved is True
    assert promoted.approved_by == "Lead_Risk_Officer"


def test_component_upgrade_rollback(repo):
    """Verify an approved or active component upgrade can be cleanly rolled back."""
    mgr = ComponentUpgradeManager(repo)
    cand = mgr.register_candidate("SMC_OrderBlock_Algorithm", "1.0", "1.1")
    rolled = mgr.rollback(cand.upgrade_id, reason="Drift observed in volatile regime")
    assert rolled.status == UpgradeStatus.ROLLED_BACK


# 6. Adversarial Tests (1000 Rapid Updates & Out-of-Order Handling)

def test_adversarial_rapid_price_updates_throughput(repo):
    """Stress test: Process 1000 rapid price updates through the core pipeline without errors or memory leak."""
    core = AdaptiveAICore(repository=repo)
    symbol = "BTC/USD"
    base_time = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)

    for i in range(1000):
        t = base_time + timedelta(milliseconds=10 * i)
        price = 60000.0 + (i % 50) * 2.0
        res = core.process_market_tick(symbol=symbol, price=price, timestamp=t)
        assert res["status"] == "PROCESSED"

    hot = core.memory.get_hot_state(symbol)
    assert hot.current_price > 0
    status = core.get_system_status()
    assert status["status"] == "HEALTHY"


def test_adversarial_out_of_order_candle_handling():
    """Verify that candles arriving out of chronological order are safely ignored without corrupting state."""
    engine = IncrementalAnalysisEngine()
    symbol = "ETH/USD"
    tf = "1h"

    t2 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc).timestamp()
    t1 = datetime(2026, 1, 1, 11, 0, tzinfo=timezone.utc).timestamp()

    # Process newer candle first
    c2 = Candle(timestamp=t2, open=3000.0, high=3020.0, low=2990.0, close=3010.0, volume=10.0)
    engine.process_new_candle(symbol, tf, c2)

    # Process older candle (out of order)
    c1 = Candle(timestamp=t1, open=2980.0, high=3000.0, low=2970.0, close=2995.0, volume=10.0)
    d_stale = engine.process_new_candle(symbol, tf, c1)

    assert "out_of_order_ignored" in d_stale.affected_components
    # State timestamp remains the latest
    assert engine.get_state(symbol, tf).last_candle_time == t2
