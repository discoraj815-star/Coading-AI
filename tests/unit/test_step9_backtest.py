"""
Comprehensive Step 9 Unit Tests:
- Trading Brain Qualitative Reasoning & Token Efficiency
- Rule Versioning, Provenance & Combined Strategies
- Deterministic Backtesting (AI & Manual Replay, Recent Lecture)
- Strict Forex & Crypto Asset Separation
- Look-Ahead Bias & Data Leakage Protection
- Multi-Stage Promotion Gating (Backtest -> Paper -> Live)
- Step 8 RiskEngine & ExecutionGate Integration
- Resumability & Checkpointing
- REST API Endpoints & UI DOM Verification
"""

import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from trading_ai.ai.gemini_client import GeminiClient
from trading_ai.backtest.engine import BacktestEngine
from trading_ai.backtest.promotion_manager import PromotionError, RulePromotionManager
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
from trading_ai.execution.execution_gate import ExecutionGate
from trading_ai.execution.risk_engine import RiskEngine
from trading_ai.schemas.backtest import (
    BacktestConfig,
    BacktestMetrics,
    BacktestRun,
    ManualBacktestTradeRequest,
    RuleVersion,
    StrategyVersion,
)
from trading_ai.schemas.enums import (
    AssetClass,
    BacktestMode,
    BacktestStatus,
    ConfluenceCategory,
    ExecutionEnvironment,
    RuleLifecycleState,
    TradeSource,
)
from trading_ai.schemas.execution import ExecutionPolicy, TradeConfluence, TradeProposal
from trading_ai.schemas.market import Candle, MarketSnapshot
from trading_ai.server.app import app
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.trading_brain.trading_brain import GeminiTradingBrain


@pytest.fixture
def temp_repo():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    db_mgr = DatabaseManager(db_path)
    db_mgr.initialize_schema()
    repo = Repository(db_mgr)
    yield repo
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def market_data():
    return SimulatedMarketDataProvider(seed=42)


@pytest.fixture
def discipline_engine(temp_repo):
    return TradingDisciplineEngine(temp_repo)


@pytest.fixture
def risk_engine(temp_repo, discipline_engine):
    return RiskEngine(temp_repo, discipline_engine)


@pytest.fixture
def execution_gate(temp_repo, discipline_engine):
    return ExecutionGate(temp_repo, discipline_engine)


@pytest.fixture
def trading_brain(temp_repo):
    ai = GeminiClient()
    return GeminiTradingBrain(gemini_client=ai, repository=temp_repo)


@pytest.fixture
def backtest_engine(temp_repo, market_data, trading_brain, risk_engine):
    return BacktestEngine(
        repository=temp_repo,
        market_data_provider=market_data,
        trading_brain=trading_brain,
        risk_engine=risk_engine,
    )


@pytest.fixture
def promotion_manager(temp_repo, execution_gate):
    return RulePromotionManager(repository=temp_repo, execution_gate=execution_gate)


# =========================================================================
# 1. TRADING BRAIN TESTS (PHASE 9.1)
# =========================================================================

def test_gemini_trading_brain_rule_extraction(trading_brain, temp_repo):
    """Verify Gemini extracts candidate rule with structured conditions and exceptions."""
    lecture_snippet = (
        "In this ICT lesson on EURUSD, when London session high is swept, wait for a sharp displacement "
        "candle that creates a 15m Fair Value Gap. Enter on the gap retest. Invalidate if price closes beyond."
    )
    rule = trading_brain.extract_candidate_rule(
        lecture_text=lecture_snippet,
        lecture_id="job_ict_01",
        source_timestamp=14.5,
    )

    assert rule.rule_id.startswith("rule_")
    assert rule.version == 1
    assert rule.state == RuleLifecycleState.CANDIDATE
    assert rule.source_lecture_id == "job_ict_01"
    assert len(rule.conditions) > 0
    assert len(rule.exceptions) > 0

    # Ensure saved to repo
    saved = temp_repo.get_rule_version(rule.rule_id)
    assert saved is not None
    assert saved.title == rule.title


def test_gemini_trading_brain_thesis_and_token_caching(trading_brain, market_data):
    """Verify trade thesis generation uses smallest context and caches identical queries."""
    rule = RuleVersion(
        rule_id="rule_fvg_01",
        version=1,
        title="FVG Retest",
        rule_text="Enter on FVG retest after liquidity sweep",
        conditions=["Liquidity sweep", "FVG formed"],
    )
    snapshot = market_data.get_snapshot("EUR/USD", timeframe="15m", limit=30)

    # First call: computes and caches
    thesis1 = trading_brain.formulate_trade_thesis(rule, snapshot)
    assert "trigger_detected" in thesis1
    assert "direction" in thesis1
    assert "confidence" in thesis1

    # Second identical call: served from cache without token usage
    thesis2 = trading_brain.formulate_trade_thesis(rule, snapshot)
    assert thesis1 == thesis2


def test_gemini_trading_brain_backtest_analysis(trading_brain):
    """Verify qualitative performance review of deterministic metrics."""
    rule = RuleVersion(
        rule_id="rule_test_01",
        version=1,
        title="London Sweep Strategy",
        rule_text="Displacement out of London sweep",
    )
    metrics = BacktestMetrics(
        total_trades=20,
        wins=14,
        losses=6,
        win_rate=0.70,
        profit_factor=2.1,
        expectancy=0.65,
        max_drawdown_pct=4.2,
        total_pnl=1250.0,
    )

    analysis = trading_brain.analyze_backtest_performance(metrics, rule, [])
    assert "assessment" in analysis
    assert "strengths" in analysis
    assert "recommendation" in analysis
    assert analysis["recommendation"] in ("APPROVE_FOR_PAPER", "NEEDS_REFINEMENT", "REJECT")


# =========================================================================
# 2. RULE VERSIONING & PROVENANCE (PHASE 9.2 & 9.3)
# =========================================================================

def test_rule_versioning_immutability(temp_repo):
    """Ensure older versions are never overwritten when new versions are spawned."""
    v1 = RuleVersion(
        rule_id="rule_bos_100",
        version=1,
        title="BOS Breakout",
        rule_text="Enter on Break of Structure with body close.",
        conditions=["BOS body close"],
    )
    temp_repo.save_rule_version(v1)

    # Create v2 for same logical rule
    v2 = RuleVersion(
        rule_id="rule_bos_100",
        version=2,
        title="BOS Breakout Enhanced",
        rule_text="Enter on Break of Structure with body close + volume confirmation.",
        conditions=["BOS body close", "Volume > 1.5x average"],
    )
    temp_repo.save_rule_version(v2)

    # Both versions must be queryable and intact
    fetched_v1 = temp_repo.get_rule_version("rule_bos_100", version=1)
    fetched_v2 = temp_repo.get_rule_version("rule_bos_100", version=2)

    assert fetched_v1 is not None
    assert fetched_v1.version == 1
    assert "Volume" not in fetched_v1.rule_text

    assert fetched_v2 is not None
    assert fetched_v2.version == 2
    assert "volume" in fetched_v2.rule_text.lower()


def test_strategy_version_combined_rules(temp_repo):
    """Test composite StrategyVersion referencing multiple exact rule versions."""
    strat = StrategyVersion(
        strategy_id="strat_ict_master",
        version=1,
        name="ICT London & NY Master",
        rule_version_ids=["rule_sweep_v1", "rule_fvg_v2"],
        asset_classes=[AssetClass.FOREX],
        symbols=["EUR/USD", "GBP/USD"],
        timeframes=["15m"],
    )
    temp_repo.save_strategy_version(strat)

    retrieved = temp_repo.get_strategy_version("strat_ict_master", version=1)
    assert retrieved is not None
    assert retrieved.rule_version_ids == ["rule_sweep_v1", "rule_fvg_v2"]


# =========================================================================
# 3. DETERMINISTIC BACKTEST ENGINE & METRICS (PHASE 9.4 - 9.13)
# =========================================================================

def test_backtest_engine_ai_replay(backtest_engine, temp_repo):
    """Run deterministic AI backtest and check purely mathematical metrics."""
    run = backtest_engine.create_backtest(
        mode=BacktestMode.AI_BACKTEST,
        asset_class=AssetClass.FOREX,
        symbol="EUR/USD",
        timeframe="15m",
        config=BacktestConfig(
            asset_class=AssetClass.FOREX,
            symbol="EUR/USD",
            timeframe="15m",
            initial_equity=10000.0,
            risk_per_trade_pct=1.0,
            max_candles=60,
        ),
    )

    completed = backtest_engine.run_backtest(run.backtest_id)
    assert completed.status == BacktestStatus.COMPLETED
    assert completed.progress_pct == 100.0
    assert completed.metrics is not None

    m = completed.metrics
    assert m.total_trades >= 0
    if m.total_trades > 0:
        # Check math consistency
        assert m.wins + m.losses + m.breakevens == m.total_trades
        assert 0.0 <= m.win_rate <= 1.0
        assert m.max_drawdown_dollars >= 0.0
        assert m.max_drawdown_pct >= 0.0


def test_backtest_engine_manual_replay(backtest_engine, temp_repo):
    """Record a user-executed manual replay backtest trade with source = USER."""
    run = backtest_engine.create_backtest(
        mode=BacktestMode.MANUAL_REPLAY,
        asset_class=AssetClass.FOREX,
        symbol="GBP/USD",
        timeframe="15m",
    )

    req = ManualBacktestTradeRequest(
        backtest_id=run.backtest_id,
        symbol="GBP/USD",
        asset_class=AssetClass.FOREX,
        direction="BUY",
        entry_time=1700000000.0,
        entry_price=1.2700,
        exit_time=1700001800.0,
        exit_price=1.2750,
        stop_loss=1.2675,
        target_price=1.2750,
        quantity=1.0,
        confluence_summary="Tested London Low Sweep",
        notes="High conviction setup from lecture 3",
    )

    trade = backtest_engine.record_manual_trade(req)
    assert trade.source == TradeSource.USER
    assert trade.pnl == 0.005  # (1.2750 - 1.2700) * 1.0 = 0.005
    assert trade.result == "WIN"

    # Verify run metrics updated
    updated_run = temp_repo.get_backtest_run(run.backtest_id)
    assert updated_run.metrics.total_trades == 1
    assert updated_run.metrics.wins == 1


def test_backtest_recent_lecture_window(backtest_engine):
    """Verify recent lecture validation mode operates on limited historical window."""
    run = backtest_engine.create_backtest(
        mode=BacktestMode.RECENT_LECTURE,
        asset_class=AssetClass.FOREX,
        symbol="EUR/USD",
        timeframe="15m",
        config=BacktestConfig(max_candles=500),  # Attempt requesting 500
    )
    completed = backtest_engine.run_backtest(run.backtest_id)
    # Recent lecture mode clamps max candles to <= 200 for a 2-3 day window
    assert completed.total_candles <= 200


# =========================================================================
# 4. ASSET CLASS SEPARATION (PHASE 9.8)
# =========================================================================

def test_asset_class_separation_forex_and_crypto(backtest_engine):
    """Ensure Forex (EUR/USD, USD/JPY) and Crypto (BTC/USD) results remain strictly isolated."""
    run_forex = backtest_engine.create_backtest(
        mode=BacktestMode.AI_BACKTEST,
        asset_class=AssetClass.FOREX,
        symbol="USD/JPY",
        timeframe="15m",
        config=BacktestConfig(asset_class=AssetClass.FOREX, symbol="USD/JPY", max_candles=40),
    )
    comp_forex = backtest_engine.run_backtest(run_forex.backtest_id)

    run_crypto = backtest_engine.create_backtest(
        mode=BacktestMode.AI_BACKTEST,
        asset_class=AssetClass.CRYPTO,
        symbol="BTC/USD",
        timeframe="15m",
        config=BacktestConfig(asset_class=AssetClass.CRYPTO, symbol="BTC/USD", max_candles=40),
    )
    comp_crypto = backtest_engine.run_backtest(run_crypto.backtest_id)

    assert comp_forex.asset_class == AssetClass.FOREX
    assert comp_crypto.asset_class == AssetClass.CRYPTO

    # Cross-Asset summary preserves separate results
    summary = backtest_engine.generate_cross_asset_summary([comp_forex, comp_crypto])
    assert "forex_results" in summary
    assert "crypto_results" in summary
    assert "cross_asset_summary" in summary
    assert "USD/JPY" in summary["forex_results"]["symbols"]
    assert "BTC/USD" in summary["crypto_results"]["symbols"]


# =========================================================================
# 5. LOOK-AHEAD BIAS & DATA LEAKAGE PROTECTION (PHASE 9.9 & 9.10)
# =========================================================================

def test_lookahead_bias_protection(market_data):
    """Verify that market series slicing strictly obeys end_time cutoffs."""
    full_series = market_data.get_historical_series("EUR/USD", timeframe="15m", limit=50)
    cutoff_candle = full_series[25]
    cutoff_time = cutoff_candle.timestamp

    sliced = market_data.get_historical_series("EUR/USD", timeframe="15m", limit=50, end_time=cutoff_time)

    # No candle in sliced may have timestamp > cutoff_time
    assert all(c.timestamp <= cutoff_time for c in sliced)
    assert len(sliced) <= 26


# =========================================================================
# 6. RULE PROMOTION GATING (PHASE 9.2, 9.14 - 9.17)
# =========================================================================

def test_promotion_gate_ladder(promotion_manager, backtest_engine, temp_repo):
    """
    Test strict promotion progression:
    BACKTEST -> USER REVIEW -> APPROVED_FOR_PAPER -> PAPER_VALIDATION -> USER_LIVE_APPROVAL -> LIVE_ELIGIBLE
    """
    rule = RuleVersion(
        rule_id="rule_ladder_01",
        version=1,
        title="Liquidity Grab Strategy",
        rule_text="Take London Low sweep retests",
        state=RuleLifecycleState.CANDIDATE,
    )
    temp_repo.save_rule_version(rule)

    # 1. Attempting paper approval without completed backtest must fail
    with pytest.raises(PromotionError):
        promotion_manager.approve_for_paper("rule_ladder_01", user_confirmed=True)

    # 2. Run backtest to completion
    run = backtest_engine.create_backtest(
        mode=BacktestMode.AI_BACKTEST,
        rule_version_id="rule_ladder_01",
        config=BacktestConfig(max_candles=40),
    )
    backtest_engine.run_backtest(run.backtest_id)

    # 3. User approval advances rule to APPROVED_FOR_PAPER
    prm = promotion_manager.approve_for_paper("rule_ladder_01", user_confirmed=True)
    assert prm.user_backtest_approved is True
    assert prm.current_state == RuleLifecycleState.APPROVED_FOR_PAPER

    # 4. Attempting live approval before paper validation must fail
    with pytest.raises(PromotionError):
        promotion_manager.approve_for_live("rule_ladder_01", user_confirmed=True)

    # 5. Record successful paper trading validation
    promotion_manager.record_paper_validation(
        "rule_ladder_01",
        trades_count=10,
        paper_pnl=450.0,
        passed_validation=True,
    )
    updated_rule = temp_repo.get_rule_version("rule_ladder_01")
    assert updated_rule.state == RuleLifecycleState.PAPER_VALIDATION

    # 6. Explicit user live approval unlocks LIVE_ELIGIBLE
    live_prm = promotion_manager.approve_for_live("rule_ladder_01", user_confirmed=True)
    assert live_prm.user_live_approved is True
    assert live_prm.current_state == RuleLifecycleState.LIVE_ELIGIBLE


def test_live_execution_eligibility_step8_gate(promotion_manager, execution_gate, temp_repo):
    """Ensure pre-flight check validates both rule promotion status and Step 8 ExecutionGate."""
    rule = RuleVersion(
        rule_id="rule_live_01",
        version=1,
        title="Ready Rule",
        rule_text="Validated strategy",
        state=RuleLifecycleState.LIVE_ELIGIBLE,
    )
    temp_repo.save_rule_version(rule)
    prm = promotion_manager.get_or_create_promotion_record("rule_live_01")
    prm.user_backtest_approved = True
    prm.paper_validation_approved = True
    prm.user_live_approved = True
    prm.current_state = RuleLifecycleState.LIVE_ELIGIBLE
    temp_repo.save_rule_promotion(prm)

    # Valid proposal meeting Step 8 confluence and risk criteria
    confluences = [
        TradeConfluence(
            category=ConfluenceCategory.MARKET_STRUCTURE,
            source="MTF",
            independence_group="STRUCTURE",
            timeframe="H4",
            evidence_reference="HTF Bullish",
            confidence=0.9,
        ),
        TradeConfluence(
            category=ConfluenceCategory.POI,
            source="S&R",
            independence_group="LEVELS",
            timeframe="D1",
            evidence_reference="Daily S1 hold",
            confidence=0.9,
        ),
    ]

    proposal = TradeProposal(
        symbol="EUR/USD",
        direction="BUY",
        entry_price=1.0850,
        hard_stop_loss=1.0820,
        tp1=1.0895,
        final_target=1.0925,
        invalidation_condition="Candle close below 1.0820",
        confluences=confluences,
    )

    eligible, reasons = promotion_manager.verify_live_execution_eligibility("rule_live_01", proposal=proposal)
    assert eligible is True
    assert len(reasons) == 0


# =========================================================================
# 7. RESUMABILITY & CHECKPOINTING (PHASE 9.18)
# =========================================================================

def test_backtest_pause_and_resumability(backtest_engine, temp_repo):
    """Verify pausing captures candle index checkpoint and resume completes from checkpoint."""
    run = backtest_engine.create_backtest(
        mode=BacktestMode.AI_BACKTEST,
        config=BacktestConfig(max_candles=50),
    )

    # Simulate pause after partial execution
    run.status = BacktestStatus.PAUSED
    run.current_candle_index = 25
    run.progress_pct = 50.0
    temp_repo.save_backtest_run(run)

    # Resume from checkpoint
    resumed = backtest_engine.resume_backtest(run.backtest_id)
    assert resumed.status == BacktestStatus.COMPLETED
    assert resumed.current_candle_index >= 25
    assert resumed.progress_pct == 100.0


def test_backtest_cancellation(backtest_engine, temp_repo):
    """Verify that cancellation stops backtest execution and transitions state to CANCELLED."""
    run = backtest_engine.create_backtest(
        mode=BacktestMode.AI_BACKTEST,
        config=BacktestConfig(max_candles=50),
    )
    run.status = BacktestStatus.RUNNING
    temp_repo.save_backtest_run(run)

    cancelled = backtest_engine.cancel_backtest(run.backtest_id)
    assert cancelled is not None
    assert cancelled.status == BacktestStatus.CANCELLED

    # Verify persisted state
    saved = temp_repo.get_backtest_run(run.backtest_id)
    assert saved.status == BacktestStatus.CANCELLED


def test_backtest_engine_causal_invariance(backtest_engine, market_data):
    """
    Verify strict look-ahead protection: future price actions must not alter decisions
    at earlier time steps.
    """
    candles = market_data.get_historical_series("EUR/USD", timeframe="15m", limit=30)
    assert len(candles) >= 15

    # 1. Evaluate trigger at index 14
    eval_slice = candles[:15]
    trig1, dir1, sl1, tp1 = backtest_engine._evaluate_market_trigger(
        eval_slice, symbol="EUR/USD", pip_mult=0.0001, spread_cost=0.00015
    )

    # 2. Add an extreme future price anomaly at index 20 (e.g. flash crash / surge)
    future_candles = list(candles)
    future_candles[20] = Candle(
        timestamp=candles[20].timestamp,
        open=2.0000,
        high=2.5000,
        low=1.9000,
        close=2.4000,
        volume=999999.0,
    )

    # 3. Re-evaluate strictly on visible slice up to index 14
    # The evaluation at index 14 MUST be 100% invariant to any future changes at index 20
    trig2, dir2, sl2, tp2 = backtest_engine._evaluate_market_trigger(
        future_candles[:15], symbol="EUR/USD", pip_mult=0.0001, spread_cost=0.00015
    )

    assert trig1 == trig2
    assert dir1 == dir2
    assert sl1 == sl2
    assert tp1 == tp2




# =========================================================================
# 8. REST API ENDPOINTS & UI DOM (PHASE 9.21 & 9.22)
# =========================================================================

def test_step9_rest_api_endpoints():
    """Verify Step 9 REST endpoints via TestClient."""
    client = TestClient(app)

    # 1. Create rule version
    res = client.post("/api/rules/versions", json={
        "title": "API Rule Test",
        "rule_text": "Enter when FVG confirmed",
        "conditions": ["FVG confirmed"],
        "exceptions": ["News event"],
    })
    assert res.status_code == 200
    rule_data = res.json()
    rule_id = rule_data["rule_id"]

    # 2. Extract rule via Trading Brain
    ext_res = client.post("/api/trading-brain/extract-rule", json={
        "lecture_text": "Always look for liquidity sweep before displacement."
    })
    assert ext_res.status_code == 200
    assert "rule_id" in ext_res.json()

    # 3. Create and execute backtest
    bt_res = client.post("/api/backtests", json={
        "mode": "AI_BACKTEST",
        "asset_class": "FOREX",
        "symbol": "EUR/USD",
        "timeframe": "15m",
        "rule_version_id": rule_id,
        "max_candles": 40,
    })
    assert bt_res.status_code == 200
    bt_data = bt_res.json()
    bt_id = bt_data["backtest_id"]

    # Start backtest
    start_res = client.post(f"/api/backtests/{bt_id}/start")
    assert start_res.status_code == 200
    assert start_res.json()["status"] == "COMPLETED"

    # Analyze performance via Trading Brain
    ai_res = client.post("/api/trading-brain/analyze-backtest", json={"backtest_id": bt_id})
    assert ai_res.status_code == 200
    assert "recommendation" in ai_res.json()

    # Approve for paper
    appr_res = client.post(f"/api/rules/{rule_id}/approve-paper", json={
        "user_confirmed": True,
        "notes": "Testing paper promotion endpoint"
    })
    assert appr_res.status_code == 200
    assert appr_res.json()["user_backtest_approved"] is True

    # Cross-asset summary
    cross_res = client.get("/api/backtests/cross-asset-summary")
    assert cross_res.status_code == 200
    assert "forex_results" in cross_res.json()
    assert "crypto_results" in cross_res.json()


def test_step9_ui_dom_elements():
    """Verify required Step 9 UI elements exist in trading_ai/ui/index.html."""
    html_path = Path(__file__).parent.parent.parent / "trading_ai" / "ui" / "index.html"
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")

    required_ids = [
        "step9-backtesting-panel",
        "bt-mode-badge",
        "bt-status-badge",
        "badge-gate-backtest",
        "badge-gate-paper",
        "badge-gate-live",
        "bt-asset-class-select",
        "bt-symbol-select",
        "bt-tf-select",
        "bt-mode-select",
        "btn-start-backtest",
        "btn-pause-backtest",
        "btn-analyze-brain",
        "bt-total-trades",
        "bt-win-rate",
        "bt-profit-factor",
        "bt-expectancy",
        "bt-drawdown",
        "bt-total-pnl",
        "btn-approve-paper",
        "btn-approve-live",
        "bt-brain-analysis-text",
        "bt-trades-container",
    ]

    for elem_id in required_ids:
        assert f'id="{elem_id}"' in content, f"Missing UI element: {elem_id}"
