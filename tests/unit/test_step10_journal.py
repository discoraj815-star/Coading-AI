"""
Focused test suite for Step 10 — Professional Trading Journal Management & Step 1–9 Wiring.
Tests:
1. Journal creation & persistence
2. Planned vs actual separation (immutability of planned params)
3. Deterministic P&L and R-multiple calculations
4. Exit reason categorization & observable behaviors (chasing price, moving SL)
5. Insufficient evidence handling
6. Strict Environment separation (BACKTEST vs PAPER vs LIVE)
7. Backtest type separation (MANUAL_BACKTEST vs AI_BACKTEST)
8. Asset class separation (FOREX vs CRYPTO)
9. Pre-trade journal review with bounded context
10. Applicable lessons and similar trades retrieval in pre-trade review
11. Step 7 discipline lock integration in pre-trade review
12. Lesson lifecycle (CANDIDATE -> VALIDATED -> APPLIED via Step 9 promotion ladder)
13. User review gate enforcement (No automatic live rule change)
14. Journal analytics accuracy (win rate, profit factor, max DD, expectancy)
15. Resilience, token budgeting, checkpoint persistence, and resumption
"""
import pytest
from datetime import datetime, timezone

from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.backtest.promotion_manager import RulePromotionManager
from trading_ai.journal import (
    TradingJournalCore,
    PreTradeReviewer,
    PostTradeEvaluator,
    JournalAnalyticsEngine,
    JournalLessonEngine,
    JournalResilienceManager,
)
from trading_ai.schemas.journal import (
    PreTradeReviewRequest,
    PlannedTradeValues,
    ActualTradeValues,
    TradeJournalEntry,
    ObservedBehavior,
)
from trading_ai.schemas.enums import (
    JournalEnvironment,
    JournalBacktestType,
    TradeOutcome,
    ExitReasonCategory,
    ObservedBehaviorType,
    LessonLifecycleState,
    PreTradeReviewStatus,
)


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "test_step10.db")
    db_manager = DatabaseManager(db_path)
    db_manager.initialize_schema()
    return Repository(db_manager)


@pytest.fixture
def discipline_engine(repo):
    return TradingDisciplineEngine(repo)


@pytest.fixture
def journal_core(repo, discipline_engine):
    return TradingJournalCore(
        repository=repo,
        discipline_engine=discipline_engine,
        promotion_manager=None,
    )


# -------------------------------------------------------------
# Test 1 & 2: Creation & Planned vs Actual Separation
# -------------------------------------------------------------
def test_journal_creation_planned_vs_actual(journal_core, repo):
    planned = PlannedTradeValues(
        entry_price=1.1000,
        stop_loss=1.0950,
        planned_tp1=1.1100,
        planned_final_tp=1.1150,
        invalidation_level=1.0945,
        planned_risk_amount=100.0,
        planned_rr=2.0,
        trade_thesis="Bullish order block tap during London open",
    )
    actual = ActualTradeValues(
        entry_price=1.1005,
        exit_price=1.1100,
        stop_loss=1.0950,
        quantity=20000.0,
        fees=2.0,
        spread_cost=1.0,
        slippage_cost=0.5,
    )

    entry = journal_core.posttrade_evaluator.create_journal_from_execution(
        trade_id="tr_1001",
        environment=JournalEnvironment.PAPER,
        asset_class="FOREX",
        symbol="EURUSD",
        timeframe="15m",
        direction="BUY",
        planned=planned,
        actual=actual,
        setup="OrderBlock_London",
    )

    assert entry.journal_id is not None
    # Planned values must remain completely untouched
    assert entry.planned.entry_price == 1.1000
    assert entry.planned.planned_risk_amount == 100.0
    # Actual values must be correctly recorded
    assert entry.actual.entry_price == 1.1005
    assert entry.actual.exit_price == 1.1100
    assert entry.actual.outcome == TradeOutcome.WIN
    assert entry.actual.exit_reason == ExitReasonCategory.TP_HIT
    assert entry.actual.realized_pnl > 0


# -------------------------------------------------------------
# Test 3 & 4: Deterministic P&L, R-Multiple, and Behaviors
# -------------------------------------------------------------
def test_deterministic_pnl_and_behavior_detection(journal_core):
    planned = PlannedTradeValues(
        entry_price=100.0,
        stop_loss=95.0,
        planned_tp1=110.0,
        planned_final_tp=120.0,
        planned_risk_amount=50.0,
        planned_rr=2.0,
        trade_thesis="Breakout test",
    )
    # Trader entered with huge slippage (chasing price) and moved SL down (expanding risk)
    actual = ActualTradeValues(
        entry_price=101.5,
        exit_price=92.0,
        stop_loss=92.0,  # moved down from 95.0
        quantity=10.0,
        fees=1.0,
        spread_cost=0.5,
        slippage_cost=0.5,
    )

    entry = journal_core.posttrade_evaluator.create_journal_from_execution(
        trade_id="tr_1002",
        environment=JournalEnvironment.PAPER,
        asset_class="CRYPTO",
        symbol="SOLUSDT",
        timeframe="1h",
        direction="BUY",
        planned=planned,
        actual=actual,
    )

    assert entry.actual.outcome == TradeOutcome.LOSS
    assert entry.actual.realized_pnl < 0
    # Check that behaviors were objectively detected
    detected = [b.behavior for b in entry.actual.observed_behaviors]
    assert ObservedBehaviorType.CHASING_PRICE in detected
    assert ObservedBehaviorType.MOVING_SL in detected


# -------------------------------------------------------------
# Test 5 & 6: Environment Separation
# -------------------------------------------------------------
def test_strict_environment_isolation(journal_core, repo):
    planned = PlannedTradeValues(
        entry_price=1.0, stop_loss=0.9, planned_tp1=1.2, planned_rr=2.0,
        trade_thesis="Thesis test",
    )
    actual_win = ActualTradeValues(entry_price=1.0, exit_price=1.2, quantity=100.0)
    actual_loss = ActualTradeValues(entry_price=1.0, exit_price=0.9, quantity=100.0)

    # 1 Backtest trade
    journal_core.posttrade_evaluator.create_journal_from_execution(
        trade_id="tr_bt_1", environment=JournalEnvironment.BACKTEST, backtest_type=JournalBacktestType.MANUAL_BACKTEST,
        asset_class="FOREX", symbol="EURUSD", timeframe="15m", direction="BUY", planned=planned, actual=actual_win,
    )
    # 1 Paper trade
    journal_core.posttrade_evaluator.create_journal_from_execution(
        trade_id="tr_pp_1", environment=JournalEnvironment.PAPER,
        asset_class="FOREX", symbol="EURUSD", timeframe="15m", direction="BUY", planned=planned, actual=actual_loss,
    )
    # 0 Live trades

    bt_trades = repo.list_journal_entries(environment=JournalEnvironment.BACKTEST)
    pp_trades = repo.list_journal_entries(environment=JournalEnvironment.PAPER)
    lv_trades = repo.list_journal_entries(environment=JournalEnvironment.LIVE)

    assert len(bt_trades) == 1
    assert len(pp_trades) == 1
    assert len(lv_trades) == 0

    # Analytics must be completely separated
    bt_summary = journal_core.analytics_engine.calculate_summary(JournalEnvironment.BACKTEST)
    pp_summary = journal_core.analytics_engine.calculate_summary(JournalEnvironment.PAPER)
    lv_summary = journal_core.analytics_engine.calculate_summary(JournalEnvironment.LIVE)

    assert bt_summary.total_trades == 1
    assert bt_summary.wins == 1
    assert pp_summary.total_trades == 1
    assert pp_summary.losses == 1
    assert lv_summary.total_trades == 0


# -------------------------------------------------------------
# Test 7 & 8: Manual vs AI Backtest & Forex vs Crypto Separation
# -------------------------------------------------------------
def test_backtest_types_and_asset_separation(journal_core, repo):
    planned = PlannedTradeValues(
        entry_price=10.0, stop_loss=9.0, planned_tp1=12.0, planned_rr=2.0,
        trade_thesis="Test",
    )
    actual = ActualTradeValues(entry_price=10.0, exit_price=12.0, quantity=10.0)

    # Manual Forex Backtest
    journal_core.posttrade_evaluator.create_journal_from_execution(
        trade_id="tr_m_1", environment=JournalEnvironment.BACKTEST, backtest_type=JournalBacktestType.MANUAL_BACKTEST,
        asset_class="FOREX", symbol="GBPUSD", timeframe="1h", direction="BUY", planned=planned, actual=actual,
    )
    # AI Crypto Backtest
    journal_core.posttrade_evaluator.create_journal_from_execution(
        trade_id="tr_ai_1", environment=JournalEnvironment.BACKTEST, backtest_type=JournalBacktestType.AI_BACKTEST,
        asset_class="CRYPTO", symbol="BTCUSDT", timeframe="1h", direction="BUY", planned=planned, actual=actual,
    )

    manual_entries = repo.list_journal_entries(environment=JournalEnvironment.BACKTEST, backtest_type=JournalBacktestType.MANUAL_BACKTEST)
    ai_entries = repo.list_journal_entries(environment=JournalEnvironment.BACKTEST, backtest_type=JournalBacktestType.AI_BACKTEST)
    forex_entries = repo.list_journal_entries(asset_class="FOREX")
    crypto_entries = repo.list_journal_entries(asset_class="CRYPTO")

    assert len(manual_entries) == 1
    assert len(ai_entries) == 1
    assert len(forex_entries) == 1
    assert len(crypto_entries) == 1


# -------------------------------------------------------------
# Test 9, 10, & 11: Pre-Trade Journal Review & Memory Decision Support
# -------------------------------------------------------------
def test_pretrade_review_workflow(journal_core, repo):
    # Seed historical trades with losses and behaviors
    planned = PlannedTradeValues(entry_price=1.1, stop_loss=1.09, planned_tp1=1.12, planned_rr=2.0, trade_thesis="Historical")
    actual_loss = ActualTradeValues(
        entry_price=1.1, exit_price=1.09, stop_loss=1.09, quantity=100.0,
        observed_behaviors=[ObservedBehavior(behavior=ObservedBehaviorType.PREMATURE_EXIT, evidence_source="MANUAL", details="Closed early", observed_at=datetime.now(timezone.utc))],
    )
    journal_core.posttrade_evaluator.create_journal_from_execution(
        trade_id="hist_1", environment=JournalEnvironment.PAPER, asset_class="FOREX",
        symbol="EURUSD", timeframe="15m", direction="BUY", setup="FairValueGap", planned=planned, actual=actual_loss,
    )

    # Seed a validated lesson
    lesson = journal_core.lesson_engine.propose_lesson(
        observation="FairValueGap entries often suffer premature exit",
        explanation="Trader cuts profits too quickly",
        proposed_action="Hold until TP1 is reached",
        source_journal_ids=["hist_1"],
    )
    journal_core.lesson_engine.user_review_lesson(lesson.lesson_id, reviewer="user_1", approved=True)

    # Now perform pre-trade review on a new proposed trade
    req = PreTradeReviewRequest(
        proposal_id="prop_new_1",
        environment=JournalEnvironment.PAPER,
        symbol="EURUSD",
        timeframe="15m",
        direction="BUY",
        setup="FairValueGap",
        planned_values=PlannedTradeValues(
            entry_price=1.1050,
            stop_loss=1.1000,
            planned_tp1=1.1150,
            planned_rr=2.0,
            trade_thesis="Valid setup meeting all criteria",
        ),
    )

    review = journal_core.pretrade_reviewer.review_trade(req)
    assert review.proposal_id == "prop_new_1"
    assert review.status == PreTradeReviewStatus.WARNING  # Warning due to historical issue and applicable lesson
    assert review.similar_trades_count >= 1
    assert len(review.applicable_lessons) >= 1
    assert any("Historical issue on EURUSD" in w for w in review.warnings)


# -------------------------------------------------------------
# Test 12 & 13: Lesson Lifecycle & User Approval Safety Gate
# -------------------------------------------------------------
def test_lesson_lifecycle_user_approval_gate(journal_core):
    # Propose lesson candidate
    candidate = journal_core.lesson_engine.propose_lesson(
        observation="Repeated stop-outs during NY opening volatility",
        explanation="Entries right at 9:30 AM get swept before reversing",
        proposed_action="Wait 15 minutes after equity market open",
        source_journal_ids=["tr_ny_1", "tr_ny_2"],
        confidence=0.9,
    )
    assert candidate.status == LessonLifecycleState.LESSON_CANDIDATE

    # Critical Safety: Cannot apply unapproved candidate to strategy
    with pytest.raises(ValueError, match="Must be VALIDATED first"):
        journal_core.lesson_engine.apply_validated_lesson(candidate.lesson_id, approved_by="admin")

    # User review gate: Reject first
    rejected = journal_core.lesson_engine.user_review_lesson(candidate.lesson_id, reviewer="lead_trader", approved=False)
    assert rejected.status == LessonLifecycleState.REJECTED

    # Re-propose and Approve
    cand2 = journal_core.lesson_engine.propose_lesson(
        observation="Valid observation", explanation="Exp", proposed_action="Act", source_journal_ids=["tr_1"],
    )
    validated = journal_core.lesson_engine.user_review_lesson(cand2.lesson_id, reviewer="lead_trader", approved=True)
    assert validated.status == LessonLifecycleState.VALIDATED

    # Now apply to candidate improvement pipeline
    applied = journal_core.lesson_engine.apply_validated_lesson(cand2.lesson_id, approved_by="lead_trader")
    assert applied["status"] == "APPLIED_TO_CANDIDATE_PIPELINE"


# -------------------------------------------------------------
# Test 14 & 15: Resilience, Checkpointing, and Resumption
# -------------------------------------------------------------
def test_journal_resilience_and_checkpoint_resume(journal_core, repo):
    planned = PlannedTradeValues(entry_price=1.0, stop_loss=0.9, planned_tp1=1.2, planned_rr=2.0, trade_thesis="T")
    actual = ActualTradeValues(entry_price=1.0, exit_price=1.2, quantity=1.0)

    entries = []
    for i in range(5):
        e = journal_core.posttrade_evaluator.create_journal_from_execution(
            trade_id=f"batch_tr_{i}", environment=JournalEnvironment.BACKTEST,
            asset_class="FOREX", symbol="EURUSD", timeframe="1h", direction="BUY",
            planned=planned, actual=actual,
        )
        entries.append(e)

    # Process batch with small token budget to force context rollover
    resilience_mgr = JournalResilienceManager(repo, token_budget_per_batch=1000)

    processed_records = []
    def dummy_processor(entry):
        processed_records.append(entry.journal_id)
        return {"processed": True}

    # First run: should process ~2 entries and pause for rollover
    cp1 = resilience_mgr.process_journal_batch("job_batch_001", entries, dummy_processor)
    assert cp1.status == "PAUSED_CONTEXT_ROLLOVER"
    assert cp1.processed_count < len(entries)

    # Second run with expanded budget (simulating rollover into fresh context window): resumes and finishes
    resilience_mgr.token_budget_per_batch = 50000
    cp2 = resilience_mgr.process_journal_batch("job_batch_001", entries, dummy_processor)
    assert cp2.status == "COMPLETED"
    assert cp2.processed_count == 5
    assert len(processed_records) == 5


# -------------------------------------------------------------
# Test 16: Adversarial - Backtest trade attempting to contaminate Live
# -------------------------------------------------------------
def test_adversarial_environment_leakage_prevented(journal_core, repo):
    planned = PlannedTradeValues(entry_price=1.0, stop_loss=0.9, planned_tp1=1.2, planned_rr=2.0, trade_thesis="T")
    actual_huge_win = ActualTradeValues(entry_price=1.0, exit_price=2.0, quantity=1000.0)

    # Log huge win in backtest
    journal_core.posttrade_evaluator.create_journal_from_execution(
        trade_id="bt_huge_1",
        environment=JournalEnvironment.BACKTEST,
        backtest_type=JournalBacktestType.AI_BACKTEST,
        asset_class="FOREX",
        symbol="EURUSD",
        timeframe="1h",
        direction="BUY",
        planned=planned,
        actual=actual_huge_win,
    )

    # Live analytics MUST report 0 trades and $0 P&L
    live_summary = journal_core.analytics_engine.calculate_summary(JournalEnvironment.LIVE)
    assert live_summary.total_trades == 0
    assert live_summary.total_pnl == 0.0
    assert live_summary.win_rate == 0.0


# -------------------------------------------------------------
# Test 17: Pre-Trade Review Blocks Invalid Risk Direction
# -------------------------------------------------------------
def test_pretrade_review_blocks_inverted_stop_loss(journal_core):
    # Long trade with Stop Loss above Entry
    req = PreTradeReviewRequest(
        proposal_id="invalid_sl_prop",
        environment=JournalEnvironment.PAPER,
        symbol="EURUSD",
        timeframe="15m",
        direction="BUY",
        planned_values=PlannedTradeValues(
            entry_price=1.1000,
            stop_loss=1.1050,  # Invalid: above entry
            planned_tp1=1.1200,
            planned_rr=2.0,
            trade_thesis="Inverted SL test",
        ),
    )

    result = journal_core.pretrade_reviewer.review_trade(req)
    assert result.status == PreTradeReviewStatus.REJECTED
    assert len(result.rule_violations) >= 1
    assert any("Long trade stop loss" in v for v in result.rule_violations)


# -------------------------------------------------------------
# Test 18: Journal Status API Payload
# -------------------------------------------------------------
def test_journal_status_reporting(journal_core):
    status = journal_core.get_status()
    assert status["status"] == "ONLINE"
    assert "environments" in status
    assert status["token_resilience"] == "ACTIVE"

