"""
Post-Trade Evaluation Engine for Step 10.
Evaluates closed trades, separates planned vs actual parameters, computes deterministic
P&L and R-multiples, attributes win/loss reasons deterministically, and identifies observable behaviors.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import uuid

from trading_ai.schemas.journal import (
    TradeJournalEntry,
    PlannedTradeValues,
    ActualTradeValues,
    ObservedBehavior,
)
from trading_ai.schemas.enums import (
    JournalEnvironment,
    JournalBacktestType,
    TradeOutcome,
    ExitReasonCategory,
    ObservedBehaviorType,
)
from trading_ai.storage.repository import Repository


class PostTradeEvaluator:
    """
    Constructs and finalizes post-trade journal entries from actual execution and order data.
    Never overwrites planned parameters with actual values.
    """

    def __init__(self, repository: Repository):
        self.repository = repository

    def create_journal_from_execution(
        self,
        trade_id: str,
        environment: JournalEnvironment,
        asset_class: str,
        symbol: str,
        timeframe: str,
        direction: str,
        planned: PlannedTradeValues,
        actual: ActualTradeValues,
        setup: Optional[str] = None,
        strategy_name: Optional[str] = None,
        rule_version_id: Optional[str] = None,
        lecture_source_id: Optional[str] = None,
        market_regime: Optional[str] = None,
        htf_bias: Optional[str] = None,
        mtf_structure: Optional[str] = None,
        ltf_trigger: Optional[str] = None,
        confluences: Optional[List[str]] = None,
        backtest_type: Optional[JournalBacktestType] = None,
        backtest_id: Optional[str] = None,
        dataset_id: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
        opened_at: Optional[datetime] = None,
        closed_at: Optional[datetime] = None,
    ) -> TradeJournalEntry:
        """
        Creates a completed or in-progress TradeJournalEntry with strict separation of planned and actual data.
        """
        # Ensure deterministic R calculation if actual is closed
        self.finalize_actual_metrics(planned, actual, direction)

        journal_id = f"jour_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        entry = TradeJournalEntry(
            journal_id=journal_id,
            trade_id=trade_id,
            environment=environment,
            backtest_type=backtest_type or JournalBacktestType.NOT_APPLICABLE,
            asset_class=asset_class.upper(),
            symbol=symbol.upper(),
            timeframe=timeframe,
            direction=direction.upper(),
            strategy_name=strategy_name,
            rule_version_id=rule_version_id,
            lecture_source_id=lecture_source_id,
            setup=setup or "DISCRETIONARY_SETUP",
            market_regime=market_regime or "BALANCED",
            htf_bias=htf_bias,
            mtf_structure=mtf_structure,
            ltf_trigger=ltf_trigger,
            confluences=confluences or [],
            planned=planned,
            actual=actual,
            backtest_id=backtest_id,
            dataset_id=dataset_id,
            evidence=evidence or {},
            created_at=now,
            updated_at=now,
            opened_at=opened_at,
            closed_at=closed_at,
        )

        self.repository.save_journal_entry(entry)
        return entry

    def finalize_actual_metrics(
        self,
        planned: PlannedTradeValues,
        actual: ActualTradeValues,
        direction: str,
    ) -> None:
        """
        Deterministically computes realized P&L, realized R-multiple, outcome, and exit reason.
        """
        if actual.exit_price is None or actual.entry_price is None:
            return  # Trade still open

        # 1. P&L calculation (net of fees and spread/slippage)
        # Gross price delta
        is_long = direction.upper() in ["BUY", "LONG"]
        price_diff = (actual.exit_price - actual.entry_price) if is_long else (actual.entry_price - actual.exit_price)
        gross_pnl = price_diff * actual.quantity
        net_pnl = gross_pnl - actual.fees - (actual.spread_cost + actual.slippage_cost)
        actual.realized_pnl = round(net_pnl, 4)

        # 2. Realized R calculation
        # Risk amount planned: |entry - SL| * quantity
        risk_per_unit = abs(planned.entry_price - planned.stop_loss)
        planned_risk = planned.risk_amount or (planned.planned_risk_amount or 0.0)
        if risk_per_unit > 0 and planned_risk > 0:
            actual.realized_r = round(actual.realized_pnl / planned_risk, 2)
        elif risk_per_unit > 0:
            calc_risk = risk_per_unit * (actual.quantity or 1.0)
            actual.realized_r = round(actual.realized_pnl / calc_risk, 2)
        else:
            actual.realized_r = 0.0

        # 3. Outcome categorization
        if actual.realized_pnl > 0.5:
            actual.outcome = TradeOutcome.WIN
        elif actual.realized_pnl < -0.5:
            actual.outcome = TradeOutcome.LOSS
        else:
            actual.outcome = TradeOutcome.BREAKEVEN

        # 4. Exit reason deduction from price levels if not already set or unknown
        tp_target = planned.tp1 or (planned.planned_tp1 or 0.0)
        if actual.exit_reason in [ExitReasonCategory.UNKNOWN, ExitReasonCategory.INSUFFICIENT_EVIDENCE]:
            if is_long:
                if tp_target > 0 and actual.exit_price >= tp_target - 0.0001:
                    actual.exit_reason = ExitReasonCategory.TP_HIT
                elif actual.exit_price <= planned.stop_loss + 0.0001:
                    actual.exit_reason = ExitReasonCategory.SL_HIT
            else:
                if tp_target > 0 and actual.exit_price <= tp_target + 0.0001:
                    actual.exit_reason = ExitReasonCategory.TP_HIT
                elif actual.exit_price >= planned.stop_loss - 0.0001:
                    actual.exit_reason = ExitReasonCategory.SL_HIT

        # 5. Observable Behavior Detection based on objective facts
        self.detect_observed_behaviors(planned, actual, is_long)

    def detect_observed_behaviors(
        self,
        planned: PlannedTradeValues,
        actual: ActualTradeValues,
        is_long: bool,
    ) -> None:
        """
        Detects objective execution discrepancies:
        - Slippage / chasing price
        - Moving SL farther (increased risk)
        - Premature exit before planned TP or SL
        """
        now = datetime.now(timezone.utc)
        existing_behaviors = {b.behavior for b in actual.observed_behaviors}

        # Check Chasing Price / Late Entry
        if actual.entry_price and planned.entry_price:
            entry_slippage_pips = abs(actual.entry_price - planned.entry_price)
            if entry_slippage_pips > 0.0010:  # ~10 pips or substantial deviation
                if ObservedBehaviorType.CHASING_PRICE not in existing_behaviors:
                    actual.observed_behaviors.append(
                        ObservedBehavior(
                            behavior=ObservedBehaviorType.CHASING_PRICE,
                            evidence_source="EXECUTION_FILL",
                            details=f"Entry filled at {actual.entry_price}, planned was {planned.entry_price} (diff {entry_slippage_pips:.5f})",
                            observed_at=now,
                        )
                    )

        # Check Moving Stop Loss (increasing risk)
        if actual.stop_loss and planned.stop_loss:
            if is_long and actual.stop_loss < planned.stop_loss:
                if ObservedBehaviorType.MOVING_SL not in existing_behaviors:
                    actual.observed_behaviors.append(
                        ObservedBehavior(
                            behavior=ObservedBehaviorType.MOVING_SL,
                            evidence_source="ORDER_HISTORY",
                            details=f"Stop loss moved down from {planned.stop_loss} to {actual.stop_loss}, expanding trade risk.",
                            observed_at=now,
                        )
                    )
            elif not is_long and actual.stop_loss > planned.stop_loss:
                if ObservedBehaviorType.MOVING_SL not in existing_behaviors:
                    actual.observed_behaviors.append(
                        ObservedBehavior(
                            behavior=ObservedBehaviorType.MOVING_SL,
                            evidence_source="ORDER_HISTORY",
                            details=f"Stop loss moved up from {planned.stop_loss} to {actual.stop_loss}, expanding trade risk.",
                            observed_at=now,
                        )
                    )

        # Check Premature Exit
        if actual.exit_reason == ExitReasonCategory.PREMATURE_EXIT:
            if ObservedBehaviorType.PREMATURE_EXIT not in existing_behaviors:
                actual.observed_behaviors.append(
                    ObservedBehavior(
                        behavior=ObservedBehaviorType.PREMATURE_EXIT,
                        evidence_source="MANUAL_CLOSE",
                        details="Position closed manually before reaching planned stop loss or target.",
                        observed_at=now,
                    )
                )
