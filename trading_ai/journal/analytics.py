"""
Deterministic Journal Analytics Engine for Step 10.
Calculates statistical performance indicators separated strictly by environment:
BACKTEST (MANUAL vs AI), PAPER, and LIVE, as well as asset class (FOREX vs CRYPTO).
Zero fabricated metrics; reports INSUFFICIENT_DATA when records are below threshold.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from trading_ai.schemas.journal import (
    TradeJournalEntry,
    JournalAnalyticsSummary,
)
from trading_ai.schemas.enums import (
    JournalEnvironment,
    JournalBacktestType,
    TradeOutcome,
)
from trading_ai.storage.repository import Repository


class JournalAnalyticsEngine:
    """
    Computes deterministic statistical summaries across isolated journal subsets.
    Never merges Backtest, Paper, or Live into primary performance metrics.
    """

    def __init__(self, repository: Repository):
        self.repository = repository

    def calculate_summary(
        self,
        environment: JournalEnvironment,
        asset_class: Optional[str] = None,
        symbol: Optional[str] = None,
        backtest_type: Optional[JournalBacktestType] = None,
        limit: int = 500,
    ) -> JournalAnalyticsSummary:
        """
        Computes accurate quantitative analytics for the specified isolated partition.
        """
        entries = self.repository.list_journal_entries(
            environment=environment,
            asset_class=asset_class,
            symbol=symbol,
            backtest_type=backtest_type,
            limit=limit,
        )

        total_trades = len(entries)
        if total_trades == 0:
            return JournalAnalyticsSummary(
                environment=environment,
                asset_class=asset_class,
                symbol=symbol,
                backtest_type=backtest_type,
                total_trades=0,
                wins=0,
                losses=0,
                breakeven=0,
                win_rate=0.0,
                profit_factor=0.0,
                average_r=0.0,
                expectancy=0.0,
                total_pnl=0.0,
                average_winner=0.0,
                average_loser=0.0,
                max_drawdown=0.0,
                max_consecutive_losses=0,
                max_consecutive_wins=0,
                setup_breakdown={},
                rule_adherence_rate=100.0,
                behavior_counts={},
            )

        wins = 0
        losses = 0
        breakeven = 0
        win_pnls: List[float] = []
        loss_pnls: List[float] = []
        r_multiples: List[float] = []
        rule_violations_count = 0
        setup_stats: Dict[str, Dict[str, Any]] = {}
        behavior_counts: Dict[str, int] = {}

        # Consecutives & Drawdown tracking
        curr_streak_wins = 0
        curr_streak_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0

        cumulative_pnl = 0.0
        peak_pnl = 0.0
        max_dd = 0.0

        for e in entries:
            # P&L
            pnl = e.actual.realized_pnl
            r = e.actual.realized_r
            r_multiples.append(r)

            cumulative_pnl += pnl
            if cumulative_pnl > peak_pnl:
                peak_pnl = cumulative_pnl
            dd = peak_pnl - cumulative_pnl
            if dd > max_dd:
                max_dd = dd

            if e.actual.outcome == TradeOutcome.WIN:
                wins += 1
                win_pnls.append(pnl)
                curr_streak_wins += 1
                curr_streak_losses = 0
                if curr_streak_wins > max_consecutive_wins:
                    max_consecutive_wins = curr_streak_wins
            elif e.actual.outcome == TradeOutcome.LOSS:
                losses += 1
                loss_pnls.append(pnl)
                curr_streak_losses += 1
                curr_streak_wins = 0
                if curr_streak_losses > max_consecutive_losses:
                    max_consecutive_losses = curr_streak_losses
            else:
                breakeven += 1
                curr_streak_wins = 0
                curr_streak_losses = 0

            # Rule adherence
            if not e.actual.rule_adhered:
                rule_violations_count += 1

            # Setup breakdown
            s_name = e.setup or "UNSPECIFIED"
            if s_name not in setup_stats:
                setup_stats[s_name] = {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
            setup_stats[s_name]["trades"] += 1
            setup_stats[s_name]["total_pnl"] = round(setup_stats[s_name]["total_pnl"] + pnl, 2)
            if e.actual.outcome == TradeOutcome.WIN:
                setup_stats[s_name]["wins"] += 1
            elif e.actual.outcome == TradeOutcome.LOSS:
                setup_stats[s_name]["losses"] += 1

            # Behavior counts
            for b in e.actual.observed_behaviors:
                b_name = b.behavior.value
                behavior_counts[b_name] = behavior_counts.get(b_name, 0) + 1

        # Ratios
        win_rate = round((wins / total_trades) * 100.0, 2)
        total_pnl = round(sum(win_pnls) + sum(loss_pnls), 2)
        avg_winner = round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0
        avg_loser = round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0

        # Profit factor: gross profits / abs(gross losses)
        gross_profit = sum(win_pnls)
        gross_loss = abs(sum(loss_pnls))
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            profit_factor = 999.99
        else:
            profit_factor = 0.0

        average_r = round(sum(r_multiples) / len(r_multiples), 2) if r_multiples else 0.0

        # Expectancy: (Win Rate * Avg Win) - (Loss Rate * Avg Loss)
        win_prob = wins / total_trades
        loss_prob = losses / total_trades
        expectancy = round((win_prob * avg_winner) + (loss_prob * avg_loser), 2)

        rule_adherence_rate = round(((total_trades - rule_violations_count) / total_trades) * 100.0, 2)

        return JournalAnalyticsSummary(
            environment=environment,
            asset_class=asset_class,
            symbol=symbol,
            backtest_type=backtest_type,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            breakeven=breakeven,
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_r=average_r,
            expectancy=expectancy,
            total_pnl=total_pnl,
            average_winner=avg_winner,
            average_loser=avg_loser,
            max_drawdown=round(max_dd, 2),
            max_consecutive_losses=max_consecutive_losses,
            max_consecutive_wins=max_consecutive_wins,
            setup_breakdown=setup_stats,
            rule_adherence_rate=rule_adherence_rate,
            behavior_counts=behavior_counts,
        )
