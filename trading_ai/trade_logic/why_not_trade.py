"""Mandatory 'Why Not Trade?' Negative Validation Engine for Step 14.

Systematically hunts for disqualifying conditions, contradictions, and missing confirmations
before any trade proposal can be synthesized.

If a blocking reason exists:
- should_block = True
- Trade proposal generation is aborted.
- Output is strictly NO TRADE / INSUFFICIENT_CONFLUENCE / INVALIDATED.
"""
from typing import List, Optional

from trading_ai.schemas.trade_logic import WhyNotTradeResult


class WhyNotTradeEngine:
    """Rigorous negative validation layer searching for reasons NOT to trade."""

    def __init__(self, min_acceptable_rr: float = 1.5):
        self.min_acceptable_rr = min_acceptable_rr

    def evaluate_reasons_not_to_trade(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        htf_bias: str = "BULLISH",
        is_poi_valid: bool = True,
        is_confirmation_present: bool = True,
        news_risk_level: str = "NO_RISK",
        active_exceptions: Optional[List[str]] = None,
        journal_warnings: Optional[List[str]] = None,
        daily_loss_limit_hit: bool = False,
    ) -> WhyNotTradeResult:
        """Run negative validation checks across technical, MTF, news, discipline, and risk domains."""
        blocking_reasons: List[str] = []
        warnings: List[str] = []
        checks_count = 0

        # Check 1: HTF Alignment
        checks_count += 1
        has_htf_contradiction = False
        dir_upper = direction.upper()
        if (dir_upper == "BUY" and htf_bias.upper() in ("BEARISH", "STRONG_BEARISH")) or (
            dir_upper == "SELL" and htf_bias.upper() in ("BULLISH", "STRONG_BULLISH")
        ):
            has_htf_contradiction = True
            blocking_reasons.append(
                f"HTF Bias Contradiction: Proposed {dir_upper} directly opposes {htf_bias} higher-timeframe trend."
            )

        # Check 2: POI Validity
        checks_count += 1
        if not is_poi_valid:
            blocking_reasons.append("POI Invalidated: Price has traded through and violated the structural zone.")

        # Check 3: Mandatory Confirmation Trigger
        checks_count += 1
        has_missing_conf = False
        if not is_confirmation_present:
            has_missing_conf = True
            blocking_reasons.append("Missing Confirmation: Lower-timeframe displacement or CHOCH has not printed.")

        # Check 4: Imminent News Risk
        checks_count += 1
        has_news = False
        if news_risk_level.upper() in ("HIGH_RISK", "CRITICAL_RISK", "IMMINENT_EVENT"):
            has_news = True
            blocking_reasons.append(
                f"High-Impact News Risk: Economic release imminent ({news_risk_level}). Slippage and spread hazard."
            )

        # Check 5: Mathematical R:R Viability
        checks_count += 1
        if entry_price > 0 and stop_loss > 0 and tp1 > 0:
            risk = abs(entry_price - stop_loss)
            reward = abs(tp1 - entry_price)
            if risk > 0:
                rr = reward / risk
                if rr < self.min_acceptable_rr:
                    blocking_reasons.append(
                        f"Unacceptable Risk/Reward: Calculated R:R ({rr:.2f}) is below minimum policy ({self.min_acceptable_rr:.2f})."
                    )
            else:
                blocking_reasons.append("Invalid Entry/SL: Zero risk distance between entry and stop loss.")
        else:
            blocking_reasons.append("Incomplete trade geometry: Entry, Stop Loss, or Target is zero/missing.")

        # Check 6: Directional Stop Loss Geometry
        checks_count += 1
        if dir_upper == "BUY" and stop_loss >= entry_price:
            blocking_reasons.append(f"Geometry Error: Long stop loss ({stop_loss}) must be below entry ({entry_price}).")
        elif dir_upper == "SELL" and stop_loss <= entry_price:
            blocking_reasons.append(f"Geometry Error: Short stop loss ({stop_loss}) must be above entry ({entry_price}).")

        # Check 7: Mentorship / Explicit Rule Exceptions
        checks_count += 1
        has_mentorship_exc = False
        if active_exceptions:
            for exc in active_exceptions:
                has_mentorship_exc = True
                blocking_reasons.append(f"Active Rule Exception: {exc}")

        # Check 8: Historical Journal Warnings
        checks_count += 1
        has_journal_warn = False
        if journal_warnings:
            for jw in journal_warnings:
                has_journal_warn = True
                warnings.append(f"Historical Journal Precedent: {jw}")

        # Check 9: Daily Risk Limit Protection
        checks_count += 1
        if daily_loss_limit_hit:
            blocking_reasons.append("Risk Policy Block: Daily loss limit or maximum drawdown limit reached.")

        should_block = len(blocking_reasons) > 0

        return WhyNotTradeResult(
            should_block=should_block,
            blocking_reasons=blocking_reasons,
            non_blocking_warnings=warnings,
            evaluated_checks_count=checks_count,
            has_htf_contradiction=has_htf_contradiction,
            has_news_risk=has_news,
            has_missing_confirmation=has_missing_conf,
            has_journal_warning=has_journal_warn,
            has_mentorship_exception=has_mentorship_exc,
        )

    def evaluate_negative_conditions(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "15m",
        direction: str = "BUY",
        entry_price: float = 1.0850,
        stop_loss: float = 1.0800,
        tp1: float = 1.0950,
        high_timeframe_bias: str = "BULLISH",
        confluence_state: Optional[str] = None,
        has_invalidation: bool = False,
        has_pending_high_impact_news: bool = False,
        risk_limit_exceeded: bool = False,
        missing_confirmations: Optional[List[str]] = None,
        active_exceptions: Optional[List[str]] = None,
        journal_warnings: Optional[List[str]] = None,
    ) -> WhyNotTradeResult:
        """Convenience method wrapping negative validation with default geometries."""
        is_poi_valid = not has_invalidation
        is_confirmation_present = missing_confirmations is None or len(missing_confirmations) == 0
        news_risk = "HIGH_RISK" if has_pending_high_impact_news else "NO_RISK"

        # Adjust default SL geometry for SELL direction if defaults were passed
        if direction.upper() == "SELL" and stop_loss <= entry_price:
            stop_loss = entry_price + 0.0050
            tp1 = entry_price - 0.0100

        res = self.evaluate_reasons_not_to_trade(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            htf_bias=high_timeframe_bias,
            is_poi_valid=is_poi_valid,
            is_confirmation_present=is_confirmation_present,
            news_risk_level=news_risk,
            active_exceptions=active_exceptions,
            journal_warnings=journal_warnings,
            daily_loss_limit_hit=risk_limit_exceeded,
        )

        if confluence_state in ("CONFLICTED", "NO_SETUP", "INSUFFICIENT_CONFLUENCE"):
            res.blocking_reasons.append(f"Confluence state is {confluence_state}")
            res.should_block = True

        return res

