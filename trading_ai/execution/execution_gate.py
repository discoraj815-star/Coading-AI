"""Execution Gate: Unified gatekeeper for Manual and Auto trade execution."""

import hashlib
import logging
from typing import List, Optional, Tuple

from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.schemas.enums import (
    EngagementMode,
    ExecutionEnvironment,
    ExecutionMode,
    OrderType,
    ProposalLifecycleState,
)
from trading_ai.schemas.execution import (
    ExecutionIntent,
    ExecutionPolicy,
    TradeProposal,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class ExecutionGate:
    """
    Institutional Execution Gate.

    Hard Rules:
      1. Manual and Auto execution MUST use the exact same validation pipeline.
      2. AUTO execution must NEVER bypass risk validation.
      3. No valid SL -> NO TRADE.
      4. Insufficient independent confluence -> NO TRADE.
      5. RR below minimum -> NO TRADE.
      6. Duplicate execution intent -> BLOCKED.
      7. AI reasoning cannot directly submit orders; it must produce an authorized ExecutionIntent here.
    """

    def __init__(self, repository: Repository, discipline_engine: TradingDisciplineEngine):
        self._repo = repository
        self._discipline_engine = discipline_engine

    def validate_and_authorize_execution(
        self,
        proposal: TradeProposal,
        policy: ExecutionPolicy,
        user_confirmed: bool = False,
        user_id: str = "default_user",
    ) -> Tuple[bool, Optional[ExecutionIntent], List[str]]:
        """
        Evaluate TradeProposal against execution rules and generate ExecutionIntent if authorized.
        """
        rejection_reasons: List[str] = []

        # 1. State check
        if proposal.lifecycle_state in (
            ProposalLifecycleState.FILLED,
            ProposalLifecycleState.CLOSED,
            ProposalLifecycleState.CANCELLED,
            ProposalLifecycleState.EXPIRED,
            ProposalLifecycleState.REJECTED,
        ):
            return False, None, [f"INVALID_STATE: Proposal is already in terminal state {proposal.lifecycle_state}."]

        # 2. Hard Stop Loss requirement
        if not proposal.hard_stop_loss or proposal.hard_stop_loss <= 0:
            rejection_reasons.append("NO_VALID_STOP_LOSS: Hard stop loss is missing or invalid.")

        # 3. Trade Thesis requirement
        if not proposal.trade_thesis or not proposal.trade_thesis.summary:
            rejection_reasons.append("NO_VALID_TRADE_THESIS: Trade proposal lacks an approved thesis.")

        # 4. Independent Confluence check
        if len(proposal.confluences) < policy.min_confluence_count:
            rejection_reasons.append(
                f"INSUFFICIENT_CONFLUENCE: Proposal has {len(proposal.confluences)} confluence(s); "
                f"minimum required is {policy.min_confluence_count}."
            )

        # 5. Risk and RR check
        if not proposal.risk_assessment:
            rejection_reasons.append("MISSING_RISK_ASSESSMENT: Risk calculation has not been performed.")
        else:
            if not proposal.risk_assessment.is_risk_acceptable:
                rejection_reasons.extend(proposal.risk_assessment.warnings)
            if not proposal.risk_assessment.is_rr_acceptable:
                rejection_reasons.append(
                    f"RR_UNACCEPTABLE: Risk-to-reward ratio 1:{proposal.risk_assessment.risk_reward_ratio} "
                    f"is below required minimum 1:{policy.min_rr_ratio}."
                )

        # 6. Engagement Mode check
        if proposal.engagement_mode == EngagementMode.RISK_OFF:
            rejection_reasons.append("ENGAGEMENT_RISK_OFF: High-impact event risk blocks order submission.")

        # 7. Step 7 Trading Discipline check
        disp_state, disp_violations = self._discipline_engine.evaluate_discipline_status(user_id)
        if disp_violations:
            for v in disp_violations:
                if any(critical in v for critical in ("DAILY_LOSS", "MAX_TRADES", "COOLDOWN", "OVERTRADING")):
                    rejection_reasons.append(f"DISCIPLINE_GATE_BLOCK: {v}")

        # 8. Execution Environment & Authorization Check (B3 & B8)
        proposal.execution_environment = policy.execution_environment

        if policy.execution_environment == ExecutionEnvironment.LIVE:
            if not policy.live_execution_enabled:
                rejection_reasons.append("LIVE_EXECUTION_DISABLED: Live execution is disabled in execution policy.")

        if policy.execution_mode == ExecutionMode.MANUAL:
            if not user_confirmed and not proposal.is_approved:
                rejection_reasons.append(
                    "USER_CONFIRMATION_REQUIRED: Manual execution policy mandates explicit user approval."
                )
        elif policy.execution_mode == ExecutionMode.AUTO:
            if not policy.auto_execution_enabled:
                rejection_reasons.append("AUTO_EXECUTION_DISABLED: Automated trading is currently disabled in policy.")
            if policy.allowed_symbols and proposal.symbol not in policy.allowed_symbols:
                rejection_reasons.append(f"SYMBOL_NOT_WHITELISTED: {proposal.symbol} is not permitted for auto-execution.")
            if proposal.risk_assessment and proposal.risk_assessment.risk_percentage > policy.max_auto_risk_pct:
                rejection_reasons.append(
                    f"AUTO_RISK_EXCEEDED: Proposed risk {proposal.risk_assessment.risk_percentage:.2f}% "
                    f"exceeds auto ceiling {policy.max_auto_risk_pct:.2f}%."
                )

        # If any validation failed, reject execution or mark awaiting approval
        if rejection_reasons:
            if all("USER_CONFIRMATION_REQUIRED" in r for r in rejection_reasons):
                proposal.lifecycle_state = ProposalLifecycleState.AWAITING_APPROVAL
                proposal.rejection_reason = None
            else:
                proposal.lifecycle_state = ProposalLifecycleState.REJECTED
                proposal.rejection_reason = " | ".join(rejection_reasons)
            self._repo.save_trade_proposal(proposal)
            return False, None, rejection_reasons

        # 9. Anti-Duplicate Token Check (includes environment to prevent cross-env clash)
        token_src = (
            f"{proposal.proposal_id}:{proposal.symbol}:{proposal.direction}:"
            f"{proposal.entry_price}:{proposal.hard_stop_loss}:{policy.execution_environment.value}"
        )
        anti_dup_token = hashlib.sha256(token_src.encode("utf-8")).hexdigest()

        existing_intent = self._repo.get_execution_intent_by_token(anti_dup_token)
        if existing_intent:
            return False, None, [f"DUPLICATE_EXECUTION_BLOCKED: Intent already exists for proposal {proposal.proposal_id}."]

        # Calculate final authorized quantity
        qty = proposal.proposed_quantity
        if qty <= 0 and proposal.risk_assessment:
            qty = proposal.risk_assessment.rounded_quantity

        if qty <= 0:
            return False, None, ["INVALID_QUANTITY: Sized trade quantity must be greater than zero."]

        # All checks passed: create ExecutionIntent
        direction_clean = "BUY" if proposal.direction.upper() in ("BUY", "LONG", "BULLISH") else "SELL"
        intent = ExecutionIntent(
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            execution_mode=policy.execution_mode,
            execution_environment=policy.execution_environment,
            order_type=OrderType.MARKET,
            direction=direction_clean,
            quantity=qty,
            entry_price=proposal.entry_price,
            stop_loss=proposal.hard_stop_loss,
            take_profit=proposal.final_target,
            tp1=proposal.tp1,
            status="AUTHORIZED",
            anti_duplicate_token=anti_dup_token,
        )

        proposal.lifecycle_state = ProposalLifecycleState.APPROVED
        proposal.is_approved = True
        self._repo.save_trade_proposal(proposal)
        self._repo.save_execution_intent(intent)

        return True, intent, []
