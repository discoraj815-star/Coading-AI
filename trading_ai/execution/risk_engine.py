"""Risk Engine: Deterministic position sizing, RR validation, engagement modes, and risk limits."""

import logging
import math
from typing import Dict, List, Optional, Tuple

from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.schemas.enums import DisciplineViolationType, EngagementMode
from trading_ai.schemas.execution import ExecutionPolicy, Position, RiskAssessment
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class RiskEngine:
    """
    Institutional Hedge Fund Risk Engine.

    Hard Rules:
      1. No valid SL -> NO TRADE.
      2. Minimum acceptable RR = 1:2 (default), preferred RR = 1:3.
      3. Deterministic position sizing based on account equity, stop distance, and risk %.
      4. Engagement Mode governs sizing:
         - DEFENSIVE: max 25% base size.
         - RISK_OFF: 0% / WAIT.
         - CALCULATED_AGGRESSION: 100% base size.
         - PYRAMIDING: Winning positions only with protected SL; NEVER average down!
      5. Reuses Step 7 TradingDisciplineEngine for user-defined limits and daily losses.
    """

    def __init__(self, repository: Repository, discipline_engine: TradingDisciplineEngine):
        self._repo = repository
        self._discipline_engine = discipline_engine

    def assess_risk_and_size(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: Optional[float],
        target_price: Optional[float],
        account_equity: float,
        policy: ExecutionPolicy,
        engagement_mode: EngagementMode = EngagementMode.DEFENSIVE,
        existing_positions: Optional[List[Position]] = None,
        user_id: str = "default_user",
    ) -> RiskAssessment:
        """
        Perform complete mathematical risk calculation, position sizing, and RR validation.
        """
        warnings: List[str] = []
        is_risk_acceptable = True
        is_rr_acceptable = True

        # Rule 1: No valid SL -> NO TRADE
        if stop_loss is None or stop_loss <= 0:
            return RiskAssessment(
                account_equity=account_equity,
                risk_percentage=0.0,
                risk_amount=0.0,
                entry_price=entry_price,
                stop_loss=0.0,
                stop_distance=0.0,
                stop_distance_pct=0.0,
                target_price=target_price or 0.0,
                reward_distance=0.0,
                risk_reward_ratio=0.0,
                calculated_quantity=0.0,
                rounded_quantity=0.0,
                engagement_mode=engagement_mode,
                sizing_factor=0.0,
                is_rr_acceptable=False,
                is_risk_acceptable=False,
                warnings=["NO_VALID_STOP_LOSS: A hard, non-negotiable stop loss is required."],
            )

        is_buy = direction.upper() in ("BUY", "LONG", "BULLISH")
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_pct = (stop_distance / entry_price) * 100.0 if entry_price > 0 else 0.0

        # Verify stop loss direction
        if is_buy and stop_loss >= entry_price:
            warnings.append("INVALID_SL_DIRECTION: Long stop loss must be strictly below entry price.")
            is_risk_acceptable = False
        elif not is_buy and stop_loss <= entry_price:
            warnings.append("INVALID_SL_DIRECTION: Short stop loss must be strictly above entry price.")
            is_risk_acceptable = False

        if stop_distance <= 0:
            warnings.append("ZERO_STOP_DISTANCE: Stop distance cannot be zero.")
            is_risk_acceptable = False

        # Calculate Reward and RR
        if target_price and target_price > 0:
            reward_distance = abs(target_price - entry_price)
            if is_buy and target_price <= entry_price:
                warnings.append("INVALID_TARGET_DIRECTION: Long target must be above entry price.")
                is_rr_acceptable = False
            elif not is_buy and target_price >= entry_price:
                warnings.append("INVALID_TARGET_DIRECTION: Short target must be below entry price.")
                is_rr_acceptable = False

            rr_ratio = (reward_distance / stop_distance) if stop_distance > 0 else 0.0
            if rr_ratio < policy.min_rr_ratio:
                warnings.append(
                    f"RR_BELOW_MINIMUM: Proposed RR 1:{rr_ratio:.2f} is below configured minimum "
                    f"1:{policy.min_rr_ratio:.2f}."
                )
                is_rr_acceptable = False
        else:
            # Fallback to configured target (e.g. 1:2)
            reward_distance = stop_distance * policy.min_rr_ratio
            target_price = entry_price + reward_distance if is_buy else entry_price - reward_distance
            rr_ratio = policy.min_rr_ratio

        # Determine Engagement Mode Sizing Factor
        sizing_factor = 1.0
        if engagement_mode == EngagementMode.DEFENSIVE:
            sizing_factor = policy.defensive_size_pct
        elif engagement_mode == EngagementMode.RISK_OFF:
            sizing_factor = 0.0
            warnings.append("ENGAGEMENT_RISK_OFF: Trading paused due to high-impact event/macro uncertainty.")
            is_risk_acceptable = False
        elif engagement_mode == EngagementMode.CALCULATED_AGGRESSION:
            sizing_factor = 1.0
        elif engagement_mode == EngagementMode.PYRAMIDING:
            # Rule: Pyramiding allowed ONLY on profitable positions with protected SL
            can_pyramid, pyramid_warnings = self.validate_pyramiding_eligibility(symbol, direction, existing_positions)
            if not can_pyramid:
                sizing_factor = 0.0
                is_risk_acceptable = False
                warnings.extend(pyramid_warnings)
            else:
                sizing_factor = 0.5  # Add 50% of base risk to existing winner

        # Base Risk Percentage
        base_risk_pct = policy.max_auto_risk_pct if policy.auto_execution_enabled else 1.0
        effective_risk_pct = base_risk_pct * sizing_factor
        risk_amount = (account_equity * (effective_risk_pct / 100.0)) if account_equity > 0 else 0.0

        # Calculate Position Size (Quantity)
        calculated_quantity = 0.0
        rounded_quantity = 0.0
        if stop_distance > 0 and risk_amount > 0:
            calculated_quantity = risk_amount / stop_distance
            # Rounding to 2 decimal places (standard contract/lot granularity)
            rounded_quantity = round(calculated_quantity, 2)
            if rounded_quantity <= 0 and calculated_quantity > 0:
                rounded_quantity = 0.01

        # Check against Step 7 Discipline Engine
        disp_state, disp_violations = self._discipline_engine.evaluate_discipline_status(user_id)
        if disp_violations:
            for v in disp_violations:
                warnings.append(f"DISCIPLINE_RULE_BREACH: {v}")
                if "DAILY_LOSS" in v or "MAX_TRADES" in v or "COOLDOWN" in v:
                    is_risk_acceptable = False

        return RiskAssessment(
            account_equity=account_equity,
            risk_percentage=effective_risk_pct,
            risk_amount=risk_amount,
            entry_price=entry_price,
            stop_loss=stop_loss,
            stop_distance=stop_distance,
            stop_distance_pct=stop_distance_pct,
            target_price=target_price,
            reward_distance=reward_distance,
            risk_reward_ratio=round(rr_ratio, 2),
            calculated_quantity=calculated_quantity,
            rounded_quantity=rounded_quantity,
            engagement_mode=engagement_mode,
            sizing_factor=sizing_factor,
            is_rr_acceptable=is_rr_acceptable,
            is_risk_acceptable=is_risk_acceptable and is_rr_acceptable and (rounded_quantity > 0 or sizing_factor == 0),
            warnings=warnings,
        )

    def validate_pyramiding_eligibility(
        self,
        symbol: str,
        direction: str,
        existing_positions: Optional[List[Position]],
    ) -> Tuple[bool, List[str]]:
        """
        Enforce strict anti-averaging down rules:
        - Must have an existing OPEN position in the same symbol and direction.
        - Existing position must be in profit (P&L > 0).
        - Stop loss on existing position must be at least moved to entry (risk protected).
        """
        if not existing_positions:
            return False, ["PYRAMIDING_BLOCKED: No existing active position found for pyramiding."]

        matching = [p for p in existing_positions if p.symbol == symbol and p.status.value == "OPEN"]
        if not matching:
            return False, [f"PYRAMIDING_BLOCKED: No open position found for {symbol}."]

        primary_pos = matching[0]

        # Direction check
        if primary_pos.direction.upper() != direction.upper():
            return False, [
                f"PYRAMIDING_BLOCKED: Cannot pyramid opposite direction position ({primary_pos.direction} vs {direction})."
            ]

        # Profitability check: NEVER average down a losing position!
        if primary_pos.unrealized_pnl <= 0:
            return False, [
                f"NO_PYRAMIDING_ON_LOSING_POSITION: Existing position P&L is {primary_pos.unrealized_pnl:.2f}. "
                "Averaging down or pyramiding into losses is strictly prohibited."
            ]

        # Protected SL check
        if not primary_pos.sl_moved_to_entry and primary_pos.unrealized_pnl < (primary_pos.entry_price * 0.005):
            return False, [
                "PYRAMIDING_BLOCKED: Existing position stop loss must be moved to entry or protected before adding size."
            ]

        return True, []
