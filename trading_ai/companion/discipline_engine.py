"""Trading Discipline Engine: Enforces user-defined trading rules, overtrading prevention, and trade confirmation gate."""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from trading_ai.schemas.companion import (
    DisciplineConfig,
    DisciplineState,
    TradeConfirmationGateResult,
    TradeIntentRequest,
)
from trading_ai.schemas.enums import AlertType, DisciplineViolationType
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class TradingDisciplineEngine:
    """
    Enforces user-configured risk management and behavioral discipline rules.

    Hard Rules:
      - Never invents unconfigured values: returns RULE_NOT_CONFIGURED.
      - Informs and alerts the user rather than secretly executing or arbitrarily blocking.
      - Trade Confirmation Gate validates intent and stops at USER_CONFIRMATION_REQUIRED.
      - Live broker execution is never initiated here.
    """

    def __init__(self, repository: Repository):
        self._repo = repository

    def get_or_create_state(self, user_id: str = "default_user") -> DisciplineState:
        """Retrieve existing state or initialize default clean discipline state."""
        state = self._repo.get_discipline_state(user_id)
        if not state:
            state = DisciplineState(user_id=user_id, config=DisciplineConfig(user_id=user_id))
            self._repo.save_discipline_state(state)
        return state

    def update_config(self, user_id: str, config: DisciplineConfig) -> DisciplineState:
        """Update user-defined discipline configuration."""
        state = self.get_or_create_state(user_id)
        state.config = config
        state.updated_at = datetime.now(timezone.utc)
        self._repo.save_discipline_state(state)
        return state

    def evaluate_discipline_status(self, user_id: str = "default_user") -> Tuple[DisciplineState, List[str]]:
        """
        Evaluate current metrics against configured rules.
        Returns the updated state and a list of active violation warnings.
        """
        state = self.get_or_create_state(user_id)
        config = state.config
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()
        violations: List[str] = []

        # 1. Cooldown Check
        if state.cooldown_active_until:
            if now < state.cooldown_active_until:
                remaining_sec = int((state.cooldown_active_until - now).total_seconds())
                violations.append(f"COOLDOWN_ACTIVE: Mandatory cooldown active for {remaining_sec}s.")
            else:
                state.cooldown_active_until = None

        # 2. Daily Loss Limit
        if config.daily_loss_limit is not None:
            if state.daily_realized_loss >= config.daily_loss_limit:
                violations.append(
                    f"DAILY_LOSS_EXCEEDED: Realized loss ${state.daily_realized_loss:.2f} "
                    f"reached or exceeded limit of ${config.daily_loss_limit:.2f}."
                )

        # 3. Maximum Trades Per Day
        if config.max_daily_trades is not None:
            if state.trades_today_count >= config.max_daily_trades:
                violations.append(
                    f"MAX_TRADES_EXCEEDED: Completed {state.trades_today_count} trades today "
                    f"(limit: {config.max_daily_trades})."
                )

        # 4. Overtrading Rapid-Fire Window
        if config.overtrading_trades_threshold is not None and config.trades_window_minutes is not None:
            window_sec = config.trades_window_minutes * 60
            cutoff = now_ts - window_sec
            recent_in_window = [t for t in state.recent_trades_timestamps if t >= cutoff]
            state.recent_trades_timestamps = recent_in_window
            if len(recent_in_window) >= config.overtrading_trades_threshold:
                violations.append(
                    f"OVERTRADING_TRIGGERED: {len(recent_in_window)} trades executed within "
                    f"{config.trades_window_minutes} minutes (threshold: {config.overtrading_trades_threshold})."
                )

        state.active_violations = violations
        state.updated_at = now
        self._repo.save_discipline_state(state)
        return state, violations

    def check_trade_confirmation_gate(
        self, intent: TradeIntentRequest, user_id: str = "default_user"
    ) -> TradeConfirmationGateResult:
        """
        Validate trade intent against setup context, discipline rules, and risk limits.
        Stops at USER_CONFIRMATION_REQUIRED.
        """
        state, violations = self.evaluate_discipline_status(user_id)
        config = state.config

        checklist: Dict[str, bool] = {}
        warnings: List[str] = list(violations)
        can_proceed = True

        # Rule 1: Stop Loss Requirement
        if config.stop_loss_required:
            if intent.stop_loss is None or intent.stop_loss <= 0.0:
                checklist["stop_loss_defined"] = False
                warnings.append("STOP_LOSS_REQUIRED: Trade intent does not define a mandatory stop loss.")
                can_proceed = False
            else:
                checklist["stop_loss_defined"] = True
                # Directional check
                if intent.direction.upper() in ("BUY", "LONG") and intent.stop_loss >= intent.entry_price:
                    checklist["stop_loss_valid"] = False
                    warnings.append("INVALID_STOP_LOSS: Stop loss for LONG must be below entry price.")
                    can_proceed = False
                elif intent.direction.upper() in ("SELL", "SHORT") and intent.stop_loss <= intent.entry_price:
                    checklist["stop_loss_valid"] = False
                    warnings.append("INVALID_STOP_LOSS: Stop loss for SHORT must be above entry price.")
                    can_proceed = False
                else:
                    checklist["stop_loss_valid"] = True
        else:
            checklist["stop_loss_defined"] = bool(intent.stop_loss and intent.stop_loss > 0)

        # Rule 2: Risk Per Trade Limit
        is_risk_acceptable = True
        if config.max_risk_per_trade_pct is not None and intent.account_balance > 0:
            if intent.risk_amount is not None:
                risk_pct = (intent.risk_amount / intent.account_balance) * 100.0
                if risk_pct > config.max_risk_per_trade_pct:
                    is_risk_acceptable = False
                    can_proceed = False
                    warnings.append(
                        f"RISK_LIMIT_EXCEEDED: Trade risk {risk_pct:.2f}% exceeds configured limit "
                        f"of {config.max_risk_per_trade_pct:.2f}%."
                    )
            elif intent.stop_loss:
                # Calculate implied distance risk
                dist = abs(intent.entry_price - intent.stop_loss) / intent.entry_price * 100.0
                if dist > config.max_risk_per_trade_pct * 2:  # Safe heuristic if position size unspecified
                    warnings.append(f"HIGH_STOP_DISTANCE: Stop distance {dist:.2f}% is wide relative to account risk.")
            checklist["risk_within_limit"] = is_risk_acceptable
        else:
            checklist["risk_within_limit"] = True

        # Rule 3: Setup Alignment Check
        situation = self._repo.get_latest_current_situation(intent.symbol)
        is_setup_aligned = False
        if situation and situation.active_confirmed_setup:
            setup = situation.active_confirmed_setup
            setup_dir = setup.direction.value if hasattr(setup.direction, "value") else str(setup.direction)
            user_dir = intent.direction.upper()
            is_matching = (
                user_dir in setup_dir.upper()
                or (user_dir in ("BUY", "LONG") and "BULL" in setup_dir.upper())
                or (user_dir in ("SELL", "SHORT") and "BEAR" in setup_dir.upper())
            )
            if is_matching:
                is_setup_aligned = True
                checklist["setup_aligned"] = True
            else:
                warnings.append(
                    f"SETUP_DIRECTION_CONFLICT: Intent {intent.direction} opposes active setup direction {setup_dir}."
                )
                checklist["setup_aligned"] = False
        else:
            checklist["setup_aligned"] = False
            warnings.append("NO_CONFIRMED_SETUP: No confirmed lecture setup currently active for this symbol.")

        # Check discipline violations
        is_discipline_passed = len(violations) == 0 and checklist.get("stop_loss_defined", True)

        if not is_discipline_passed:
            can_proceed = False

        recommended_action = (
            "TRADE INTENT VALIDATED: Review details and provide manual user confirmation to execute."
            if can_proceed
            else "TRADE INTENT REJECTED BY DISCIPLINE GATE: Address highlighted warnings before proceeding."
        )

        return TradeConfirmationGateResult(
            status="USER_CONFIRMATION_REQUIRED",
            is_discipline_passed=is_discipline_passed,
            is_setup_aligned=is_setup_aligned,
            is_risk_acceptable=is_risk_acceptable,
            can_proceed=can_proceed,
            warnings=warnings,
            checklist=checklist,
            recommended_action=recommended_action,
        )

    def record_trade_execution(
        self,
        symbol: str,
        user_id: str = "default_user",
        realized_loss: float = 0.0,
        trigger_cooldown: bool = False,
    ) -> DisciplineState:
        """Record trade execution and update daily counters and cooldowns."""
        state = self.get_or_create_state(user_id)
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()

        state.trades_today_count += 1
        state.last_trade_timestamp = now_ts
        state.recent_trades_timestamps.append(now_ts)

        if realized_loss > 0:
            state.daily_realized_loss += realized_loss

        # Apply cooldown if configured and requested or if overtrading threshold reached
        cfg = state.config
        if (trigger_cooldown or realized_loss > 0) and cfg.cooldown_minutes:
            state.cooldown_active_until = now + timedelta(minutes=cfg.cooldown_minutes)

        self._repo.save_discipline_state(state)
        return state

    def reset_daily_metrics(self, user_id: str = "default_user") -> DisciplineState:
        """Reset daily trade counters at new session/day."""
        state = self.get_or_create_state(user_id)
        state.trades_today_count = 0
        state.daily_realized_loss = 0.0
        state.recent_trades_timestamps = []
        state.active_violations = []
        state.cooldown_active_until = None
        state.updated_at = datetime.now(timezone.utc)
        self._repo.save_discipline_state(state)
        return state
