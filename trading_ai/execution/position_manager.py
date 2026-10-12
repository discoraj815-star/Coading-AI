"""Position Manager: Lifecycle, TP1 partial close, SL-to-entry, trailing SL, and audit logging."""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.schemas.enums import (
    ExecutionEnvironment,
    PositionManagementAction,
    PositionStatus,
)
from trading_ai.schemas.execution import Position, PositionEvent
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages active trading positions, partial exits, and protective stop moves.

    Hard Rules:
      - Default TP1 = 1.5R: closes 50% of position and automatically moves remaining SL to Entry.
      - Final Target (TP2): closes remaining 50% position.
      - Stop loss hits: closes position and records realized loss with Step 7 Discipline Engine.
      - Never fabricates closed positions or fills without price action evidence.
      - All modifications emit persistent auditable PositionEvents.
    """

    def __init__(self, repository: Repository, discipline_engine: TradingDisciplineEngine):
        self._repo = repository
        self._discipline_engine = discipline_engine

    def update_position_on_price(
        self,
        position: Position,
        current_price: float,
        user_id: str = "default_user",
    ) -> Tuple[Position, List[PositionEvent]]:
        """
        Evaluate current price against SL, TP1, and Final Target.
        Executes trade management transitions automatically.
        """
        events: List[PositionEvent] = []
        if position.status != PositionStatus.OPEN and position.status != PositionStatus.PARTIALLY_CLOSED:
            return position, events

        is_buy = position.direction.upper() in ("BUY", "LONG")
        now = datetime.now(timezone.utc)

        # 1. Check Stop Loss Hit
        is_sl_hit = (current_price <= position.current_sl) if is_buy else (current_price >= position.current_sl)
        if is_sl_hit:
            exit_price = position.current_sl
            pnl_per_unit = (exit_price - position.entry_price) if is_buy else (position.entry_price - exit_price)
            loss_amount = pnl_per_unit * position.current_quantity

            position.realized_pnl += loss_amount
            position.unrealized_pnl = 0.0
            position.current_quantity = 0.0
            position.status = PositionStatus.CLOSED
            position.closed_at = now
            position.updated_at = now

            event = PositionEvent(
                position_id=position.position_id,
                action=PositionManagementAction.POSITION_CLOSED,
                price=exit_price,
                quantity_affected=position.current_quantity,
                realized_pnl_delta=loss_amount,
                notes=f"Stop loss triggered at {exit_price:.5f}. Position fully closed.",
            )
            events.append(event)
            self._repo.save_position_event(event)
            self._repo.save_position(position)

            # Record with Discipline Engine strictly for LIVE trades (B7: Paper P&L never alters live discipline/risk state)
            if position.execution_environment == ExecutionEnvironment.LIVE:
                realized_loss_val = abs(loss_amount) if loss_amount < 0 else 0.0
                self._discipline_engine.record_trade_execution(
                    symbol=position.symbol,
                    user_id=user_id,
                    realized_loss=realized_loss_val,
                    trigger_cooldown=(realized_loss_val > 0),
                )
            return position, events

        # 2. Check TP1 (1.5R) Reached
        is_tp1_hit = (current_price >= position.tp1) if is_buy else (current_price <= position.tp1)
        if is_tp1_hit and not position.tp1_reached:
            # Book 50% partial
            close_qty = round(position.initial_quantity * 0.5, 2)
            if close_qty > position.current_quantity:
                close_qty = position.current_quantity

            pnl_per_unit = (position.tp1 - position.entry_price) if is_buy else (position.entry_price - position.tp1)
            realized_delta = pnl_per_unit * close_qty

            position.current_quantity = round(position.current_quantity - close_qty, 2)
            position.realized_pnl += realized_delta
            position.tp1_reached = True
            position.status = PositionStatus.PARTIALLY_CLOSED

            event_tp1 = PositionEvent(
                position_id=position.position_id,
                action=PositionManagementAction.TP1_REACHED_PARTIAL_BOOKED,
                price=position.tp1,
                quantity_affected=close_qty,
                realized_pnl_delta=realized_delta,
                notes=f"TP1 achieved at {position.tp1:.5f}. Booked 50% partial ({close_qty} units).",
            )
            events.append(event_tp1)
            self._repo.save_position_event(event_tp1)

            # Automatically move remaining SL to Entry
            old_sl = position.current_sl
            position.current_sl = position.entry_price
            position.sl_moved_to_entry = True

            event_sl = PositionEvent(
                position_id=position.position_id,
                action=PositionManagementAction.SL_MOVED_TO_ENTRY,
                price=position.entry_price,
                quantity_affected=position.current_quantity,
                realized_pnl_delta=0.0,
                notes=f"Stop loss moved from {old_sl:.5f} to Entry {position.entry_price:.5f} (Risk-Free).",
            )
            events.append(event_sl)
            self._repo.save_position_event(event_sl)

        # 3. Check Final Target Reached
        is_final_hit = (current_price >= position.final_target) if is_buy else (current_price <= position.final_target)
        if is_final_hit and position.current_quantity > 0:
            remaining_qty = position.current_quantity
            pnl_per_unit = (position.final_target - position.entry_price) if is_buy else (position.entry_price - position.final_target)
            realized_delta = pnl_per_unit * remaining_qty

            position.realized_pnl += realized_delta
            position.unrealized_pnl = 0.0
            position.current_quantity = 0.0
            position.status = PositionStatus.CLOSED
            position.closed_at = now
            position.updated_at = now

            event_final = PositionEvent(
                position_id=position.position_id,
                action=PositionManagementAction.FINAL_TARGET_REACHED,
                price=position.final_target,
                quantity_affected=remaining_qty,
                realized_pnl_delta=realized_delta,
                notes=f"Final target reached at {position.final_target:.5f}. Full position closed in profit.",
            )
            events.append(event_final)
            self._repo.save_position_event(event_final)
            self._repo.save_position(position)

            # Record successful trade with Discipline Engine strictly for LIVE trades
            if position.execution_environment == ExecutionEnvironment.LIVE:
                self._discipline_engine.record_trade_execution(
                    symbol=position.symbol,
                    user_id=user_id,
                    realized_loss=0.0,
                )
            return position, events

        # Update unrealized P&L for remaining open units
        if position.current_quantity > 0:
            unit_gain = (current_price - position.entry_price) if is_buy else (position.entry_price - current_price)
            position.unrealized_pnl = round(unit_gain * position.current_quantity, 2)
            position.updated_at = now

        self._repo.save_position(position)
        return position, events

    def manual_update_sl(
        self,
        position_id: str,
        new_sl: float,
        reason: str = "Manual Trailing SL Adjustment",
    ) -> Tuple[Optional[Position], Optional[str]]:
        """Allow trader to update stop loss safely."""
        pos = self._repo.get_position(position_id)
        if not pos:
            return None, "Position not found."

        if pos.status == PositionStatus.CLOSED:
            return None, "Cannot update SL of closed position."

        is_buy = pos.direction.upper() in ("BUY", "LONG")
        # Validate that new SL is not worse than entry if already moved to entry
        if pos.sl_moved_to_entry:
            if is_buy and new_sl < pos.entry_price:
                return None, "Cannot move SL below entry once risk-free status is achieved."
            if not is_buy and new_sl > pos.entry_price:
                return None, "Cannot move SL above entry once risk-free status is achieved."

        old_sl = pos.current_sl
        pos.current_sl = new_sl
        pos.updated_at = datetime.now(timezone.utc)
        self._repo.save_position(pos)

        event = PositionEvent(
            position_id=pos.position_id,
            action=PositionManagementAction.TRAILING_SL_UPDATED,
            price=new_sl,
            quantity_affected=pos.current_quantity,
            realized_pnl_delta=0.0,
            notes=f"{reason}: SL updated from {old_sl:.5f} to {new_sl:.5f}.",
        )
        self._repo.save_position_event(event)
        return pos, None

    def manual_close_position(
        self,
        position_id: str,
        current_price: float,
        user_id: str = "default_user",
        reason: str = "Manual Trader Discretion Exit",
    ) -> Tuple[Optional[Position], Optional[str]]:
        """Manually exit an open position."""
        pos = self._repo.get_position(position_id)
        if not pos:
            return None, "Position not found."

        if pos.status == PositionStatus.CLOSED:
            return None, "Position already closed."

        is_buy = pos.direction.upper() in ("BUY", "LONG")
        remaining_qty = pos.current_quantity
        pnl_per_unit = (current_price - pos.entry_price) if is_buy else (pos.entry_price - current_price)
        realized_delta = pnl_per_unit * remaining_qty

        pos.realized_pnl += realized_delta
        pos.unrealized_pnl = 0.0
        pos.current_quantity = 0.0
        pos.status = PositionStatus.CLOSED
        pos.closed_at = datetime.now(timezone.utc)
        pos.updated_at = datetime.now(timezone.utc)
        self._repo.save_position(pos)

        event = PositionEvent(
            position_id=pos.position_id,
            action=PositionManagementAction.POSITION_CLOSED,
            price=current_price,
            quantity_affected=remaining_qty,
            realized_pnl_delta=realized_delta,
            notes=f"{reason} at {current_price:.5f}.",
        )
        self._repo.save_position_event(event)

        if pos.execution_environment == ExecutionEnvironment.LIVE:
            realized_loss_val = abs(realized_delta) if realized_delta < 0 else 0.0
            self._discipline_engine.record_trade_execution(
                symbol=pos.symbol,
                user_id=user_id,
                realized_loss=realized_loss_val,
            )

        return pos, None
