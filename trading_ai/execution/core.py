"""Hedge Fund Execution Core: Institutional facade for proposals, risk, gate, broker, and position management."""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.execution.broker_adapter import (
    BrokerAdapter,
    LiveBrokerAdapter,
    PaperBrokerAdapter,
)
from trading_ai.execution.confluence_engine import ConfluenceEngine
from trading_ai.execution.execution_gate import ExecutionGate
from trading_ai.execution.position_manager import PositionManager
from trading_ai.execution.risk_engine import RiskEngine
from trading_ai.schemas.enums import (
    AlertQuality,
    EngagementMode,
    ExecutionEnvironment,
    ExecutionMode,
    OrderStatus,
    OrderType,
    PositionStatus,
    ProposalLifecycleState,
    PositionManagementAction,
)
from trading_ai.schemas.execution import (
    ExecutionIntent,
    ExecutionPolicy,
    Order,
    Position,
    PositionEvent,
    RiskAssessment,
    TradeProposal,
    TradeThesis,
)
from trading_ai.schemas.market import CurrentSituationSnapshot
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class HedgeFundExecutionCore:
    """
    Unified Institutional Execution & Risk Management Core.

    Hard Rules:
      - AI reasoning produces TradeProposal; NEVER directly routes orders to brokers.
      - Execution path is strictly:
          AI / Step 5 & 6 Situation
          -> TradeProposal
          -> Confluence Validation
          -> RiskEngine
          -> ExecutionGate
          -> BrokerAdapter
          -> PositionManager
      - Manual and Auto execution use the exact same validation pipeline.
      - Stop Loss is non-negotiable (No SL -> No Trade).
      - Losing positions are never pyramided / no averaging down.
    """

    def __init__(
        self,
        repository: Repository,
        discipline_engine: TradingDisciplineEngine,
        broker_adapter: Optional[BrokerAdapter] = None,
        live_broker_adapter: Optional[BrokerAdapter] = None,
    ):
        self._repo = repository
        self._discipline_engine = discipline_engine
        self._paper_broker = broker_adapter or PaperBrokerAdapter()
        self._live_broker = live_broker_adapter or LiveBrokerAdapter()
        self._confluence_engine = ConfluenceEngine()
        self._risk_engine = RiskEngine(repository, discipline_engine)
        self._execution_gate = ExecutionGate(repository, discipline_engine)
        self._position_manager = PositionManager(repository, discipline_engine)

    @property
    def broker_adapter(self) -> BrokerAdapter:
        return self._paper_broker

    @property
    def paper_broker(self) -> BrokerAdapter:
        return self._paper_broker

    @property
    def paper_broker_adapter(self) -> BrokerAdapter:
        return self._paper_broker

    @property
    def live_broker(self) -> BrokerAdapter:
        return self._live_broker

    @property
    def live_broker_adapter(self) -> BrokerAdapter:
        return self._live_broker

    @property
    def position_manager(self) -> PositionManager:
        return self._position_manager

    @property
    def risk_engine(self) -> RiskEngine:
        return self._risk_engine

    @property
    def execution_gate(self) -> ExecutionGate:
        return self._execution_gate

    @property
    def confluence_engine(self) -> ConfluenceEngine:
        return self._confluence_engine

    def get_or_create_policy(self, user_id: str = "default_user") -> ExecutionPolicy:
        """Retrieve user execution policy or return defaults."""
        policy = self._repo.get_execution_policy(user_id)
        if not policy:
            policy = ExecutionPolicy(user_id=user_id)
            self._repo.save_execution_policy(policy)
        return policy

    def update_policy(self, policy: ExecutionPolicy) -> ExecutionPolicy:
        """Update and persist execution policy."""
        policy.updated_at = datetime.now(timezone.utc)
        self._repo.save_execution_policy(policy)
        return policy

    def generate_trade_proposal(
        self,
        symbol: str,
        situation: CurrentSituationSnapshot,
        direction: Optional[str] = None,
        engagement_mode: EngagementMode = EngagementMode.DEFENSIVE,
        account_equity: float = 10000.0,
        user_id: str = "default_user",
    ) -> TradeProposal:
        """
        Generate a fully formed, auditable TradeProposal from current market situation.
        """
        policy = self.get_or_create_policy(user_id)

        # 1. Determine trade direction from confirmed setup or overall bias
        if not direction:
            if situation.active_confirmed_setup:
                setup_dir = situation.active_confirmed_setup.direction
                direction = "BUY" if "BULL" in str(setup_dir).upper() else "SELL"
            else:
                bias_str = str(situation.overall_bias)
                direction = "BUY" if "BULL" in bias_str.upper() else ("SELL" if "BEAR" in bias_str.upper() else "BUY")

        is_buy = direction.upper() in ("BUY", "LONG", "BULLISH")
        entry_price = situation.current_price

        # 2. Derive Hard Stop Loss
        # Prefer confirmed invalidation level from setup, else POI or 0.75% structural cushion
        if situation.invalidation_level and situation.invalidation_level > 0:
            hard_sl = situation.invalidation_level
        elif situation.current_poi and hasattr(situation.current_poi, "price") and situation.current_poi.price > 0:
            poi_p = situation.current_poi.price
            hard_sl = (poi_p * 0.995) if is_buy else (poi_p * 1.005)
        else:
            hard_sl = (entry_price * 0.9925) if is_buy else (entry_price * 1.0075)

        # Ensure correct directional positioning of SL
        if is_buy and hard_sl >= entry_price:
            hard_sl = entry_price * 0.995
        elif not is_buy and hard_sl <= entry_price:
            hard_sl = entry_price * 1.005

        stop_dist = abs(entry_price - hard_sl)

        # 3. Calculate Targets (TP1 at 1.5R, Final Target at configured min_rr_ratio)
        tp1 = (entry_price + stop_dist * policy.default_tp1_r_multiple) if is_buy else (entry_price - stop_dist * policy.default_tp1_r_multiple)
        final_target = (entry_price + stop_dist * policy.min_rr_ratio) if is_buy else (entry_price - stop_dist * policy.min_rr_ratio)

        # 4. Confluence Validation
        confluences, is_confluence_ok, conf_warnings = self._confluence_engine.evaluate_confluences(
            symbol=symbol,
            situation=situation,
            direction=direction,
        )

        # 5. Formulate Trade Thesis
        regime = "HIGH_VOLATILITY" if situation.news_risk in ("HIGH_RISK", "CRITICAL_RISK") else "BALANCED_STRUCTURE"
        thesis_summary = (
            f"{symbol} {direction} thesis based on {situation.overall_bias.value if hasattr(situation.overall_bias, 'value') else situation.overall_bias} "
            f"structure and {len(confluences)} verified independent confluence(s)."
        )
        invalidation_premise = (
            f"Thesis invalidated if price breaks structural stop loss at {hard_sl:.5f} or setup conditions fail."
        )
        thesis = TradeThesis(
            summary=thesis_summary,
            catalyst=situation.next_high_impact_event.title if situation.next_high_impact_event else None,
            invalidation_premise=invalidation_premise,
            market_regime=regime,
        )

        # 6. Risk Assessment and Position Sizing
        existing_pos = self._repo.get_open_positions(symbol)
        risk_assessment = self._risk_engine.assess_risk_and_size(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=hard_sl,
            target_price=final_target,
            account_equity=account_equity,
            policy=policy,
            engagement_mode=engagement_mode,
            existing_positions=existing_pos,
            user_id=user_id,
        )

        # Lifecycle State determination
        if not is_confluence_ok:
            state = ProposalLifecycleState.PROPOSED
        elif not risk_assessment.is_risk_acceptable:
            state = ProposalLifecycleState.CONFLUENCE_VALIDATED
        else:
            state = ProposalLifecycleState.RISK_VALIDATED

        quality = AlertQuality.HIGH_QUALITY if len(confluences) >= 3 and risk_assessment.is_risk_acceptable else AlertQuality.MEDIUM_QUALITY

        proposal = TradeProposal(
            symbol=symbol,
            direction=direction,
            market_regime=regime,
            htf_context=str(situation.overall_bias),
            mtf_context=situation.bias_summary,
            overall_bias=situation.overall_bias,
            liquidity_context=f"Nearest liquidity: {situation.nearest_major_liquidity.price if situation.nearest_major_liquidity else 'None'}",
            poi_context=f"Current POI: {situation.current_poi.price if situation.current_poi else 'None'}",
            lecture_rule_evidence=situation.active_confirmed_setup.conditions_satisfied if situation.active_confirmed_setup else [],
            confluences=confluences,
            entry_trigger="CONFIRMED_STRUCTURE",
            entry_price=entry_price,
            hard_stop_loss=hard_sl,
            tp1=tp1,
            final_target=final_target,
            invalidation_condition=invalidation_premise,
            risk_assessment=risk_assessment,
            proposed_quantity=risk_assessment.rounded_quantity,
            engagement_mode=engagement_mode,
            execution_environment=policy.execution_environment,
            news_event_context=f"Risk: {situation.news_risk}, Time dist: {situation.news_time_distance_minutes}m",
            trade_thesis=thesis,
            quality=quality,
            lifecycle_state=state,
            user_approval_required=(policy.execution_mode == ExecutionMode.MANUAL),
            is_approved=False,
        )

        self._repo.save_trade_proposal(proposal)
        return proposal

    def execute_proposal(
        self,
        proposal_id: str,
        user_confirmed: bool = False,
        user_id: str = "default_user",
    ) -> Tuple[bool, Optional[Order], Optional[Position], List[str]]:
        """
        Route proposal through ExecutionGate and submit to BrokerAdapter if authorized.
        """
        proposal = self._repo.get_trade_proposal(proposal_id)
        if not proposal:
            return False, None, None, ["PROPOSAL_NOT_FOUND: Proposal ID does not exist."]

        policy = self.get_or_create_policy(user_id)

        # Pass through Execution Gate
        authorized, intent, gate_reasons = self._execution_gate.validate_and_authorize_execution(
            proposal=proposal,
            policy=policy,
            user_confirmed=user_confirmed,
            user_id=user_id,
        )

        if not authorized or not intent:
            return False, None, None, gate_reasons

        # Create Order for Broker Submission
        proposal.lifecycle_state = ProposalLifecycleState.SUBMITTING
        self._repo.save_trade_proposal(proposal)

        order = Order(
            intent_id=intent.intent_id,
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            execution_environment=intent.execution_environment,
            is_simulated=(intent.execution_environment == ExecutionEnvironment.PAPER),
            order_type=intent.order_type,
            direction=intent.direction,
            quantity=intent.quantity,
            limit_price=proposal.entry_price,
            stop_price=proposal.hard_stop_loss,
            status=OrderStatus.PENDING,
        )
        self._repo.save_order(order)

        # B2: Route to Strictly Isolated Broker Adapter (No cross-environment fallback)
        if intent.execution_environment == ExecutionEnvironment.LIVE:
            adapter = self._live_broker
        else:
            adapter = self._paper_broker

        status, brk_id, fill_px, resp_meta, err = adapter.submit_order(order)
        order.status = status
        order.broker_order_id = brk_id
        order.average_fill_price = fill_px
        order.broker_response = resp_meta
        order.error_message = err
        order.updated_at = datetime.now(timezone.utc)
        self._repo.save_order(order)

        # B6: Ambiguous / Network Timeout Handling (Never assume UNKNOWN -> FILLED)
        if status == OrderStatus.UNKNOWN:
            proposal.lifecycle_state = ProposalLifecycleState.UNKNOWN
            proposal.rejection_reason = f"Execution state uncertain: {err or 'Gateway timeout'}"
            self._repo.save_trade_proposal(proposal)
            return False, order, None, [f"EXECUTION_STATE_UNKNOWN: {err or 'Uncertain broker response'}"]

        if status != OrderStatus.FILLED:
            proposal.lifecycle_state = (
                ProposalLifecycleState.BROKER_ERROR
                if status == OrderStatus.BROKER_ERROR
                else ProposalLifecycleState.REJECTED
            )
            proposal.rejection_reason = f"Broker submission failed: {err or 'Rejected'}"
            self._repo.save_trade_proposal(proposal)
            return False, order, None, [f"BROKER_SUBMISSION_FAILED: {err or 'Rejected by broker'}"]

        # Order Filled: Create and Track Managed Position
        order.filled_quantity = order.quantity
        self._repo.save_order(order)

        proposal.lifecycle_state = ProposalLifecycleState.FILLED
        self._repo.save_trade_proposal(proposal)

        effective_fill = fill_px or proposal.entry_price
        position = Position(
            proposal_id=proposal.proposal_id,
            symbol=proposal.symbol,
            direction=intent.direction,
            execution_environment=intent.execution_environment,
            entry_price=effective_fill,
            current_quantity=order.quantity,
            initial_quantity=order.quantity,
            initial_sl=proposal.hard_stop_loss,
            current_sl=proposal.hard_stop_loss,
            tp1=proposal.tp1,
            final_target=proposal.final_target,
            status=PositionStatus.OPEN,
            engagement_mode=proposal.engagement_mode,
        )
        self._repo.save_position(position)

        # Log initial position event
        init_event = PositionEvent(
            position_id=position.position_id,
            action=PositionManagementAction.INITIAL_SL_SET,
            price=effective_fill,
            quantity_affected=order.quantity,
            notes=f"Position opened in {intent.execution_environment.value} environment via {policy.execution_mode.value} execution. Initial SL at {proposal.hard_stop_loss:.5f}.",
        )
        self._repo.save_position_event(init_event)

        return True, order, position, []

    def reject_proposal(self, proposal_id: str, reason: str = "Manual Trader Rejection") -> Optional[TradeProposal]:
        """Explicitly reject a pending trade proposal."""
        proposal = self._repo.get_trade_proposal(proposal_id)
        if not proposal:
            return None
        proposal.lifecycle_state = ProposalLifecycleState.REJECTED
        proposal.rejection_reason = reason
        proposal.updated_at = datetime.now(timezone.utc)
        self._repo.save_trade_proposal(proposal)
        return proposal

    def modify_proposal(
        self,
        proposal_id: str,
        new_sl: Optional[float] = None,
        new_target: Optional[float] = None,
        new_mode: Optional[EngagementMode] = None,
        account_equity: float = 10000.0,
        user_id: str = "default_user",
    ) -> Optional[TradeProposal]:
        """Modify proposal parameters and deterministically recalculate risk."""
        proposal = self._repo.get_trade_proposal(proposal_id)
        if not proposal:
            return None

        if proposal.lifecycle_state in (ProposalLifecycleState.FILLED, ProposalLifecycleState.CLOSED):
            return proposal

        if new_sl is not None:
            proposal.hard_stop_loss = new_sl
        if new_target is not None:
            proposal.final_target = new_target
        if new_mode is not None:
            proposal.engagement_mode = new_mode

        # Re-assess risk
        policy = self.get_or_create_policy(user_id)
        existing_pos = self._repo.get_open_positions(proposal.symbol)
        risk = self._risk_engine.assess_risk_and_size(
            symbol=proposal.symbol,
            direction=proposal.direction,
            entry_price=proposal.entry_price,
            stop_loss=proposal.hard_stop_loss,
            target_price=proposal.final_target,
            account_equity=account_equity,
            policy=policy,
            engagement_mode=proposal.engagement_mode,
            existing_positions=existing_pos,
            user_id=user_id,
        )
        proposal.risk_assessment = risk
        proposal.proposed_quantity = risk.rounded_quantity
        proposal.updated_at = datetime.now(timezone.utc)
        proposal.lifecycle_state = ProposalLifecycleState.RISK_VALIDATED if risk.is_risk_acceptable else ProposalLifecycleState.PROPOSED
        self._repo.save_trade_proposal(proposal)
        return proposal
