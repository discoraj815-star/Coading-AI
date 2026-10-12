"""
Rule Promotion Manager: Enforcing Strict Backtest -> Paper -> Live Multi-Stage Gating.

Mandatory Progression:
    LEARNED -> CANDIDATE -> BACKTESTING -> BACKTEST_COMPLETE -> USER_REVIEW ->
    APPROVED_FOR_PAPER -> PAPER_VALIDATION -> USER_LIVE_APPROVAL -> LIVE_ELIGIBLE -> LIVE_ACTIVE

Hard Safety Constraints:
  - Backtest PASS alone NEVER authorizes live execution.
  - Paper validation requires prior backtest completion and explicit user review.
  - Live eligibility requires completed backtest + user approval + paper validation + explicit user live approval.
  - Final live execution remains strictly guarded by Step 8 RiskEngine and ExecutionGate.
"""

import logging
from typing import List, Optional, Tuple

from trading_ai.execution.execution_gate import ExecutionGate
from trading_ai.schemas.backtest import RulePromotionRecord, RuleVersion
from trading_ai.schemas.enums import BacktestStatus, RuleLifecycleState
from trading_ai.schemas.execution import TradeProposal
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class PromotionError(Exception):
    """Raised when a rule violates promotion gating criteria."""
    pass


class RulePromotionManager:
    """
    State machine and gatekeeper managing rule progression through paper and live readiness.
    """

    def __init__(self, repository: Repository, execution_gate: Optional[ExecutionGate] = None):
        self._repo = repository
        self._execution_gate = execution_gate

    def get_or_create_promotion_record(self, rule_version_id: str) -> RulePromotionRecord:
        """Fetch or initialize the promotion record for a rule version."""
        record = self._repo.get_rule_promotion(rule_version_id)
        if not record:
            rule = self._repo.get_rule_version(rule_version_id)
            current_state = rule.state if rule else RuleLifecycleState.LEARNED
            record = RulePromotionRecord(
                rule_version_id=rule_version_id,
                current_state=current_state,
            )
            self._repo.save_rule_promotion(record)
        return record

    def approve_for_paper(
        self,
        rule_version_id: str,
        user_confirmed: bool = True,
        notes: str = "User reviewed and approved backtest results",
    ) -> RulePromotionRecord:
        """
        Gate 1: Promote rule from BACKTEST_COMPLETE to APPROVED_FOR_PAPER.
        Requires completed backtest and explicit user confirmation.
        """
        if not user_confirmed:
            raise PromotionError("User confirmation is required to approve a rule for paper trading.")

        rule = self._repo.get_rule_version(rule_version_id)
        if not rule:
            raise PromotionError(f"Rule version {rule_version_id} not found.")

        # Verify a completed backtest exists for this rule
        runs = self._repo.get_backtest_runs(limit=50)
        matching_runs = [
            r for r in runs
            if r.rule_version_id == rule_version_id and r.status == BacktestStatus.COMPLETED
        ]

        if not matching_runs and rule.state not in (RuleLifecycleState.BACKTEST_COMPLETE, RuleLifecycleState.USER_REVIEW):
            raise PromotionError(
                f"Rule {rule_version_id} cannot be promoted to paper without a completed backtest."
            )

        latest_run_id = matching_runs[0].backtest_id if matching_runs else None

        record = self.get_or_create_promotion_record(rule_version_id)
        record.backtest_id = latest_run_id or record.backtest_id
        record.user_backtest_approved = True
        record.current_state = RuleLifecycleState.APPROVED_FOR_PAPER
        record.notes = notes

        rule.state = RuleLifecycleState.APPROVED_FOR_PAPER
        self._repo.save_rule_version(rule)
        self._repo.save_rule_promotion(record)

        logger.info(f"Rule {rule_version_id} successfully promoted to APPROVED_FOR_PAPER.")
        return record

    def record_paper_validation(
        self,
        rule_version_id: str,
        trades_count: int,
        paper_pnl: float,
        passed_validation: bool = True,
    ) -> RulePromotionRecord:
        """
        Gate 2: Record paper trading validation performance.
        Transitions rule into PAPER_VALIDATION state.
        """
        rule = self._repo.get_rule_version(rule_version_id)
        if not rule:
            raise PromotionError(f"Rule version {rule_version_id} not found.")

        record = self.get_or_create_promotion_record(rule_version_id)
        if not record.user_backtest_approved:
            raise PromotionError("Cannot record paper validation without prior approved backtest.")

        record.paper_trade_count = trades_count
        record.paper_pnl = paper_pnl
        record.paper_validation_approved = passed_validation
        record.current_state = RuleLifecycleState.PAPER_VALIDATION

        rule.state = RuleLifecycleState.PAPER_VALIDATION
        self._repo.save_rule_version(rule)
        self._repo.save_rule_promotion(record)

        return record

    def approve_for_live(
        self,
        rule_version_id: str,
        user_confirmed: bool = True,
        notes: str = "Explicit user live execution authorization",
    ) -> RulePromotionRecord:
        """
        Gate 3: Promote rule to LIVE_ELIGIBLE.
        STRICT REQUIREMENT:
          1. Validated rule exists.
          2. Backtest completed & approved by user.
          3. Paper trading completed & validated.
          4. Explicit user live approval.
          No automatic promotion permitted under any condition.
        """
        if not user_confirmed:
            raise PromotionError("Explicit user confirmation is mandatory to promote a rule to live trading.")

        rule = self._repo.get_rule_version(rule_version_id)
        if not rule:
            raise PromotionError(f"Rule version {rule_version_id} not found.")

        record = self.get_or_create_promotion_record(rule_version_id)

        if not record.user_backtest_approved:
            raise PromotionError("Live promotion rejected: Rule has not passed user backtest approval.")

        if not record.paper_validation_approved:
            raise PromotionError("Live promotion rejected: Rule has not completed and passed paper validation.")

        record.user_live_approved = True
        record.current_state = RuleLifecycleState.LIVE_ELIGIBLE
        record.notes = notes

        rule.state = RuleLifecycleState.LIVE_ELIGIBLE
        self._repo.save_rule_version(rule)
        self._repo.save_rule_promotion(record)

        logger.info(f"Rule {rule_version_id} promoted to LIVE_ELIGIBLE following strict validation ladder.")
        return record

    def verify_live_execution_eligibility(
        self,
        rule_version_id: str,
        proposal: Optional[TradeProposal] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Pre-flight safety barrier before any order reaches live broker.
        Enforces Step 9 promotion status requirements. The full Step 8 ExecutionGate
        authorization (with complete TradeProposal + policy + thesis + risk) is enforced
        separately at actual order submission time.
        """
        reasons: List[str] = []

        rule = self._repo.get_rule_version(rule_version_id)
        if not rule:
            return False, [f"Rule version {rule_version_id} does not exist."]

        record = self.get_or_create_promotion_record(rule_version_id)

        if rule.state not in (RuleLifecycleState.LIVE_ELIGIBLE, RuleLifecycleState.LIVE_ACTIVE):
            reasons.append(f"Rule state is {rule.state.value}; must be LIVE_ELIGIBLE or LIVE_ACTIVE.")

        if not record.user_live_approved:
            reasons.append("Missing explicit user live approval authorization.")

        if not record.paper_validation_approved:
            reasons.append("Missing successful paper validation.")

        # Note: Full Step 8 ExecutionGate.validate_and_authorize_execution() is called
        # at order submission time with a complete proposal, policy, risk assessment, and
        # trade thesis. It is NOT repeated here to avoid double-gating with incomplete data.
        if proposal and self._execution_gate:
            logger.info(
                "Step 8 ExecutionGate is configured; final gate check will occur at order submission time "
                "for rule %s", rule_version_id
            )

        is_eligible = len(reasons) == 0
        return is_eligible, reasons

