"""
Component Upgrade Manager.
Governs controlled, future-ready upgrades for models, algorithms, and data providers.
STRICT SAFETY RULE: Automatic promotion to Live is prohibited; explicit user approval is required.
"""

from datetime import datetime, timezone
import logging
from typing import List, Optional
import uuid

from trading_ai.schemas.enums import UpgradeStatus
from trading_ai.schemas.performance import ComponentUpgradeRecord
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class ComponentUpgradeManager:
    """
    Manages safe lifecycle of algorithm, model, and system component upgrades:
    CANDIDATE -> SANDBOX_TESTING -> BENCHMARK_PASSED -> PAPER_VALIDATED -> AWAITING_USER_APPROVAL -> APPROVED.
    """

    def __init__(self, repository: Repository):
        self._repo = repository

    def register_candidate(
        self,
        component_name: str,
        current_version: str,
        candidate_version: str,
        notes: str = "",
    ) -> ComponentUpgradeRecord:
        """Register a new candidate for evaluation."""
        upgrade_id = f"upg_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc)
        record = ComponentUpgradeRecord(
            upgrade_id=upgrade_id,
            component_name=component_name,
            current_version=current_version,
            candidate_version=candidate_version,
            status=UpgradeStatus.CANDIDATE,
            benchmark_accuracy=0.0,
            benchmark_latency_ms=0.0,
            regression_pass_rate=0.0,
            user_approved=False,
            approved_by=None,
            notes=notes,
            created_at=now,
            updated_at=now,
        )
        self._repo.save_component_upgrade(record)
        return record

    def run_sandbox_benchmark(
        self,
        upgrade_id: str,
        accuracy: float,
        latency_ms: float,
        regression_pass_rate: float,
    ) -> ComponentUpgradeRecord:
        """Record sandbox evaluation results and update status."""
        record = self._repo.get_component_upgrade(upgrade_id)
        if not record:
            raise ValueError(f"Upgrade record {upgrade_id} not found.")

        record.benchmark_accuracy = accuracy
        record.benchmark_latency_ms = latency_ms
        record.regression_pass_rate = regression_pass_rate
        record.updated_at = datetime.now(timezone.utc)

        # Require >= 95% regression pass rate and >= 80% accuracy
        if regression_pass_rate >= 0.95 and accuracy >= 0.80:
            record.status = UpgradeStatus.BENCHMARK_PASSED
        else:
            record.status = UpgradeStatus.REJECTED
            record.notes += f" | Rejected in sandbox: pass_rate={regression_pass_rate}, acc={accuracy}"

        self._repo.save_component_upgrade(record)
        return record

    def record_paper_validation(self, upgrade_id: str, passed: bool) -> ComponentUpgradeRecord:
        """Record paper environment validation result."""
        record = self._repo.get_component_upgrade(upgrade_id)
        if not record:
            raise ValueError(f"Upgrade record {upgrade_id} not found.")

        if record.status != UpgradeStatus.BENCHMARK_PASSED:
            raise ValueError(f"Cannot record paper validation when status is {record.status}.")

        record.updated_at = datetime.now(timezone.utc)
        if passed:
            record.status = UpgradeStatus.AWAITING_USER_APPROVAL
            record.notes += " | Passed paper validation, awaiting user sign-off."
        else:
            record.status = UpgradeStatus.REJECTED
            record.notes += " | Failed paper validation."

        self._repo.save_component_upgrade(record)
        return record

    def approve_and_promote(
        self,
        upgrade_id: str,
        approved_by: str,
        is_user: bool,
    ) -> ComponentUpgradeRecord:
        """
        User approval gate.
        Enforces strict safety: AI cannot promote itself. is_user MUST be True.
        """
        if not is_user or not approved_by:
            raise PermissionError("Automatic self-promotion to live is strictly prohibited. User approval is required.")

        record = self._repo.get_component_upgrade(upgrade_id)
        if not record:
            raise ValueError(f"Upgrade record {upgrade_id} not found.")

        if record.status != UpgradeStatus.AWAITING_USER_APPROVAL:
            raise ValueError(f"Cannot approve upgrade in state {record.status}. Must be in AWAITING_USER_APPROVAL.")

        record.status = UpgradeStatus.APPROVED
        record.user_approved = True
        record.approved_by = approved_by
        record.updated_at = datetime.now(timezone.utc)
        record.notes += f" | Approved by user '{approved_by}'."

        self._repo.save_component_upgrade(record)
        logger.info(f"Upgrade {upgrade_id} for {record.component_name} approved and promoted to production.")
        return record

    def rollback(self, upgrade_id: str, reason: str) -> ComponentUpgradeRecord:
        """Rollback an upgrade to the prior version."""
        record = self._repo.get_component_upgrade(upgrade_id)
        if not record:
            raise ValueError(f"Upgrade record {upgrade_id} not found.")

        record.status = UpgradeStatus.ROLLED_BACK
        record.notes += f" | Rolled back: {reason}"
        record.updated_at = datetime.now(timezone.utc)
        self._repo.save_component_upgrade(record)
        return record

    def list_upgrades(self, limit: int = 50) -> List[ComponentUpgradeRecord]:
        """List upgrade history."""
        return self._repo.list_component_upgrades(limit=limit)
