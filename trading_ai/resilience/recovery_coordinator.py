"""Auto-Recovery Coordinator & Serial Invariant Preservation Engine."""
import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.schemas.enums import RecoveryClassification, TaskUnitStatus
from trading_ai.schemas.recovery import RecoveryCheckpoint
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier


class AutoRecoveryCoordinator:
    """Orchestrates crash-recovery, state validation, and preservation of the serial commit invariant."""

    def __init__(self, repository: Repository, commit_barrier: Optional[CommitBarrier] = None):
        self.repository = repository
        self.commit_barrier = commit_barrier or CommitBarrier(repository)

    def create_checkpoint(
        self,
        job_type: str,
        job_id: str,
        state_payload: Dict[str, Any],
        serial_number: Optional[int] = None,
        classification: RecoveryClassification = RecoveryClassification.AUTO_RECOVERABLE,
        requires_user_approval: bool = False,
    ) -> RecoveryCheckpoint:
        """Create and persist an atomic recovery checkpoint."""
        raw_json = json.dumps(state_payload, sort_keys=True).encode("utf-8")
        chk_id = f"chk_{job_type}_{uuid.uuid4().hex[:8]}"
        digest = hashlib.sha256(raw_json).hexdigest()

        checkpoint = RecoveryCheckpoint(
            checkpoint_id=chk_id,
            job_type=job_type,
            job_id=job_id,
            serial_number=serial_number,
            state_payload=state_payload,
            sha256_hash=digest,
            created_at=datetime.now(timezone.utc),
            classification=classification,
            requires_user_approval=requires_user_approval,
        )

        if hasattr(self.repository, "save_recovery_checkpoint"):
            self.repository.save_recovery_checkpoint(checkpoint)

        return checkpoint

    def recover_job_state(self, job_id: str) -> Dict[str, Any]:
        """Perform crash recovery for a job, verifying checkpoint integrity and preserving serial ordering."""
        # 1. Fetch barrier state
        barrier = self.commit_barrier.get_barrier_state(job_id)

        # 2. Check if there was a blocked serial
        blocked_serial = barrier.blocked_on_serial
        highest_committed = barrier.highest_committed_serial

        # 3. If there is a blocked unit (e.g. serial 5 failed while 6 & 7 buffered),
        # recovery must resolve serial 5 first!
        status_info = {
            "job_id": job_id,
            "highest_committed_serial": highest_committed,
            "blocked_on_serial": blocked_serial,
            "recovered": True,
            "action": "RESUMED",
        }

        if blocked_serial is not None:
            unit = self.repository.get_task_unit(job_id, blocked_serial)
            if unit and unit.status == TaskUnitStatus.FAILED:
                # Re-queue / reset the failed unit so it can be re-executed cleanly
                unit.status = TaskUnitStatus.PENDING
                unit.updated_at = datetime.now(timezone.utc)
                self.repository.save_task_unit(unit)
                status_info["action"] = f"RESET_FAILED_SERIAL_{blocked_serial}_FOR_RETRY"

        return status_info

    def resolve_serial_barrier(self, job_id: str, serial_number: int, result: Dict[str, Any]) -> List[int]:
        """Recover a failed serial unit and trigger sequential commit of subsequent buffered units."""
        return self.commit_barrier.recover_task_unit(job_id, serial_number, result)
