"""Persistent Knowledge Commit Barrier ensuring strict serial commit order."""

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from trading_ai.schemas import (
    CommitBarrierState,
    ErrorSeverity,
    KnowledgeItem,
    PipelineStage,
    TaskUnit,
    TaskUnitStatus,
)
from trading_ai.storage.repository import Repository


class CommitBarrier:
    """Orchestrates sequential task execution and enforces strict Knowledge Commit ordering.

    Guarantees that knowledge generated out-of-order or following an unrecovered failure
    remains strictly BUFFERED and uncommitted until preceding units are fully recovered
    and accepted. Ordering and barrier state survive process restart.
    """

    def __init__(self, repository: Repository):
        self._repo = repository

    def get_barrier_state(self, job_id: str) -> CommitBarrierState:
        """Fetch current persisted barrier state for a job."""
        return self._repo.get_commit_barrier(job_id)

    def register_task_unit(
        self,
        job_id: str,
        stage: PipelineStage,
        start_time: float,
        end_time: float,
        payload: Optional[Dict] = None,
    ) -> TaskUnit:
        """Create and persist a new sequential task unit."""
        existing_units = self._repo.list_task_units(job_id)
        next_serial = len(existing_units) + 1

        unit = TaskUnit(
            unit_id=str(uuid.uuid4()),
            job_id=job_id,
            serial_number=next_serial,
            stage=stage,
            start_time=start_time,
            end_time=end_time,
            status=TaskUnitStatus.PENDING,
            payload=payload or {},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self._repo.save_task_unit(unit)
        return unit

    def complete_task_unit(
        self,
        job_id: str,
        serial_number: int,
        result: Dict,
        knowledge_items: Optional[List[KnowledgeItem]] = None,
    ) -> List[int]:
        """Record completion of a task unit and commit knowledge if barrier is clear.

        Returns list of serial numbers that were committed as a result of this operation.
        """
        unit = self._repo.get_task_unit(job_id, serial_number)
        if not unit:
            raise ValueError(f"TaskUnit with serial {serial_number} for job {job_id} not found.")

        # Save knowledge items as uncommitted first
        if knowledge_items:
            for item in knowledge_items:
                self._repo.save_knowledge_item(
                    item, serial_number=serial_number, is_committed=False
                )

        barrier = self._repo.get_commit_barrier(job_id)
        committed_serials: List[int] = []

        # Check if this unit can be immediately committed
        can_commit = (
            barrier.blocked_on_serial is None
            and serial_number == barrier.highest_committed_serial + 1
        )

        if can_commit:
            # Commit this unit
            unit.status = TaskUnitStatus.COMMITTED
            unit.result = result
            unit.updated_at = datetime.now(timezone.utc)
            self._repo.save_task_unit(unit)
            self._repo.commit_knowledge_for_serial(job_id, serial_number)

            barrier.highest_committed_serial = serial_number
            committed_serials.append(serial_number)

            # Check if any previously buffered units can now be committed sequentially
            buffered_set = set(barrier.buffered_units)
            next_candidate = barrier.highest_committed_serial + 1

            while next_candidate in buffered_set:
                candidate_unit = self._repo.get_task_unit(job_id, next_candidate)
                if candidate_unit and candidate_unit.status == TaskUnitStatus.BUFFERED:
                    candidate_unit.status = TaskUnitStatus.COMMITTED
                    candidate_unit.updated_at = datetime.now(timezone.utc)
                    self._repo.save_task_unit(candidate_unit)
                    self._repo.commit_knowledge_for_serial(job_id, next_candidate)

                    barrier.highest_committed_serial = next_candidate
                    committed_serials.append(next_candidate)
                    buffered_set.remove(next_candidate)
                    next_candidate += 1
                else:
                    break

            barrier.buffered_units = sorted(list(buffered_set))
            self._repo.save_commit_barrier(barrier)
        else:
            # Must remain buffered behind a missing or failed predecessor
            unit.status = TaskUnitStatus.BUFFERED
            unit.result = result
            unit.updated_at = datetime.now(timezone.utc)
            self._repo.save_task_unit(unit)

            if serial_number not in barrier.buffered_units:
                barrier.buffered_units.append(serial_number)
                barrier.buffered_units.sort()
            self._repo.save_commit_barrier(barrier)

        return committed_serials

    def fail_task_unit(
        self,
        job_id: str,
        serial_number: int,
        error_message: str,
        severity: ErrorSeverity = ErrorSeverity.RECOVERABLE,
    ) -> None:
        """Mark a task unit as failed, establishing a commit barrier for subsequent units."""
        unit = self._repo.get_task_unit(job_id, serial_number)
        if not unit:
            raise ValueError(f"TaskUnit with serial {serial_number} for job {job_id} not found.")

        unit.status = TaskUnitStatus.FAILED
        unit.retry_count += 1
        unit.last_error = error_message
        unit.error_severity = severity
        unit.updated_at = datetime.now(timezone.utc)
        self._repo.save_task_unit(unit)

        self._repo.log_error(
            job_id=job_id,
            serial_number=serial_number,
            stage=unit.stage.value,
            error_message=error_message,
            severity=severity,
            retry_count=unit.retry_count,
        )

        barrier = self._repo.get_commit_barrier(job_id)
        if barrier.blocked_on_serial is None or serial_number < barrier.blocked_on_serial:
            barrier.blocked_on_serial = serial_number
        self._repo.save_commit_barrier(barrier)

    def recover_task_unit(
        self,
        job_id: str,
        serial_number: int,
        recovered_result: Dict,
        knowledge_items: Optional[List[KnowledgeItem]] = None,
    ) -> List[int]:
        """Recover a failed task unit and release the commit barrier for buffered units.

        Returns list of all newly committed serial numbers.
        """
        unit = self._repo.get_task_unit(job_id, serial_number)
        if not unit:
            raise ValueError(f"TaskUnit with serial {serial_number} for job {job_id} not found.")

        # Save recovered knowledge items
        if knowledge_items:
            for item in knowledge_items:
                self._repo.save_knowledge_item(
                    item, serial_number=serial_number, is_committed=False
                )

        barrier = self._repo.get_commit_barrier(job_id)

        # Clear blocked_on_serial if this was the blocking unit
        if barrier.blocked_on_serial == serial_number:
            barrier.blocked_on_serial = None

        committed_serials: List[int] = []

        # If this unit is now exactly the next expected unit to commit:
        if serial_number == barrier.highest_committed_serial + 1:
            unit.status = TaskUnitStatus.COMMITTED
            unit.result = recovered_result
            unit.last_error = None
            unit.updated_at = datetime.now(timezone.utc)
            self._repo.save_task_unit(unit)
            self._repo.commit_knowledge_for_serial(job_id, serial_number)

            barrier.highest_committed_serial = serial_number
            committed_serials.append(serial_number)

            # Cascade through buffered units in order
            buffered_set = set(barrier.buffered_units)
            next_candidate = barrier.highest_committed_serial + 1

            while next_candidate in buffered_set:
                candidate_unit = self._repo.get_task_unit(job_id, next_candidate)
                if candidate_unit and candidate_unit.status == TaskUnitStatus.BUFFERED:
                    candidate_unit.status = TaskUnitStatus.COMMITTED
                    candidate_unit.updated_at = datetime.now(timezone.utc)
                    self._repo.save_task_unit(candidate_unit)
                    self._repo.commit_knowledge_for_serial(job_id, next_candidate)

                    barrier.highest_committed_serial = next_candidate
                    committed_serials.append(next_candidate)
                    buffered_set.remove(next_candidate)
                    next_candidate += 1
                else:
                    break

            barrier.buffered_units = sorted(list(buffered_set))
            self._repo.save_commit_barrier(barrier)
        else:
            # Even after recovery, if there is an even earlier unrecovered unit, keep buffered
            unit.status = TaskUnitStatus.BUFFERED
            unit.result = recovered_result
            unit.updated_at = datetime.now(timezone.utc)
            self._repo.save_task_unit(unit)

            if serial_number not in barrier.buffered_units:
                barrier.buffered_units.append(serial_number)
                barrier.buffered_units.sort()
            self._repo.save_commit_barrier(barrier)

        return committed_serials
