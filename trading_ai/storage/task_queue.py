"""Step 16 / C3: Task Queue Adapter and Local Persistent Queue Implementation."""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from trading_ai.schemas.cloud import TaskQueueJob
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier

logger = logging.getLogger(__name__)


class TaskQueueAdapter(ABC):
    """
    Abstract Task Queue interface (C3 foundation).
    Decouples background jobs and workers from concrete brokers (Redis/Celery/Arq).
    Guarantees strict idempotency, priority ordering, and CommitBarrier serial preservation.
    """

    @abstractmethod
    def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 2,
        serial_number: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> TaskQueueJob:
        """Enqueue a background task."""
        pass

    @abstractmethod
    def dequeue(self, limit: int = 1) -> List[TaskQueueJob]:
        """Fetch pending tasks by priority."""
        pass

    @abstractmethod
    def acknowledge(self, job_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        """Mark job completed."""
        pass

    @abstractmethod
    def retry(self, job_id: str, error_message: str) -> bool:
        """Mark job for retry or move to dead-letter if max retries exceeded."""
        pass

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        pass

    @abstractmethod
    def get_status(self, job_id: str) -> Optional[TaskQueueJob]:
        """Fetch status of job."""
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Return queue health and metrics."""
        pass


class LocalPersistentTaskQueue(TaskQueueAdapter):
    """
    Production-grade SQLite-backed persistent task queue.
    Authoritative default backend that guarantees zero external dependency overhead.
    Strictly preserves CommitBarrier monotonic serial invariants for large video/learning jobs.
    """

    def __init__(self, repository: Repository, commit_barrier: Optional[CommitBarrier] = None):
        self._repo = repository
        self._barrier = commit_barrier or CommitBarrier(repository)
        self._is_paused = False

    def enqueue(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 2,
        serial_number: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> TaskQueueJob:
        jid = job_id or f"task_{uuid.uuid4().hex[:10]}"
        job = TaskQueueJob(
            job_id=jid,
            task_type=task_type,
            priority=priority,
            payload=payload,
            status="PENDING",
            serial_number=serial_number,
        )
        self._repo.save_task_queue_job(job.model_dump(mode="json"))
        return job

    def dequeue(self, limit: int = 1) -> List[TaskQueueJob]:
        if self._is_paused:
            return []
        pending = self._repo.get_pending_task_queue_jobs(limit=limit)
        jobs: List[TaskQueueJob] = []
        for p in pending:
            # Mark RUNNING in DB
            self._repo.save_task_queue_job({
                **p,
                "status": "RUNNING",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            jobs.append(TaskQueueJob(**p))
        return jobs

    def acknowledge(self, job_id: str, result: Optional[Dict[str, Any]] = None) -> bool:
        job_dict = self._repo.get_task_queue_job(job_id)
        if not job_dict:
            return False

        # Invariant check: If this task has a parent job and serial number, verify CommitBarrier
        parent_job_id = job_dict.get("payload", {}).get("parent_job_id")
        serial = job_dict.get("serial_number")
        if parent_job_id and serial is not None:
            # Check commit barrier status
            state = self._barrier.get_state(parent_job_id)
            if state and serial > state.last_committed_serial + 1:
                logger.warning(
                    "Task %s serial %d cannot commit before serial %d completes",
                    job_id, serial, state.last_committed_serial + 1
                )
                # Keep state as WAITING_BARRIER rather than completing
                self._repo.save_task_queue_job({
                    **job_dict,
                    "status": "WAITING_BARRIER",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                return False

        self._repo.save_task_queue_job({
            **job_dict,
            "status": "COMPLETED",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return True

    def retry(self, job_id: str, error_message: str) -> bool:
        job_dict = self._repo.get_task_queue_job(job_id)
        if not job_dict:
            return False
        curr_retries = int(job_dict.get("retry_count", 0)) + 1
        max_retries = int(job_dict.get("max_retries", 3))
        status = "RETRYING" if curr_retries < max_retries else "DEAD_LETTER"
        self._repo.save_task_queue_job({
            **job_dict,
            "status": status,
            "retry_count": curr_retries,
            "error_message": error_message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return True

    def cancel(self, job_id: str) -> bool:
        job_dict = self._repo.get_task_queue_job(job_id)
        if not job_dict:
            return False
        self._repo.save_task_queue_job({
            **job_dict,
            "status": "CANCELLED",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        return True

    def pause(self) -> bool:
        self._is_paused = True
        return True

    def resume(self) -> bool:
        self._is_paused = False
        return True

    def get_status(self, job_id: str) -> Optional[TaskQueueJob]:
        data = self._repo.get_task_queue_job(job_id)
        return TaskQueueJob(**data) if data else None

    def health(self) -> Dict[str, Any]:
        pending = self._repo.get_pending_task_queue_jobs(limit=100)
        return {
            "backend": "LocalPersistentTaskQueue",
            "status": "HEALTHY" if not self._is_paused else "PAUSED",
            "pending_count": len(pending),
            "is_paused": self._is_paused,
            "external_dependency": "none",
        }
