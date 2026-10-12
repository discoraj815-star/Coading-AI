"""Anti-Hang Task Watchdog & Adaptive Progress Monitor."""
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.schemas.enums import EventPriority, SystemEventType, TaskWatchdogState
from trading_ai.schemas.performance import SystemEvent
from trading_ai.schemas.recovery import TaskWatchdogEntry

logger = logging.getLogger(__name__)


class AntiHangWatchdog:
    """Monitors running tasks for hangs using adaptive progress-based timeouts and graceful cancellation."""

    def __init__(self, event_bus: Optional[ReactiveEventBus] = None):
        self._tasks: Dict[str, TaskWatchdogEntry] = {}
        self._cancellation_callbacks: Dict[str, Callable[[], None]] = {}
        self._lock = threading.Lock()
        self.event_bus = event_bus

    def register_task(
        self,
        task_id: str,
        task_name: str,
        timeout_seconds: float = 10.0,
        is_critical_operation: bool = False,
        cancellation_cb: Optional[Callable[[], None]] = None,
    ) -> TaskWatchdogEntry:
        """Register a new task to be monitored by the watchdog."""
        with self._lock:
            now = datetime.now(timezone.utc)
            entry = TaskWatchdogEntry(
                task_id=task_id,
                task_name=task_name,
                started_at=now,
                last_progress_at=now,
                timeout_seconds=timeout_seconds,
                state=TaskWatchdogState.RUNNING,
                is_critical_operation=is_critical_operation,
            )
            self._tasks[task_id] = entry
            if cancellation_cb:
                self._cancellation_callbacks[task_id] = cancellation_cb
            return entry

    def record_progress(self, task_id: str) -> bool:
        """Update last progress timestamp for an active task."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.state == TaskWatchdogState.RUNNING:
                task.last_progress_at = datetime.now(timezone.utc)
                return True
            return False

    def complete_task(self, task_id: str) -> None:
        """Mark a task as successfully completed and remove from active monitoring."""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
            if task_id in self._cancellation_callbacks:
                del self._cancellation_callbacks[task_id]

    def check_for_hangs(self) -> List[TaskWatchdogEntry]:
        """Audit all active tasks. Trigger graceful cancellation for tasks exceeding their progress timeout."""
        hung_tasks: List[TaskWatchdogEntry] = []
        now = datetime.now(timezone.utc)

        with self._lock:
            for task_id, task in list(self._tasks.items()):
                if task.state != TaskWatchdogState.RUNNING:
                    continue

                elapsed_since_progress = (now - task.last_progress_at).total_seconds()
                if elapsed_since_progress > task.timeout_seconds:
                    # Critical operations (DB commit, broker execution) are exempted from force kill
                    if task.is_critical_operation:
                        task.state = TaskWatchdogState.SUSPECTED_HANG
                        hung_tasks.append(task)
                        continue

                    # Gracefully cancel non-critical hung tasks
                    task.state = TaskWatchdogState.GRACEFUL_CANCEL
                    hung_tasks.append(task)

                    cb = self._cancellation_callbacks.get(task_id)
                    if cb:
                        try:
                            cb()
                        except Exception as e:
                            logger.warning(f"Error executing watchdog cancellation callback for task {task_id}: {e}")

                    # Notify event bus
                    if self.event_bus:
                        evt = SystemEvent(
                            event_type=SystemEventType.TASK_HANG_DETECTED,
                            payload={
                                "task_id": task.task_id,
                                "task_name": task.task_name,
                                "elapsed": elapsed_since_progress,
                                "timeout": task.timeout_seconds,
                            },
                            priority=EventPriority.HIGH,
                            provenance="AntiHangWatchdog",
                        )
                        self.event_bus.publish(evt)

        return hung_tasks

    def get_task(self, task_id: str) -> Optional[TaskWatchdogEntry]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_active_tasks(self) -> List[TaskWatchdogEntry]:
        with self._lock:
            return list(self._tasks.values())
