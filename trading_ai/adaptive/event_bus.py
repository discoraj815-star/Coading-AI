"""Reactive Event Bus for High-Performance Event-Driven Market Intelligence."""

import heapq
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from trading_ai.schemas.enums import EventPriority, SystemEventType
from trading_ai.schemas.performance import SystemEvent

logger = logging.getLogger(__name__)

# Priority integer ordering (lower integer = higher priority)
PRIORITY_ORDER = {
    EventPriority.CRITICAL: 0,
    EventPriority.HIGH: 1,
    EventPriority.MEDIUM: 2,
    EventPriority.LOW: 3,
    EventPriority.BACKGROUND: 4,
}


class ReactiveEventBus:
    """High-throughput, priority-ordered, idempotent event bus."""

    def __init__(self, max_dedup_cache: int = 5000):
        self._max_dedup_cache = max_dedup_cache
        self._seen_event_ids: Set[str] = set()
        self._seen_queue: deque = deque()
        # Internal min-heap queue items: (priority_int, sequence_counter, SystemEvent)
        self._heap: List[tuple] = []
        self._seq: int = 0
        self._subscribers: Dict[SystemEventType, List[Callable[[SystemEvent], Any]]] = {}
        self._global_subscribers: List[Callable[[SystemEvent], Any]] = []
        self._dispatched_count: int = 0
        self._dropped_duplicates: int = 0

    def subscribe(
        self,
        event_type: Optional[SystemEventType],
        handler: Callable[[SystemEvent], Any],
    ) -> None:
        """Subscribe a callback to a specific event type or all events if event_type is None."""
        if event_type is None:
            self._global_subscribers.append(handler)
        else:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(handler)

    def publish(self, event: SystemEvent) -> bool:
        """
        Publish an event to the priority bus.
        Returns True if accepted, False if dropped due to duplicate event_id.
        """
        if not event.event_id:
            event.event_id = f"evt_{uuid.uuid4().hex[:12]}"

        # Idempotency check
        if event.event_id in self._seen_event_ids:
            self._dropped_duplicates += 1
            logger.debug(f"Event {event.event_id} dropped as duplicate.")
            return False

        self._seen_event_ids.add(event.event_id)
        self._seen_queue.append(event.event_id)
        if len(self._seen_queue) > self._max_dedup_cache:
            oldest = self._seen_queue.popleft()
            self._seen_event_ids.discard(oldest)

        priority_val = PRIORITY_ORDER.get(event.priority, 2)
        self._seq += 1
        heapq.heappush(self._heap, (priority_val, self._seq, event))
        return True

    def dispatch_next(self) -> Optional[SystemEvent]:
        """Pop and dispatch the highest-priority event immediately."""
        if not self._heap:
            return None

        _, _, event = heapq.heappop(self._heap)
        self._execute_dispatch(event)
        return event

    def dispatch_all(self, max_events: int = 500) -> int:
        """Dispatch up to max_events from the priority queue in order."""
        count = 0
        while self._heap and count < max_events:
            self.dispatch_next()
            count += 1
        return count

    def _execute_dispatch(self, event: SystemEvent) -> None:
        """Deliver event to registered type-specific and global handlers safely."""
        self._dispatched_count += 1
        handlers: List[Callable[[SystemEvent], Any]] = []

        if event.event_type in self._subscribers:
            handlers.extend(self._subscribers[event.event_type])
        handlers.extend(self._global_subscribers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error executing event handler for {event.event_type}: {e}", exc_info=True)

    @property
    def queue_size(self) -> int:
        """Current number of pending events waiting in queue."""
        return len(self._heap)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Return breakdown of pending items by priority and overall stats."""
        counts = {p.name: 0 for p in EventPriority}
        for prio_val, _, _ in self._heap:
            for p, val in PRIORITY_ORDER.items():
                if val == prio_val:
                    counts[p.name] += 1
                    break

        return {
            "total_pending": len(self._heap),
            "dispatched_total": self._dispatched_count,
            "dropped_duplicates": self._dropped_duplicates,
            "priority_breakdown": counts,
        }

    def clear(self) -> None:
        """Clear the event queue."""
        self._heap.clear()
