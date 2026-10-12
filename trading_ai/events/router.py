"""Step 16: Event Router bridging EventBus to WebSocket Gateway and Notifications."""
import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional
import uuid

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.events.gateway import WebSocketGateway
from trading_ai.schemas.cloud import CloudStreamEvent
from trading_ai.schemas.enums import EventPriority, EventPriorityLevel, NotificationDeliveryStatus, SystemEventType
from trading_ai.schemas.performance import SystemEvent
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)

# Map Step 11 EventPriority to Step 16 EventPriorityLevel
PRIORITY_MAP = {
    EventPriority.CRITICAL: EventPriorityLevel.P0_SYSTEM_CRITICAL,
    EventPriority.HIGH: EventPriorityLevel.P2_ACTIVE_POSITION_RISK,
    EventPriority.MEDIUM: EventPriorityLevel.P3_CONFIRMED_SETUP,
    EventPriority.LOW: EventPriorityLevel.P4_HIGH_VALUE_MARKET_EVENT,
    EventPriority.BACKGROUND: EventPriorityLevel.P5_INFORMATIONAL,
}


class EventRouter:
    """
    Authoritative bridge between existing ReactiveEventBus, WebSocket Gateway, and Notification Orchestrator.
    Guarantees no duplicate event buses are created.
    """

    def __init__(
        self,
        event_bus: Optional[ReactiveEventBus] = None,
        repository: Optional[Repository] = None,
        gateway: Optional[WebSocketGateway] = None,
        notification_consumer: Optional[Callable[[CloudStreamEvent], Any]] = None,
    ):
        self._event_bus = event_bus
        self._repo = repository
        self._gateway = gateway
        self._notification_consumer = notification_consumer
        self._running = False
        self._routed_count = 0

        # Subscribe to all events on existing EventBus if provided
        if self._event_bus:
            self._event_bus.subscribe(None, self._on_event_bus_event)

    def set_notification_consumer(self, consumer: Callable[[CloudStreamEvent], Any]) -> None:
        """Attach downstream Notification Orchestrator."""
        self._notification_consumer = consumer

    def _on_event_bus_event(self, system_event: SystemEvent) -> None:
        """Handler invoked synchronously by ReactiveEventBus upon dispatch."""
        stream_event = self._translate_system_event(system_event)
        self.route_event(stream_event)

    def _translate_system_event(self, event: SystemEvent) -> CloudStreamEvent:
        """Map Step 11 SystemEvent into Step 16 CloudStreamEvent."""
        prio = PRIORITY_MAP.get(event.priority, EventPriorityLevel.P5_INFORMATIONAL)
        # Specialize priority based on event type if critical
        if event.event_type in [SystemEventType.SECURITY_ALERT, SystemEventType.THREAT_DETECTED]:
            prio = EventPriorityLevel.P1_SECURITY
        elif event.event_type in [SystemEventType.RISK_CHANGED, SystemEventType.DISCIPLINE_WARNING]:
            prio = EventPriorityLevel.P2_ACTIVE_POSITION_RISK
        elif event.event_type in [SystemEventType.SETUP_CONFIRMED, SystemEventType.TRADE_PROPOSED]:
            prio = EventPriorityLevel.P3_CONFIRMED_SETUP
        elif event.event_type in [SystemEventType.SETUP_CHANGED, SystemEventType.NEWS_EVENT]:
            prio = EventPriorityLevel.P4_HIGH_VALUE_MARKET_EVENT

        symbol = event.details.get("symbol") if event.details else None
        dedup_key = f"{symbol or 'GLOBAL'}_{event.event_type.value}_{event.event_id}"

        return CloudStreamEvent(
            event_id=event.event_id,
            event_type=event.event_type.value,
            timestamp=event.timestamp,
            symbol=symbol,
            priority=prio,
            source="ReactiveEventBus",
            confidence=float(event.details.get("confidence", 1.0)) if event.details else 1.0,
            evidence_reference=event.details.get("evidence", {}) if event.details else {},
            correlation_id=event.details.get("correlation_id", uuid.uuid4().hex[:8]) if event.details else uuid.uuid4().hex[:8],
            deduplication_key=dedup_key,
            payload=event.details,
            delivery_status=NotificationDeliveryStatus.QUEUED,
        )

    def route_event(self, stream_event: CloudStreamEvent) -> None:
        """Persist, push to WebSocket clients, and forward to Notification Orchestrator."""
        self._routed_count += 1
        # 1. Persist to authoritative SQLite log
        try:
            self._repo.save_stream_event(stream_event)
        except Exception as e:
            logger.error("Failed persisting stream event %s: %s", stream_event.event_id, e)

        # 2. Asynchronously push to connected WebSocket clients
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._gateway.broadcast_event(stream_event))
            else:
                loop.run_until_complete(self._gateway.broadcast_event(stream_event))
        except RuntimeError:
            pass
        except Exception as e:
            logger.debug("Async push error: %s", e)

        # 3. Forward to Notification Orchestrator if registered
        if self._notification_consumer:
            try:
                self._notification_consumer(stream_event)
            except Exception as e:
                logger.error("Notification consumer error for event %s: %s", stream_event.event_id, e)

    def health(self) -> Dict[str, Any]:
        return {
            "status": "HEALTHY",
            "total_routed_events": self._routed_count,
            "gateway_status": self._gateway.health(),
        }
