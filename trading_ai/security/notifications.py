"""Cross-Device Push Notifications with Privacy Minimization & Priority Handling."""

from datetime import datetime, timezone
import logging
from typing import Dict, List, Optional
import uuid

from trading_ai.adaptive.event_bus import ReactiveEventBus
from trading_ai.schemas.enums import EventPriority, NotificationPriority, PCCapability, SystemEventType
from trading_ai.schemas.performance import SystemEvent
from trading_ai.schemas.security import NotificationItem
from trading_ai.security.audit import SecurityAuditLogger, redact_sensitive_text
from trading_ai.security.capability_gate import CapabilityPolicyGate
from trading_ai.security.device_manager import DeviceManager

logger = logging.getLogger(__name__)

# Map NotificationPriority to Step 11 EventPriority
PRIORITY_MAP = {
    NotificationPriority.CRITICAL: EventPriority.CRITICAL,
    NotificationPriority.HIGH: EventPriority.HIGH,
    NotificationPriority.MEDIUM: EventPriority.MEDIUM,
    NotificationPriority.LOW: EventPriority.LOW,
}


class CrossDeviceNotificationService:
    """
    Manages push notifications to paired client devices.

    Guarantees:
      - Sensitive data minimization: never includes passwords, private keys, or raw confidential screenshots in notification payloads.
      - Priority-ordered dispatch through ReactiveEventBus.
      - Gated by PCCapability.NOTIFICATION_SEND.
      - Maintains local delivery history and audit trail.
    """

    def __init__(
        self,
        device_manager: DeviceManager,
        capability_gate: CapabilityPolicyGate,
        event_bus: Optional[ReactiveEventBus] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
    ):
        self._dev_mgr = device_manager
        self._gate = capability_gate
        self._bus = event_bus
        self._audit = audit_logger or SecurityAuditLogger()
        self._notifications: List[NotificationItem] = []

    def dispatch_notification(
        self,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        category: str = "system",
        target_device_ids: Optional[List[str]] = None,
    ) -> NotificationItem:
        """
        Create, minimize, and dispatch a push notification.
        """
        # 1. Capability Permission Check
        if not self._gate.check_permission(PCCapability.NOTIFICATION_SEND, "CrossDeviceNotificationService"):
            self._audit.log(
                action="NOTIFICATION_BLOCKED",
                decision="DENIED",
                requesting_component="CrossDeviceNotificationService",
                capability="NOTIFICATION_SEND",
                details={"reason": "NOTIFICATION_SEND capability is not authorized."},
            )
            raise PermissionError("NOTIFICATION_SEND capability is not authorized by user.")

        # 2. Privacy Minimization: Strip sensitive strings
        clean_title = redact_sensitive_text(title)
        clean_msg = redact_sensitive_text(message)

        # 3. Resolve active recipient devices
        all_devices = self._dev_mgr.list_devices()
        active_recipients = [
            d.device_id for d in all_devices
            if d.status.value == "ACTIVE" and (target_device_ids is None or d.device_id in target_device_ids)
        ]

        item = NotificationItem(
            notification_id=f"notif_{uuid.uuid4().hex[:8]}",
            title=clean_title,
            message=clean_msg,
            priority=priority,
            category=category,
            delivered_devices=active_recipients,
            timestamp=datetime.now(timezone.utc),
        )
        self._notifications.append(item)
        if len(self._notifications) > 500:
            self._notifications.pop(0)

        # 4. Publish to Step 11 EventBus if present
        if self._bus:
            evt_priority = PRIORITY_MAP.get(priority, EventPriority.MEDIUM)
            bus_event = SystemEvent(
                event_type=SystemEventType.NOTIFICATION_DISPATCHED,
                priority=evt_priority,
                source_component="NotificationService",
                payload={
                    "notification_id": item.notification_id,
                    "title": item.title,
                    "priority": item.priority.value,
                    "recipients_count": len(active_recipients),
                },
            )
            self._bus.publish(bus_event)

        # 5. Audit
        self._audit.log(
            action="NOTIFICATION_DISPATCHED",
            decision="ALLOWED",
            requesting_component="CrossDeviceNotificationService",
            capability="NOTIFICATION_SEND",
            details={
                "notification_id": item.notification_id,
                "priority": priority.value,
                "recipients": active_recipients,
            },
        )
        return item

    def list_notifications(self, limit: int = 50) -> List[NotificationItem]:
        """Return recent notifications."""
        return list(reversed(self._notifications[-limit:]))

    def mark_all_read(self) -> int:
        """Mark all notifications as read."""
        count = 0
        for n in self._notifications:
            if not n.read:
                n.read = True
                count += 1
        return count
