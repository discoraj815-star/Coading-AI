"""Step 16: Notification Orchestrator — Multi-stage qualification, deduplication, and multi-channel routing."""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from trading_ai.events.gateway import WebSocketGateway
from trading_ai.notifications.preferences import PreferencesManager
from trading_ai.notifications.providers import (
    EmailNotificationProvider,
    NotificationProvider,
    SMSNotificationProvider,
    VoiceNotificationProvider,
    WebPushNotificationProvider,
    WebSocketNotificationProvider,
)
from trading_ai.notifications.templates import NotificationTemplateFormatter
from trading_ai.notifications.voice import CloudVoiceSynthesizer
from trading_ai.schemas.cloud import CloudStreamEvent, NotificationRecord, UserNotificationPreferences
from trading_ai.schemas.enums import EventPriorityLevel, NotificationChannel, NotificationDeliveryStatus
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class NotificationOrchestrator:
    """
    Central coordinator for all user notifications.

    Pipeline:
      Event
      → Importance Filter
      → Deduplication Check
      → User Preference Filter
      → Evidence Validation
      → Hindi Formatter (4-Layer Epistemic Separation)
      → Delivery Router
      → Multi-Channel Providers (WebSocket, Push, SMS, Voice, Email)
      → Audit Persistence
    """

    def __init__(
        self,
        repository: Repository,
        gateway: WebSocketGateway,
        preferences_manager: Optional[PreferencesManager] = None,
        voice_synthesizer: Optional[CloudVoiceSynthesizer] = None,
    ):
        self._repo = repository
        self._gateway = gateway
        self._prefs_mgr = preferences_manager or PreferencesManager(repository)
        self._formatter = NotificationTemplateFormatter(greeting_name="Alam")
        self._synth = voice_synthesizer or CloudVoiceSynthesizer()

        # Instantiate providers
        self._providers: Dict[NotificationChannel, NotificationProvider] = {
            NotificationChannel.WEBSOCKET: WebSocketNotificationProvider(gateway),
            NotificationChannel.PUSH: WebPushNotificationProvider(repository),
            NotificationChannel.SMS: SMSNotificationProvider(repository),
            NotificationChannel.VOICE: VoiceNotificationProvider(self._synth),
            NotificationChannel.EMAIL: EmailNotificationProvider(),
        }

        self._dispatched_records: int = 0
        self._suppressed_duplicates: int = 0

    @property
    def preferences_manager(self) -> PreferencesManager:
        return self._prefs_mgr

    @property
    def providers(self) -> Dict[NotificationChannel, NotificationProvider]:
        return self._providers

    def process_event(self, event: CloudStreamEvent) -> List[NotificationRecord]:
        """
        Evaluate and dispatch an approved CloudStreamEvent through the full notification pipeline.
        Returns the list of generated and dispatched NotificationRecords.
        """
        prefs = self._prefs_mgr.get_preferences()
        self._formatter.greeting_name = prefs.greeting_name

        # 1. Importance Filter: drop low-confidence informational events
        if event.priority == EventPriorityLevel.P5_INFORMATIONAL and event.confidence < prefs.minimum_alert_confidence:
            return []

        # 2. Deterministic Deduplication Key
        symbol = event.symbol or "MARKET"
        setup_id = event.payload.get("setup_id", event.event_id)
        version = event.payload.get("setup_version", "1")
        confirm_state = event.payload.get("confirmation_state", event.event_type)
        dedup_key = f"{symbol}_{setup_id}_{version}_{event.event_type}_{confirm_state}"

        # Check deduplication window in DB (1 hour)
        if self._repo.is_notification_duplicate(dedup_key, within_seconds=3600):
            self._suppressed_duplicates += 1
            logger.info("Notification suppressed as duplicate: %s", dedup_key)
            return []

        # 3. Evidence Validation: Confirmed setups require supporting evidence reference
        if event.priority == EventPriorityLevel.P3_CONFIRMED_SETUP:
            evidence = event.evidence_reference or event.payload.get("evidence", {})
            if not evidence and not event.payload.get("pattern_name"):
                logger.warning("Setup notification %s missing evidence, suppressed.", event.event_id)
                return []

        # 4. Synthesize Hindi and English Messages with 4-Layer Epistemic Separation
        title, hindi_msg, english_msg = self._formatter.format_alert(
            event_type=event.event_type,
            priority=event.priority,
            symbol=event.symbol,
            details=event.payload,
            privacy_level=prefs.privacy_level,
            verbosity=prefs.notification_verbosity,
        )

        # 5. Route to Target Channels based on Priority and User Preferences
        target_channels = self._select_channels_for_priority(event.priority, prefs)
        dispatched: List[NotificationRecord] = []

        for channel in target_channels:
            if not self._prefs_mgr.should_deliver(event.priority, channel):
                continue

            record = NotificationRecord(
                notification_id=f"notif_{uuid.uuid4().hex[:10]}",
                event_id=event.event_id,
                priority=event.priority,
                title=title,
                message_hindi=hindi_msg,
                message_english=english_msg,
                symbol=event.symbol,
                channel=channel,
                privacy_level=prefs.privacy_level,
                deduplication_key=dedup_key,
                delivery_status=NotificationDeliveryStatus.QUEUED,
            )

            # Invoke provider
            provider = self._providers.get(channel)
            if provider:
                status = provider.send(record)
                record.delivery_status = status
                if status == NotificationDeliveryStatus.SUCCESS:
                    self._prefs_mgr.record_channel_dispatch(channel)

            # Persist delivery audit log
            self._repo.save_notification_record(record)
            dispatched.append(record)
            self._dispatched_records += 1

        return dispatched

    def _select_channels_for_priority(
        self,
        priority: EventPriorityLevel,
        prefs: UserNotificationPreferences,
    ) -> List[NotificationChannel]:
        """Map event priority to delivery channels."""
        if priority == EventPriorityLevel.P0_SYSTEM_CRITICAL:
            # Critical system fault: Push + SMS + Voice
            return [NotificationChannel.PUSH, NotificationChannel.SMS, NotificationChannel.VOICE, NotificationChannel.WEBSOCKET]

        elif priority == EventPriorityLevel.P1_SECURITY:
            # Security breach / incident: Push + SMS + Voice
            return [NotificationChannel.PUSH, NotificationChannel.SMS, NotificationChannel.VOICE, NotificationChannel.WEBSOCKET]

        elif priority == EventPriorityLevel.P2_ACTIVE_POSITION_RISK:
            # Immediate risk to active capital: Push + Voice (optionally SMS if enabled)
            channels = [NotificationChannel.PUSH, NotificationChannel.VOICE, NotificationChannel.WEBSOCKET]
            if prefs.sms_enabled:
                channels.append(NotificationChannel.SMS)
            return channels

        elif priority == EventPriorityLevel.P3_CONFIRMED_SETUP:
            # Valid high-confluence setup: Push + In-app + WebSocket (optionally Voice)
            channels = [NotificationChannel.PUSH, NotificationChannel.WEBSOCKET]
            if prefs.voice_enabled:
                channels.append(NotificationChannel.VOICE)
            return channels

        elif priority == EventPriorityLevel.P4_HIGH_VALUE_MARKET_EVENT:
            # High-impact news, POI reaction: Push + WebSocket
            return [NotificationChannel.PUSH, NotificationChannel.WEBSOCKET]

        else:
            # P5 Informational: WebSocket / In-App only
            return [NotificationChannel.WEBSOCKET]

    def health(self) -> Dict[str, Any]:
        return {
            "status": "HEALTHY",
            "dispatched_count": self._dispatched_records,
            "suppressed_duplicates": self._suppressed_duplicates,
            "providers": {c.value: "ONLINE" for c in self._providers.keys()},
        }
