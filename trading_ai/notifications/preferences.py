"""Step 16: User Notification Preferences & Constraints Manager."""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional

from trading_ai.schemas.cloud import UserNotificationPreferences
from trading_ai.schemas.enums import EventPriorityLevel, NotificationChannel
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class PreferencesManager:
    """
    Evaluates user delivery constraints, quiet hours, emergency overrides, and channel quotas.
    """

    def __init__(self, repository: Repository, user_id: str = "default_user"):
        self._repo = repository
        self.user_id = user_id

    def get_preferences(self) -> UserNotificationPreferences:
        """Fetch user preferences or return default config."""
        data = self._repo.get_user_notification_preferences(self.user_id)
        if data:
            return UserNotificationPreferences(**data)
        default_prefs = UserNotificationPreferences()
        self.save_preferences(default_prefs)
        return default_prefs

    def save_preferences(self, prefs: UserNotificationPreferences) -> None:
        """Persist updated user preferences."""
        self._repo.save_user_notification_preferences(self.user_id, prefs.model_dump(mode="json"))

    def should_deliver(
        self,
        priority: EventPriorityLevel,
        channel: NotificationChannel,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Check if notification should be delivered based on quiet hours, channel enablement, and quotas.
        """
        prefs = self.get_preferences()
        now = now or datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")

        # Reset daily quotas if date has rolled over
        if prefs.last_reset_date != today_str:
            prefs.sms_sent_today = 0
            prefs.voice_alerts_sent_today = 0
            prefs.last_reset_date = today_str
            self.save_preferences(prefs)

        if not prefs.notification_enabled:
            return False

        # 1. Channel enablement check
        if channel == NotificationChannel.PUSH and not prefs.push_enabled:
            return False
        if channel == NotificationChannel.SMS and not prefs.sms_enabled:
            return False
        if channel == NotificationChannel.VOICE and not prefs.voice_enabled:
            return False
        if channel == NotificationChannel.EMAIL and not prefs.email_enabled:
            return False
        if channel == NotificationChannel.IN_APP and not prefs.in_app_enabled:
            return False

        # 2. Emergency override check for high priority (P0, P1, P2)
        is_emergency = priority in [
            EventPriorityLevel.P0_SYSTEM_CRITICAL,
            EventPriorityLevel.P1_SECURITY,
            EventPriorityLevel.P2_ACTIVE_POSITION_RISK,
        ]

        # 3. Quiet hours check
        if prefs.quiet_hours_enabled:
            curr_hour = now.hour
            in_quiet_hours = False
            if prefs.quiet_hours_start_utc > prefs.quiet_hours_end_utc:
                # Spans midnight (e.g. 22:00 to 04:00)
                if curr_hour >= prefs.quiet_hours_start_utc or curr_hour < prefs.quiet_hours_end_utc:
                    in_quiet_hours = True
            else:
                if prefs.quiet_hours_start_utc <= curr_hour < prefs.quiet_hours_end_utc:
                    in_quiet_hours = True

            if in_quiet_hours:
                if is_emergency and prefs.critical_alert_override:
                    pass  # Override allowed
                else:
                    return False

        # 4. Quotas for paid/disruptive channels (SMS & Voice)
        if channel == NotificationChannel.SMS:
            if prefs.sms_sent_today >= prefs.max_sms_per_day:
                if not is_emergency:
                    return False
        if channel == NotificationChannel.VOICE:
            if prefs.voice_alerts_sent_today >= prefs.max_voice_alerts_per_day:
                if not is_emergency:
                    return False

        return True

    def record_channel_dispatch(self, channel: NotificationChannel) -> None:
        """Increment daily counter for quota tracking."""
        prefs = self.get_preferences()
        if channel == NotificationChannel.SMS:
            prefs.sms_sent_today += 1
            self.save_preferences(prefs)
        elif channel == NotificationChannel.VOICE:
            prefs.voice_alerts_sent_today += 1
            self.save_preferences(prefs)
