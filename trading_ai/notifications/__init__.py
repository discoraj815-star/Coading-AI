"""Step 16 Multi-Channel Notifications Package."""
from trading_ai.notifications.orchestrator import NotificationOrchestrator
from trading_ai.notifications.preferences import PreferencesManager
from trading_ai.notifications.providers import (
    NotificationProvider,
    WebSocketNotificationProvider,
    WebPushNotificationProvider,
    SMSNotificationProvider,
    VoiceNotificationProvider,
    EmailNotificationProvider,
)
from trading_ai.notifications.templates import NotificationTemplateFormatter
from trading_ai.notifications.voice import CloudVoiceSynthesizer

__all__ = [
    "NotificationOrchestrator",
    "PreferencesManager",
    "NotificationProvider",
    "WebSocketNotificationProvider",
    "WebPushNotificationProvider",
    "SMSNotificationProvider",
    "VoiceNotificationProvider",
    "EmailNotificationProvider",
    "NotificationTemplateFormatter",
    "CloudVoiceSynthesizer",
]
