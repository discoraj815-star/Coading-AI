"""Step 16: Localized Hindi Notification Formatter, Epistemic Separation & Templates."""
from typing import Any, Dict, Optional, Tuple
from trading_ai.schemas.enums import EventPriorityLevel, NotificationPrivacyLevel

DEFAULT_GREETING = "Hello Alam"


class NotificationTemplateFormatter:
    """
    Synthesizes natural, respectful Hindi and English alert messages.
    Enforces strict 4-layer epistemic separation:
      - OBSERVED (क्या देखा)
      - LEARNED (क्या सीखा)
      - INFERRED (क्या निष्कर्ष निकला)
      - DECIDED (क्या निर्णय लिया गया)
    Never presents inferences as certain facts.
    """

    def __init__(self, greeting_name: str = "Alam"):
        self.greeting_name = greeting_name

    def format_alert(
        self,
        event_type: str,
        priority: EventPriorityLevel,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        privacy_level: NotificationPrivacyLevel = NotificationPrivacyLevel.NORMAL,
        verbosity: str = "detailed",
    ) -> Tuple[str, str, str]:
        """
        Returns (title, message_hindi, message_english).
        """
        details = details or {}
        sym = symbol or details.get("symbol", "MARKET")
        greeting = f"Hello {self.greeting_name}"

        # 1. Handle SECURE Privacy Mode (Privacy-redacted message)
        if privacy_level == NotificationPrivacyLevel.SECURE:
            title = f"{greeting} — Trading Alert"
            hindi = f"{greeting}, आपके trading account पर एक महत्वपूर्ण alert मिला है। कृपया App खोलकर सुरक्षित विवरण देखें।"
            english = f"{greeting}, a high-priority trading alert was generated. Please open the secure app to review details."
            return title, hindi, english

        # 2. Setup Confirmation / Trade Proposal Alerts
        if event_type in ["SETUP_CONFIRMED", "TRADE_PROPOSED", "CONFIRMED_SETUP"]:
            pattern = details.get("pattern_name", details.get("setup_type", "ICT Setup"))
            direction = details.get("direction", details.get("bias", "BULLISH"))
            timeframe = details.get("timeframe", "15m")
            invalidation = details.get("invalidation_level", details.get("stop_loss", "Not Defined"))
            poi = details.get("poi_level", details.get("entry_price", "Market POI"))
            reason = details.get("reason", "Learned lecture rules satisfied")

            title = f"{greeting} — {sym} पर Confirmation मिला है"

            if verbosity == "short":
                hindi = f"{greeting}, {sym} पर {timeframe} chart में {pattern} confirmation मिला है।"
                english = f"{greeting}, {pattern} confirmation detected on {sym} {timeframe} chart."
            else:
                hindi = (
                    f"{greeting}, {sym} पर आपकी learned trading rules के अनुसार confirmation मिला है।\n\n"
                    f"• OBSERVED (देखा गया): {sym} {timeframe} पर {poi} के पास price reaction.\n"
                    f"• LEARNED (सीखा गया): {pattern} rule conditions satisfied.\n"
                    f"• INFERRED (निष्कर्ष): {direction} setup aligned with structure.\n"
                    f"• DECIDED (निर्णय): Invalidation level {invalidation} है। यह केवल एक trade suggestion है।"
                )
                english = (
                    f"{greeting}, a validated confirmation was detected on {sym} according to your learned rules.\n\n"
                    f"• OBSERVED: Price reaction near {poi} on {timeframe}.\n"
                    f"• LEARNED: {pattern} setup conditions met.\n"
                    f"• INFERRED: {direction} structure alignment.\n"
                    f"• DECIDED: Invalidation at {invalidation}. Trade suggestion only."
                )
            return title, hindi, english

        # 3. Setup Invalidated Alert
        if event_type in ["SETUP_INVALIDATED"]:
            reason = details.get("reason", "Price breached invalidation level")
            title = f"{greeting} — {sym} Setup Invalidated"
            hindi = f"{greeting}, {sym} पर setup invalidate हो गया है। कारण: {reason}।"
            english = f"{greeting}, setup on {sym} has been invalidated. Reason: {reason}."
            return title, hindi, english

        # 4. Active Position Risk Alert (P2)
        if event_type in ["RISK_CHANGED", "DISCIPLINE_WARNING", "ACTIVE_POSITION_RISK"]:
            warning = details.get("warning", details.get("message", "Risk threshold exceeded"))
            title = f"⚠️ {greeting} — Risk Warning on {sym}"
            hindi = f"{greeting}, ध्यान दें! {sym} पर active risk warning: {warning}। कृपया तुरंत review करें।"
            english = f"{greeting}, attention required! Active risk warning on {sym}: {warning}."
            return title, hindi, english

        # 5. Security Incident Alert (P1)
        if event_type in ["SECURITY_ALERT", "THREAT_DETECTED"]:
            action = details.get("action", "Security event")
            title = f"🚨 {greeting} — Security Alert"
            hindi = f"{greeting}, सुरक्षा चेतावनी! सिस्टम में सुरक्षा घटना दर्ज की गई है: {action}। Access fail-closed मोड में है।"
            english = f"{greeting}, security alert! Incident logged: {action}. System operating in fail-closed mode."
            return title, hindi, english

        # 6. Why-Not-Trade Informational Alert
        if event_type in ["WHY_NOT_TRADE"]:
            reason = details.get("reason", "Confluence criteria pending")
            title = f"{greeting} — {sym} No-Trade Status"
            hindi = f"{greeting}, {sym} पर trade नहीं लिया गया क्योंकि: {reason}।"
            english = f"{greeting}, trade was not taken on {sym} because: {reason}."
            return title, hindi, english

        # Default Informational Alert
        title = f"{greeting} — Update on {sym}"
        msg = details.get("message", f"Event {event_type} occurred on {sym}.")
        hindi = f"{greeting}, {sym} पर update: {msg}"
        english = f"{greeting}, update on {sym}: {msg}"
        return title, hindi, english
