"""Two-Way Voice Gateway with Capability-Gated Execution & Privacy Controls."""

from datetime import datetime, timezone
import logging
import re
import time
from typing import Any, Dict, Optional

from trading_ai.companion.voice_engine import (
    BaseAudioInputAdapter,
    BaseAudioOutputAdapter,
    CompanionIntentRouter,
    MockAudioInputAdapter,
    MockAudioOutputAdapter,
)
from trading_ai.schemas.enums import PCCapability, VoiceProcessingState
from trading_ai.schemas.security import VoiceCommandRequest, VoiceCommandResult
from trading_ai.security.audit import SecurityAuditLogger
from trading_ai.security.capability_gate import CapabilityPolicyGate

logger = logging.getLogger(__name__)

# Patterns that map voice commands to dangerous PC capabilities
CAPABILITY_VOICE_TRIGGERS = [
    (re.compile(r"(?i)\b(terminal|cmd|powershell|bash|shell|run command|execute command)\b"), PCCapability.TERMINAL_EXECUTE),
    (re.compile(r"(?i)\b(write file|create file|delete file|modify file|save file)\b"), PCCapability.FILE_WRITE),
    (re.compile(r"(?i)\b(system settings|change config|reboot|shutdown|registry)\b"), PCCapability.SYSTEM_SETTINGS),
    (re.compile(r"(?i)\b(screen capture|take screenshot|look at my screen|watch screen)\b"), PCCapability.SCREEN_READ),
    (re.compile(r"(?i)\b(send notification|push alert|notify phone)\b"), PCCapability.NOTIFICATION_SEND),
]


class SecureVoiceGateway:
    """
    Two-way interactive voice gateway.

    Guarantees:
      - Voice input is treated as UNTRUSTED user input.
      - Any voice command invoking PC actions is gated through CapabilityPolicyGate.
      - Dangerous commands strictly require user authorization.
      - Supports cancellation, noise filtering, and structured TTS output synthesis.
    """

    def __init__(
        self,
        capability_gate: CapabilityPolicyGate,
        stt_adapter: Optional[BaseAudioInputAdapter] = None,
        tts_adapter: Optional[BaseAudioOutputAdapter] = None,
        audit_logger: Optional[SecurityAuditLogger] = None,
        chat_service: Optional[Any] = None,
    ):
        self._gate = capability_gate
        self._stt = stt_adapter or MockAudioInputAdapter()
        self._tts = tts_adapter or MockAudioOutputAdapter()
        self._audit = audit_logger or SecurityAuditLogger()
        self._chat_service = chat_service
        self._state = VoiceProcessingState.IDLE
        self._active_operation_cancelled = False

    def set_chat_service(self, chat_service: Any) -> None:
        """Inject or link the Unified AI Brain (AIChatService)."""
        self._chat_service = chat_service

    def cancel_active_operation(self) -> None:
        """Cancel ongoing voice processing or speech output."""
        self._active_operation_cancelled = True
        self._state = VoiceProcessingState.CANCELLED
        self._audit.log(
            action="VOICE_OPERATION_CANCELLED",
            decision="CANCELLED",
            requesting_component="SecureVoiceGateway",
        )

    def process_voice_command(self, request: VoiceCommandRequest) -> VoiceCommandResult:
        """
        Execute full voice pipeline:
        VOICE INPUT -> STT -> Intent/Command Processing -> Permission/Policy Check -> AI Response -> TTS -> Voice Output
        """
        start_time = time.perf_counter()
        self._active_operation_cancelled = False
        self._state = VoiceProcessingState.LISTENING

        # 1. STT: Extract transcript
        if request.audio_base64:
            transcript = self._stt.transcribe(request.audio_base64)
        else:
            transcript = request.command_text.strip()

        if not transcript:
            self._state = VoiceProcessingState.IDLE
            return VoiceCommandResult(
                intent="EMPTY",
                raw_query="",
                response_text="No voice input detected.",
                state=VoiceProcessingState.IDLE,
                authorized=True,
                latency_ms=0.0,
            )

        self._state = VoiceProcessingState.PROCESSING

        # 2. Capability Gate Check: Does command attempt OS/PC actions?
        for pattern, required_cap in CAPABILITY_VOICE_TRIGGERS:
            if pattern.search(transcript):
                has_perm = self._gate.check_permission(required_cap, "SecureVoiceGateway")
                if not has_perm:
                    self._state = VoiceProcessingState.RESPONDING
                    denial_msg = f"Voice command requires capability '{required_cap.value}', which is currently DENIED. Please grant approval in the Security Panel."
                    audio_url = self._tts.synthesize(denial_msg)
                    self._audit.log(
                        action="VOICE_COMMAND_DENIED",
                        decision="DENIED",
                        requesting_component="SecureVoiceGateway",
                        capability=required_cap.value,
                        details={"transcript": transcript, "required_capability": required_cap.value},
                    )
                    return VoiceCommandResult(
                        intent="CAPABILITY_INVOCATION",
                        raw_query=transcript,
                        response_text=denial_msg,
                        audio_output_url=audio_url,
                        state=VoiceProcessingState.RESPONDING,
                        authorized=False,
                        denial_reason=f"Capability '{required_cap.value}' denied.",
                        latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    )

        # 3. Intent Routing (using existing Step 7 CompanionIntentRouter)
        intent = CompanionIntentRouter.route_intent(transcript)

        # 4. Generate Structured Response
        if self._active_operation_cancelled:
            return VoiceCommandResult(
                intent=intent.value,
                raw_query=transcript,
                response_text="Voice response cancelled by user.",
                state=VoiceProcessingState.CANCELLED,
                authorized=True,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        if self._chat_service is not None:
            from trading_ai.schemas.unified import UnifiedAIRequest, InputModality, OutputModality, UnifiedAudioMetadata
            unified_req = UnifiedAIRequest(
                session_id=request.session_id or request.device_id or "default_session",
                user_id="trader",
                input_modality=InputModality.VOICE,
                output_modality=OutputModality.BOTH,
                input_text=transcript,
                audio_metadata=UnifiedAudioMetadata(
                    confidence=1.0,
                    source="voice_gateway",
                ),
                source="secure_voice_gateway",
            )
            unified_resp = self._chat_service.process_unified_request(unified_req)
            response_text = unified_resp.response_text
            speech_to_synthesize = unified_resp.speech_text or response_text
            audio_url = self._tts.synthesize(speech_to_synthesize)
        else:
            response_text = f"Companion analysis for '{transcript}': market status is stable and risk guardrails are active."
            audio_url = self._tts.synthesize(response_text)
        self._state = VoiceProcessingState.IDLE

        latency = (time.perf_counter() - start_time) * 1000.0

        self._audit.log(
            action="VOICE_COMMAND_EXECUTED",
            decision="ALLOWED",
            requesting_component="SecureVoiceGateway",
            details={"intent": intent.value, "latency_ms": latency},
        )

        return VoiceCommandResult(
            intent=intent.value,
            raw_query=transcript,
            response_text=response_text,
            audio_output_url=audio_url,
            state=VoiceProcessingState.RESPONDING,
            authorized=True,
            latency_ms=latency,
        )
