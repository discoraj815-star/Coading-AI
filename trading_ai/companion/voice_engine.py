"""Interactive Voice & Companion Conversational Engine: Intent routing, minimal-context reasoning, audio adapters."""

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.companion.context_builder import CompanionContextBuilder
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.companion.scenario_engine import ScenarioEngine
from trading_ai.schemas.companion import (
    CompanionInteractionRecord,
    CompanionQueryRequest,
    CompanionQueryResponse,
    ScenarioCondition,
    TradingCompanionContext,
)
from trading_ai.schemas.enums import VoiceIntent
from trading_ai.schemas.unified import (
    InputModality,
    OutputModality,
    UnifiedAIRequest,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class BaseAudioInputAdapter(ABC):
    """Abstract interface for speech-to-text input (hardware or provider)."""

    @abstractmethod
    def transcribe(self, audio_data: Optional[str]) -> str:
        """Convert audio payload into text transcript."""
        pass


class MockAudioInputAdapter(BaseAudioInputAdapter):
    """Mock audio input adapter supporting direct text transcripts and simulation payloads."""

    def transcribe(self, audio_data: Optional[str]) -> str:
        if not audio_data:
            return ""
        # If payload is prefixed with mock:
        if audio_data.startswith("mock_voice:"):
            return audio_data.replace("mock_voice:", "").strip()
        return "Market status query"


class BaseAudioOutputAdapter(ABC):
    """Abstract interface for text-to-speech output synthesis."""

    @abstractmethod
    def synthesize(self, text: str) -> Optional[str]:
        """Convert answer text into an audio file URL or base64 stream."""
        pass


class MockAudioOutputAdapter(BaseAudioOutputAdapter):
    """Mock audio output adapter for clean testing without requiring audio hardware."""

    def synthesize(self, text: str) -> Optional[str]:
        # Return a deterministic mock audio descriptor URL
        if not text:
            return None
        safe_snippet = text[:30].replace(" ", "_").lower()
        return f"mock_speech://audio/response_{safe_snippet}.mp3"


class CompanionIntentRouter:
    """Classifies user natural language or voice queries into structured companion intents."""

    @staticmethod
    def route_intent(query: str) -> VoiceIntent:
        q = query.lower().strip()

        # Market Status / Current Activity
        if any(w in q for w in ["kya chal raha", "what is the market doing", "market status", "current status", "right now", "overview"]):
            return VoiceIntent.MARKET_STATUS

        # Explain Bias
        if any(w in q for w in ["kyun hai", "why is", "explain bias", "bias reason", "why bullish", "why bearish"]):
            return VoiceIntent.EXPLAIN_BIAS

        # Explain Setup
        if ("setup" in q and ("explain" in q or "detail" in q or "what" in q or "active" in q)) or any(w in q for w in ["explain setup", "what setup", "active setup details", "lecture setup"]):
            return VoiceIntent.EXPLAIN_SETUP

        # News Query
        if any(w in q for w in ["news", "economic event", "calendar", "high impact", "forex factory", "nfp", "cpi", "fomc"]):
            return VoiceIntent.NEWS_QUERY

        # Scenario Analysis (IF / THEN / Breaks / Agar)
        if any(w in q for w in ["agar", "if level", "breaks", "what happens if", "what if", "scenario"]):
            return VoiceIntent.SCENARIO_ANALYSIS

        # Setup Validation
        if any(w in q for w in ["is this setup valid", "validate setup", "confirmation check", "setup confirmed"]):
            return VoiceIntent.SETUP_VALIDATION

        # Market History
        if any(w in q for w in ["what did the market do", "yesterday", "previous session", "history", "what happened"]):
            return VoiceIntent.MARKET_HISTORY

        # Risk Status
        if any(w in q for w in ["risk status", "how much risk", "risk check", "account risk"]):
            return VoiceIntent.RISK_STATUS

        # Discipline Status
        if any(w in q for w in ["discipline", "overtrading", "daily loss", "trades count", "cooldown", "rules"]):
            return VoiceIntent.DISCIPLINE_STATUS

        # Why Alert
        if any(w in q for w in ["why alert", "why was i alerted", "alert reason", "explain alert"]):
            return VoiceIntent.WHY_ALERT

        return VoiceIntent.GENERAL_MARKET_QA


class InteractiveVoiceEngine:
    """
    Orchestrates user queries and voice inputs with token-efficient reasoning
    and deterministic factual grounding.
    """

    def __init__(
        self,
        repository: Repository,
        context_builder: Optional[CompanionContextBuilder] = None,
        scenario_engine: Optional[ScenarioEngine] = None,
        discipline_engine: Optional[TradingDisciplineEngine] = None,
        input_adapter: Optional[BaseAudioInputAdapter] = None,
        output_adapter: Optional[BaseAudioOutputAdapter] = None,
        chat_service: Optional[Any] = None,
    ):
        self._repo = repository
        self._context_builder = context_builder or CompanionContextBuilder(repository)
        self._scenario_engine = scenario_engine or ScenarioEngine(repository)
        self._discipline_engine = discipline_engine or TradingDisciplineEngine(repository)
        self._input_adapter = input_adapter or MockAudioInputAdapter()
        self._output_adapter = output_adapter or MockAudioOutputAdapter()
        self._chat_service = chat_service

    def process_query(self, req: CompanionQueryRequest) -> CompanionQueryResponse:
        """
        Process speech or text prompt using minimal sufficient context.
        When connected to the Unified AI Brain, routes through the single intelligence core.
        """
        raw_text = req.query.strip()
        if not raw_text and req.audio_base64:
            raw_text = self._input_adapter.transcribe(req.audio_base64)

        if not raw_text:
            raw_text = "Market Status"

        intent = CompanionIntentRouter.route_intent(raw_text)
        symbol = req.symbol.strip() or "EUR/USD"

        # If connected to Unified AI Brain, route through shared intelligence
        if self._chat_service is not None:
            unified_req = UnifiedAIRequest(
                session_id=getattr(req, "session_id", "default_session"),
                user_id=req.user_id,
                input_modality=InputModality.VOICE if req.audio_base64 else InputModality.TEXT,
                output_modality=OutputModality.BOTH,
                input_text=raw_text,
                audio_base64=req.audio_base64,
                symbol=symbol,
                source="companion_voice_engine",
            )
            unified_res = self._chat_service.process_unified_request(unified_req)

            try:
                v_intent = VoiceIntent(unified_res.intent)
            except Exception:
                v_intent = intent

            audio_url = unified_res.audio_output_url or self._output_adapter.synthesize(unified_res.speech_text or unified_res.response_text)
            return CompanionQueryResponse(
                intent=v_intent,
                symbol=symbol,
                title=f"{symbol} Analysis",
                answer=unified_res.response_text,
                context_summary=unified_res.context_summary,
                evidence=unified_res.evidence,
                scenarios=[],
                audio_transcript=raw_text,
                synthesized_audio_mock=audio_url,
                created_at=datetime.now(timezone.utc),
            )

        # Build smallest sufficient context (standalone fallback)
        ctx = self._context_builder.build_context(symbol=symbol, user_id=req.user_id)

        # Generate deterministic, factual response based on intent
        title, answer, evidence, scenarios = self._generate_response_for_intent(
            intent=intent, raw_query=raw_text, symbol=symbol, ctx=ctx, user_id=req.user_id
        )

        # Synthesize audio mock
        audio_url = self._output_adapter.synthesize(answer)

        response = CompanionQueryResponse(
            intent=intent,
            symbol=symbol,
            title=title,
            answer=answer,
            context_summary=f"{symbol} @ {ctx.current_price:.5f} | Bias: {ctx.bias} | Setup: {ctx.setup_confirmation_state}",
            evidence=evidence,
            scenarios=scenarios,
            audio_transcript=raw_text,
            synthesized_audio_mock=audio_url,
            created_at=datetime.now(timezone.utc),
        )

        # Persist interaction audit log
        self._repo.save_companion_interaction(
            CompanionInteractionRecord(
                symbol=symbol,
                user_query=raw_text,
                intent=intent,
                answer=answer,
                context_summary=response.context_summary,
                evidence=evidence,
                scenarios=scenarios,
                timestamp=datetime.now(timezone.utc).timestamp(),
            )
        )

        return response

    def _generate_response_for_intent(
        self,
        intent: VoiceIntent,
        raw_query: str,
        symbol: str,
        ctx: TradingCompanionContext,
        user_id: str,
    ) -> Tuple[str, str, List[str], List[ScenarioCondition]]:
        """Synthesize response using authoritative local data without LLM hallucination."""
        scenarios: List[ScenarioCondition] = []
        evidence: List[str] = [
            f"Symbol: {symbol}",
            f"Current Price: {ctx.current_price:.5f}",
            f"Overall Bias: {ctx.bias}",
        ]

        if intent == VoiceIntent.MARKET_STATUS:
            title = f"{symbol} Current Market Status"
            news_txt = (
                f"Next high-impact news in {int(ctx.news_time_distance_minutes)}m ({ctx.next_high_impact_event.get('title', '')})"
                if ctx.next_high_impact_event and ctx.news_time_distance_minutes is not None
                else "No imminent high-impact news."
            )
            answer = (
                f"{symbol} is currently trading at {ctx.current_price:.5f}. "
                f"Market structure is {ctx.bias} ({ctx.bias_summary}). "
                f"Active Setup State: {ctx.setup_confirmation_state}. "
                f"Fundamental Context: {ctx.news_risk} risk. {news_txt}"
            )
            if ctx.recent_changes:
                evidence.append(f"Recent Changes: {'; '.join(ctx.recent_changes[:2])}")
            return title, answer, evidence, scenarios

        elif intent == VoiceIntent.EXPLAIN_BIAS:
            title = f"{symbol} Bias Explanation ({ctx.bias})"
            answer = (
                f"{symbol} bias is currently evaluated as {ctx.bias}. "
                f"Key structural rationale: {ctx.bias_summary}. "
                f"Technical-Fundamental correlation: {ctx.tech_fund_relationship}."
            )
            evidence.append(f"Bias Rationale: {ctx.bias_summary}")
            return title, answer, evidence, scenarios

        elif intent == VoiceIntent.EXPLAIN_SETUP:
            title = f"{symbol} Active Setup Explanation"
            if ctx.active_setup:
                setup = ctx.active_setup
                answer = (
                    f"Active setup is '{setup.get('name', 'Identified Pattern')}' "
                    f"({setup.get('direction', 'DIRECTION')}). "
                    f"State: {ctx.setup_confirmation_state}. "
                    f"Invalidation Level: {ctx.invalidation_level or 'Unspecified'}. "
                    f"Confirmation Level: {ctx.confirmation_level or 'Unspecified'}."
                )
                evidence.append(f"Setup ID: {setup.get('setup_id', 'N/A')}")
            else:
                answer = f"No active confirmed setup found for {symbol} at this time. Monitoring background conditions."
            return title, answer, evidence, scenarios

        elif intent == VoiceIntent.NEWS_QUERY:
            title = f"{symbol} Fundamental News Briefing"
            if ctx.next_high_impact_event:
                ev = ctx.next_high_impact_event
                dist_str = f"in {int(ctx.news_time_distance_minutes)} minutes" if ctx.news_time_distance_minutes else "imminent"
                answer = (
                    f"Next relevant high-impact event is '{ev.get('title', 'Economic Event')}' ({ev.get('currency', '')}) "
                    f"scheduled {dist_str}. Volatility Risk: {ctx.news_risk}. "
                    f"Relationship: {ctx.tech_fund_relationship}."
                )
                evidence.append(f"Event: {ev.get('title')} ({ev.get('currency')})")
            else:
                answer = f"No high-impact economic news scheduled soon for {symbol} currencies. Fundamental risk is {ctx.news_risk}."
            return title, answer, evidence, scenarios

        elif intent == VoiceIntent.SCENARIO_ANALYSIS:
            title = f"{symbol} Conditional Scenario Analysis"
            # Extract possible level from query if present (e.g. 1.0850)
            level_match = re.search(r"\b(\d+\.\d+)\b", raw_query)
            custom_lvl = float(level_match.group(1)) if level_match else None

            scenarios = self._scenario_engine.generate_scenarios_for_symbol(
                symbol=symbol, custom_level=custom_lvl
            )
            if scenarios:
                top_scn = scenarios[0]
                answer = (
                    f"Scenario: {top_scn.if_condition}. "
                    f"{top_scn.then_outcome} "
                    f"{top_scn.unless_condition}. "
                    f"Invalidation: {top_scn.invalidated_when}"
                )
                evidence.extend(top_scn.evidence)
            else:
                answer = f"No key POI or boundary levels currently registered to compute conditional scenarios for {symbol}."
            return title, answer, evidence, scenarios

        elif intent == VoiceIntent.SETUP_VALIDATION:
            title = f"{symbol} Setup Validation Status"
            if ctx.active_setup:
                answer = (
                    f"Active setup '{ctx.active_setup.get('name')}' is currently in state '{ctx.setup_confirmation_state}'. "
                    f"Invalidation boundary is at {ctx.invalidation_level or 'Not defined'}. "
                    f"Current price is {ctx.current_price:.5f}."
                )
            else:
                answer = f"Validation check complete: No valid setup currently active for {symbol}."
            return title, answer, evidence, scenarios

        elif intent == VoiceIntent.MARKET_HISTORY:
            title = f"{symbol} Market History Brief"
            # Never invent missing data: report DATA_UNAVAILABLE for unrecorded sessions
            if ctx.recent_changes:
                answer = (
                    f"Recent recorded market actions for {symbol}: "
                    f"{'; '.join(ctx.recent_changes)}. "
                    f"Structure established bias as {ctx.bias}."
                )
                evidence.extend(ctx.recent_changes)
            else:
                answer = (
                    f"Recent detailed historical session orderflow for {symbol} is DATA_UNAVAILABLE in local memory. "
                    f"Only current structure ({ctx.bias} @ {ctx.current_price:.5f}) is actively cached."
                )
            return title, answer, evidence, scenarios

        elif intent in (VoiceIntent.RISK_STATUS, VoiceIntent.DISCIPLINE_STATUS):
            title = "Trading Discipline & Risk Status"
            state, violations = self._discipline_engine.evaluate_discipline_status(user_id=user_id)
            cfg = state.config

            loss_lim_str = f"${cfg.daily_loss_limit:.2f}" if cfg.daily_loss_limit else "RULE_NOT_CONFIGURED"
            max_trd_str = str(cfg.max_daily_trades) if cfg.max_daily_trades else "RULE_NOT_CONFIGURED"
            sl_str = "MANDATORY" if cfg.stop_loss_required else "OPTIONAL"

            viol_str = f"Active Warnings: {'; '.join(violations)}" if violations else "No active discipline violations."

            answer = (
                f"Today's Trades: {state.trades_today_count} (Limit: {max_trd_str}). "
                f"Realized Daily Loss: ${state.daily_realized_loss:.2f} (Limit: {loss_lim_str}). "
                f"Stop Loss Rule: {sl_str}. {viol_str}"
            )
            evidence.append(f"Trades Today: {state.trades_today_count}")
            evidence.append(f"Daily Loss: ${state.daily_realized_loss:.2f}")
            if violations:
                evidence.extend(violations)
            return title, answer, evidence, scenarios

        elif intent == VoiceIntent.WHY_ALERT:
            title = f"{symbol} Alert Investigation"
            recent_alerts = self._repo.get_alerts(symbol=symbol, limit=1)
            if recent_alerts:
                top_alert = recent_alerts[0]
                answer = (
                    f"Recent alert '{top_alert.title}' ({top_alert.alert_type.value}) was generated because: "
                    f"{top_alert.message} Quality: {top_alert.quality.value}. Priority: {top_alert.priority.value}."
                )
                evidence.append(f"Alert ID: {top_alert.alert_id}")
            else:
                answer = f"No alerts recorded for {symbol} in the recent active window."
            return title, answer, evidence, scenarios

        # Fallback General Market QA
        title = f"{symbol} Trading Assistant Guidance"
        answer = (
            f"For {symbol}, current price is {ctx.current_price:.5f} with {ctx.bias} bias. "
            f"Setup confirmation state: {ctx.setup_confirmation_state}. "
            f"You can ask for market status, bias explanation, upcoming news, scenario breakdowns, or discipline checks."
        )
        return title, answer, evidence, scenarios
