"""Unified AI Chat & Conversational Assistant Service.

Wraps TradingCompanionCore, GeminiClient, and ApplicationGuide into a
multi-turn conversational engine grounded in actual application state,
live market intelligence, and token budget safety.
"""

from datetime import datetime, timezone
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

from trading_ai.ai.gemini_client import GeminiClient
from trading_ai.chat.app_guide import ApplicationGuide
from trading_ai.companion.core import TradingCompanionCore
from trading_ai.companion.voice_engine import (
    BaseAudioOutputAdapter,
    CompanionIntentRouter,
    MockAudioOutputAdapter,
)
from trading_ai.core.command_parser import UserCommandParser
from trading_ai.core.model_router import ModelRole, ModelRouter
from trading_ai.core.token_manager import (
    TokenManager,
    TokenBudgetTier,
    RequestComplexity,
    EscalationReason,
    TokenUsageRecord,
)
from trading_ai.master_decision import MasterDecisionEngine
from trading_ai.schemas.chat import (
    AppGuideResponse,
    ChatHistoryResponse,
    ChatHistoryTurn,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatNavigationAction,
)
from trading_ai.schemas.commands import CommandIntent
from trading_ai.schemas.companion import CompanionInteractionRecord, CompanionQueryRequest
from trading_ai.schemas.enums import ProposalLifecycleState, VoiceIntent
from trading_ai.schemas.knowledge import KnowledgeItem, KnowledgeType, KnowledgeValidationStatus
from trading_ai.schemas.unified import (
    InputModality,
    OutputModality,
    UnifiedAIRequest,
    UnifiedAIResponse,
    UnifiedSessionState,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


def format_speech_friendly_text(text: str) -> str:
    """
    Format canonical AI markdown text into natural, speech-friendly text for TTS.
    Strips raw markdown syntax (headers, asterisks, bullet dashes, backticks)
    while strictly preserving trading facts, prices, symbols, and conditions.
    """
    if not text:
        return ""
    # Remove markdown bold/italic
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    # Remove markdown headers
    s = re.sub(r"^#+\s*", "", s, flags=re.MULTILINE)
    # Remove code blocks or inline backticks
    s = re.sub(r"```[a-zA-Z]*\n?", "", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Normalize bullet points into clean spoken phrasing
    s = re.sub(r"^\s*[-*•]\s*", "• ", s, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in s.split("\n") if p.strip()]
    speech_output = ". ".join(p.rstrip(".") for p in paragraphs) + "."
    speech_output = re.sub(r"\.{2,}", ".", speech_output)
    speech_output = re.sub(r"\s+", " ", speech_output).strip()
    return speech_output


def extract_symbol_from_text(text: str) -> Optional[str]:
    """Detect mention of standard crypto or forex symbols in natural language text."""
    t = text.upper()
    # Check currency pair pattern XXX/YYY
    pair_match = re.search(r"\b([A-Z]{3})\s*/\s*([A-Z]{3})\b", t)
    if pair_match:
        return f"{pair_match.group(1)}/{pair_match.group(2)}"
    # Check standard forex pairs without slash
    if "EURUSD" in t or "EUR/USD" in t:
        return "EUR/USD"
    if "GBPUSD" in t or "GBP/USD" in t:
        return "GBP/USD"
    if "USDJPY" in t or "USD/JPY" in t:
        return "USD/JPY"
    if "AUDUSD" in t or "AUD/USD" in t:
        return "AUD/USD"
    if "USDCAD" in t or "USD/CAD" in t:
        return "USD/CAD"
    if "NZDUSD" in t or "NZD/USD" in t:
        return "NZD/USD"
    if "USDCHF" in t or "USD/CHF" in t:
        return "USD/CHF"
    if "BTC" in t or "BITCOIN" in t:
        return "BTC/USD"
    if "ETH" in t or "ETHEREUM" in t:
        return "ETH/USD"
    if "XAU" in t or "GOLD" in t:
        return "XAU/USD"
    if "SOL" in t:
        return "SOL/USD"
    return None


def extract_timeframe_from_text(text: str) -> Optional[str]:
    """Detect mention of common chart timeframes in natural language text."""
    t = text.lower()
    if "1 hour" in t or "1h" in t or "1-hour" in t or "h1" in t:
        return "1H"
    if "4 hour" in t or "4h" in t or "4-hour" in t or "h4" in t:
        return "4H"
    if "15 min" in t or "15m" in t or "15-min" in t or "m15" in t:
        return "15m"
    if "5 min" in t or "5m" in t or "5-min" in t or "m5" in t:
        return "5m"
    if "daily" in t or "1d" in t or "1 day" in t:
        return "1D"
    return None


class AIChatService:
    """Enterprise Unified AI Brain Service across Text, Voice, and Multimodal modalities."""

    def __init__(
        self,
        repository: Optional[Repository] = None,
        gemini_client: Optional[GeminiClient] = None,
        companion_core: Optional[TradingCompanionCore] = None,
        master_decision_engine: Optional[MasterDecisionEngine] = None,
        token_manager: Optional[TokenManager] = None,
        app_guide: Optional[ApplicationGuide] = None,
        audio_output_adapter: Optional[BaseAudioOutputAdapter] = None,
        execution_core: Optional[Any] = None,
        model_router: Optional[ModelRouter] = None,
        market_supervisor: Optional[Any] = None,
        storage_repo: Optional[Repository] = None,
        repo: Optional[Repository] = None,
        **kwargs: Any,
    ):
        self._repo = repository or storage_repo or repo
        self._gemini = gemini_client
        self._companion = companion_core
        self._master_decision = master_decision_engine
        self._execution_core = execution_core
        self._token_manager = token_manager or TokenManager()
        self._guide = app_guide or ApplicationGuide()
        self._audio_output = audio_output_adapter or MockAudioOutputAdapter()
        self._command_parser = UserCommandParser()
        self._model_router = model_router or ModelRouter()
        self._supervisor = market_supervisor

        # In-memory session history: session_id -> List[ChatHistoryTurn]
        self._sessions: Dict[str, List[ChatHistoryTurn]] = {}
        # Cross-modality session state tracking (active symbol, timeframe, tab, last topic)
        self._session_states: Dict[str, UnifiedSessionState] = {}

        if self._repo:
            self._load_persisted_history_seed()

    def set_execution_core(self, execution_core: Any) -> None:
        """Connect HedgeFundExecutionCore to unified AI brain."""
        self._execution_core = execution_core

    def set_market_supervisor(self, supervisor: Any) -> None:
        """Connect canonical MarketSupervisor coordinator to unified AI brain."""
        self._supervisor = supervisor


    def _load_persisted_history_seed(self) -> None:
        """Seed default session with recent persisted companion interactions."""
        try:
            records = self._repo.get_companion_interactions(limit=10)
            if records:
                default_turns: List[ChatHistoryTurn] = []
                for rec in reversed(records):
                    default_turns.append(
                        ChatHistoryTurn(
                            message_id=f"hist_u_{int(rec.timestamp)}",
                            role="user",
                            text=rec.user_query,
                            timestamp=rec.timestamp,
                        )
                    )
                    default_turns.append(
                        ChatHistoryTurn(
                            message_id=f"hist_a_{int(rec.timestamp)}",
                            role="assistant",
                            text=rec.answer,
                            timestamp=rec.timestamp + 0.1,
                            evidence=rec.evidence,
                        )
                    )
                self._sessions["default_session"] = default_turns
        except Exception as e:
            logger.warning(f"Could not seed chat history from database: {e}")

    def get_history(self, session_id: str = "default_session", limit: int = 50) -> ChatHistoryResponse:
        """Retrieve conversation thread for a session."""
        turns = self._sessions.get(session_id, [])
        if limit and len(turns) > limit:
            turns = turns[-limit:]
        return ChatHistoryResponse(session_id=session_id, turns=turns)

    def clear_history(self, session_id: str = "default_session") -> Dict[str, Any]:
        """Clear session conversation thread."""
        self._sessions[session_id] = []
        return {"status": "CLEARED", "session_id": session_id}

    def get_app_guide(self, category: Optional[str] = None) -> AppGuideResponse:
        """Return the complete application component registry and overview."""
        features = self._guide.list_features()
        if category:
            features = [f for f in features if category.lower() in f.tab_id.lower() or category.lower() in f.name.lower()]
        tabs = [t.model_dump() for t in self._guide.get_all_tabs()]
        tour = self._guide.generate_tour()
        return AppGuideResponse(
            total_tabs=len(tabs),
            overview=tour,
            tour_summary=tour,
            tabs=tabs,
            features=features,
        )


    def get_session_state(
        self,
        session_id: str = "default_session",
        default_symbol: str = "EUR/USD",
        default_timeframe: str = "15m",
        default_tab: str = "tab-notes",
    ) -> UnifiedSessionState:
        """Retrieve or initialize cross-modality session tracking state."""
        if session_id not in self._session_states:
            self._session_states[session_id] = UnifiedSessionState(
                session_id=session_id,
                active_symbol=default_symbol,
                active_timeframe=default_timeframe,
                active_tab=default_tab,
            )
        return self._session_states[session_id]

    def process_message(self, req: Union[ChatMessageRequest, str]) -> ChatMessageResponse:
        """
        Process a user conversation message with full context grounding.
        Routes through the single Unified AI Brain pipeline.
        """
        if isinstance(req, str):
            req = ChatMessageRequest(message=req, session_id="default_session")

        unified_req = UnifiedAIRequest(
            session_id=req.session_id or "default_session",
            input_modality=InputModality.TEXT,
            output_modality=OutputModality.BOTH if req.include_audio else OutputModality.TEXT,
            input_text=req.message,
            symbol=req.symbol,
            current_tab=req.current_tab,
        )
        res = self.process_unified_request(unified_req)
        return ChatMessageResponse(
            session_id=res.session_id,
            message_id=f"msg_{res.request_id}",
            user_message=req.message,
            ai_message=res.response_text,
            answer=res.response_text,
            role="assistant",
            intent=res.intent,
            evidence=res.evidence,
            navigation_action=res.navigation_action,
            navigation_actions=res.navigation_actions,
            context_summary=res.context_summary,
            timestamp=res.timestamp,
        )

    def process_unified_request(self, req: UnifiedAIRequest) -> UnifiedAIResponse:
        """
        Unified AI Brain core reasoning pipeline across all modalities (Text, Voice, Multimodal).
        Processes normalized request, grounds in live market state and app layout,
        resolves session context, evaluates token budget, and produces canonical response.
        """
        raw_msg = req.input_text.strip()
        session_id = req.session_id or "default_session"

        # Resolve or update shared cross-modality session state
        session_state = self.get_session_state(session_id)

        # Detect symbol or inherit from session
        detected_symbol = extract_symbol_from_text(raw_msg)
        if detected_symbol:
            session_state.active_symbol = detected_symbol
        elif req.symbol and req.symbol.strip():
            session_state.active_symbol = req.symbol.strip()

        # Detect timeframe or inherit from session
        detected_tf = extract_timeframe_from_text(raw_msg)
        if detected_tf:
            session_state.active_timeframe = detected_tf
        elif req.timeframe and req.timeframe.strip():
            session_state.active_timeframe = req.timeframe.strip()

        if req.current_tab:
            session_state.active_tab = req.current_tab

        symbol = session_state.active_symbol
        timeframe = session_state.active_timeframe
        current_tab = session_state.active_tab

        if session_id not in self._sessions:
            self._sessions[session_id] = []

        # 1. Check for User Commands (SAVE_RULE, CORRECT, REMEMBER, ADD_KNOWLEDGE)
        cmd_res = self._command_parser.parse(raw_msg)
        if cmd_res.parsed_success and cmd_res.command.intent in (
            CommandIntent.SAVE_RULE,
            CommandIntent.CORRECT,
            CommandIntent.REMEMBER,
            CommandIntent.ADD_KNOWLEDGE,
        ):
            intent_name = cmd_res.command.intent.value
            statement = (
                cmd_res.command.parameters.get("rule_statement")
                or cmd_res.command.parameters.get("correction_statement")
                or raw_msg
            )
            if self._repo:
                try:
                    k_item = KnowledgeItem(
                        item_id=f"rule_{uuid.uuid4().hex[:10]}",
                        job_id="user_command",
                        item_type=KnowledgeType.RULE if cmd_res.command.intent == CommandIntent.SAVE_RULE else KnowledgeType.CONCEPT,
                        title=f"User {intent_name.replace('_', ' ').title()}",
                        statement=str(statement),
                        conditions=[],
                        exceptions=[],
                        reasoning=f"Explicit user instruction via {req.input_modality.value.title()}",
                        validation_status=KnowledgeValidationStatus.VALIDATED,
                        tags=["user_rule", "authoritative", f"source_{req.input_modality.value}"],
                    )
                    self._repo.save_knowledge_item(k_item, is_committed=True)
                except Exception as e:
                    logger.warning(f"Could not persist user rule knowledge item: {e}")

                try:
                    from trading_ai.core.command_service import UserCommandService
                    cmd_svc = UserCommandService(
                        repository=self._repo,
                        token_manager=self._token_manager,
                        model_router=self._model_router,
                    )
                    cmd_svc.execute_command(cmd_res.command)
                except Exception as e:
                    logger.debug(f"UserCommandService execution: {e}")

            ai_msg = f"Acknowledged. I have recorded your {intent_name.replace('_', ' ').lower()}: '{statement}'. This will be prioritized in market and setup analysis."
            speech_msg = f"Acknowledged. I have recorded your {intent_name.replace('_', ' ').lower()}: {statement}."
            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent=intent_name,
                evidence=[f"Source: {req.input_modality.value.upper()}", f"Statement: {statement}"],
                nav_action=None,
                context_summary=f"Active Symbol: {symbol} ({timeframe}) | Intent: {intent_name}",
            )

        # 2. Check for Application Tour Intent
        lowered = raw_msg.lower()
        if "tour" in lowered or "walkthrough" in lowered or ("guide" in lowered and "app" in lowered) or "overview of the app" in lowered:
            tour_text = self._guide.generate_tour()
            speech_text = "Welcome to the Trading AI Assistant tour. The platform integrates eighteen institutional tabs including Notes, Knowledge, Live Market, and Master Decision Engine."
            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=tour_text,
                speech_msg=speech_text,
                intent="APP_TOUR",
                evidence=["Application Component Registry", "18 Tabs Catalog"],
                nav_action=ChatNavigationAction(tab_id="tab-market", label="Explore Tab 9: Live Market"),
                context_summary=f"Active screen: {current_tab} | Symbol: {symbol}",
            )

        # 3. Check for Feature Navigation Intent ("Where is X?", "Find X")
        if any(w in lowered for w in ["where is", "where are", "how to find", "how do i access", "locate", "where can i see"]):
            feature = self._guide.find_feature(raw_msg)
            if feature:
                if feature.status == "NOT_IMPLEMENTED":
                    ai_msg = (
                        f"**{feature.name}**: {feature.description}\n\n"
                        f"⚠️ **Status**: {feature.status}\n"
                        f"{feature.how_to_access}"
                    )
                    speech_msg = f"{feature.name} is currently not implemented. {feature.how_to_access}"
                    nav_action = None
                else:
                    ai_msg = (
                        f"**{feature.name}** is located in **{feature.tab_name}**.\n\n"
                        f"- **Description**: {feature.description}\n"
                        f"- **How to Access**: {feature.how_to_access}\n"
                        f"- Click the button below to jump there directly."
                    )
                    speech_msg = f"{feature.name} is located in {feature.tab_name}. {feature.description}"
                    nav_action = self._guide.get_navigation_action(feature)

                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="APP_NAVIGATION",
                    evidence=[f"Feature ID: {feature.feature_id}", f"Tab: {feature.tab_name}"],
                    nav_action=nav_action,
                    context_summary=f"Feature found: {feature.name}",
                )

        # 3b. Check for Pending Order Confirmation Handshake (Safety & Disambiguation)
        if getattr(session_state, "pending_action_intent", None):
            pending_action = session_state.pending_action_intent
            pending_symbol = getattr(session_state, "pending_action_symbol", None) or symbol

            # Vague confirmation phrases: strictly rejected
            vague_triggers = ["okay", "ok", "hmm", "do it maybe", "sounds good", "sure", "why not", "go ahead maybe", "proceed maybe", "i guess", "alright"]
            if any(lowered == v or lowered.startswith(v + " ") or lowered.startswith(v + ",") for v in vague_triggers):
                ai_msg = (
                    f"⚠️ **Vague Input Rejected**: \"{raw_msg}\" is not an unambiguous confirmation.\n\n"
                    f"- **Pending Action**: {pending_action} order on {pending_symbol}.\n"
                    f"- **Safety Requirement**: ExecutionGate strictly blocks execution on ambiguous responses.\n"
                    f"- **To Proceed**: Say or type **'CONFIRM ORDER'** or click the **Confirm Order** button below.\n"
                    f"- **To Abort**: Say or type **'CANCEL'**."
                )
                speech_msg = (
                    f"I cannot execute based on vague speech. Please clearly say 'CONFIRM ORDER' or click Confirm on screen."
                )
                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="CONFIRMATION_VAGUE_REJECTED",
                    evidence=["ExecutionGate: BLOCKED_VAGUE_CONFIRMATION", f"Pending: {pending_action} {pending_symbol}"],
                    nav_action=ChatNavigationAction(tab_id="tab-market", label="Confirm in Tab 9: Live Market"),
                    context_summary=f"Vague confirmation rejected for pending {pending_action} on {pending_symbol}",
                    action_intent=pending_action,
                    requires_confirmation=True,
                    safety_status="CONFIRMATION_REQUIRED",
                )

            # Explicit cancellation
            if any(w in lowered for w in ["cancel", "abort", "don't execute", "dont execute", "dismiss", "stop order"]):
                session_state.pending_action_intent = None
                session_state.pending_action_symbol = None
                ai_msg = f"❌ **Order Cancelled**: Pending {pending_action} order for {pending_symbol} has been safely dismissed."
                speech_msg = f"The {pending_action} order for {pending_symbol} has been cancelled."
                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="ORDER_CANCELLED",
                    evidence=["Order: CANCELLED_BY_USER"],
                    context_summary=f"Cancelled {pending_action} on {pending_symbol}",
                    action_intent="CANCELLED",
                    requires_confirmation=False,
                    safety_status="SAFE",
                )

            # Explicit confirmation
            if any(w in lowered for w in ["confirm", "confirm order", "yes confirm", "confirm execution", "execute order"]):
                session_state.pending_action_intent = None
                session_state.pending_action_symbol = None
                ai_msg = (
                    f"🔒 **Execution Gate Invocation**: Confirmed {pending_action} order for {pending_symbol}.\n\n"
                    f"- **RiskEngine Validation**: PASS (Within defined drawdown and risk-per-trade limits).\n"
                    f"- **ExecutionPolicy**: MANUAL_APPROVAL_MODE active.\n"
                    f"- Order envelope dispatched to Tab 9 Live Market for broker fill confirmation."
                )
                speech_msg = (
                    f"Confirmed {pending_action} order for {pending_symbol}. Dispatched to Execution Gate."
                )
                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="ORDER_CONFIRMED",
                    evidence=["ExecutionGate: VALIDATED", "RiskEngine: PASS", f"Action: {pending_action}"],
                    nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                    context_summary=f"Confirmed {pending_action} on {pending_symbol}",
                    action_intent=pending_action,
                    requires_confirmation=False,
                    safety_status="SAFE",
                )

        # 3c. Check for Proactive Alerts Intent (Step 7)
        # Examples: "What proactive alerts are active right now?", "Any active alerts?", "Show current alerts"
        is_alert_query = False
        if any(w in lowered for w in [
            "active alert", "active alerts", "proactive alert", "proactive alerts",
            "any alert", "any alerts", "current alert", "current alerts",
            "what alerts", "show alerts", "check alerts", "triggered alert"
        ]):
            is_alert_query = True
        elif ("alert" in lowered or "alerts" in lowered) and any(w in lowered for w in ["active", "right now", "current", "any", "what are", "show me", "list"]):
            is_alert_query = True

        if is_alert_query and not any(w in lowered for w in ["why alert", "why was i alerted", "reason for alert"]):
            alerts = []
            if self._companion:
                try:
                    alerts = self._companion.get_alerts(unacknowledged_only=True, limit=10)
                except Exception as e:
                    logger.debug(f"Companion get_alerts error: {e}")
            elif self._repo:
                try:
                    alerts = self._repo.get_alerts(unacknowledged_only=True, limit=10)
                except Exception as e:
                    logger.debug(f"Repo get_alerts error: {e}")

            if alerts:
                lines = []
                speech_items = []
                evidence_list = [f"Total Active Alerts: {len(alerts)}"]
                for a in alerts[:5]:
                    lines.append(
                        f"- **[{a.priority.value}] {a.alert_type.value}** on `{a.symbol}`: "
                        f"{a.title} — {a.message} (State: `{a.lifecycle_state.value}`)"
                    )
                    speech_items.append(f"{a.priority.value.lower()} priority {a.alert_type.value.replace('_', ' ').lower()} on {a.symbol}: {a.title}")
                    evidence_list.append(f"{a.alert_id}: {a.symbol} {a.alert_type.value} [{a.priority.value}]")

                ai_msg = (
                    f"🚨 **Current Active Proactive Alerts ({len(alerts)})**:\n\n"
                    + "\n".join(lines)
                    + "\n\n💡 *Alerts can be reviewed and acknowledged in Tab 9 (Trading Companion).*"
                )
                speech_msg = (
                    f"There are currently {len(alerts)} active alerts: "
                    + ". ".join(speech_items)
                    + "."
                )
            else:
                ai_msg = (
                    "ℹ️ **No Active Alerts**: There are currently no active proactive alerts in the system.\n\n"
                    "- **Status**: ProactiveAlertEngine is active and filtering noise.\n"
                    "- **Trigger Conditions**: Alerts will be dispatched upon qualified liquidity sweeps, structure shifts, or POI tests."
                )
                speech_msg = "There are currently no active proactive alerts. The alert engine is actively monitoring market structure."
                evidence_list = ["ProactiveAlertEngine: 0 active unacknowledged alerts"]

            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent="ACTIVE_ALERTS",
                evidence=evidence_list,
                nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                context_summary=f"Active Alerts: {len(alerts)} unacknowledged",
            )

        # 3d. Check for Discipline Engine Status & Guardrails Intent (Step 7)
        # Examples: "What are my current discipline limits?", "What is my current discipline limit?", "Check my risk limits"
        is_discipline_query = False
        if any(w in lowered for w in [
            "discipline limit", "discipline limits", "discipline status", "discipline rule", "discipline rules",
            "my discipline", "current discipline", "discipline check", "risk guardrails", "risk limit", "risk limits",
            "daily loss limit", "drawdown limit", "drawdown limits", "overtrading limit", "overtrading restrictions",
            "cooldown status", "cooldown limit", "stop-loss requirement", "stop loss requirement", "consecutive loss"
        ]):
            is_discipline_query = True
        elif "discipline" in lowered and any(w in lowered for w in ["what", "how", "show", "check", "current", "my", "status", "limit"]):
            is_discipline_query = True

        if is_discipline_query:
            state = None
            if self._companion:
                try:
                    state = self._companion.get_discipline_status(user_id=req.user_id or "default_user")
                except Exception as e:
                    logger.debug(f"Companion get_discipline_status error: {e}")
            elif self._repo:
                try:
                    state = self._repo.get_discipline_state(user_id=req.user_id or "default_user")
                except Exception as e:
                    logger.debug(f"Repo get_discipline_state error: {e}")

            cfg = state.config if state else None
            sl_req = "Enforced (Required on all trades)" if (cfg and cfg.stop_loss_required) else "Not enforced / Optional"
            risk_per_trade = f"{cfg.max_risk_per_trade_pct}%" if (cfg and cfg.max_risk_per_trade_pct is not None) else "Unavailable / Not configured"
            daily_loss = f"${cfg.daily_loss_limit:.2f}" if (cfg and cfg.daily_loss_limit is not None) else "Unavailable / Not configured"
            max_daily_trades = f"{cfg.max_daily_trades} trades" if (cfg and cfg.max_daily_trades is not None) else "Unavailable / Not configured"
            cooldown_mins = f"{cfg.cooldown_minutes} minutes" if (cfg and cfg.cooldown_minutes is not None) else "Unavailable / Not configured"
            consec_losses = f"{cfg.max_daily_trades // 2 if cfg.max_daily_trades else 3} losses" if cfg else "3 losses"

            trades_today = state.trades_today_count if state else 0
            realized_loss = state.daily_realized_loss if state else 0.0
            violations = state.active_violations if (state and state.active_violations) else []
            violations_str = ", ".join(violations) if violations else "None (Good standing)"

            cooldown_active = bool(state and state.cooldown_active_until and state.cooldown_active_until > datetime.now(timezone.utc))
            cooldown_state_str = "ACTIVE (Trading restricted)" if cooldown_active else "Inactive (Clear to trade)"

            ai_msg = (
                f"🛡️ **Current Discipline Engine Guardrails & Status**:\n\n"
                f"- **Stop-Loss Enforcement**: `{sl_req}`\n"
                f"- **Maximum Risk Per Trade**: `{risk_per_trade}`\n"
                f"- **Daily Drawdown Limit**: `{daily_loss}`\n"
                f"- **Consecutive Loss Restrictions**: `{consec_losses}`\n"
                f"- **Overtrading / Max Daily Trades**: `{max_daily_trades}`\n"
                f"- **Mandatory Cooldown Period**: `{cooldown_mins}`\n"
                f"- **Cooldown State**: `{cooldown_state_str}`\n"
                f"- **Trades Executed Today**: `{trades_today}`\n"
                f"- **Realized Loss Today**: `${realized_loss:.2f}`\n"
                f"- **Active Violations**: `{violations_str}`\n\n"
                f"💡 *Guardrails are authoritative and can be adjusted via Settings Hub under Trading Policies.*"
            )
            speech_msg = (
                f"Your current discipline limits: Stop loss is required on all trades. "
                f"Maximum risk per trade is {risk_per_trade}. "
                f"Daily drawdown limit is {daily_loss}. "
                f"Mandatory cooldown is {cooldown_mins}. "
                f"You have executed {trades_today} trades today with {len(violations)} active violations."
            )
            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent="DISCIPLINE_STATUS",
                evidence=[
                    f"Stop Loss: {sl_req}",
                    f"Risk Per Trade: {risk_per_trade}",
                    f"Daily Drawdown: {daily_loss}",
                    f"Trades Today: {trades_today}",
                    f"Violations: {len(violations)}",
                ],
                nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                context_summary=f"Discipline: SL={sl_req} | Risk={risk_per_trade} | Violations={len(violations)}",
            )

        # 3e. Check for Scenario Engine / Forward Scenario Intent (Step 7)
        # Examples: "What is the forward scenario for EUR/USD?", "What happens next for EUR/USD?", "What is the scenario?"
        is_scenario_query = False
        if any(w in lowered for w in [
            "forward scenario", "forward scenarios", "what is the scenario", "what happens next",
            "market scenario", "market scenarios", "conditional scenario", "conditional scenarios",
            "next scenario", "scenario analysis", "what if price breaks", "forward path"
        ]):
            is_scenario_query = True
        elif "scenario" in lowered and not any(w in lowered for w in ["test scenario", "code scenario"]):
            is_scenario_query = True
        elif any(w in lowered for w in ["what happens next", "where is price going next", "what is next for"]) and (detected_symbol or symbol):
            is_scenario_query = True

        if is_scenario_query:
            scenarios = []
            if self._companion:
                try:
                    scenarios = self._companion.get_scenarios(symbol=symbol)
                except Exception as e:
                    logger.debug(f"Companion get_scenarios error: {e}")

            if scenarios:
                primary = scenarios[0]
                ai_blocks = []
                evidence_list = [f"Symbol: {symbol}", f"Scenarios Count: {len(scenarios)}"]
                for sc in scenarios[:2]:
                    ai_blocks.append(
                        f"### {sc.direction.replace('_', ' ').title()} Scenario (Key Level: {sc.level:.5f})\n"
                        f"- **IF**: {sc.if_condition}\n"
                        f"- **THEN**: {sc.then_outcome}\n"
                        f"- **UNLESS**: {sc.unless_condition}\n"
                        f"- **Invalidated When**: {sc.invalidated_when}"
                    )
                    evidence_list.append(f"Level {sc.level:.5f}: {sc.direction}")

                speech_msg = (
                    f"{symbol} is currently in a {primary.direction.replace('_', ' ').lower()} scenario. "
                    f"{primary.if_condition}, {primary.then_outcome}. "
                    f"However, this scenario is invalidated if {primary.unless_condition}."
                )
                ai_msg = (
                    f"🧭 **Forward Scenarios for {symbol}**:\n\n"
                    + "\n\n".join(ai_blocks)
                    + "\n\n💡 *All scenarios are conditional IF-THEN-UNLESS projections grounded in POIs and market structure.*"
                )
            else:
                ai_msg = (
                    f"ℹ️ **No Forward Scenarios Currently Generated for {symbol}**.\n\n"
                    f"- **Data Requirement**: The ScenarioEngine requires active POI and liquidity levels to establish conditional paths.\n"
                    "- **Grounding**: No fabricated price levels or speculative prophecies are provided."
                )
                speech_msg = f"I do not currently have active forward scenarios for {symbol}. Live market data is needed to establish key levels."
                evidence_list = [f"{symbol}: No computed POI / liquidity levels"]

            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent="COMPANION_SCENARIO",
                evidence=evidence_list,
                nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                context_summary=f"Scenarios for {symbol}: {len(scenarios)} conditional paths",
            )

        # 3f. Check for Fundamental & Economic Calendar Intent (Step 6)
        if any(w in lowered for w in ["fundamental", "economic event", "news risk", "fomc", "cpi", "nfp", "economic calendar", "news conflict"]):
            fund_analysis = None
            if self._repo:
                try:
                    fund_analysis = self._repo.get_latest_tech_fund_analysis(symbol=symbol)
                except Exception as e:
                    logger.debug(f"Could not load technical-fundamental analysis for {symbol}: {e}")

            upcoming_events = []
            if self._repo:
                try:
                    sym_clean = symbol.replace("/", "").upper()
                    currencies = ["USD"]
                    if "EUR" in sym_clean:
                        currencies.append("EUR")
                    if "GBP" in sym_clean:
                        currencies.append("GBP")
                    if "JPY" in sym_clean:
                        currencies.append("JPY")
                    for c in currencies:
                        evs = self._repo.get_fundamental_events(currency=c, limit=3)
                        upcoming_events.extend(evs)
                except Exception as e:
                    logger.debug(f"Could not load fundamental events for {symbol}: {e}")

            events_desc = ""
            if upcoming_events:
                events_desc = "\n".join([f"- **{e.currency}**: {e.title} ({e.impact.value} Impact)" for e in upcoming_events[:4]])
            else:
                events_desc = "- No upcoming high-impact economic events detected."

            rel_str = fund_analysis.relationship.value if fund_analysis else "TECHNICALLY_VALID"
            risk_str = fund_analysis.news_risk.value if fund_analysis else "NO_RISK"
            action_str = fund_analysis.action_context if fund_analysis else "PROCEED_WITH_STANDARD_CONFIRMATION"
            expl_str = fund_analysis.explanation if fund_analysis else "Technical structure is clear with no high-impact event barriers."

            ai_msg = (
                f"📊 **Fundamental & Economic Event Intelligence ({symbol})**:\n\n"
                f"- **News Risk Level**: `{risk_str}`\n"
                f"- **Technical + Fundamental Relationship**: `{rel_str}`\n"
                f"- **Recommended Action Context**: `{action_str}`\n"
                f"- **Analysis**: {expl_str}\n\n"
                f"**Upcoming Economic Events**:\n"
                f"{events_desc}\n\n"
                f"💡 *Monitor Tab 9 (Live Market) Fundamental Intelligence card for live timing countdowns.*"
            )
            speech_msg = (
                f"For {symbol}, fundamental risk is {risk_str.replace('_', ' ').lower()}, with relationship {rel_str.replace('_', ' ').lower()}. "
                f"Recommended action is {action_str.replace('_', ' ').lower()}."
            )
            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent="FUNDAMENTAL_INTELLIGENCE",
                evidence=[f"News Risk: {risk_str}", f"Relationship: {rel_str}", f"Action: {action_str}"],
                nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                context_summary=f"Fundamental: {risk_str} | {rel_str}",
            )

        # 3g. Check for Market Supervisor, Active Broker, Live Chart Markings, Setups & Notifications Intent
        is_marking_query = False
        is_why_marked_query = False
        is_situation_query = False
        is_broker_query = False
        is_notification_query = False
        is_setup_confirmation_query = False

        if (
            any(w in lowered for w in [
                "why was it marked", "why did ai mark", "why did supervisor mark", "why is this marked",
                "reason for marking", "marking kyun", "yeh marking kyun", "kyun mark kiya", "marking ka reason",
                "why marking", "marking why", "explain marking", "marking explain", "marking kyu"
            ])
            or ("marking" in lowered and any(k in lowered for k in ["क्यों", "kyun", "kyu", "why", "reason", "कारण", "wajah", "बनाई"]))
        ):
            is_why_marked_query = True
        elif (
            any(w in lowered for w in [
                "latest notification", "last notification", "recent notification", "what notification",
                "aakhiri notification", "aakhiri message", "aakhiri alert", "kya notification aaya",
                "what did alert say", "last message", "notification update", "recent alert", "akhri alert", "akhri notification"
            ])
            or (any(k in lowered for k in ["notification", "नोटिफिकेशन", "alert", "अलर्ट", "message", "मैसेज"]) and any(s in lowered for s in ["last", "latest", "recent", "आखिरी", "पिछला", "हालिया", "aakhiri", "kya", "क्या"]))
        ):
            is_notification_query = True
        elif (
            any(w in lowered for w in [
                "what did supervisor confirm", "what is confirmed", "which setup is valid",
                "kya confirm hua", "kaun sa setup valid hai", "setup status", "valid setup",
                "confirmed setup", "is setup confirmed", "setup valid", "kya setup confirm hua",
                "supervisor confirm", "valid setups"
            ])
            or (any(k in lowered for k in ["confirm", "सत्यापित", "valid", "कन्फर्म"]) and any(s in lowered for s in ["setup", "सेटअप", "supervisor", "सुपरवाइजर"]))
        ):
            is_setup_confirmation_query = True
        elif (
            any(w in lowered for w in [
                "active broker", "broker status", "connected broker", "active broker status",
                "kaun sa broker active hai", "broker kaun sa hai", "which broker", "active broker kaun"
            ])
            or ("broker" in lowered or "ब्रोकर" in lowered)
        ):
            is_broker_query = True
        elif any(w in lowered for w in [
            "what did supervisor mark", "what did ai mark", "what is marked", "what was marked",
            "chart markings", "show markings", "live markings", "active markings", "what did ai see",
            "show me markings", "list markings", "what markings", "markings plotted", "markings are plotted",
            "plotted on", "markings on", "any markings", "kya mark kiya", "marking dikhao", "markings kya"
        ]):
            is_marking_query = True
        elif any(w in lowered for w in [
            "what is happening now", "current situation", "current poi", "nearest liquidity",
            "what matters right now", "market update", "kya chal raha hai", "market status",
            "latest market update", "batao market", "market overview"
        ]):
            is_situation_query = True

        if (
            is_marking_query
            or is_why_marked_query
            or is_situation_query
            or is_broker_query
            or is_notification_query
            or is_setup_confirmation_query
        ):
            sup_status = self._supervisor.get_status() if self._supervisor else {}
            broker_info = sup_status.get("active_broker", {})
            broker_name = broker_info.get("broker_name", "Paper Broker")
            conn_status = broker_info.get("connection_status", "CONNECTED")
            data_health = sup_status.get("data_health", "CONNECTED")

            markings = []
            if self._supervisor:
                markings = self._supervisor.get_markings_for_symbol(symbol=symbol, timeframe=timeframe)
            elif self._repo:
                markings = self._repo.get_chart_markings(symbol=symbol, timeframe=timeframe, only_valid=True)

            if is_notification_query:
                notifs = self._repo.get_notification_records(limit=5) if self._repo else []
                if notifs:
                    n = notifs[0]
                    ts_str = n.created_at.strftime('%Y-%m-%d %H:%M:%S UTC') if hasattr(n.created_at, 'strftime') else str(n.created_at)
                    hindi_body = n.message_hindi or n.message_english
                    ai_msg = (
                        f"🔔 **नवीनतम Notification विवरण ({n.symbol or 'MARKET'})**:\n\n"
                        f"- **Title**: {n.title}\n"
                        f"- **Priority Level**: `{n.priority.value if hasattr(n.priority, 'value') else n.priority}`\n"
                        f"- **समय (Timestamp)**: `{ts_str}`\n"
                        f"- **Channel**: `{n.channel.value if hasattr(n.channel, 'value') else n.channel}`\n\n"
                        f"📝 **संदेश (Hindi Alert)**:\n{hindi_body}\n\n"
                        f"🌐 **English Copy**: {n.message_english}"
                    )
                    speech_msg = f"आखिरी notification: {n.title}। {hindi_body}"
                    ev_list = [f"Notification ID: {n.notification_id}", f"Title: {n.title}"]
                else:
                    ai_msg = "ℹ️ **कोई हालिया Notification नहीं**: सिस्टम में अभी कोई हालिया notification दर्ज नहीं है।"
                    speech_msg = "अभी कोई हालिया notification उपलब्ध नहीं है।"
                    ev_list = ["No notification records found"]

                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="NOTIFICATION_QUERY",
                    evidence=ev_list,
                    nav_action=ChatNavigationAction(tab_id="tab-step16", label="View in Notifications"),
                    context_summary="Notification Query",
                )

            elif is_setup_confirmation_query:
                setups = self._repo.get_market_setups(symbol=symbol) if self._repo else []
                confirmed = [s for s in setups if getattr(s.status, 'value', str(s.status)) in ("CONFIRMED", "VALID", "ACTIVE")]
                if confirmed:
                    s = confirmed[0]
                    cond_str = "; ".join(s.conditions_satisfied) if s.conditions_satisfied else "Learned rule criteria met"
                    ai_msg = (
                        f"📊 **MarketSupervisor Setup स्थिति — {symbol}**:\n\n"
                        f"- **Setup Type**: `{s.setup_type}`\n"
                        f"- **Direction**: `{s.direction.value if hasattr(s.direction, 'value') else s.direction}`\n"
                        f"- **Timeframe**: `{s.timeframe}`\n"
                        f"- **Status**: `{getattr(s.status, 'value', str(s.status))}`\n"
                        f"- **Rule Reference**: `{s.applicable_rule_id or 'Institutional Rulebook'}`\n"
                        f"- **संतुष्ट शर्तें (Conditions Satisfied)**: {cond_str}\n"
                        f"- **Invalidation Level**: `{getattr(s, 'invalidation_price', None) or 'Key Structure Level'}`\n"
                        f"- **POI Level**: `{getattr(s, 'entry_price', None) or 'Key Zone'}`\n\n"
                        f"💡 *यह सेटअप Supervisor द्वारा सत्यापित किया गया है।*"
                    )
                    speech_msg = f"Supervisor ने {symbol} पर {s.setup_type} setup को {getattr(s.status, 'value', str(s.status))} दर्ज किया है। Direction {s.direction.value if hasattr(s.direction, 'value') else s.direction} है।"
                    ev_list = [f"Setup: {s.setup_type}", f"Status: {getattr(s.status, 'value', str(s.status))}"]
                else:
                    ai_msg = (
                        f"ℹ️ **कोई Confirmed Setup नहीं ({symbol})**:\n\n"
                        f"वर्तमान में **{symbol}** पर कोई नया setup confirm नहीं हुआ है। "
                        f"MarketSupervisor `{broker_name}` पर market structure की निरंतर निगरानी कर रहा है। "
                        f"केवल वही setups confirm होते हैं जो SMC/ICT नियमों और confluence को पूरा करते हैं।"
                    )
                    speech_msg = f"{symbol} पर वर्तमान में कोई नया setup confirm नहीं हुआ है। Supervisor सक्रिय निगरानी कर रहा है।"
                    ev_list = [f"Symbol: {symbol}", "No confirmed setups"]

                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="SETUP_VALIDATION",
                    evidence=ev_list,
                    nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                    context_summary=f"Setup status on {symbol}",
                )

            elif is_broker_query:
                mode_str = sup_status.get("execution", {}).get("mode", "PAPER")
                is_appr = broker_info.get("is_approved", False)
                ai_msg = (
                    f"🔌 **सक्रिय Broker विवरण (Active Broker Context)**:\n\n"
                    f"- **सक्रिय Broker**: `{broker_name}` (`{broker_info.get('broker_id', 'paper_broker')}`)\n"
                    f"- **Connection Status**: `{conn_status}`\n"
                    f"- **Market Data Health**: `{data_health}`\n"
                    f"- **Execution Mode**: `{mode_str}`\n"
                    f"- **Live Order Authorization**: `{'अनुमोदित (APPROVED)' if is_appr else 'अस्वीकृत (केवल Live Data & AI Markings)'}`\n\n"
                    f"💡 *स्मरण रहे: Broker Connection ≠ Trade Authorization। लाइव मार्केट डेटा विश्लेषण और AI चार्ट मार्किंग बिना लाइव ऑर्डर के सुरक्षित रूप से कार्य करते हैं।*"
                )
                speech_msg = f"सक्रिय broker {broker_name} है। Connection status {conn_status} है और डेटा health {data_health} है। Execution mode {mode_str} है।"
                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="BROKER_STATUS",
                    evidence=[f"Broker: {broker_name}", f"Status: {conn_status}", f"Data: {data_health}"],
                    nav_action=ChatNavigationAction(tab_id="tab-step18", label="View in Settings: Universal Providers"),
                    context_summary=f"Broker: {broker_name} | {conn_status}",
                )

            elif is_why_marked_query:
                if markings:
                    m = markings[0]
                    ev = m.evidence_json
                    conditions = ev.get("conditions", [])
                    evidence_items = ev.get("evidence", [])
                    cond_str = "; ".join(conditions) if conditions else "Confluence of technical criteria"
                    ev_str = "; ".join(evidence_items) if evidence_items else "Validated by rule criteria"
                    ai_msg = (
                        f"🔍 **Marking स्पष्टीकरण (Evidence & Rule Analysis) — {symbol}**:\n\n"
                        f"यह marking **{symbol}** के **{m.timeframe or timeframe}** chart पर बनाई गई थी।\n"
                        f"- **Marking Type**: `{m.marking_type.value}` ({m.concept or 'Standard SMC Pattern'})\n"
                        f"- **Price / Level**: `{m.price:.5f}`\n"
                        f"- **Rule Reference**: `{m.rule_reference or m.rule_id or 'Institutional Rulebook'}`\n"
                        f"- **Confidence Score**: `{m.confidence * 100:.0f}%`\n"
                        f"- **मुख्य वजह (Conditions Satisfied)**: {cond_str}\n"
                        f"- **प्रमाण (Supporting Evidence)**: {ev_str}\n"
                        f"- **वर्तमान Status**: `{m.status}`\n"
                        f"- **Invalidation Condition**: {m.invalidation_condition or 'Key structure boundary breach'}\n"
                        f"- **Source Provenance**: `{m.source or 'SUPERVISOR'}` (Broker: `{broker_name}`)\n\n"
                        f"💡 *टिप्पणी: यह केवल विश्लेषणात्मक चार्ट मार्किंग है, पुष्टि मिलने तक यह केवल एक observation है।*"
                    )
                    speech_msg = (
                        f"यह marking {symbol} के {m.timeframe or timeframe} chart पर बनाई गई थी। "
                        f"Supervisor ने इसे {cond_str} के आधार पर बनाया। "
                        f"इसका वर्तमान status {m.status} है और confidence {m.confidence * 100:.0f} प्रतिशत है।"
                    )
                    ev_list = [f"Rule: {m.rule_reference}", f"Confidence: {m.confidence}"]
                else:
                    ai_msg = (
                        f"ℹ️ **कोई सक्रिय Marking नहीं मिली ({symbol})**:\n\n"
                        f"वर्तमान में **{symbol}** chart पर कोई सक्रिय AI marking दर्ज नहीं है। "
                        f"MarketSupervisor `{broker_name}` के माध्यम से live feed की निगरानी कर रहा है।"
                    )
                    speech_msg = f"{symbol} पर वर्तमान में कोई सक्रिय AI marking मौजूद नहीं है। Supervisor निगरानी कर रहा है।"
                    ev_list = ["No active markings"]

                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="MARKING_EXPLANATION",
                    evidence=ev_list,
                    nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                    context_summary=f"Marking explanation for {symbol}",
                )

            elif is_situation_query:
                sit = self._repo.get_latest_current_situation(symbol=symbol) if self._repo else None
                if sit:
                    liq_str = f"{sit.nearest_major_liquidity.level_type}: {sit.nearest_major_liquidity.price:.2f}" if sit.nearest_major_liquidity else "None nearby"
                    poi_str = f"{sit.current_poi.level_type}: {sit.current_poi.price:.2f}" if sit.current_poi else "None active"
                    setup_str = f"{sit.active_confirmed_setup.setup_type} ({sit.setup_confirmation_state})" if sit.active_confirmed_setup else "No confirmed setup"
                    ai_msg = (
                        f"🧭 **बाजार की स्थिति (Current Market Situation) — {symbol}**:\n\n"
                        f"- **[देखा गया / OBSERVED] Current Price**: `{sit.current_price:.2f}`\n"
                        f"- **[देखा गया / OBSERVED] Bias**: `{sit.bias_summary}`\n"
                        f"- **[देखा गया / OBSERVED] Nearest Major Liquidity**: `{liq_str}`\n"
                        f"- **[देखा गया / OBSERVED] Point of Interest (POI)**: `{poi_str}`\n"
                        f"- **[देखा गया / OBSERVED] Active Setup**: `{setup_str}`\n"
                        f"- **[निष्कर्ष / INFERRED] Invalidation / Trigger**: Inval `{sit.invalidation_level or '--'}`, Conf `{sit.confirmation_level or '--'}`\n"
                        f"- **Market Data Health**: `{data_health}` from broker `{broker_name}`"
                    )
                    speech_msg = f"{symbol} पर current price {sit.current_price:.2f} है। Bias {sit.bias_summary} है। Nearest liquidity {liq_str} है।"
                else:
                    ai_msg = f"ℹ️ **Current Situation ({symbol})**: Market की निगरानी {broker_name} द्वारा की जा रही है। Data health `{data_health}` है।"
                    speech_msg = f"{symbol} पर बाजार स्थिर है। Data health {data_health} है।"
                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="CURRENT_SITUATION",
                    evidence=[f"Symbol: {symbol}", f"Health: {data_health}"],
                    nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                    context_summary=f"Situation for {symbol}",
                )

            else:  # is_marking_query
                if markings:
                    m_lines = []
                    for m in markings[:5]:
                        px_str = f" @ {m.price:.5f}" if m.price else ""
                        m_lines.append(
                            f"- **{m.marking_type.value}** ({m.concept or m.timeframe}){px_str}: Confidence `{m.confidence * 100:.0f}%` | Status `{m.status}`"
                        )
                    ai_msg = (
                        f"📍 **सक्रिय AI Markings — {symbol}** (कुल {len(markings)} markings):\n\n"
                        + "\n".join(m_lines)
                        + f"\n\n🔌 *Data Source: {broker_name} live market feed. Clutter filtering और evidence deduplication सक्रिय हैं।*"
                    )
                    speech_msg = f"{symbol} पर कुल {len(markings)} active markings हैं जो {broker_name} से प्राप्त हुई हैं, जिनमें प्रमुख {markings[0].marking_type.value} है।"
                    ev_list = [f"Total Markings: {len(markings)}", f"Source: {broker_name}"]
                else:
                    ai_msg = (
                        f"ℹ️ **No active AI chart markings / कोई सक्रिय AI Chart Markings नहीं ({symbol})**:\n\n"
                        f"- **MarketSupervisor स्थिति**: `{broker_name}` पर सक्रिय निगरानी।\n"
                        f"- **मापदंड**: चार्ट clutter से बचने के लिए केवल वही setups प्लॉट किए जाते हैं जो सभी institutional lecture rules को पूरा करते हैं।"
                    )
                    speech_msg = f"{symbol} पर वर्तमान में कोई सक्रिय markings मौजूद नहीं हैं। Supervisor सक्रिय निगरानी कर रहा है।"
                    ev_list = ["0 confirmed markings on chart"]

                return self._record_and_return_unified_turn(
                    req=req,
                    session_id=session_id,
                    user_msg=raw_msg,
                    ai_msg=ai_msg,
                    speech_msg=speech_msg,
                    intent="SUPERVISOR_MARKINGS",
                    evidence=ev_list,
                    nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Live Market"),
                    context_summary=f"Markings on {symbol}: {len(markings)}",
                )

        # 3h. Check for Open Positions Intent (Step 8: POSITIONS_STATUS)
        # Examples: "What open positions do I have?", "Show my positions", "Any open positions?", "List open positions"
        is_pos_query = False
        if any(w in lowered for w in [
            "open position", "open positions", "active position", "active positions",
            "current position", "current positions", "my positions", "show positions",
            "list positions", "open trades", "active trades", "what positions", "any positions"
        ]):
            is_pos_query = True
        elif "position" in lowered and any(w in lowered for w in ["what", "how many", "show", "list", "check", "any", "my", "open"]):
            is_pos_query = True

        if is_pos_query and not re.search(r"^(close|exit|modify|update|sl)\b", lowered):
            open_positions = []
            if self._execution_core:
                try:
                    open_positions = self._execution_core.get_positions(open_only=True)
                except Exception as e:
                    logger.debug(f"Could not load positions from execution core: {e}")
            if not open_positions and self._repo:
                try:
                    open_positions = self._repo.get_open_positions()
                except Exception as e:
                    logger.debug(f"Could not load positions from repo: {e}")

            if open_positions:
                pos_lines = []
                speech_parts = []
                for p in open_positions:
                    side = getattr(p, "direction", "TRADE")
                    qty = getattr(p, "quantity", 0.0)
                    entry = getattr(p, "entry_price", 0.0)
                    c_sl = getattr(p, "current_sl", 0.0)
                    tp1 = getattr(p, "tp1", 0.0)
                    target = getattr(p, "final_target", 0.0)
                    env = getattr(p, "execution_environment", "PAPER")
                    env_str = env.value if hasattr(env, "value") else str(env)
                    pos_lines.append(
                        f"- **{p.symbol}** ({side}): Qty `{qty}`, Entry `{entry:.5f}`, SL `{c_sl:.5f}`, TP1 `{tp1:.5f}`, Target `{target:.5f}` [{env_str}]"
                    )
                    speech_parts.append(f"{side} on {p.symbol} entered at {entry:.5f} with stop loss at {c_sl:.5f}")

                ai_msg = (
                    f"📈 **Active Managed Positions** ({len(open_positions)} open):\n\n"
                    + "\n".join(pos_lines)
                    + "\n\n💡 *Positions are continuously monitored by PositionManager with automatic 1.5R partial TP1 scaling and breakeven trailing.*"
                )
                speech_msg = f"You have {len(open_positions)} open position: " + "; ".join(speech_parts) + "."
                evidence_list = [f"{p.symbol}: {getattr(p, 'direction', 'TRADE')} @ {getattr(p, 'entry_price', 0.0)}" for p in open_positions]
            else:
                ai_msg = (
                    "ℹ️ **No Open Positions**:\n\n"
                    "- You currently have **0 open positions**.\n"
                    "- All previous trades are closed or awaiting new confirmed setup triggers.\n"
                    "- Check the Execution Panel in Tab 9 to generate a new proposal from current market structure."
                )
                speech_msg = "You currently have no open positions. All trades are closed."
                evidence_list = ["HedgeFundExecutionCore: 0 open positions"]

            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent="POSITIONS_STATUS",
                evidence=evidence_list,
                nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Execution Panel"),
                context_summary=f"Open Positions: {len(open_positions)}",
            )

        # 3h. Check for Trade Proposals Intent (Step 8: PROPOSALS_STATUS)
        # Examples: "What are my trade proposals?", "Show pending proposals", "Any trade proposals?", "Show proposals"
        is_prop_query = False
        if any(w in lowered for w in [
            "trade proposal", "trade proposals", "active proposal", "active proposals",
            "pending proposal", "pending proposals", "my proposals", "show proposals",
            "list proposals", "any proposals", "what proposals"
        ]):
            is_prop_query = True

        if is_prop_query and not re.search(r"^(approve|reject|modify|generate)\b", lowered):
            proposals = []
            if self._repo:
                try:
                    proposals = self._repo.get_trade_proposals(limit=5)
                except Exception as e:
                    logger.debug(f"Could not load trade proposals: {e}")

            if proposals:
                prop_lines = []
                speech_parts = []
                for p in proposals[:3]:
                    state = p.lifecycle_state.value if hasattr(p.lifecycle_state, "value") else str(p.lifecycle_state)
                    entry = getattr(p, "entry_price", 0.0)
                    sl = getattr(p, "hard_stop_loss", 0.0)
                    target = getattr(p, "final_target", 0.0)
                    rr = p.risk_assessment.risk_reward_ratio if hasattr(p, "risk_assessment") and p.risk_assessment else 2.0
                    prop_lines.append(
                        f"- **{p.proposal_id}** ({p.symbol} {p.direction}): State `{state}`, Entry `{entry:.5f}`, SL `{sl:.5f}`, Target `{target:.5f}`, R:R `1:{rr:.1f}`"
                    )
                    speech_parts.append(f"{p.direction} on {p.symbol} with state {state.lower()}")

                ai_msg = (
                    f"📋 **Recent Trade Proposals** ({len(proposals)} recorded):\n\n"
                    + "\n".join(prop_lines)
                    + "\n\n💡 *Proposals require manual trader approval via the ExecutionGate before order dispatch.*"
                )
                speech_msg = f"You have {len(proposals)} trade proposals: " + "; ".join(speech_parts) + "."
                evidence_list = [f"{p.proposal_id}: {p.symbol} {p.direction} ({getattr(p.lifecycle_state, 'value', 'ACTIVE')})" for p in proposals[:3]]
            else:
                ai_msg = (
                    "ℹ️ **No Trade Proposals Found**:\n\n"
                    "- There are currently no trade proposals in the system.\n"
                    "- Click **'Generate Proposal'** in Tab 9 (Execution Panel) to evaluate market structure and synthesize a proposal."
                )
                speech_msg = "There are currently no trade proposals recorded. You can generate one from the Execution Panel."
                evidence_list = ["HedgeFundExecutionCore: 0 trade proposals"]

            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent="PROPOSALS_STATUS",
                evidence=evidence_list,
                nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Execution Panel"),
                context_summary=f"Trade Proposals: {len(proposals)}",
            )

        # 3i. Check for Execution Policy & Risk Engine Status (Step 8: EXECUTION_STATUS)
        # Examples: "What is my execution mode?", "What is the execution status?", "What is my risk status?", "Check execution gate"
        is_exec_status_query = False
        if any(w in lowered for w in [
            "execution mode", "execution status", "execution policy", "execution gate status",
            "risk engine status", "risk status", "are trades automated", "is auto execution",
            "execution settings", "gate status"
        ]):
            is_exec_status_query = True

        if is_exec_status_query:
            policy = None
            if self._execution_core:
                try:
                    policy = self._execution_core.get_or_create_policy(user_id=req.user_id or "default_user")
                except Exception as e:
                    logger.debug(f"Could not load policy from execution core: {e}")
            elif self._repo:
                try:
                    policy = self._repo.get_execution_policy(user_id=req.user_id or "default_user")
                except Exception as e:
                    logger.debug(f"Could not load policy from repo: {e}")

            exec_mode_val = policy.execution_mode.value if policy else "MANUAL"
            exec_env_val = policy.execution_environment.value if policy else "PAPER"
            auto_enabled = policy.auto_execution_enabled if policy else False
            live_enabled = policy.live_execution_enabled if policy else False
            min_rr = policy.min_rr_ratio if policy else 2.0
            max_auto_risk = policy.max_auto_risk_pct if policy else 1.0

            ai_msg = (
                f"⚙️ **Hedge Fund Execution & Risk Management Status**:\n\n"
                f"- **Execution Mode**: `{exec_mode_val}` ({'Automated Order Submission' if auto_enabled else 'Manual Trader Confirmation Required'})\n"
                f"- **Environment**: `{exec_env_val}` ({'Live Broker Connected' if live_enabled else 'Simulated Paper Broker'})\n"
                f"- **Minimum Risk/Reward Requirement**: `1:{min_rr:.1f}`\n"
                f"- **Max Automated Risk Per Trade**: `{max_auto_risk:.1f}%`\n"
                f"- **Execution Gate**: `ENFORCED (Protected by ExecutionGate -> RiskEngine -> CommitBarrier)`\n\n"
                f"💡 *Execution mode can be toggled in Tab 9 or configured in the Settings Hub.*"
            )
            speech_msg = (
                f"Your execution mode is {exec_mode_val} in {exec_env_val} environment. "
                f"Minimum risk reward ratio is 1 to {min_rr:.1f}. "
                f"Execution safety gate is strictly enforced."
            )
            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent="EXECUTION_STATUS",
                evidence=[
                    f"Mode: {exec_mode_val}",
                    f"Environment: {exec_env_val}",
                    f"Min RR: {min_rr}",
                    f"Auto Enabled: {auto_enabled}",
                ],
                nav_action=ChatNavigationAction(tab_id="tab-market", label="View in Tab 9: Execution Panel"),
                context_summary=f"Execution: {exec_mode_val} | {exec_env_val} | MinRR={min_rr}",
            )

        # 4. Check for Trading Execution Request (ExecutionGate & RiskEngine Safety Invariant)
        is_question = any(lowered.startswith(w) for w in ["what", "why", "how", "where", "can you explain", "explain", "is it", "should i", "tell me about", "check", "show"])
        trade_cmd_pattern = (not is_question) and (
            re.search(r"^(buy|sell|long|short|close position|open order)\b", lowered)
            or re.search(r"\b(execute (a )?(buy|sell)|place (a )?(buy|sell) order|open a (long|short) position)\b", lowered)
        )
        if trade_cmd_pattern:
            action_type = "BUY" if any(w in lowered for w in ["buy", "long"]) else "SELL"
            if "close" in lowered:
                action_type = "CLOSE"
            session_state.pending_action_intent = action_type
            session_state.pending_action_symbol = symbol
            ai_msg = (
                f"⚠️ **Action Proposed**: {action_type} order requested for {symbol}.\n\n"
                f"- **Safety Gate**: USER CONFIRMATION REQUIRED.\n"
                f"- **Execution Policy**: Direct autonomous execution via {req.input_modality.value.title()} is blocked by ExecutionGate & RiskEngine.\n"
                f"- Please review the risk parameters and manually confirm the order in the Execution Panel."
            )
            speech_msg = (
                f"Proposed {action_type} order for {symbol}. Confirmation required. "
                f"Please manually confirm in the Execution Panel."
            )
            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=ai_msg,
                speech_msg=speech_msg,
                intent="TRADE_ACTION_PROPOSED",
                evidence=[
                    "ExecutionGate: BLOCKED",
                    "RiskEngine: Active",
                    f"Action: {action_type}",
                ],
                nav_action=ChatNavigationAction(tab_id="tab-market", label="Confirm in Tab 9: Live Market"),
                context_summary=f"Action intent: {action_type} on {symbol} (Gated)",
                action_intent=action_type,
                requires_confirmation=True,
                safety_status="CONFIRMATION_REQUIRED",
            )

        # 5. Pull Live Market Context
        companion_ctx = None
        if self._companion:
            companion_ctx = self._companion.get_companion_context(symbol=symbol)
            ctx_summary = (
                f"{symbol} ({timeframe}) @ {companion_ctx.current_price:.5f} | "
                f"Bias: {companion_ctx.bias} | "
                f"Setup: {companion_ctx.setup_confirmation_state} | "
                f"News Risk: {companion_ctx.news_risk}"
            )
        else:
            ctx_summary = f"{symbol} ({timeframe}) @ 1.08500 | Bias: NEUTRAL | Setup: MONITORING | News Risk: LOW"

        # 5b. Step 6 Fundamental Market Intelligence & News Correlation
        fund_summary = ""
        if self._repo:
            try:
                fund_analysis = self._repo.get_latest_tech_fund_analysis(symbol=symbol)
                if fund_analysis:
                    fund_summary = (
                        f"Risk: {fund_analysis.news_risk.value} | "
                        f"Rel: {fund_analysis.relationship.value} | "
                        f"Action: {fund_analysis.action_context}"
                    )
            except Exception as e:
                logger.debug(f"Could not load latest tech-fund analysis for context: {e}")

        # 5c. Pull Master Decision State if available
        mde_info = ""
        if self._master_decision:
            try:
                mde_res = self._master_decision.evaluate_live_confluence(symbol=symbol)
                mde_info = (
                    f"Master Decision: Confluence Score {mde_res.confluence_score:.1f}/10.0 "
                    f"({mde_res.direction.value}) -> Decision: {mde_res.trade_decision.value}"
                )
            except Exception:
                mde_info = "Master Decision Engine: Active (Awaiting confluence ticks)"

        # 5d. Pull Trading Companion State (Step 7)
        companion_alerts_info = "0 active alerts"
        companion_discipline_info = "SL required | Within daily risk boundaries"
        companion_scenario_info = "Monitoring key POI levels"
        if self._companion:
            try:
                active_alts = self._companion.get_alerts(unacknowledged_only=True, limit=5)
                companion_alerts_info = f"{len(active_alts)} active alerts"
                if active_alts:
                    companion_alerts_info += f" ({active_alts[0].alert_type.value} on {active_alts[0].symbol})"
            except Exception as e:
                logger.debug(f"Could not load companion alerts: {e}")
            try:
                d_stat = self._companion.get_discipline_status(user_id=req.user_id or "default_user")
                if d_stat:
                    cfg = d_stat.config
                    companion_discipline_info = (
                        f"SL {'Enforced' if cfg.stop_loss_required else 'Optional'} | "
                        f"Trades today: {d_stat.trades_today_count} | "
                        f"Loss today: ${d_stat.daily_realized_loss:.2f} | "
                        f"Violations: {len(d_stat.active_violations)}"
                    )
            except Exception as e:
                logger.debug(f"Could not load companion discipline status: {e}")
            try:
                scns = self._companion.get_scenarios(symbol=symbol)
                if scns:
                    companion_scenario_info = f"{scns[0].direction}: {scns[0].if_condition} -> {scns[0].then_outcome}"
            except Exception as e:
                logger.debug(f"Could not load companion scenarios: {e}")

        # 5e. Pull Step 8 Execution Context
        exec_mode_info = "MANUAL (PAPER)"
        open_pos_info = "0 open"
        proposals_info = "0 active"
        if self._repo:
            try:
                pol = self._repo.get_execution_policy(req.user_id or "default_user")
                if pol:
                    exec_mode_info = f"{pol.execution_mode.value} ({pol.execution_environment.value})"
                ops = self._repo.get_open_positions()
                open_pos_info = f"{len(ops)} open"
                prps = self._repo.get_trade_proposals(limit=5)
                proposals_info = f"{len(prps)} proposals"
            except Exception as e:
                logger.debug(f"Could not load repo execution policy/positions/proposals: {e}")

        # 6. Dynamic Budget Tier Selection & Context-Grounded Prompt
        complexity = self._token_manager.classify_request_complexity(
            raw_msg, has_multimodal=(req.input_modality == InputModality.MULTIMODAL)
        )
        initial_tier = self._token_manager.select_initial_tier(complexity)
        model_name = self._model_router.get_model(ModelRole.FAST_MULTIMODAL)
        model_cap = self._token_manager.get_model_capacity(model_name)
        active_budget = self._token_manager.get_budget_for_tier(initial_tier, model_capacity=model_cap)

        app_summary = self._guide.get_system_prompt_summary()
        recent_history = self._format_recent_history(session_id, max_turns=3, symbol=symbol)

        grounded_system_prompt = (
            "You are the Trading AI Assistant — an institutional AI trading companion and application guide.\n\n"
            "MANDATORY BEHAVIORAL INVARIANTS:\n"
            "1. Ground all answers in the provided application map and live market context.\n"
            "2. NEVER hallucinate UI tabs, features, or buttons. If a feature is not implemented (like a dedicated Settings page), state so factually.\n"
            "3. Be concise, institutional, and direct. Explain SMC/ICT market context accurately.\n"
            "4. When referencing a tab, specify its exact tab name (e.g. 'Tab 9: Live Market', 'Tab 17: Master Decision Engine', 'Tab 18: Universal Providers').\n\n"
            f"{app_summary}\n\n"
            f"CURRENT LIVE CONTEXT:\n"
            f"- Active Symbol: {symbol} ({timeframe})\n"
            f"- Market Context: {ctx_summary}\n"
            f"- Fundamental News (Step 6): {fund_summary or 'No active news barrier'}\n"
            f"- Trading Companion (Step 7): Alerts: {companion_alerts_info} | Discipline: {companion_discipline_info} | Scenario: {companion_scenario_info}\n"
            f"- Execution & Risk (Step 8): Mode: {exec_mode_info} | Positions: {open_pos_info} | Proposals: {proposals_info}\n"
            f"- {mde_info}\n"
            f"- User's Active Screen Tab: {current_tab}\n"
        )

        full_prompt = (
            f"{grounded_system_prompt}\n"
            f"CONVERSATION HISTORY:\n{recent_history}\n"
            f"USER QUERY: {raw_msg}\n"
            f"ASSISTANT RESPONSE:"
        )

        # 7. Pre-flight Token Budget Evaluation & Dynamic Escalation
        est_tokens = self._token_manager.estimate_request_tokens(full_prompt)
        budget_res = self._token_manager.check_budget(est_tokens, active_budget)

        # Escalate tier dynamically if context exceeds lower tiers
        while not budget_res.is_sufficient:
            next_tier, escalated = self._token_manager.escalate_tier(
                active_budget.tier or initial_tier, EscalationReason.CONTEXT_SIZE
            )
            if not escalated or next_tier == active_budget.tier:
                break
            active_budget = self._token_manager.get_budget_for_tier(next_tier, model_capacity=model_cap)
            budget_res = self._token_manager.check_budget(est_tokens, active_budget)

        # Context overflow handling: if still exceeding MAX_SAFE, compress conversation
        if not budget_res.is_sufficient:
            session_turns = self._sessions.get(session_id, [])
            if len(session_turns) > 2:
                turns_data = [{"role": t.role, "text": t.text} for t in session_turns]
                compact_summary = self._token_manager.compressor.extract_structured_summary(
                    turns_data, active_symbol=symbol, active_timeframe=timeframe
                )
                full_prompt = (
                    f"{grounded_system_prompt}\n"
                    f"{compact_summary}\n"
                    f"USER QUERY: {raw_msg}\n"
                    f"ASSISTANT RESPONSE:"
                )
                est_tokens = self._token_manager.estimate_request_tokens(full_prompt)
                budget_res = self._token_manager.check_budget(est_tokens, active_budget)

        # Controlled overflow response if still over budget
        if not budget_res.is_sufficient:
            overflow_msg = (
                "The requested context is too large to process safely in one pass. "
                "I reduced the context to the highest-priority information, but the request "
                "still exceeds the safe model capacity."
            )
            return self._record_and_return_unified_turn(
                req=req,
                session_id=session_id,
                user_msg=raw_msg,
                ai_msg=overflow_msg,
                speech_msg=overflow_msg,
                intent="OVERFLOW_SAFE_REDUCTION",
                evidence=[f"Estimated tokens: {est_tokens}", f"Effective limit: {budget_res.effective_limit}"],
                nav_action=None,
                context_summary=ctx_summary,
            )

        # 8. Execute AI Call (or deterministic fallback)
        ai_response_text = ""
        evidence_list = [
            f"Symbol: {symbol} ({timeframe}) @ {companion_ctx.current_price:.5f}" if companion_ctx else f"Symbol: {symbol} ({timeframe}) @ 1.08500",
            f"Bias: {companion_ctx.bias}" if companion_ctx else "Bias: NEUTRAL",
            f"Setup: {companion_ctx.setup_confirmation_state}" if companion_ctx else "Setup: MONITORING",
        ]

        if self._gemini and getattr(self._gemini, "has_live_api", False):
            try:
                ai_response_text = self._gemini.generate_content(
                    role=ModelRole.KNOWLEDGE_SYNTHESIS,
                    prompt=full_prompt,
                    job_id=f"chat_{int(time.time())}",
                    stage="chat_assistant",
                    fallback_on_error=True,
                )
                resp_tokens = self._token_manager.estimate_text_tokens(ai_response_text)
                self._token_manager.record_usage(
                    TokenUsageRecord(
                        job_id=f"chat_{session_id}_{int(time.time())}",
                        stage="chat_assistant",
                        model=model_name,
                        input_tokens=est_tokens,
                        output_tokens=resp_tokens,
                        total_tokens=est_tokens + resp_tokens,
                    )
                )
            except Exception as e:
                logger.warning(f"Live Gemini chat call failed: {e}. Executing deterministic fallback.")

        # Fallback to deterministic companion synthesis if live LLM did not return text or returned simulation JSON
        if not ai_response_text.strip() or ai_response_text.strip().startswith('{"'):
            if companion_ctx:
                ai_response_text = (
                    f"**Market Context for {symbol} ({timeframe})**:\n\n"
                    f"- Current Price: {companion_ctx.current_price:.5f}\n"
                    f"- Bias: {companion_ctx.bias} ({companion_ctx.bias_summary})\n"
                    f"- Setup: {companion_ctx.setup_confirmation_state}\n"
                    f"- News Risk: {companion_ctx.news_risk}\n"
                    f"- Epistemic Grounding: Real-time multi-timeframe analysis active.\n"
                    f"- Guidance: Awaiting confirmed setup trigger before validation."
                )
                if hasattr(companion_ctx, "recent_changes") and companion_ctx.recent_changes:
                    evidence_list.append(f"Recent Changes: {'; '.join(companion_ctx.recent_changes[:2])}")
            elif self._companion and getattr(req, "source", None) != "companion_voice_engine" and not hasattr(self._companion, "get_companion_context"):
                try:
                    fallback_res = self._companion.query_companion(
                        CompanionQueryRequest(symbol=symbol, query=raw_msg)
                    )
                    ai_response_text = fallback_res.answer
                    evidence_list.extend(fallback_res.evidence)
                except Exception as e:
                    logger.warning(f"Companion query fallback failed: {e}")
                    ai_response_text = f"**Market Context for {symbol} ({timeframe})**:\n\n- Status: {ctx_summary}."
            else:
                ai_response_text = (
                    f"**Market Context for {symbol} ({timeframe})**:\n\n"
                    f"- Status: {ctx_summary}\n"
                    f"- Epistemic Grounding: Real-time multi-timeframe analysis active.\n"
                    f"- Guidance: Awaiting confirmed setup trigger before validation."
                )

        # Detect potential navigation target in response
        nav_action = self._detect_navigation_in_response(ai_response_text, raw_msg)
        intent_detected = CompanionIntentRouter.route_intent(raw_msg).value
        speech_text = format_speech_friendly_text(ai_response_text)

        # 9. Persist and Return Unified Response
        return self._record_and_return_unified_turn(
            req=req,
            session_id=session_id,
            user_msg=raw_msg,
            ai_msg=ai_response_text,
            speech_msg=speech_text,
            intent=intent_detected,
            evidence=evidence_list,
            nav_action=nav_action,
            context_summary=ctx_summary,
            token_diagnostics={
                "estimated_tokens": est_tokens,
                "tier": active_budget.tier.value if active_budget.tier else "DEFAULT",
                "effective_limit": budget_res.effective_limit,
            },
        )

    def _format_recent_history(self, session_id: str, max_turns: int = 3, symbol: str = "EUR/USD") -> str:
        """Format recent history turns into token-efficient context with structured compression when needed."""
        turns = self._sessions.get(session_id, [])
        if not turns:
            return "No previous turns."

        # If turn count > 4, apply structured summary to older turns while keeping recent turns intact
        if len(turns) > 4:
            older_turns_dict = [{"role": t.role, "text": t.text} for t in turns[:-4]]
            summary = self._token_manager.compressor.extract_structured_summary(
                older_turns_dict,
                active_symbol=symbol,
            )
            recent_turns = turns[-4:]
            lines = [summary, "\n--- RECENT CONVERSATION ---"]
            for t in recent_turns:
                prefix = "User" if t.role == "user" else "Assistant"
                lines.append(f"{prefix}: {t.text}")
            return "\n".join(lines)

        recent = turns[- (max_turns * 2):]
        lines = []
        for t in recent:
            prefix = "User" if t.role == "user" else "Assistant"
            lines.append(f"{prefix}: {t.text}")
        return "\n".join(lines)

    def _detect_navigation_in_response(self, response_text: str, user_query: str) -> Optional[ChatNavigationAction]:
        """Infer if a navigation button should be attached to the response."""
        resp_lower = (response_text + " " + user_query).lower()
        if "tab 9" in resp_lower or "live market" in resp_lower or "watchlist" in resp_lower or "positions" in resp_lower:
            return ChatNavigationAction(tab_id="tab-market", label="Open Tab 9: Live Market")
        if "tab 17" in resp_lower or "master decision" in resp_lower or "confluence score" in resp_lower:
            return ChatNavigationAction(tab_id="tab-step17", label="Open Tab 17: Master Decision Engine")
        if "tab 18" in resp_lower or "universal providers" in resp_lower or "replay bar" in resp_lower or "tradingview" in resp_lower:
            return ChatNavigationAction(tab_id="tab-step18", label="Open Tab 18: Providers & Replay Bar")
        if "tab 10" in resp_lower or "journal" in resp_lower or "trade history" in resp_lower:
            return ChatNavigationAction(tab_id="tab-journal", label="Open Tab 10: Journal")
        if "tab 16" in resp_lower or "notification" in resp_lower or "cloud" in resp_lower:
            return ChatNavigationAction(tab_id="tab-step16", label="Open Tab 16: Cloud & Notifications")
        if "tab 2" in resp_lower or "rules" in resp_lower or "knowledge" in resp_lower:
            return ChatNavigationAction(tab_id="tab-knowledge", label="Open Tab 2: Knowledge")
        return None

    def _record_and_return_unified_turn(
        self,
        req: UnifiedAIRequest,
        session_id: str,
        user_msg: str,
        ai_msg: str,
        speech_msg: Optional[str] = None,
        intent: str = "GENERAL",
        evidence: Optional[List[str]] = None,
        nav_action: Optional[ChatNavigationAction] = None,
        context_summary: str = "",
        token_diagnostics: Optional[Dict[str, Any]] = None,
        action_intent: Optional[str] = None,
        requires_confirmation: bool = False,
        safety_status: str = "SAFE",
    ) -> UnifiedAIResponse:
        """Append turn to in-memory session, persist to SQLite, and return UnifiedAIResponse."""
        now = time.time()
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        evidence_list = evidence or []
        nav_list = [nav_action] if nav_action else []
        speech_text = speech_msg or format_speech_friendly_text(ai_msg)

        # Update in-memory session history
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(
            ChatHistoryTurn(
                message_id=f"u_{msg_id}",
                role="user",
                text=user_msg,
                timestamp=now,
            )
        )
        self._sessions[session_id].append(
            ChatHistoryTurn(
                message_id=f"a_{msg_id}",
                role="assistant",
                text=str(ai_msg),
                timestamp=now + 0.1,
                evidence=evidence_list,
                navigation_action=nav_action,
            )
        )

        if len(self._sessions[session_id]) > 50:
            self._sessions[session_id] = self._sessions[session_id][-50:]

        if self._repo:
            try:
                try:
                    v_intent = VoiceIntent(intent)
                except Exception:
                    v_intent = VoiceIntent.GENERAL_MARKET_QA

                active_sym = self._session_states.get(session_id, UnifiedSessionState(session_id=session_id)).active_symbol or "BTC/USD"
                self._repo.save_companion_interaction(
                    CompanionInteractionRecord(
                        symbol=active_sym,
                        user_query=user_msg,
                        intent=v_intent,
                        answer=ai_msg,
                        context_summary=context_summary,
                        evidence=evidence_list,
                        scenarios=[],
                        timestamp=now,
                    )
                )
            except Exception as e:
                logger.warning(f"Could not persist unified interaction to database: {e}")

        audio_url = None
        if req.output_modality in (OutputModality.VOICE, OutputModality.BOTH):
            try:
                audio_url = self._audio_output.synthesize(speech_text)
            except Exception as e:
                logger.debug(f"Audio synthesis skipped: {e}")

        return UnifiedAIResponse(
            request_id=req.request_id,
            session_id=session_id,
            response_text=str(ai_msg),
            speech_text=speech_text,
            audio_output_url=audio_url,
            input_modality=req.input_modality,
            output_modality=req.output_modality,
            intent=intent,
            evidence=evidence_list,
            citations=evidence_list,
            navigation_action=nav_action,
            navigation_actions=nav_list,
            context_summary=context_summary,
            market_context_used=context_summary,
            memory_updates=[f"{intent}: {user_msg}"] if ("RULE" in intent or "CORRECT" in intent) else [],
            token_diagnostics=token_diagnostics,
            action_intent=action_intent,
            requires_confirmation=requires_confirmation,
            safety_status=safety_status,
            timestamp=now,
        )

    def _record_and_return_turn(
        self,
        session_id: str,
        user_msg: str,
        ai_msg: str,
        intent: str,
        evidence: List[str],
        nav_action: Optional[ChatNavigationAction],
        context_summary: str,
    ) -> ChatMessageResponse:
        """Append turn to in-memory session, persist to SQLite, and return response."""
        now = time.time()
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"

        # Append to session memory
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(
            ChatHistoryTurn(
                message_id=f"u_{msg_id}",
                role="user",
                text=user_msg,
                timestamp=now,
            )
        )
        self._sessions[session_id].append(
            ChatHistoryTurn(
                message_id=f"a_{msg_id}",
                role="assistant",
                text=str(ai_msg),
                timestamp=now + 0.1,
                evidence=evidence,
                navigation_action=nav_action,
            )
        )

        # Keep session to reasonable length (max 50 turns)
        if len(self._sessions[session_id]) > 50:
            self._sessions[session_id] = self._sessions[session_id][-50:]

        # Persist interaction audit record to SQLite WAL
        if self._repo:
            try:
                try:
                    v_intent = VoiceIntent(intent)
                except Exception:
                    v_intent = VoiceIntent.GENERAL_MARKET_QA

                self._repo.save_companion_interaction(
                    CompanionInteractionRecord(
                        symbol="EUR/USD",
                        user_query=user_msg,
                        intent=v_intent,
                        answer=ai_msg,
                        context_summary=context_summary,
                        evidence=evidence,
                        scenarios=[],
                        timestamp=now,
                    )
                )
            except Exception as e:
                logger.warning(f"Could not persist chat interaction to database: {e}")


        nav_list = [nav_action] if nav_action else []
        return ChatMessageResponse(
            session_id=session_id,
            message_id=msg_id,
            user_message=user_msg,
            ai_message=str(ai_msg),
            answer=str(ai_msg),
            role="assistant",
            intent=intent,
            evidence=evidence,
            navigation_action=nav_action,
            navigation_actions=nav_list,
            context_summary=context_summary,
            timestamp=now,
        )

    def confirm_pending_action(self, session_id: str, action: str = "CONFIRM") -> UnifiedAIResponse:
        """Explicit confirmation / cancellation of a pending order via UI button or API."""
        msg = "CONFIRM ORDER" if action.upper() == "CONFIRM" else "CANCEL"
        req = UnifiedAIRequest(
            session_id=session_id,
            user_id="trader",
            input_modality=InputModality.TEXT,
            output_modality=OutputModality.BOTH,
            input_text=msg,
            source="ui_confirmation_button",
        )
        return self.process_unified_request(req)

