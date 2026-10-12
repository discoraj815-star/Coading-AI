"""Trading Companion Core: Central unified orchestrator for alerts, discipline, scenarios, and voice engine."""

import logging
from typing import Any, Dict, List, Optional

from trading_ai.companion.alert_engine import ProactiveAlertEngine
from trading_ai.companion.context_builder import CompanionContextBuilder
from trading_ai.companion.discipline_engine import TradingDisciplineEngine
from trading_ai.companion.scenario_engine import ScenarioEngine
from trading_ai.companion.voice_engine import InteractiveVoiceEngine
from trading_ai.schemas.companion import (
    AlertEvent,
    CompanionQueryRequest,
    CompanionQueryResponse,
    DisciplineConfig,
    DisciplineState,
    ScenarioCondition,
    TradeConfirmationGateResult,
    TradeIntentRequest,
    TradingCompanionContext,
)
from trading_ai.schemas.market import CurrentSituationSnapshot
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class TradingCompanionCore:
    """
    Unified entry point for the Step 7 AI Trading Companion.

    Coordinates:
      - CompanionContextBuilder (unified minimal sufficient decision context)
      - ProactiveAlertEngine (low-noise alert pipeline with persistent deduplication)
      - TradingDisciplineEngine (user risk rule enforcement & trade confirmation gate)
      - ScenarioEngine (conditional IF-THEN-UNLESS market breakdowns)
      - InteractiveVoiceEngine (speech/text intent routing & concise responses)

    Does NOT create a duplicate market analysis brain; seamlessly consumes
    Step 5 Current Situation and Step 6 Fundamental Intelligence.
    """

    def __init__(self, repository: Repository):
        self._repo = repository
        self._context_builder = CompanionContextBuilder(repository)
        self._alert_engine = ProactiveAlertEngine(repository)
        self._discipline_engine = TradingDisciplineEngine(repository)
        self._scenario_engine = ScenarioEngine(repository)
        self._voice_engine = InteractiveVoiceEngine(
            repository=repository,
            context_builder=self._context_builder,
            scenario_engine=self._scenario_engine,
            discipline_engine=self._discipline_engine,
        )

    @property
    def alert_engine(self) -> ProactiveAlertEngine:
        return self._alert_engine

    @property
    def discipline_engine(self) -> TradingDisciplineEngine:
        return self._discipline_engine

    @property
    def scenario_engine(self) -> ScenarioEngine:
        return self._scenario_engine

    @property
    def voice_engine(self) -> InteractiveVoiceEngine:
        return self._voice_engine

    def set_chat_service(self, chat_service: Any) -> None:
        """Connect unified AI brain chat service to the companion voice engine."""
        self._voice_engine._chat_service = chat_service

    def get_companion_context(self, symbol: str, user_id: str = "default_user") -> TradingCompanionContext:
        """Fetch unified decision context for symbol."""
        return self._context_builder.build_context(symbol=symbol, user_id=user_id)

    def evaluate_symbol_alerts(
        self,
        symbol: str,
        current_situation: Optional[CurrentSituationSnapshot] = None,
        previous_situation: Optional[CurrentSituationSnapshot] = None,
    ) -> List[AlertEvent]:
        """Run proactive alert qualification and noise filtering."""
        return self._alert_engine.evaluate_and_generate_alerts(
            symbol=symbol,
            current_situation=current_situation,
            previous_situation=previous_situation,
        )

    def query_companion(self, req: CompanionQueryRequest) -> CompanionQueryResponse:
        """Process conversational user inquiry or voice prompt."""
        return self._voice_engine.process_query(req)

    def get_scenarios(self, symbol: str, custom_level: Optional[float] = None) -> List[ScenarioCondition]:
        """Retrieve or compute conditional scenarios for symbol."""
        return self._scenario_engine.generate_scenarios_for_symbol(symbol=symbol, custom_level=custom_level)

    def check_trade_intent(self, intent: TradeIntentRequest, user_id: str = "default_user") -> TradeConfirmationGateResult:
        """Validate trade intent through the confirmation gate."""
        return self._discipline_engine.check_trade_confirmation_gate(intent=intent, user_id=user_id)

    def get_discipline_status(self, user_id: str = "default_user") -> DisciplineState:
        """Fetch current discipline configuration and violation status."""
        state, _ = self._discipline_engine.evaluate_discipline_status(user_id=user_id)
        return state

    def update_discipline_config(self, user_id: str, config: DisciplineConfig) -> DisciplineState:
        """Save updated user discipline parameters."""
        return self._discipline_engine.update_config(user_id=user_id, config=config)

    def get_alerts(
        self,
        symbol: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> List[AlertEvent]:
        """Fetch persisted alerts."""
        return self._repo.get_alerts(symbol=symbol, unacknowledged_only=unacknowledged_only, limit=limit)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        return self._repo.acknowledge_alert(alert_id)
