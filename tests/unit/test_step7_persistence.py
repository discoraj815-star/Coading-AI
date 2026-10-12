"""Unit tests verifying Step 7 database persistence and restart recovery."""

import os
import tempfile
import pytest
from datetime import datetime, timezone

from trading_ai.schemas.companion import (
    AlertEvent,
    AlertStateRecord,
    CompanionInteractionRecord,
    DisciplineConfig,
    DisciplineState,
    ScenarioCondition,
)
from trading_ai.schemas.enums import (
    AlertDecision,
    AlertLifecycleState,
    AlertPriority,
    AlertQuality,
    AlertType,
    VoiceIntent,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


def test_step7_restart_persistence():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # 1. Initialize first DB & Repo instance and save data
        db1 = DatabaseManager(db_path)
        repo1 = Repository(db1)

        # Save Alert
        alert = AlertEvent(
            alert_id="alt_persist_1",
            symbol="EUR/USD",
            alert_type=AlertType.SETUP_HIGH_PROBABILITY,
            priority=AlertPriority.HIGH,
            lifecycle_state=AlertLifecycleState.ALERTED,
            decision=AlertDecision.ALERT,
            quality=AlertQuality.HIGH_QUALITY,
            title="Setup Confirmed EURUSD",
            message="ICT Bullish OB confirmed at 1.0850",
            dedup_key="EURUSD:setup_1:CONFIRMED",
        )
        repo1.save_alert_event(alert)

        # Save Alert Deduplication State
        state = AlertStateRecord(
            dedup_key="EURUSD:setup_1:CONFIRMED",
            symbol="EUR/USD",
            last_alerted_at=1234567.89,
            last_lifecycle_state=AlertLifecycleState.ALERTED,
            alert_count=1,
        )
        repo1.save_alert_state(state)

        # Save Discipline
        disc = DisciplineState(
            user_id="trader_john",
            config=DisciplineConfig(user_id="trader_john", daily_loss_limit=450.0, max_daily_trades=6),
            trades_today_count=3,
            daily_realized_loss=120.0,
        )
        repo1.save_discipline_state(disc)

        # Save Scenario
        scenario = ScenarioCondition(
            scenario_id="scn_persist_1",
            symbol="EUR/USD",
            level=1.0900,
            direction="BREAKOUT",
            if_condition="IF price breaks 1.0900",
            then_outcome="THEN target 1.0950",
            unless_condition="UNLESS fakeout occurs",
            invalidated_when="Invalidated on close below 1.0890",
        )
        repo1.save_scenario_analysis(scenario)

        # Save Interaction
        inter = CompanionInteractionRecord(
            interaction_id="int_persist_1",
            symbol="EUR/USD",
            user_query="Market status",
            intent=VoiceIntent.MARKET_STATUS,
            answer="EURUSD is Bullish @ 1.0850",
            context_summary="Context summary string",
            timestamp=1234567.0,
        )
        repo1.save_companion_interaction(inter)

        # Simulate shutdown
        del repo1
        del db1

        # 2. Reopen fresh DB & Repo from disk
        db2 = DatabaseManager(db_path)
        repo2 = Repository(db2)

        # Verify Alert recovered
        loaded_alert = repo2.get_alert_event("alt_persist_1")
        assert loaded_alert is not None
        assert loaded_alert.symbol == "EUR/USD"
        assert loaded_alert.alert_type == AlertType.SETUP_HIGH_PROBABILITY
        assert loaded_alert.title == "Setup Confirmed EURUSD"

        # Verify Alert State recovered
        loaded_state = repo2.get_alert_state("EURUSD:setup_1:CONFIRMED")
        assert loaded_state is not None
        assert loaded_state.alert_count == 1
        assert loaded_state.last_alerted_at == 1234567.89

        # Verify Discipline recovered
        loaded_disc = repo2.get_discipline_state("trader_john")
        assert loaded_disc is not None
        assert loaded_disc.config.daily_loss_limit == 450.0
        assert loaded_disc.config.max_daily_trades == 6
        assert loaded_disc.trades_today_count == 3
        assert loaded_disc.daily_realized_loss == 120.0

        # Verify Scenario recovered
        scenarios = repo2.get_scenarios("EUR/USD")
        assert len(scenarios) == 1
        assert scenarios[0].level == 1.0900

        # Verify Interaction recovered
        interactions = repo2.get_companion_interactions("EUR/USD")
        assert len(interactions) == 1
        assert interactions[0].intent == VoiceIntent.MARKET_STATUS
        assert interactions[0].answer == "EURUSD is Bullish @ 1.0850"

    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
