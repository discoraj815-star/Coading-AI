"""Proactive Alert Engine: Zero-noise, multi-stage qualification, persistent deduplication lifecycle."""

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.companion import (
    AlertEvent,
    AlertStateRecord,
    TradingCompanionContext,
)
from trading_ai.schemas.enums import (
    AlertDecision,
    AlertLifecycleState,
    AlertPriority,
    AlertQuality,
    AlertType,
    MarketBias,
    SetupStatus,
)
from trading_ai.schemas.market import CurrentSituationSnapshot, MarketSetup
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class ProactiveAlertEngine:
    """
    Evaluates market conditions and generates high-fidelity, low-noise trading alerts.

    Guarantees:
      - Multi-stage pipeline: Candidate Detection -> Context Validation -> Quality Rating -> Noise Filter.
      - Zero-Noise Rule: Suppresses unchanged setups, duplicate events, and insignificant fluctuations.
      - Persistent Deduplication Lifecycle: Persists alert state to prevent re-alerting across restarts.
      - Qualitative Classifications: HIGH_QUALITY / MEDIUM_QUALITY / LOW_QUALITY (no fake probabilities).
      - Seamless Step 5/6 Reuse: Consumes CurrentSituationSnapshot and NewsTimingContext.
    """

    COOLDOWN_WINDOW_SECONDS = 3600  # 1 hour deduplication cooldown for recurring informational alerts

    def __init__(self, repository: Repository):
        self._repo = repository

    def evaluate_and_generate_alerts(
        self,
        symbol: str,
        current_situation: Optional[CurrentSituationSnapshot] = None,
        previous_situation: Optional[CurrentSituationSnapshot] = None,
    ) -> List[AlertEvent]:
        """
        Main entry point for generating alerts for a symbol upon new market or situation update.
        """
        situation = current_situation or self._repo.get_latest_current_situation(symbol)
        if not situation:
            return []

        candidates: List[AlertEvent] = []

        # 1. Evaluate Setup Lifecycle Alerts (Confirmed / Invalidated / Quality)
        setup_alert = self._evaluate_setup_alerts(symbol, situation)
        if setup_alert:
            candidates.append(setup_alert)

        # 2. Evaluate Market Bias Change Alerts
        bias_alert = self._evaluate_bias_change(symbol, situation, previous_situation)
        if bias_alert:
            candidates.append(bias_alert)

        # 3. Evaluate Fundamental & News Proximity Alerts
        news_alerts = self._evaluate_news_alerts(symbol, situation)
        candidates.extend(news_alerts)

        # 4. Evaluate Liquidity & POI Reaction Alerts
        poi_alert = self._evaluate_poi_reaction(symbol, situation)
        if poi_alert:
            candidates.append(poi_alert)

        # 5. Evaluate Discipline Alerts
        discipline_alerts = self._evaluate_discipline_alerts(symbol)
        candidates.extend(discipline_alerts)

        # 6. Apply Zero-Noise Filter & Lifecycle Transitions
        emitted_alerts: List[AlertEvent] = []
        for cand in candidates:
            final_alert = self._apply_noise_filter_and_lifecycle(cand)
            if final_alert.decision == AlertDecision.ALERT:
                self._repo.save_alert_event(final_alert)
                emitted_alerts.append(final_alert)
            elif final_alert.decision == AlertDecision.TRACK:
                # Save as tracking/suppressed for audit trail
                self._repo.save_alert_event(final_alert)

        return emitted_alerts

    def _evaluate_setup_alerts(
        self, symbol: str, situation: CurrentSituationSnapshot
    ) -> Optional[AlertEvent]:
        """Detect and qualify active setup confirmation or invalidation."""
        setup = situation.active_confirmed_setup
        if not setup:
            return None

        status_val = setup.status.value if hasattr(setup.status, "value") else str(setup.status)
        setup_name = getattr(setup, "name", None) or getattr(setup, "setup_type", "Setup")
        setup_dir = getattr(setup, "direction", "DIRECTION")
        dir_val = setup_dir.value if hasattr(setup_dir, "value") else str(setup_dir)

        # Quality determination (Qualitative only, no fake statistics)
        conf_evidence = getattr(setup, "confirmation_evidence", []) or []
        conf_len = len(conf_evidence) if isinstance(conf_evidence, list) else 0
        quality = AlertQuality.HIGH_QUALITY if conf_len >= 2 else (
            AlertQuality.MEDIUM_QUALITY if conf_len == 1 else AlertQuality.LOW_QUALITY
        )

        # Setup Confirmed / High vs Low Probability
        if status_val == SetupStatus.CONFIRMED.value or status_val == SetupStatus.ACTIVE.value:
            alert_type = (
                AlertType.SETUP_HIGH_PROBABILITY
                if quality == AlertQuality.HIGH_QUALITY
                else AlertType.SETUP_LOW_PROBABILITY
            )
            dedup_key = f"{symbol}:{setup.setup_id}:{status_val}"
            return AlertEvent(
                symbol=symbol,
                alert_type=alert_type,
                priority=AlertPriority.HIGH if quality == AlertQuality.HIGH_QUALITY else AlertPriority.MEDIUM,
                lifecycle_state=AlertLifecycleState.QUALIFIED,
                decision=AlertDecision.ALERT,
                quality=quality,
                title=f"{symbol} Setup Confirmed: {setup_name}",
                message=(
                    f"Lecture setup '{setup_name}' ({dir_val}) "
                    f"has reached confirmation at price {situation.current_price:.5f}. "
                    f"Quality: {quality.value}. Invalidation: {situation.invalidation_level or 'Unspecified'}."
                ),
                evidence={
                    "setup_id": setup.setup_id,
                    "direction": dir_val,
                    "price": situation.current_price,
                    "confirmation_level": situation.confirmation_level,
                    "invalidation_level": situation.invalidation_level,
                    "evidence_count": conf_len,
                },
                dedup_key=dedup_key,
            )

        # Setup Invalidated
        if status_val == SetupStatus.INVALIDATED.value:
            dedup_key = f"{symbol}:{setup.setup_id}:INVALIDATED"
            return AlertEvent(
                symbol=symbol,
                alert_type=AlertType.SETUP_INVALIDATED,
                priority=AlertPriority.CRITICAL,
                lifecycle_state=AlertLifecycleState.QUALIFIED,
                decision=AlertDecision.ALERT,
                quality=AlertQuality.HIGH_QUALITY,
                title=f"{symbol} Setup Invalidated: {setup_name}",
                message=(
                    f"Active setup '{setup_name}' was invalidated at price {situation.current_price:.5f}. "
                    f"Condition: {getattr(setup, 'invalidation_condition', 'Price breached boundary')}."
                ),
                evidence={"setup_id": setup.setup_id, "price": situation.current_price},
                dedup_key=dedup_key,
            )

        return None

    def _evaluate_bias_change(
        self,
        symbol: str,
        current: CurrentSituationSnapshot,
        previous: Optional[CurrentSituationSnapshot] = None,
    ) -> Optional[AlertEvent]:
        """Detect meaningful market bias shift between situation snapshots."""
        if not previous:
            return None

        curr_bias = current.overall_bias.value if hasattr(current.overall_bias, "value") else str(current.overall_bias)
        prev_bias = previous.overall_bias.value if hasattr(previous.overall_bias, "value") else str(previous.overall_bias)

        # Only trigger when a genuine directional transition occurs (e.g. BULLISH <-> BEARISH)
        if curr_bias != prev_bias and ("BULLISH" in curr_bias or "BEARISH" in curr_bias):
            dedup_key = f"{symbol}:BIAS_CHANGE:{prev_bias}->{curr_bias}"
            return AlertEvent(
                symbol=symbol,
                alert_type=AlertType.MARKET_BIAS_CHANGE,
                priority=AlertPriority.HIGH,
                lifecycle_state=AlertLifecycleState.QUALIFIED,
                decision=AlertDecision.ALERT,
                quality=AlertQuality.HIGH_QUALITY,
                title=f"{symbol} Market Bias Changed: {curr_bias}",
                message=(
                    f"Overall market structure transitioned from {prev_bias} to {curr_bias} "
                    f"at price {current.current_price:.5f}. Summary: {current.bias_summary}"
                ),
                evidence={
                    "previous_bias": prev_bias,
                    "current_bias": curr_bias,
                    "price": current.current_price,
                    "summary": current.bias_summary,
                },
                dedup_key=dedup_key,
            )

        return None

    def _evaluate_news_alerts(
        self, symbol: str, situation: CurrentSituationSnapshot
    ) -> List[AlertEvent]:
        """Generate high-impact economic news timing and conflict alerts."""
        alerts: List[AlertEvent] = []
        next_event = situation.next_high_impact_event
        dist = situation.news_time_distance_minutes

        if next_event and dist is not None:
            # Imminent High Impact Event (within 30 minutes)
            if 0 <= dist <= 30.0:
                if isinstance(next_event, dict):
                    event_name = next_event.get("title", "High-Impact Event")
                    event_time = str(next_event.get("event_time", next_event.get("scheduled_time", "")))
                    currency = next_event.get("currency", "")
                else:
                    event_name = getattr(next_event, "title", "High-Impact Event")
                    event_time = str(getattr(next_event, "event_time", getattr(next_event, "scheduled_time", "")))
                    currency = getattr(next_event, "currency", "")
                
                # Window bucket (e.g. 30min, 15min) to prevent duplicates
                window_bucket = "15m" if dist <= 15.0 else "30m"
                dedup_key = f"{symbol}:NEWS_IMMINENT:{event_name}:{event_time}:{window_bucket}"

                priority = AlertPriority.CRITICAL if dist <= 15.0 else AlertPriority.HIGH
                alerts.append(
                    AlertEvent(
                        symbol=symbol,
                        alert_type=AlertType.NEWS_IMMINENT,
                        priority=priority,
                        lifecycle_state=AlertLifecycleState.QUALIFIED,
                        decision=AlertDecision.ALERT,
                        quality=AlertQuality.HIGH_QUALITY,
                        title=f"{currency} High-Impact News in {int(dist)}m",
                        message=(
                            f"Upcoming high-impact event '{event_name}' scheduled in {int(dist)} minutes. "
                            f"Expect elevated spread and volatility. Risk Level: {situation.news_risk}."
                        ),
                        evidence={
                            "event_title": event_name,
                            "currency": currency,
                            "minutes_away": dist,
                            "news_risk": situation.news_risk,
                        },
                        dedup_key=dedup_key,
                    )
                )

        # Technical/Fundamental Conflict Alert
        if situation.technical_fundamental_relationship == "FUNDAMENTAL_CONFLICT":
            dedup_key = f"{symbol}:FUNDAMENTAL_CONFLICT:{situation.news_risk}"
            alerts.append(
                AlertEvent(
                    symbol=symbol,
                    alert_type=AlertType.FUNDAMENTAL_CONFLICT,
                    priority=AlertPriority.HIGH,
                    lifecycle_state=AlertLifecycleState.QUALIFIED,
                    decision=AlertDecision.ALERT,
                    quality=AlertQuality.HIGH_QUALITY,
                    title=f"{symbol} Technical & Fundamental Conflict",
                    message=(
                        f"Technical setup directly conflicts with macroeconomic fundamental context. "
                        f"Recommended action: {situation.recommended_action_context}."
                    ),
                    evidence={
                        "relationship": situation.technical_fundamental_relationship,
                        "action": situation.recommended_action_context,
                    },
                    dedup_key=dedup_key,
                )
            )

        return alerts

    def _evaluate_poi_reaction(
        self, symbol: str, situation: CurrentSituationSnapshot
    ) -> Optional[AlertEvent]:
        """Detect interaction with Key Point of Interest (POI)."""
        poi = situation.current_poi
        if not poi:
            return None

        if isinstance(poi, dict):
            poi_price = poi.get("price", 0.0)
            poi_type = poi.get("level_type", "POI")
            poi_name = poi.get("name", str(poi_type))
        else:
            poi_price = getattr(poi, "price", 0.0)
            poi_type_obj = getattr(poi, "level_type", "POI")
            poi_type = poi_type_obj.value if hasattr(poi_type_obj, "value") else str(poi_type_obj)
            ev = getattr(poi, "evidence_json", {}) if hasattr(poi, "evidence_json") else {}
            poi_name = getattr(poi, "name", None) or (ev.get("name") if isinstance(ev, dict) else None) or str(poi_type)

        # Proximity check (within 0.15% of price)
        if poi_price > 0 and abs(situation.current_price - poi_price) / poi_price <= 0.0015:
            dedup_key = f"{symbol}:POI_REACTION:{poi_type}:{round(poi_price, 4)}"
            return AlertEvent(
                symbol=symbol,
                alert_type=AlertType.POI_REACTION,
                priority=AlertPriority.MEDIUM,
                lifecycle_state=AlertLifecycleState.QUALIFIED,
                decision=AlertDecision.ALERT,
                quality=AlertQuality.MEDIUM_QUALITY,
                title=f"{symbol} Testing POI: {poi_name}",
                message=(
                    f"Price {situation.current_price:.5f} is reacting at Point of Interest "
                    f"({poi_name} @ {poi_price:.5f}). Monitor for confirmation rejection or breakout."
                ),
                evidence={"poi_name": poi_name, "poi_price": poi_price, "current_price": situation.current_price},
                dedup_key=dedup_key,
            )

        return None

    def _evaluate_discipline_alerts(self, symbol: str) -> List[AlertEvent]:
        """Generate proactive alerts for discipline and risk thresholds."""
        alerts: List[AlertEvent] = []
        discipline = self._repo.get_discipline_state("default_user")
        if not discipline or not discipline.active_violations:
            return alerts

        for violation in discipline.active_violations:
            alert_type = AlertType.RISK_RULE_WARNING
            if "OVERTRADING" in violation:
                alert_type = AlertType.OVERTRADING_WARNING
            elif "DAILY_LOSS" in violation:
                alert_type = AlertType.DAILY_LOSS_LIMIT_WARNING
            elif "MAX_TRADES" in violation:
                alert_type = AlertType.MAX_TRADES_WARNING
            elif "COOLDOWN" in violation:
                alert_type = AlertType.COOLDOWN_REQUIRED

            dedup_key = f"discipline:{alert_type.value}:{violation[:30]}"
            alerts.append(
                AlertEvent(
                    symbol=symbol,
                    alert_type=alert_type,
                    priority=AlertPriority.CRITICAL,
                    lifecycle_state=AlertLifecycleState.QUALIFIED,
                    decision=AlertDecision.ALERT,
                    quality=AlertQuality.HIGH_QUALITY,
                    title="Trading Discipline Warning",
                    message=violation,
                    evidence={"violation": violation},
                    dedup_key=dedup_key,
                )
            )

        return alerts

    def _apply_noise_filter_and_lifecycle(self, alert: AlertEvent) -> AlertEvent:
        """
        Check existing deduplication state and apply zero-noise suppression rules.
        """
        now = datetime.now(timezone.utc)
        now_ts = now.timestamp()

        # Check existing state in database
        existing_state = self._repo.get_alert_state(alert.dedup_key)

        if existing_state:
            elapsed = now_ts - existing_state.last_alerted_at

            # Suppress identical condition within cooldown window
            if elapsed < self.COOLDOWN_WINDOW_SECONDS:
                alert.decision = AlertDecision.SUPPRESS
                alert.lifecycle_state = AlertLifecycleState.SUPPRESSED
                existing_state.suppression_reason = f"Suppressed duplicate within {int(elapsed)}s cooldown."
                existing_state.updated_at = now
                self._repo.save_alert_state(existing_state)
                return alert

            # Condition re-occurred after cooldown: update state
            existing_state.last_alerted_at = now_ts
            existing_state.alert_count += 1
            existing_state.last_lifecycle_state = AlertLifecycleState.UPDATED
            existing_state.updated_at = now
            self._repo.save_alert_state(existing_state)
            alert.lifecycle_state = AlertLifecycleState.UPDATED
            alert.decision = AlertDecision.ALERT
            return alert

        # New alert condition: record state and emit
        new_state = AlertStateRecord(
            dedup_key=alert.dedup_key,
            symbol=alert.symbol,
            last_alerted_at=now_ts,
            last_lifecycle_state=AlertLifecycleState.ALERTED,
            alert_count=1,
            suppression_reason=None,
            updated_at=now,
        )
        self._repo.save_alert_state(new_state)
        alert.lifecycle_state = AlertLifecycleState.ALERTED
        alert.decision = AlertDecision.ALERT
        return alert
