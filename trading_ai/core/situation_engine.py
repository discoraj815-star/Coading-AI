"""
Step 5 Intelligent Situation & Display Engine:
Synthesizes background market intelligence into the minimal decision-relevant
'What matters RIGHT NOW?' situation snapshot, and manages selective chart visualization.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.core.chart_marker import ChartMarker
from trading_ai.core.tech_fund_correlator import TechnicalFundamentalCorrelator
from trading_ai.fundamental.news_timing import NewsTimingEngine
from trading_ai.fundamental.provider import FundamentalDataProvider
from trading_ai.schemas.enums import (
    LevelType,
    MarketBias,
    MarketDirection,
    MarkingType,
    MarkingValidity,
    SetupStatus,
)
from trading_ai.schemas.fundamental import (
    FundamentalEvent,
    FundamentalProviderState,
    NewsTimingContext,
    TechnicalFundamentalAnalysis,
)
from trading_ai.schemas.market import (
    Candle,
    ChartMarking,
    CurrentSituationSnapshot,
    DisplayPolicy,
    MarketLevel,
    MarketSetup,
    MarketSnapshot,
    UnifiedMarketAnalysis,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class CurrentSituationEngine:
    """
    Synthesizes multi-timeframe analysis, deep technical structure, background tracking,
    and fundamental news event risk into a lightweight CurrentSituationSnapshot.
    """

    def __init__(
        self,
        repository: Repository,
        chart_marker: Optional[ChartMarker] = None,
        fundamental_provider: Optional[FundamentalDataProvider] = None,
        news_timing_engine: Optional[NewsTimingEngine] = None,
        correlator: Optional[TechnicalFundamentalCorrelator] = None,
    ):
        self._repo = repository
        self._marker = chart_marker or ChartMarker()
        self._fund_provider = fundamental_provider
        self._timing_engine = news_timing_engine or NewsTimingEngine()
        self._correlator = correlator or TechnicalFundamentalCorrelator()

    def evaluate_situation(
        self,
        symbol: str,
        snapshot: MarketSnapshot,
        unified_analysis: UnifiedMarketAnalysis,
        display_policy: Optional[DisplayPolicy] = None,
    ) -> CurrentSituationSnapshot:
        """
        Produce a concise current decision context snapshot.
        Extracts:
          - Overall bias
          - Current price
          - Nearest major liquidity (Buy-side / Sell-side)
          - Current POI (Order block / FVG nearest to price)
          - Active / confirmed setup
          - Invalidation & confirmation levels
          - Timeframe conflicts
          - Intelligently selected visible chart markings
        """
        policy = display_policy or DisplayPolicy()
        current_price = snapshot.latest_price
        now_ts = datetime.now(timezone.utc).timestamp()

        # 1. Bias Summary
        bias_summary = f"{symbol} overall bias is {unified_analysis.overall_bias.value}."
        if unified_analysis.htf_analysis:
            bias_summary += f" HTF ({unified_analysis.htf_analysis.timeframe}) is {unified_analysis.htf_analysis.direction.value}."
        if unified_analysis.mtf_analysis:
            bias_summary += f" MTF ({unified_analysis.mtf_analysis.timeframe}) is {unified_analysis.mtf_analysis.direction.value}."

        # 2. Nearest Major Liquidity
        all_levels = self._repo.get_market_levels(symbol=symbol, only_valid=True)
        liquidity_levels = [
            lvl for lvl in all_levels
            if lvl.level_type in (
                LevelType.LIQUIDITY_BUY_SIDE,
                LevelType.LIQUIDITY_SELL_SIDE,
                LevelType.SWING_HIGH,
                LevelType.SWING_LOW,
                LevelType.PREVIOUS_HIGH,
                LevelType.PREVIOUS_LOW,
            )
        ]
        nearest_liquidity = None
        min_liq_dist = float("inf")
        for lvl in liquidity_levels:
            dist = abs(lvl.price - current_price)
            if dist < min_liq_dist:
                min_liq_dist = dist
                nearest_liquidity = lvl

        # 3. Current Point of Interest (POI)
        poi_levels = [
            lvl for lvl in all_levels
            if lvl.level_type in (
                LevelType.OB_HIGH,
                LevelType.OB_LOW,
                LevelType.FVG_TOP,
                LevelType.FVG_BOTTOM,
                LevelType.SUPPORT,
                LevelType.RESISTANCE,
            )
        ]
        current_poi = None
        min_poi_dist = float("inf")
        for lvl in poi_levels:
            dist = abs(lvl.price - current_price)
            if dist < min_poi_dist:
                min_poi_dist = dist
                current_poi = lvl

        # 4. Active Confirmed Setup
        all_setups = self._repo.get_market_setups(symbol=symbol)
        active_setup = None
        for s in all_setups:
            if s.status in (SetupStatus.CONFIRMED, SetupStatus.ACTIVE, SetupStatus.VALID):
                active_setup = s
                break

        confirmation_state = "NO_ACTIVE_SETUP"
        invalidation_level = None
        confirmation_level = None

        if active_setup:
            confirmation_state = active_setup.status.value
            # Derive key levels
            if active_setup.direction == MarketDirection.BULLISH:
                invalidation_level = current_price * 0.985
                confirmation_level = current_price * 1.015
            else:
                invalidation_level = current_price * 1.015
                confirmation_level = current_price * 0.985

        # 5. Timeframe & Fundamental Conflicts
        conflicts = list(unified_analysis.conflict_notes)
        if unified_analysis.overall_bias == MarketBias.MULTI_TIMEFRAME_CONFLICT:
            conflicts.append("HTF and MTF directional clash: trading against HTF is prohibited.")

        # 6. Step 6 Fundamental News Intelligence & Correlation
        relevant_currencies = self._timing_engine.resolve_symbol_currencies(symbol)
        events: List[FundamentalEvent] = []
        if self._fund_provider:
            events = self._fund_provider.fetch_events(currencies=relevant_currencies)
        if not events:
            events = self._repo.get_fundamental_events(limit=50)
            events = [e for e in events if e.currency.upper() in {c.upper() for c in relevant_currencies}]

        provider_state = (
            self._fund_provider.get_provider_status()
            if self._fund_provider
            else self._repo.get_provider_state("forex_factory")
        )

        timing_context = self._timing_engine.evaluate_timing(
            symbol=symbol, events=events, current_time=datetime.now(timezone.utc)
        )
        correlation = self._correlator.correlate(
            symbol=symbol,
            unified_analysis=unified_analysis,
            active_setup=active_setup,
            timing_context=timing_context,
            provider_state=provider_state,
        )
        self._repo.save_tech_fund_analysis(correlation)

        if correlation.conflict.has_conflict:
            conflicts.append(f"FUNDAMENTAL_CONFLICT: {correlation.conflict.reason}")

        # 7. Intelligent Selective Visualization (Clean Chart Filtering)
        visible_markings = self.select_visible_markings(
            snapshot=snapshot,
            all_setups=all_setups,
            all_levels=all_levels,
            active_setup=active_setup,
            policy=policy,
        )

        # 8. Recent Changes
        recent_changes = []
        last_situation = self._repo.get_latest_current_situation(symbol=symbol)
        if last_situation:
            if last_situation.overall_bias != unified_analysis.overall_bias:
                recent_changes.append(f"Bias shifted from {last_situation.overall_bias.value} to {unified_analysis.overall_bias.value}")
            if bool(last_situation.active_confirmed_setup) != bool(active_setup):
                recent_changes.append(f"Setup state changed: {confirmation_state}")
            if last_situation.news_risk != timing_context.risk_level.value:
                recent_changes.append(f"News event risk shifted to {timing_context.risk_level.value}")

        dist_minutes = (
            round(timing_context.time_to_next_seconds / 60.0, 1)
            if timing_context.time_to_next_seconds is not None
            else None
        )

        situation = CurrentSituationSnapshot(
            symbol=symbol,
            current_price=current_price,
            overall_bias=unified_analysis.overall_bias,
            bias_summary=bias_summary,
            nearest_major_liquidity=nearest_liquidity,
            current_poi=current_poi,
            active_confirmed_setup=active_setup,
            setup_confirmation_state=confirmation_state,
            invalidation_level=invalidation_level,
            confirmation_level=confirmation_level,
            timeframe_conflicts=conflicts,
            recent_changes=recent_changes,
            visible_markings=visible_markings,
            background_tracking_count=len(all_setups) + len(all_levels),
            fundamental_context=correlation.explanation,
            relevant_news=timing_context.upcoming_events[:3] + timing_context.active_events,
            next_high_impact_event=timing_context.next_high_impact_event,
            news_time_distance_minutes=dist_minutes,
            news_risk=timing_context.risk_level.value,
            technical_fundamental_relationship=correlation.relationship.value,
            technical_fundamental_conflicts=[correlation.conflict.model_dump(mode="json")] if correlation.conflict.has_conflict else [],
            recommended_action_context=correlation.action_context,
            timestamp=now_ts,
            created_at=datetime.now(timezone.utc),
        )

        self._repo.save_current_situation(situation)
        return situation

    def select_visible_markings(
        self,
        snapshot: MarketSnapshot,
        all_setups: List[MarketSetup],
        all_levels: List[MarketLevel],
        active_setup: Optional[MarketSetup],
        policy: DisplayPolicy,
    ) -> List[ChartMarking]:
        """
        Intelligent Display Selector enforcing:
        Detection alone is NEVER sufficient for prominent display.
        Only decision-relevant levels within proximity and confirmation are shown.
        """
        raw_selected: List[ChartMarking] = []
        current_price = snapshot.latest_price
        max_dist = current_price * policy.proximity_threshold_pct

        # 1. Active Setup Marking (High Priority)
        if active_setup and active_setup.status in (SetupStatus.VALID, SetupStatus.CONFIRMED, SetupStatus.ACTIVE):
            raw_selected.append(
                ChartMarking(
                    symbol=snapshot.symbol,
                    timeframe=snapshot.timeframe,
                    marking_type=MarkingType.SETUP_ZONE,
                    coordinates_json={"direction": active_setup.direction.value, "price": current_price},
                    rule_id=active_setup.applicable_rule_id,
                    evidence_json={
                        "setup_id": active_setup.setup_id,
                        "conditions": active_setup.conditions_satisfied,
                        "confirmation": active_setup.confirmation_evidence,
                    },
                    validity=MarkingValidity.VALID,
                    confidence=active_setup.confidence,
                    priority=10,
                )
            )

        # 2. Nearest Major Levels (Proximity Filtered)
        for lvl in all_levels:
            if lvl.validity != MarkingValidity.VALID:
                continue

            dist = abs(lvl.price - current_price)
            # If not configured to show all, filter by proximity to current price
            if not policy.show_all_liquidity and dist > max_dist:
                continue

            priority = 8 if lvl.level_type in (LevelType.SWING_HIGH, LevelType.SWING_LOW) else 5
            raw_selected.append(
                ChartMarking(
                    symbol=snapshot.symbol,
                    timeframe=lvl.timeframe,
                    marking_type=MarkingType.IMPORTANT_LEVEL,
                    coordinates_json={"price": lvl.price, "level_type": lvl.level_type.value},
                    rule_id=lvl.rule_id,
                    evidence_json=lvl.evidence_json,
                    validity=MarkingValidity.VALID,
                    confidence=lvl.confidence,
                    priority=priority,
                )
            )

        # 3. Suppress duplicates and cap to max visible markings
        deduped = self._marker._suppress_duplicates(raw_selected)
        deduped.sort(key=lambda m: (m.priority, m.confidence), reverse=True)
        return deduped[: policy.max_visible_markings]
