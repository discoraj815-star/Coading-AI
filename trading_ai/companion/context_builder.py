"""Trading Companion Context Builder: Unifies Step 5 Situation, Step 6 Fundamentals, Memory, and Discipline."""

import logging
from datetime import datetime, timezone
from typing import Optional

from trading_ai.schemas.companion import TradingCompanionContext
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class CompanionContextBuilder:
    """
    Constructs the unified decision-relevant TradingCompanionContext for a symbol.
    Consumes existing Step 5 CurrentSituationSnapshot and Step 6 TechnicalFundamentalAnalysis
    without duplicating market analysis brains.
    """

    def __init__(self, repository: Repository):
        self._repo = repository

    def build_context(self, symbol: str, user_id: str = "default_user") -> TradingCompanionContext:
        """Construct minimal sufficient context snapshot for companion engines."""
        situation = self._repo.get_latest_current_situation(symbol)
        tech_fund = self._repo.get_latest_tech_fund_analysis(symbol)
        discipline = self._repo.get_discipline_state(user_id)
        unacked_alerts = self._repo.get_alerts(symbol=symbol, unacknowledged_only=True, limit=50)

        # Baseline defaults if situation is not yet computed
        current_price = 0.0
        bias_str = "NEUTRAL"
        bias_summary = "Awaiting market situation computation."
        nearest_liq_dict = None
        current_poi_dict = None
        active_setup_dict = None
        setup_state = "NO_ACTIVE_SETUP"
        inv_level = None
        conf_level = None
        news_risk_str = "NO_RISK"
        tech_fund_rel_str = "TECHNICALLY_VALID"
        next_event_dict = None
        news_distance = None
        recent_changes_list = []

        if situation:
            current_price = situation.current_price
            bias_str = situation.overall_bias.value if hasattr(situation.overall_bias, "value") else str(situation.overall_bias)
            bias_summary = situation.bias_summary
            setup_state = situation.setup_confirmation_state
            inv_level = situation.invalidation_level
            conf_level = situation.confirmation_level
            news_risk_str = situation.news_risk
            tech_fund_rel_str = situation.technical_fundamental_relationship
            news_distance = situation.news_time_distance_minutes
            recent_changes_list = list(situation.recent_changes)

            if situation.nearest_major_liquidity:
                nearest_liq_dict = situation.nearest_major_liquidity.model_dump(mode="json")
            if situation.current_poi:
                current_poi_dict = situation.current_poi.model_dump(mode="json")
            if situation.active_confirmed_setup:
                active_setup_dict = situation.active_confirmed_setup.model_dump(mode="json")
            if situation.next_high_impact_event:
                next_event_dict = situation.next_high_impact_event.model_dump(mode="json")

        if tech_fund and tech_fund.timing_context and tech_fund.timing_context.upcoming_events:
            if not next_event_dict:
                first_event = tech_fund.timing_context.upcoming_events[0]
                next_event_dict = first_event.model_dump(mode="json")
                time_sec = tech_fund.timing_context.time_to_next_seconds
                news_distance = round(time_sec / 60.0, 1) if time_sec is not None else None

        discipline_dict = {
            "configured": bool(discipline and discipline.config),
            "trades_today": discipline.trades_today_count if discipline else 0,
            "daily_loss": discipline.daily_realized_loss if discipline else 0.0,
            "violations": discipline.active_violations if discipline else [],
            "cooldown_active": bool(discipline and discipline.cooldown_active_until and discipline.cooldown_active_until > datetime.now(timezone.utc)),
        }

        return TradingCompanionContext(
            symbol=symbol,
            current_price=current_price,
            bias=bias_str,
            bias_summary=bias_summary,
            nearest_liquidity=nearest_liq_dict,
            current_poi=current_poi_dict,
            active_setup=active_setup_dict,
            setup_confirmation_state=setup_state,
            invalidation_level=inv_level,
            confirmation_level=conf_level,
            news_risk=news_risk_str,
            tech_fund_relationship=tech_fund_rel_str,
            next_high_impact_event=next_event_dict,
            news_time_distance_minutes=news_distance,
            recent_changes=recent_changes_list,
            discipline_status=discipline_dict,
            active_alerts_count=len(unacked_alerts),
            timestamp=datetime.now(timezone.utc).timestamp(),
        )
