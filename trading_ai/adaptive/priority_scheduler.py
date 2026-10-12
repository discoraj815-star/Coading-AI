"""
Adaptive Compute Scheduler & Priority Allocation.
Dynamically adjusts polling frequency, reasoning depth, and Gemini gating based on
market regime, active positions, confirmed setups, and imminent news risk.
Strict rule: Gemini is NEVER called on every raw tick; deterministic fast path handles raw math.
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, Optional

from trading_ai.schemas.enums import EventPriority, NewsRiskLevel, SetupStatus
from trading_ai.schemas.performance import AdaptiveComputePolicy

logger = logging.getLogger(__name__)


class AdaptiveComputeScheduler:
    """
    Allocates system compute resources proportionally to market criticality.
    Quiet markets conserve compute/tokens; active positions and volatile setups get maximum responsiveness.
    """

    def __init__(self):
        self._last_gemini_call: Optional[datetime] = None
        self._min_gemini_interval_seconds: float = 3.0  # Throttling floor

    def evaluate_policy(
        self,
        has_active_position: bool = False,
        active_setup_status: Optional[SetupStatus] = None,
        is_volatile: bool = False,
        news_risk: NewsRiskLevel = NewsRiskLevel.NO_RISK,
        is_market_open: bool = True,
    ) -> AdaptiveComputePolicy:
        """
        Derives the optimal AdaptiveComputePolicy based on current market state.
        """
        # CRITICAL: Active live/paper position requiring continuous risk monitoring
        if has_active_position:
            return AdaptiveComputePolicy(
                priority=EventPriority.CRITICAL,
                monitoring_interval_seconds=1.0,
                technical_recalc_depth="INCREMENTAL",
                gemini_skip_ratio=0.3,
                reasoning_depth="DEEP",
                news_check_interval_seconds=5.0,
            )

        # HIGH: Confirmed setup or high news risk
        if active_setup_status == SetupStatus.CONFIRMED or news_risk in (NewsRiskLevel.HIGH_RISK, NewsRiskLevel.EXTREME_RISK):
            return AdaptiveComputePolicy(
                priority=EventPriority.HIGH,
                monitoring_interval_seconds=2.0,
                technical_recalc_depth="INCREMENTAL",
                gemini_skip_ratio=0.5,
                reasoning_depth="BALANCED",
                news_check_interval_seconds=10.0,
            )

        # MEDIUM: Volatile conditions or setup candidate forming
        if is_volatile or active_setup_status in (SetupStatus.DETECTED, SetupStatus.VALID):
            return AdaptiveComputePolicy(
                priority=EventPriority.MEDIUM,
                monitoring_interval_seconds=5.0,
                technical_recalc_depth="INCREMENTAL",
                gemini_skip_ratio=0.8,
                reasoning_depth="FAST",
                news_check_interval_seconds=30.0,
            )

        # LOW / QUIET: Quiet market with no setup
        return AdaptiveComputePolicy(
            priority=EventPriority.LOW,
            monitoring_interval_seconds=15.0,
            technical_recalc_depth="INCREMENTAL",
            gemini_skip_ratio=0.95,
            reasoning_depth="FAST",
            news_check_interval_seconds=60.0,
        )

    def should_trigger_gemini(
        self,
        policy: AdaptiveComputePolicy,
        is_setup_transition: bool = False,
        is_news_conflict: bool = False,
        is_critical_risk_alert: bool = False,
        now: Optional[datetime] = None,
    ) -> bool:
        """
        Determines whether Gemini reasoning should be invoked or handled strictly deterministically.
        Strictly prevents rate-limit hammering while ensuring high-importance triggers are addressed.
        """
        current_time = now or datetime.now(timezone.utc)

        # Throttle check
        if self._last_gemini_call:
            elapsed = (current_time - self._last_gemini_call).total_seconds()
            if elapsed < self._min_gemini_interval_seconds:
                return False

        # Explicit high-value triggers ALWAYS warrant Gemini reasoning
        if is_critical_risk_alert or is_setup_transition or is_news_conflict:
            self._last_gemini_call = current_time
            return True

        # Otherwise, respect the policy skip ratio (e.g. skip 95% of routine ticks)
        if policy.reasoning_depth == "DEEP" and policy.gemini_skip_ratio < 0.5:
            self._last_gemini_call = current_time
            return True

        return False
