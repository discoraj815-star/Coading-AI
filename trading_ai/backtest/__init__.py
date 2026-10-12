"""Backtest package for deterministic historical replay, asset separation, and rule promotion."""

from trading_ai.backtest.engine import BacktestEngine
from trading_ai.backtest.promotion_manager import RulePromotionManager
from trading_ai.backtest.replay_clock import BacktestClock, SUPPORTED_REPLAY_SPEEDS

__all__ = ["BacktestEngine", "RulePromotionManager", "BacktestClock", "SUPPORTED_REPLAY_SPEEDS"]
