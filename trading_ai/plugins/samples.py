"""Sample concrete plugins demonstrating categories, capability isolation, and broker safety."""

from typing import Any, Dict, Optional

from trading_ai.plugins.base import BasePlugin, BrokerReadPlugin, MarketDataPlugin, NewsPlugin, TechnicalAnalysisPlugin
from trading_ai.plugins.manifest import PluginCategory, PluginCapability, PluginManifest


class MockForexFactoryNewsPlugin(NewsPlugin):
    """Forex Factory Economic Calendar and News Plugin."""

    def __init__(self):
        manifest = PluginManifest(
            plugin_id="plugin_forex_factory",
            name="Forex Factory Calendar Plugin",
            version="1.0.0",
            category=PluginCategory.FOREX_FACTORY,
            description="Provides economic calendar events and news impact data for forex majors",
            author="TradingAI Official",
            capabilities=[PluginCapability.READ_NEWS],
            config={"update_interval": 300},
        )
        super().__init__(manifest)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        return True

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def fetch_upcoming_events(self, currency: Optional[str] = None) -> Dict[str, Any]:
        return {
            "events": [
                {
                    "title": "US Non-Farm Payrolls",
                    "currency": currency or "USD",
                    "impact": "HIGH",
                    "time_to_event_minutes": 120,
                    "forecast": "180K",
                    "previous": "175K",
                }
            ],
            "risk_level": "NO_RISK_IMMEDIATE",
        }

    def execute_action(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if action == "fetch_news":
            return self.fetch_upcoming_events((params or {}).get("currency"))
        return {"result": f"Action {action} handled"}


class MockTechnicalAnalysisPlugin(TechnicalAnalysisPlugin):
    """Quantitative Technical Analysis and Indicator Calculation Plugin."""

    def __init__(self):
        manifest = PluginManifest(
            plugin_id="plugin_ta_engine",
            name="Technical Analysis & Indicator Engine",
            version="1.0.0",
            category=PluginCategory.TECHNICAL_ANALYSIS,
            description="Computes RSI, EMA, Fair Value Gaps, and ATR metrics",
            author="TradingAI Official",
            capabilities=[PluginCapability.PROVIDE_TECHNICAL_ANALYSIS],
            config={"default_period": 14},
        )
        super().__init__(manifest)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        return True

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def compute_indicators(self, symbol: str, candles: list) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "rsi_14": 52.4,
            "ema_20": 1.0845,
            "ema_50": 1.0820,
            "fvg_detected": True,
            "fvg_zone": [1.0830, 1.0842],
            "trend": "BULLISH",
        }

    def execute_action(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        p = params or {}
        if action == "compute":
            return self.compute_indicators(p.get("symbol", "EURUSD"), p.get("candles", []))
        return {"result": f"Action {action} handled"}


class MockBrokerReadPlugin(BrokerReadPlugin):
    """Broker Read-Only Information Plugin.

    TRADING SAFETY:
    This plugin provides account balance, equity, and open positions.
    It CANNOT execute trades directly!
    """

    def __init__(self):
        manifest = PluginManifest(
            plugin_id="plugin_broker_read",
            name="Interactive Brokers Read Feeder",
            version="1.0.0",
            category=PluginCategory.BROKER,
            description="Read-only account balance, margin, and position monitor",
            author="TradingAI Official",
            capabilities=[PluginCapability.BROKER_READ],
            config={"paper_mode": True},
        )
        super().__init__(manifest)

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        return True

    def start(self) -> bool:
        return True

    def stop(self) -> bool:
        return True

    def get_account_summary(self) -> Dict[str, Any]:
        return {
            "account_id": "U12345678",
            "currency": "USD",
            "balance": 100000.0,
            "equity": 102340.50,
            "unrealized_pnl": 2340.50,
            "margin_used": 15000.0,
            "environment": "PAPER",
        }

    def get_open_positions(self) -> list:
        return [
            {
                "symbol": "EURUSD",
                "units": 100000,
                "direction": "BUY",
                "entry_price": 1.0815,
                "current_price": 1.0835,
                "unrealized_pnl": 200.0,
            }
        ]

    def execute_action(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if action == "account_summary":
            return self.get_account_summary()
        elif action == "open_positions":
            return {"positions": self.get_open_positions()}
        return {"result": f"Action {action} handled"}
