"""Step 18: Universal Providers Package."""
from trading_ai.providers.base import (
    BaseUniversalProvider,
    BacktestProvider,
    BrokerProvider,
    TradingTerminalProvider,
    MarketDataProvider,
    OrderflowProvider,
    FootprintProvider,
    VolumeProfileProvider,
    NewsProvider,
)
from trading_ai.providers.backtest import (
    StandardHistoricalBacktestProvider,
    TradingLeadsBacktestProvider,
)
from trading_ai.providers.broker import (
    PaperBrokerProvider,
    DeltaExchangeBrokerProvider,
    XMBrokerProvider,
)
from trading_ai.providers.terminal import MetaTrader5TerminalProvider
from trading_ai.providers.market_data import (
    SimulatedMarketDataProviderAdapter,
    LiveFeedMarketDataProvider,
)
from trading_ai.providers.tradingview import TradingViewMarketDataProvider
from trading_ai.providers.orderflow import OrderflowFeedProvider
from trading_ai.providers.footprint import FootprintFeedProvider
from trading_ai.providers.volume_profile import VolumeProfileFeedProvider
from trading_ai.providers.news import ForexFactoryNewsProvider
from trading_ai.providers.registry import UniversalProviderRegistry
from trading_ai.providers.connection_manager import (
    UniversalConnectionManager,
    mask_sensitive_config,
)

__all__ = [
    "BaseUniversalProvider",
    "BacktestProvider",
    "BrokerProvider",
    "TradingTerminalProvider",
    "MarketDataProvider",
    "OrderflowProvider",
    "FootprintProvider",
    "VolumeProfileProvider",
    "NewsProvider",
    "StandardHistoricalBacktestProvider",
    "TradingLeadsBacktestProvider",
    "PaperBrokerProvider",
    "DeltaExchangeBrokerProvider",
    "XMBrokerProvider",
    "MetaTrader5TerminalProvider",
    "SimulatedMarketDataProviderAdapter",
    "LiveFeedMarketDataProvider",
    "TradingViewMarketDataProvider",
    "OrderflowFeedProvider",
    "FootprintFeedProvider",
    "VolumeProfileFeedProvider",
    "ForexFactoryNewsProvider",
    "UniversalProviderRegistry",
    "UniversalConnectionManager",
    "mask_sensitive_config",
]
