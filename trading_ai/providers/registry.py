"""Step 18: Universal Provider Registry."""
import logging
from typing import Any, Dict, List, Optional

from trading_ai.providers.backtest import (
    StandardHistoricalBacktestProvider,
    TradingLeadsBacktestProvider,
)
from trading_ai.providers.base import BaseUniversalProvider
from trading_ai.providers.broker import (
    DeltaExchangeBrokerProvider,
    DynamicBrokerProvider,
    PaperBrokerProvider,
    XMBrokerProvider,
)
from trading_ai.providers.footprint import FootprintFeedProvider
from trading_ai.providers.market_data import (
    LiveFeedMarketDataProvider,
    SimulatedMarketDataProviderAdapter,
)
from trading_ai.providers.news import ForexFactoryNewsProvider
from trading_ai.providers.orderflow import OrderflowFeedProvider
from trading_ai.providers.terminal import MetaTrader5TerminalProvider
from trading_ai.providers.tradingview import TradingViewMarketDataProvider
from trading_ai.providers.volume_profile import VolumeProfileFeedProvider
from trading_ai.schemas.provider import (
    ConnectionTestResult,
    ProviderCategory,
    ProviderLifecycleState,
    ProviderMetadata,
    TradingProviderExecutionState,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class UniversalProviderRegistry:
    """
    Central registry and discovery engine for all Trading AI external providers.

    Guarantees:
      - AI Brain, CoreTradeLogic, RiskEngine, and ExecutionGate remain provider-independent.
      - Switching providers never mutates learned rules, memory, or journal lessons.
      - Provider states persist in SQLite WAL storage.
    """

    def __init__(self, repository: Optional[Repository] = None):
        self._repo = repository
        self._providers: Dict[str, BaseUniversalProvider] = {}
        self._active_per_category: Dict[ProviderCategory, str] = {
            ProviderCategory.BACKTEST: "trading_leads",
            ProviderCategory.BROKER: "paper_broker",
            ProviderCategory.TRADING_TERMINAL: "metatrader5",
            ProviderCategory.MARKET_DATA: "tradingview",
            ProviderCategory.ORDERFLOW: "orderflow_feed",
            ProviderCategory.FOOTPRINT: "footprint_feed",
            ProviderCategory.VOLUME_PROFILE: "volume_profile_feed",
            ProviderCategory.NEWS: "forex_factory",
        }
        self._register_default_providers()
        self._sync_with_repository()

    def _register_default_providers(self) -> None:
        """Register the complete set of initial provider targets."""
        mt5_terminal = MetaTrader5TerminalProvider()

        defaults = [
            TradingViewMarketDataProvider(),
            StandardHistoricalBacktestProvider(self._repo),
            TradingLeadsBacktestProvider(),
            PaperBrokerProvider(),
            DeltaExchangeBrokerProvider(),
            XMBrokerProvider(mt5_terminal_provider=mt5_terminal),
            mt5_terminal,
            SimulatedMarketDataProviderAdapter(),
            LiveFeedMarketDataProvider(),
            OrderflowFeedProvider(),
            FootprintFeedProvider(),
            VolumeProfileFeedProvider(),
            ForexFactoryNewsProvider(),
        ]
        for p in defaults:
            self._providers[p.provider_id] = p

    def _sync_with_repository(self) -> None:
        """Synchronize provider states and approvals from database."""
        if not self._repo:
            return
        for p in self._providers.values():
            try:
                db_record = self._repo.get_universal_provider(p.provider_id)
                if db_record:
                    p.metadata.enabled = db_record.get("enabled", p.metadata.enabled)
                    p.metadata.approved = db_record.get("approved", p.metadata.approved)
                    p.metadata.status = ProviderLifecycleState(db_record.get("status", p.metadata.status.value))
                    p.metadata.trading_state = TradingProviderExecutionState(
                        db_record.get("trading_state", p.metadata.trading_state.value)
                    )
                else:
                    self._repo.save_universal_provider(p.metadata)
            except Exception as e:
                logger.warning(f"Error syncing provider {p.provider_id} with database: {e}")

    def register(self, provider: BaseUniversalProvider) -> None:
        """Register or override a provider."""
        self._providers[provider.provider_id] = provider
        if self._repo:
            self._repo.save_universal_provider(provider.metadata)

    register_provider = register

    def get_provider(self, provider_id: str) -> Optional[BaseUniversalProvider]:
        """Fetch provider instance by ID."""
        return self._providers.get(provider_id)

    def get_provider_metadata(self, provider_id: str) -> Optional[ProviderMetadata]:
        """Fetch provider metadata by ID."""
        p = self.get_provider(provider_id)
        return p.metadata if p else None

    def get_active_provider(self, category: ProviderCategory) -> Optional[BaseUniversalProvider]:
        """Return the currently selected provider for a category."""
        active_id = self._active_per_category.get(category)
        if active_id and active_id in self._providers:
            return self._providers[active_id]
        # Fallback to first available enabled provider in category
        for p in self._providers.values():
            if p.category == category and p.is_enabled:
                return p
        return None

    def set_active_provider(self, category: ProviderCategory, provider_id: str) -> bool:
        """Switch active provider for a category without altering the AI brain."""
        if provider_id not in self._providers:
            raise ValueError(f"Provider '{provider_id}' is not registered.")
        p = self._providers[provider_id]
        if p.category != category:
            raise ValueError(f"Provider '{provider_id}' is of category {p.category.value}, not {category.value}.")
        self._active_per_category[category] = provider_id
        logger.info(f"Active provider for {category.value} switched to {provider_id}.")
        return True

    def get_active_broker(self) -> Optional[BaseUniversalProvider]:
        """Return the active connected broker context."""
        return self.get_active_provider(ProviderCategory.BROKER)

    def set_active_broker(self, broker_id: str) -> bool:
        """
        Manually switch active broker context.
        INVARIANT: Broker switching NEVER executes an order, NEVER enables Auto Live,
        and NEVER bypasses ExecutionGate or RiskEngine.
        """
        if broker_id not in self._providers:
            raise ValueError(f"Broker provider '{broker_id}' is not registered.")
        p = self._providers[broker_id]
        if p.category != ProviderCategory.BROKER:
            raise ValueError(f"Provider '{broker_id}' is not a broker (category={p.category.value}).")

        # Update active flags across all brokers
        for prov_id, prov in self._providers.items():
            if prov.category == ProviderCategory.BROKER:
                prov.metadata.is_active_broker = (prov_id == broker_id)
                if self._repo:
                    try:
                        self._repo.save_universal_provider(prov.metadata)
                    except Exception:
                        pass

        self._active_per_category[ProviderCategory.BROKER] = broker_id
        logger.info(f"Active broker context safely switched to: {broker_id} ({p.provider_name})")
        return True

    def add_broker(
        self,
        broker_id: str,
        broker_name: str = "",
        adapter_type: str = "GENERIC_BROKER",
        capabilities: Optional[List[str]] = None,
        authentication_type: str = "API_KEY",
        supported_markets: Optional[List[str]] = None,
        supported_symbols: Optional[List[str]] = None,
        live_supported: bool = False,
        paper_supported: bool = True,
        config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        execution_state: Optional[str] = None,
        **kwargs: Any,
    ) -> DynamicBrokerProvider:
        """
        Dynamically register a new broker without code changes.
        Validates capabilities and credentials securely.
        """
        clean_id = broker_id.strip().lower().replace(" ", "_")
        actual_name = (broker_name or name or kwargs.get("provider_name", broker_id)).strip()
        cfg = dict(config or {})
        for k in ("api_key", "api_secret", "server", "account_id"):
            if k in kwargs and kwargs[k]:
                cfg[k] = kwargs[k]

        provider = DynamicBrokerProvider(
            broker_id=clean_id,
            broker_name=actual_name,
            adapter_type=adapter_type,
            capabilities=capabilities,
            authentication_type=authentication_type,
            supported_markets=supported_markets,
            supported_symbols=supported_symbols,
            live_supported=live_supported,
            paper_supported=paper_supported,
            config=cfg if cfg else None,
            execution_state=execution_state or kwargs.get("execution_state"),
        )
        self.register(provider)
        logger.info(f"Dynamically registered new broker: {clean_id} ({actual_name})")
        return provider

    def list_providers(self, category: Optional[str] = None) -> List[ProviderMetadata]:
        """List provider metadata with current runtime state."""
        out = []
        for p in self._providers.values():
            if category and p.category.value != category.upper():
                continue
            out.append(p.metadata)
        return out

    def toggle_provider(self, provider_id: str, enabled: bool) -> None:
        """Enable or disable provider."""
        if provider_id in self._providers:
            self._providers[provider_id].metadata.enabled = enabled
            if self._repo:
                self._repo.toggle_universal_provider(provider_id, enabled)

    def approve_provider(self, provider_id: str, approved: bool) -> None:
        """Grant or revoke user approval for provider."""
        if provider_id in self._providers:
            self._providers[provider_id].metadata.approved = approved
            if self._repo:
                self._repo.approve_universal_provider(provider_id, approved)

    def test_provider(self, provider_id: str) -> ConnectionTestResult:
        """Safely test connection for a provider."""
        p = self.get_provider(provider_id)
        if not p:
            raise ValueError(f"Provider '{provider_id}' not found.")
        res = p.test_connection()
        if self._repo:
            new_status = ProviderLifecycleState.CONNECTED if res.success else ProviderLifecycleState.ERROR
            self._repo.update_universal_provider_status(
                provider_id=provider_id,
                status=new_status.value,
                last_error=None if res.success else res.message,
            )
            p.metadata.status = new_status
            if not res.success:
                p.metadata.last_error = res.message
        return res
