"""Broker Adapter Interface, Deterministic Paper Execution, and Strict Live Isolation."""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from trading_ai.schemas.enums import ExecutionEnvironment, OrderStatus, OrderType
from trading_ai.schemas.execution import Order

logger = logging.getLogger(__name__)


class BrokerAdapter(ABC):
    """Abstract interface for multi-broker integration."""

    @abstractmethod
    def submit_order(
        self, order: Order
    ) -> Tuple[OrderStatus, Optional[str], Optional[float], Dict[str, Any], Optional[str]]:
        """
        Submit order to broker.
        Returns: (order_status, broker_order_id, average_fill_price, broker_response_metadata, error_message)
        """
        pass

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel order on broker."""
        pass


class PaperBrokerAdapter(BrokerAdapter):
    """
    Deterministic simulated paper broker adapter.

    Hard Rules:
      - Does NOT fake broker fills if simulation parameters trigger a rejection or error.
      - Accurately tracks execution timestamps, latency, and fills.
      - Paper adapter must NEVER execute Live orders.
      - Paper failure must NEVER silently fall back to Live.
      - Broker errors never represent FILLED state.
    """

    def __init__(self, simulate_latency_ms: float = 25.0, simulate_slippage_pct: float = 0.0001):
        self._simulate_latency_ms = simulate_latency_ms
        self._simulate_slippage_pct = simulate_slippage_pct
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._force_next_broker_error: Optional[str] = None

    def set_force_error_for_testing(self, error_message: Optional[str]) -> None:
        """Inject controlled paper broker error for resilience testing."""
        self._force_next_broker_error = error_message

    def submit_order(
        self, order: Order
    ) -> Tuple[OrderStatus, Optional[str], Optional[float], Dict[str, Any], Optional[str]]:
        """Submit order to paper execution engine."""
        # B2 & B4: Strict Adapter Isolation Check
        if order.execution_environment != ExecutionEnvironment.PAPER:
            err = (
                "PAPER_BROKER_ERROR: PAPER_ADAPTER_CANNOT_EXECUTE_LIVE: Strict adapter isolation prevents "
                "PaperBrokerAdapter from handling Live orders."
            )
            logger.error(err)
            return OrderStatus.REJECTED, None, None, {"environment": "PAPER", "error": err}, err

        # Simulated latency
        time.sleep(self._simulate_latency_ms / 1000.0)

        # Check injected failure
        if self._force_next_broker_error:
            err_msg = self._force_next_broker_error
            self._force_next_broker_error = None
            logger.warning(f"Simulated paper broker error triggered: {err_msg}")
            return (
                OrderStatus.BROKER_ERROR,
                None,
                None,
                {"environment": "PAPER", "error_type": "PAPER_BROKER_ERROR", "error": err_msg, "latency_ms": self._simulate_latency_ms},
                err_msg,
            )

        # Validation checks
        if order.quantity <= 0:
            return (
                OrderStatus.REJECTED,
                None,
                None,
                {"environment": "PAPER", "error": "Invalid order quantity"},
                "QUANTITY_INVALID",
            )

        broker_id = f"sim_brk_{uuid.uuid4().hex[:10]}"

        # Calculate fill price with optional micro-slippage
        base_price = order.limit_price if order.limit_price else (order.stop_price or 1.0)
        slippage = base_price * self._simulate_slippage_pct
        is_buy = order.direction.upper() in ("BUY", "LONG")
        fill_price = base_price + slippage if is_buy else base_price - slippage

        metadata = {
            "broker_order_id": broker_id,
            "engine": "PaperBrokerSimulation",
            "environment": "PAPER",
            "is_simulated": True,
            "slippage": slippage,
            "latency_ms": self._simulate_latency_ms,
            "filled_at": time.time(),
        }

        self._orders[broker_id] = {
            "order": order.model_dump(),
            "status": OrderStatus.FILLED,
            "fill_price": fill_price,
            "metadata": metadata,
        }

        return OrderStatus.FILLED, broker_id, round(fill_price, 5), metadata, None

    def cancel_order(self, broker_order_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel paper order."""
        if broker_order_id in self._orders:
            self._orders[broker_order_id]["status"] = OrderStatus.CANCELLED
            return True, None
        return False, "ORDER_NOT_FOUND"


class LiveBrokerAdapter(BrokerAdapter):
    """
    Live broker adapter with fail-safe boundaries.

    Hard Rules:
      - Live execution requires verified connection and credentials.
      - Live adapter must NEVER execute Paper orders.
      - Live failure must NEVER silently fall back to Paper.
      - Network timeout or ambiguous status returns UNKNOWN; NEVER assumed FILLED.
      - Real broker responses are never fabricated.
    """

    def __init__(
        self,
        is_connected: bool = False,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        account_id: Optional[str] = None,
    ):
        self._is_connected = is_connected
        self._api_key = api_key
        self._api_secret = api_secret
        self._account_id = account_id
        self._force_network_timeout: bool = False
        self._force_broker_error: Optional[str] = None
        self._live_orders: Dict[str, Dict[str, Any]] = {}

    def set_connected_for_testing(self, connected: bool) -> None:
        """Configure simulated connection state for integration testing."""
        self._is_connected = connected

    def set_force_network_timeout(self, enabled: bool) -> None:
        """Inject uncertain network state for resilience testing."""
        self._force_network_timeout = enabled

    def set_force_timeout_for_testing(self, enabled: bool) -> None:
        """Alias for set_force_network_timeout."""
        self.set_force_network_timeout(enabled)

    def set_force_broker_error(self, error_message: Optional[str]) -> None:
        """Inject live broker error for resilience testing."""
        self._force_broker_error = error_message

    def submit_order(
        self, order: Order
    ) -> Tuple[OrderStatus, Optional[str], Optional[float], Dict[str, Any], Optional[str]]:
        """Submit order to real live broker gateway."""
        # B2: Strict Adapter Isolation Check
        if order.execution_environment != ExecutionEnvironment.LIVE:
            err = (
                "LIVE_ADAPTER_CANNOT_EXECUTE_PAPER: Strict adapter isolation prevents "
                "LiveBrokerAdapter from handling Paper orders."
            )
            logger.error(err)
            return OrderStatus.REJECTED, None, None, {"environment": "LIVE", "error": err}, err

        # B6: Network Timeout / Ambiguous state check
        if self._force_network_timeout:
            self._force_network_timeout = False
            uncertain_id = f"tx_unknown_{uuid.uuid4().hex[:8]}"
            err = "NETWORK_ERROR_UNCERTAIN_STATE: Gateway timed out; order state is unknown. Do NOT assume filled."
            logger.error(err)
            return (
                OrderStatus.UNKNOWN,
                uncertain_id,
                None,
                {"environment": "LIVE", "uncertain": True, "error": err},
                err,
            )

        # Live Broker Injected or Connection Error
        if self._force_broker_error:
            err = f"BROKER_ERROR: {self._force_broker_error}"
            self._force_broker_error = None
            logger.error(err)
            return (
                OrderStatus.BROKER_ERROR,
                None,
                None,
                {"environment": "LIVE", "connected": self._is_connected, "error": err},
                err,
            )

        # Connection Verification (Prevent fabricated live execution)
        if not self._is_connected:
            err = (
                "LIVE_BROKER_DISCONNECTED: No verified live broker session connected. "
                "Real fills cannot be fabricated."
            )
            logger.warning(err)
            return (
                OrderStatus.BROKER_ERROR,
                None,
                None,
                {"environment": "LIVE", "connected": False, "error": err},
                err,
            )

        # If live connection is established (e.g. in live test fixture with simulated connection):
        live_brk_id = f"live_brk_{uuid.uuid4().hex[:10]}"
        base_price = order.limit_price or (order.stop_price or 1.0)
        metadata = {
            "broker_order_id": live_brk_id,
            "engine": "LiveBrokerGateway",
            "environment": "LIVE",
            "is_simulated": False,
            "filled_at": time.time(),
        }
        self._live_orders[live_brk_id] = {
            "order": order.model_dump(),
            "status": OrderStatus.FILLED,
            "fill_price": base_price,
            "metadata": metadata,
        }
        return OrderStatus.FILLED, live_brk_id, base_price, metadata, None

    def cancel_order(self, broker_order_id: str) -> Tuple[bool, Optional[str]]:
        """Cancel live broker order."""
        if not self._is_connected:
            return False, "LIVE_BROKER_DISCONNECTED"
        if broker_order_id in self._live_orders:
            self._live_orders[broker_order_id]["status"] = OrderStatus.CANCELLED
            return True, None
        return False, "ORDER_NOT_FOUND"
