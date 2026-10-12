"""Unit tests for Step 8 Broker Adapter and Paper Simulation."""

import pytest
from trading_ai.execution.broker_adapter import PaperBrokerAdapter
from trading_ai.schemas.enums import OrderStatus, OrderType
from trading_ai.schemas.execution import Order


@pytest.fixture
def paper_broker():
    return PaperBrokerAdapter(simulate_latency_ms=5.0)


def test_paper_broker_successful_fill(paper_broker):
    order = Order(
        intent_id="int_1",
        proposal_id="prop_1",
        symbol="EUR/USD",
        order_type=OrderType.MARKET,
        direction="BUY",
        quantity=1000.0,
        limit_price=1.0850,
    )
    status, brk_id, fill_px, meta, err = paper_broker.submit_order(order)
    assert status == OrderStatus.FILLED
    assert brk_id is not None
    assert fill_px is not None
    assert fill_px > 0
    assert err is None
    assert meta["latency_ms"] == 5.0


def test_paper_broker_injected_error_not_converted_to_filled(paper_broker):
    paper_broker.set_force_error_for_testing("EXCHANGE_TIMEOUT")

    order = Order(
        intent_id="int_2",
        proposal_id="prop_2",
        symbol="EUR/USD",
        order_type=OrderType.MARKET,
        direction="BUY",
        quantity=1000.0,
        limit_price=1.0850,
    )
    status, brk_id, fill_px, meta, err = paper_broker.submit_order(order)
    assert status == OrderStatus.BROKER_ERROR
    assert status != OrderStatus.FILLED, "Broker errors must never be marked as FILLED"
    assert err == "EXCHANGE_TIMEOUT"
    assert brk_id is None
