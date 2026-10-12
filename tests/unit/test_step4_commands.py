"""
Unit tests for Step 4 user commands integration with UserCommandParser and UserCommandService.
"""

import pytest
from trading_ai.core.command_parser import UserCommandParser
from trading_ai.core.command_service import UserCommandService
from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
from trading_ai.schemas.enums import (
    CommandIntent,
    KnowledgeType,
    KnowledgeValidationStatus,
    MarketBias,
    MarketDirection,
    ScreenAccessStatus,
    SetupStatus,
    WatchlistMonitorStatus,
)
from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.schemas.market import MarketSetup
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def test_repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


@pytest.fixture
def parser():
    return UserCommandParser()


@pytest.fixture
def service(test_repo):
    provider = SimulatedMarketDataProvider(seed=42)
    return UserCommandService(
        repository=test_repo,
        data_provider=provider,
    )


def test_command_analyze_symbol(parser, service):
    parse_result = parser.parse("Analyze BTC")
    assert parse_result.command.intent == CommandIntent.ANALYZE_SYMBOL
    assert parse_result.command.parameters.get("symbol") == "BTC"

    exec_result = service.execute_command(parse_result.command)
    assert exec_result["status"] == "success"
    assert exec_result["symbol"] == "BTC"
    assert exec_result["overall_bias"] in [
        MarketBias.BULLISH_BIAS.value,
        MarketBias.BEARISH_BIAS.value,
        MarketBias.NEUTRAL_BIAS.value,
        MarketBias.MULTI_TIMEFRAME_CONFLICT.value,
    ]


def test_command_analyze_chart_respects_screen_permission(parser, service, test_repo):
    # Case A: No screen access granted (Privacy boundary enforced)
    parse_result = parser.parse("Analyze this chart")
    assert parse_result.command.intent == CommandIntent.ANALYZE_CHART

    exec_res_no_perm = service.execute_command(parse_result.command)
    assert exec_res_no_perm["status"] == "success"
    assert exec_res_no_perm["screen_access_granted"] is False
    assert "SCREEN_ACCESS_DISABLED" in exec_res_no_perm["screen_source"]

    # Case B: Screen access granted by user
    perm = service._get_screen_access_manager().request_access(app_name="TradingView", window_title="BTCUSD Chart")
    service._get_screen_access_manager().grant_access(perm.permission_id)

    exec_res_granted = service.execute_command(parse_result.command)
    assert exec_res_granted["status"] == "success"
    assert exec_res_granted["screen_access_granted"] is True
    assert "TradingView" in exec_res_granted["screen_source"]


def test_command_analyze_watchlist(parser, service):
    parse_result = parser.parse("Analyze my whole watchlist")
    assert parse_result.command.intent == CommandIntent.ANALYZE_WATCHLIST

    exec_result = service.execute_command(parse_result.command)
    assert exec_result["status"] == "success"
    assert len(exec_result["symbols_analyzed"]) >= 2
    assert "BTC/USD" in exec_result["summary"]


def test_command_check_htf(parser, service):
    parse_result = parser.parse("Check higher timeframe")
    assert parse_result.command.intent == CommandIntent.CHECK_HTF

    exec_result = service.execute_command(parse_result.command)
    assert exec_result["status"] == "success"
    assert exec_result["htf_timeframe"] == "1d"
    assert exec_result["direction"] in ["BULLISH", "BEARISH", "NEUTRAL", "RANGING", "UNKNOWN"]
    assert "key_levels" in exec_result


def test_command_filter_valid_setups(parser, service, test_repo):
    # Seed one valid setup and one invalidated setup
    s_valid = MarketSetup(
        symbol="BTC/USD",
        timeframe="15m",
        setup_type="FVG",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.VALID,
        confidence=0.9,
    )
    s_invalid = MarketSetup(
        symbol="BTC/USD",
        timeframe="15m",
        setup_type="OB",
        direction=MarketDirection.BEARISH,
        status=SetupStatus.INVALIDATED,
        confidence=0.0,
    )
    test_repo.save_market_setup(s_valid)
    test_repo.save_market_setup(s_invalid)

    parse_result = parser.parse("Show valid setups only")
    assert parse_result.command.intent == CommandIntent.FILTER_VALID_SETUPS

    exec_result = service.execute_command(parse_result.command)
    assert exec_result["status"] == "success"
    assert exec_result["count"] == 1
    assert exec_result["valid_setups"][0]["status"] == SetupStatus.VALID.value
    assert exec_result["valid_setups"][0]["setup_type"] == "FVG"


def test_command_use_lecture_rules_only(parser, service):
    parse_result = parser.parse("Use my lecture rules only")
    assert parse_result.command.intent == CommandIntent.USE_LECTURE_RULES_ONLY

    exec_result = service.execute_command(parse_result.command)
    assert exec_result["status"] == "success"
    assert exec_result["mode"] == "LECTURE_CONSTRAINED_STRICT"


def test_command_explain_setup(parser, service, test_repo):
    # Seed a lecture rule and a validated setup
    rule = KnowledgeItem(
        job_id="lecture_01",
        item_type=KnowledgeType.RULE,
        title="FVG Entry Rule",
        statement="Enter at 50% equilibrium of 15m FVG",
        conditions=["Displacement present"],
        exceptions=["Major news"],
    )
    test_repo.save_knowledge([rule])

    setup = MarketSetup(
        symbol="ETH/USD",
        timeframe="15m",
        setup_type="FVG",
        direction=MarketDirection.BULLISH,
        status=SetupStatus.VALID,
        applicable_rule_id=rule.item_id,
        conditions_satisfied=["Displacement present", "HTF alignment confirmed"],
        supporting_evidence=["Institutional impulse confirmed"],
        confidence=0.95,
    )
    test_repo.save_market_setup(setup)

    parse_result = parser.parse("Why is this setup valid?")
    assert parse_result.command.intent == CommandIntent.EXPLAIN_SETUP

    parse_result.command.parameters["symbol"] = "ETH/USD"
    exec_result = service.execute_command(parse_result.command)
    assert exec_result["status"] == "success"
    assert "VALID" in exec_result["explanation"]
    assert "Displacement present" in exec_result["explanation"]


def test_command_mark_pattern_clean_chart(parser, service, test_repo):
    parse_result = parser.parse("Mark valid FVG")
    assert parse_result.command.intent == CommandIntent.MARK_PATTERN

    exec_result = service.execute_command(parse_result.command)
    assert exec_result["status"] == "success"
    assert exec_result["pattern"] == "FVG"
    assert "clean_chart_note" in exec_result


def test_command_monitoring_controls(parser, service):
    # Stop monitoring
    stop_parse = parser.parse("Stop background monitoring")
    assert stop_parse.command.intent == CommandIntent.STOP_MONITORING
    stop_res = service.execute_command(stop_parse.command)
    assert stop_res["status"] == "success"
    assert stop_res["monitor_status"] == WatchlistMonitorStatus.STOPPED.value

    # Resume monitoring
    resume_parse = parser.parse("Resume background monitoring")
    assert resume_parse.command.intent == CommandIntent.RESUME_MONITORING
    resume_res = service.execute_command(resume_parse.command)
    assert resume_res["status"] == "success"
    assert resume_res["monitor_status"] == WatchlistMonitorStatus.MONITORING.value
