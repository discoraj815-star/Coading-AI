import pytest
from trading_ai.schemas.enums import SourceDomain, SourceAuthority, TradeDecisionState
from trading_ai.schemas.trade_logic import NormalizedKnowledgeItem
from trading_ai.trade_logic.data_fusion import DataFusionEngine
from trading_ai.trade_logic.why_not_trade import WhyNotTradeEngine
from trading_ai.trade_logic.core_trade_logic import CoreTradeLogic
from trading_ai.trade_logic.domain_manager import SourceDomainManager
from trading_ai.trade_logic.knowledge_normalizer import KnowledgeNormalizer
from trading_ai.storage.repository import Repository
from trading_ai.storage.db import DatabaseManager

def test_adversarial_youtube_buy_mentorship_sell_conflict():
    """When YouTube says BUY and Mentorship says SELL, engine must NOT average or guess; it must output CONFLICTED."""
    repo = Repository(DatabaseManager(":memory:"))
    fusion = DataFusionEngine(repo=repo)
    normalizer = KnowledgeNormalizer()

    item_buy = normalizer.normalize_rule(
        title="YT Liquidity Sweep Buy",
        content="Enter BUY on sweep of Asian lows",
        domain=SourceDomain.YOUTUBE_LECTURES,
        rule_type="SETUP",
        authority=SourceAuthority.APPROVED_LECTURE,
        direction="BUY",
        symbols=["GBPUSD"]
    )
    item_sell = normalizer.normalize_rule(
        title="Mentor Macro Sell Directive",
        content="Enter SELL on weekly supply retest; do not take long sweeps",
        domain=SourceDomain.MENTORSHIP_SESSIONS,
        rule_type="SETUP",
        authority=SourceAuthority.APPROVED_MENTOR,
        direction="SELL",
        symbols=["GBPUSD"]
    )
    repo.save_normalized_knowledge(item_buy)
    repo.save_normalized_knowledge(item_sell)

    res = fusion.evaluate_confluence(
        symbol="GBPUSD",
        timeframe="15m",
        direction="BUY",
        market_state={"sweep": True}
    )
    assert res.state == TradeDecisionState.CONFLICTED.value
    assert len(res.conflicting_rules) > 0

    # Trade logic on CONFLICTED must produce NO_SETUP / NO_TRADE
    core = CoreTradeLogic()
    why_not = WhyNotTradeEngine().evaluate_negative_conditions(
        symbol="GBPUSD",
        direction="BUY",
        confluence_state=res.state
    )
    decision = core.generate_trade_decision(
        symbol="GBPUSD",
        direction="BUY",
        confluence_evaluation=res,
        why_not_result=why_not
    )
    assert decision.decision_state in [TradeDecisionState.CONFLICTED.value, TradeDecisionState.NO_SETUP.value]
    assert decision.entry_price is None

def test_adversarial_htf_contradicts_ltf():
    """Higher-Timeframe Bearish bias must block an LTF BUY setup in WhyNotTradeEngine."""
    why_not = WhyNotTradeEngine()
    res = why_not.evaluate_negative_conditions(
        symbol="EURUSD",
        direction="BUY",
        high_timeframe_bias="BEARISH",
        confluence_state=TradeDecisionState.VALID_SETUP.value
    )
    assert res.has_blocking_reasons
    assert any("bias" in r.lower() for r in res.blocking_reasons)

def test_adversarial_invalidated_poi_blocks_trade():
    """When a Point of Interest (POI) has been breached or invalidated, trade must be rejected."""
    why_not = WhyNotTradeEngine()
    res = why_not.evaluate_negative_conditions(
        symbol="EURUSD",
        direction="BUY",
        high_timeframe_bias="BULLISH",
        confluence_state=TradeDecisionState.VALID_SETUP.value,
        has_invalidation=True
    )
    assert res.has_blocking_reasons
    assert any("invalidat" in r.lower() for r in res.blocking_reasons)

def test_adversarial_missing_confirmation_blocks():
    """When a setup lacks mandatory confirmation, execution must not proceed."""
    why_not = WhyNotTradeEngine()
    res = why_not.evaluate_negative_conditions(
        symbol="EURUSD",
        direction="BUY",
        high_timeframe_bias="BULLISH",
        confluence_state=TradeDecisionState.VALID_SETUP.value,
        missing_confirmations=["Displacement candle not closed"]
    )
    assert res.has_blocking_reasons
    assert any("confirmation" in r.lower() for r in res.blocking_reasons)

def test_adversarial_path_traversal_cross_domain_leak():
    """Attempting to write or read across physical domain boundaries must raise PermissionError."""
    manager = SourceDomainManager()
    with pytest.raises(PermissionError):
        manager.resolve_domain_path(SourceDomain.YOUTUBE_LECTURES, "../02_Mentorship_Sessions/secret_mentor_rules.json")

    with pytest.raises(PermissionError):
        manager.resolve_domain_path(SourceDomain.LIVE_TRADE_LOGS, "..\\..\\Windows\\System32\\calc.exe")

def test_adversarial_source_authority_override():
    """Mentor rule (Priority 2) must take precedence over YouTube lecture rule (Priority 4)."""
    normalizer = KnowledgeNormalizer()
    yt_rule = normalizer.normalize_rule(
        title="YouTube Basic Entry",
        content="Enter instantly on wick touch",
        domain=SourceDomain.YOUTUBE_LECTURES,
        authority=SourceAuthority.APPROVED_LECTURE,
        confidence=0.7
    )
    mentor_rule = normalizer.normalize_rule(
        title="Mentor Advanced Entry",
        content="Never enter on wick touch; require body close",
        domain=SourceDomain.MENTORSHIP_SESSIONS,
        authority=SourceAuthority.APPROVED_MENTOR,
        confidence=0.95
    )

    # Authority priorities: Approved Mentor (2) has higher authority than Approved Lecture (4)
    assert mentor_rule.authority_priority < yt_rule.authority_priority
