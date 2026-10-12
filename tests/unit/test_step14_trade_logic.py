import pytest
import os
import shutil
import tempfile
from trading_ai.schemas.enums import SourceDomain, SourceAuthority, RelationshipType, TradeDecisionState
from trading_ai.schemas.trade_logic import (
    NormalizedKnowledgeItem, KnowledgeRelationship, ConfluenceEvaluation,
    WhyNotTradeResult, CoreTradeDecision
)
from trading_ai.trade_logic.domain_manager import SourceDomainManager
from trading_ai.trade_logic.knowledge_normalizer import KnowledgeNormalizer
from trading_ai.trade_logic.relationship_graph import CrossSourceRelationshipGraph
from trading_ai.trade_logic.retriever import UnifiedKnowledgeRetriever
from trading_ai.trade_logic.data_fusion import DataFusionEngine
from trading_ai.trade_logic.why_not_trade import WhyNotTradeEngine
from trading_ai.trade_logic.core_trade_logic import CoreTradeLogic
from trading_ai.trade_logic.journal_feedback import JournalFeedbackEngine
from trading_ai.storage.repository import Repository
from trading_ai.storage.db import DatabaseManager

@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)

@pytest.fixture
def repo():
    return Repository(DatabaseManager(":memory:"))


def test_source_domain_isolation(temp_dir):
    manager = SourceDomainManager(base_dir=temp_dir)
    domains = manager.get_domain_stats()
    assert len(domains) == 4
    assert SourceDomain.YOUTUBE_LECTURES.value in domains
    assert SourceDomain.MENTORSHIP_SESSIONS.value in domains
    assert SourceDomain.LIVE_TRADE_LOGS.value in domains
    assert SourceDomain.TRADING_TECH_WORK.value in domains

    # Safe write within domain
    path = manager.save_domain_file(SourceDomain.YOUTUBE_LECTURES, "notes.txt", "Lecture notes content")
    assert os.path.exists(path)

    # Cross-domain path traversal attempt must fail
    with pytest.raises(PermissionError):
        manager.resolve_domain_path(SourceDomain.YOUTUBE_LECTURES, "../02_Mentorship_Sessions/infiltrate.txt")

    with pytest.raises(PermissionError):
        manager.save_domain_file(SourceDomain.YOUTUBE_LECTURES, "../../evil.txt", "malicious")

def test_knowledge_normalizer():
    normalizer = KnowledgeNormalizer()
    item = normalizer.normalize_rule(
        title="Liquidity Sweep Strategy",
        content="Enter on displacement after 15m sweep",
        domain=SourceDomain.YOUTUBE_LECTURES,
        rule_type="SETUP_RULE",
        authority=SourceAuthority.APPROVED_LECTURE,
        confidence=0.85,
        symbols=["EURUSD"],
        timeframes=["15m"],
        evidence_reference="lecture_12_frame_400.jpg"
    )
    assert isinstance(item, NormalizedKnowledgeItem)
    assert item.source_domain == SourceDomain.YOUTUBE_LECTURES.value
    assert item.authority == SourceAuthority.APPROVED_LECTURE.value
    assert item.confidence == 0.85
    assert item.rule_version == "1.0"
    assert "lecture_12_frame_400.jpg" in item.evidence_references

def test_relationship_graph(repo):
    graph = CrossSourceRelationshipGraph(repo=repo)
    rel = graph.link_items(
        source_item_id="item_yt_1",
        target_item_id="item_mentor_1",
        relationship_type=RelationshipType.REFINES,
        description="Mentorship refines liquidity sweep entry condition"
    )
    assert isinstance(rel, KnowledgeRelationship)
    assert rel.relationship_type == RelationshipType.REFINES.value

    # Graph retrieval
    rels = graph.get_relationships_for_item("item_yt_1")
    assert any(r.target_item_id == "item_mentor_1" for r in rels)

def test_unified_retriever(repo):
    retriever = UnifiedKnowledgeRetriever(repo=repo, max_tokens=1000)
    normalizer = KnowledgeNormalizer()
    item1 = normalizer.normalize_rule(
        title="FVG Entry",
        content="Buy when price taps into 15m FVG",
        domain=SourceDomain.YOUTUBE_LECTURES,
        rule_type="ENTRY_RULE",
        authority=SourceAuthority.APPROVED_LECTURE,
        confidence=0.9,
        symbols=["EURUSD"],
        timeframes=["15m"]
    )
    repo.save_normalized_knowledge(item1)

    results = retriever.retrieve(
        symbol="EURUSD",
        timeframe="15m",
        domain=SourceDomain.YOUTUBE_LECTURES
    )
    assert len(results) >= 1
    assert any(r.item_id == item1.item_id for r in results)

def test_data_fusion_confluence_and_conflict(repo):
    fusion = DataFusionEngine(repo=repo)
    normalizer = KnowledgeNormalizer()

    # Create matching rules
    rule_yt = normalizer.normalize_rule(
        title="Bullish Sweep Entry",
        content="Enter LONG after sell-side liquidity sweep",
        domain=SourceDomain.YOUTUBE_LECTURES,
        rule_type="ENTRY_RULE",
        authority=SourceAuthority.APPROVED_LECTURE,
        confidence=0.88,
        symbols=["EURUSD"],
        direction="BUY"
    )
    rule_mentor = normalizer.normalize_rule(
        title="Mentor HTF Filter",
        content="Only buy if 4H discount condition holds",
        domain=SourceDomain.MENTORSHIP_SESSIONS,
        rule_type="CONFIRMATION_FILTER",
        authority=SourceAuthority.APPROVED_MENTOR,
        confidence=0.95,
        symbols=["EURUSD"],
        direction="BUY"
    )
    repo.save_normalized_knowledge(rule_yt)
    repo.save_normalized_knowledge(rule_mentor)

    # 1. High confluence case
    eval_res = fusion.evaluate_confluence(
        symbol="EURUSD",
        timeframe="15m",
        direction="BUY",
        market_state={"sweep_detected": True, "discount_aligned": True}
    )
    assert eval_res.state in [TradeDecisionState.HIGH_CONFLUENCE.value, TradeDecisionState.VALID_SETUP.value]
    assert len(eval_res.matched_rules) >= 2
    assert len(eval_res.conflicting_rules) == 0

    # 2. Conflicting rules case (YouTube BUY vs Mentorship SELL)
    rule_conflict = normalizer.normalize_rule(
        title="Mentor Strict Bearish Directive",
        content="DO NOT BUY: Weekly key resistance hit, only trade SHORT",
        domain=SourceDomain.MENTORSHIP_SESSIONS,
        rule_type="DIRECTIVE",
        authority=SourceAuthority.APPROVED_MENTOR,
        confidence=0.99,
        symbols=["EURUSD"],
        direction="SELL"
    )
    repo.save_normalized_knowledge(rule_conflict)

    conflict_res = fusion.evaluate_confluence(
        symbol="EURUSD",
        timeframe="15m",
        direction="BUY",
        market_state={"sweep_detected": True}
    )
    assert conflict_res.state == TradeDecisionState.CONFLICTED.value
    assert len(conflict_res.conflicting_rules) > 0

def test_why_not_trade_engine():
    engine = WhyNotTradeEngine()

    # Clean pass
    pass_res = engine.evaluate_negative_conditions(
        symbol="EURUSD",
        direction="BUY",
        high_timeframe_bias="BULLISH",
        confluence_state=TradeDecisionState.HIGH_CONFLUENCE.value,
        has_invalidation=False,
        has_pending_high_impact_news=False,
        risk_limit_exceeded=False
    )
    assert isinstance(pass_res, WhyNotTradeResult)
    assert not pass_res.has_blocking_reasons
    assert len(pass_res.blocking_reasons) == 0

    # Blocking HTF bias contradiction
    block_res = engine.evaluate_negative_conditions(
        symbol="EURUSD",
        direction="BUY",
        high_timeframe_bias="BEARISH", # Contradiction!
        confluence_state=TradeDecisionState.HIGH_CONFLUENCE.value,
        has_invalidation=False
    )
    assert block_res.has_blocking_reasons
    assert any("bias" in r.lower() for r in block_res.blocking_reasons)

def test_core_trade_logic_generation():
    core_logic = CoreTradeLogic()
    normalizer = KnowledgeNormalizer()

    matched = [
        normalizer.normalize_rule(
            title="ICT Sweep",
            content="Sweep confirmed",
            domain=SourceDomain.YOUTUBE_LECTURES,
            rule_type="SETUP",
            authority=SourceAuthority.APPROVED_LECTURE,
            confidence=0.85
        )
    ]

    # Blocked state produces NO_SETUP / INVALIDATED
    blocked_why_not = WhyNotTradeResult(
        has_blocking_reasons=True,
        blocking_reasons=["High impact news in 5 minutes"]
    )
    decision_blocked = core_logic.generate_trade_decision(
        symbol="EURUSD",
        direction="BUY",
        current_price=1.0850,
        confluence_evaluation=ConfluenceEvaluation(
            symbol="EURUSD",
            timeframe="15m",
            state=TradeDecisionState.HIGH_CONFLUENCE.value,
            confluence_score=0.85,
            matched_rules=matched
        ),
        why_not_result=blocked_why_not
    )
    assert decision_blocked.decision_state == TradeDecisionState.NO_SETUP.value
    assert decision_blocked.entry_price is None

    # Valid pass produces exact entry, SL, TP, RR
    clean_why_not = WhyNotTradeResult(has_blocking_reasons=False)
    decision_valid = core_logic.generate_trade_decision(
        symbol="EURUSD",
        direction="BUY",
        current_price=1.0850,
        confluence_evaluation=ConfluenceEvaluation(
            symbol="EURUSD",
            timeframe="15m",
            state=TradeDecisionState.HIGH_CONFLUENCE.value,
            confluence_score=0.9,
            matched_rules=matched
        ),
        why_not_result=clean_why_not
    )
    assert decision_valid.decision_state in [TradeDecisionState.HIGH_CONFLUENCE.value, TradeDecisionState.VALID_SETUP.value]
    assert decision_valid.entry_price == 1.0850
    assert decision_valid.stop_loss < 1.0850
    assert decision_valid.take_profits[0] > 1.0850
    assert decision_valid.risk_reward_ratio >= 2.0

def test_journal_feedback_engine(repo):
    journal_engine = JournalFeedbackEngine(repo=repo)
    insight = journal_engine.get_feedback_for_setup(symbol="EURUSD", setup_type="SWEEP_ENTRY")
    assert hasattr(insight, "win_rate")
    assert hasattr(insight, "validated_lessons")
    assert hasattr(insight, "mistakes_to_avoid")

    # Absolute rule: journal feedback must never mutate live rules automatically
    assert not hasattr(journal_engine, "auto_mutate_rules")
