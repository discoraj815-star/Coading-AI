"""Step 14 Unified Trade Logic Core Facade.

Coordinates:
- Source Domain physical isolation
- Cross-source knowledge normalization & relationship graph
- Token-efficient retrieval across isolated domains
- Data Fusion & Confluence evaluation
- 'Why Not Trade?' negative validation
- Core Trade Logic & TradeProposal generation
- Journal feedback loop
"""
from typing import Any, Dict, List, Optional

from trading_ai.schemas.trade_logic import (
    ConfluenceEvaluation,
    CoreTradeDecision,
    JournalFeedbackInsight,
    NormalizedKnowledgeItem,
)
from trading_ai.storage.repository import Repository
from trading_ai.trade_logic.core_trade_logic import CoreTradeLogic
from trading_ai.trade_logic.data_fusion import DataFusionEngine
from trading_ai.trade_logic.domain_manager import SourceDomainManager
from trading_ai.trade_logic.journal_feedback import JournalFeedbackEngine
from trading_ai.trade_logic.knowledge_normalizer import KnowledgeNormalizer
from trading_ai.trade_logic.relationship_graph import CrossSourceRelationshipGraph
from trading_ai.trade_logic.retriever import UnifiedKnowledgeRetriever
from trading_ai.trade_logic.why_not_trade import WhyNotTradeEngine


class TradeLogicCore:
    """Unified facade coordinating Step 14 Separate File Architecture & Core Trade Logic."""

    def __init__(self, repository: Optional[Repository] = None, base_domains_dir: Optional[str] = None):
        self.repository = repository
        self.domain_manager = SourceDomainManager(base_dir=base_domains_dir)
        self.normalizer = KnowledgeNormalizer()
        self.relationship_graph = CrossSourceRelationshipGraph(repository=repository)
        self.retriever = UnifiedKnowledgeRetriever(repository=repository)
        self.fusion_engine = DataFusionEngine(relationship_graph=self.relationship_graph)
        self.why_not_engine = WhyNotTradeEngine()
        self.core_logic = CoreTradeLogic(
            data_fusion=self.fusion_engine,
            why_not_trade=self.why_not_engine,
        )
        self.journal_feedback = JournalFeedbackEngine(repository=repository)

    def analyze_and_propose_trade(
        self,
        symbol: str,
        timeframe: str,
        direction: str,
        current_price: float,
        structural_sl: float,
        target_tp1: float,
        target_tp2: Optional[float] = None,
        setup_type: str = "liquidity_sweep",
        htf_bias: str = "BULLISH",
        is_poi_valid: bool = True,
        is_confirmation_present: bool = True,
        news_risk_level: str = "NO_RISK",
        active_exceptions: Optional[List[str]] = None,
        daily_loss_limit_hit: bool = False,
    ) -> CoreTradeDecision:
        """Execute full end-to-end multi-source analysis and synthesize CoreTradeDecision."""
        # 1. Retrieve bounded relevant knowledge across domains
        candidate_rules = self.retriever.retrieve_relevant_knowledge(
            symbol=symbol,
            timeframe=timeframe,
            setup_type=setup_type,
            keywords=[setup_type, direction],
        )

        # 2. Query journal feedback for warnings
        feedback = self.journal_feedback.get_feedback_for_setup(
            symbol=symbol,
            setup_type=setup_type,
            direction=direction,
        )
        journal_warnings = feedback.warning_flags

        # 3. Evaluate through Core Trade Logic
        decision = self.core_logic.evaluate_and_propose(
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            current_price=current_price,
            structural_sl=structural_sl,
            target_tp1=target_tp1,
            target_tp2=target_tp2,
            candidate_rules=candidate_rules,
            htf_bias=htf_bias,
            is_poi_valid=is_poi_valid,
            is_confirmation_present=is_confirmation_present,
            news_risk_level=news_risk_level,
            active_exceptions=active_exceptions,
            journal_warnings=journal_warnings,
            daily_loss_limit_hit=daily_loss_limit_hit,
        )

        # 4. Save confluence record if repository available
        if self.repository and hasattr(self.repository, "save_cross_source_confluence"):
            confluence_eval = self.fusion_engine.evaluate_confluence(
                symbol=symbol,
                timeframe=timeframe,
                candidate_rules=candidate_rules,
                market_direction=direction,
            )
            proposal_id = decision.trade_proposal.proposal_id if decision.trade_proposal else None
            self.repository.save_cross_source_confluence(confluence_eval, proposal_id=proposal_id)

        return decision
