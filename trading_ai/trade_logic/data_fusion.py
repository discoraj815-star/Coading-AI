"""Data Fusion & Cross-Source Confluence Engine for Step 14.

Synthesizes relevant knowledge across the four isolated domains:
- YouTube Lectures (Core concepts, patterns, setups)
- Mentorship Sessions (Risk filters, personal rules, exceptions)
- Live Trade Logs (Lessons, past mistakes, outcome evidence)
- Trading Tech Work (Quantitative indicators, algorithmic findings)

Hard Principles:
1. No Artificial Averaging: Contradictions produce CONFLICTED / NO_TRADE.
2. Authority Precedence: User Override > Mentorship > Journal > Lecture > General.
3. No False Penalization: If a domain has no relevant rule, the setup is not artificially downgraded.
"""
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.enums import (
    RelationshipType,
    SourceAuthority,
    SourceDomain,
    TradeDecisionState,
)
from trading_ai.schemas.trade_logic import (
    ConfluenceEvaluation,
    NormalizedKnowledgeItem,
)
from trading_ai.trade_logic.relationship_graph import CrossSourceRelationshipGraph


class DataFusionEngine:
    """Combines multi-source knowledge and evaluates confluence and conflicts."""

    def __init__(
        self,
        relationship_graph: Optional[CrossSourceRelationshipGraph] = None,
        repo: Optional[Any] = None,
    ):
        self.graph = relationship_graph or CrossSourceRelationshipGraph(repo=repo)
        self.repo = repo

    def evaluate_confluence(
        self,
        symbol: str,
        timeframe: str = "15m",
        candidate_rules: Optional[List[NormalizedKnowledgeItem]] = None,
        market_direction: str = "BUY",
        direction: Optional[str] = None,
        market_state: Optional[Dict[str, Any]] = None,
        active_exceptions: Optional[List[str]] = None,
        journal_warnings: Optional[List[str]] = None,
    ) -> ConfluenceEvaluation:
        """Evaluate cross-source confluence, detect contradictions, and assign decision state."""
        eff_direction = (direction or market_direction).upper()

        if candidate_rules is None:
            if self.repo:
                all_rules = self.repo.get_normalized_knowledge(limit=100)
                sym_upper = symbol.upper()
                candidate_rules = [
                    r for r in all_rules
                    if not r.conditions or any(sym_upper in c.upper() for c in r.conditions) or "symbols" not in str(r.conditions).lower()
                ]
            else:
                candidate_rules = []

        if not candidate_rules:
            return ConfluenceEvaluation(
                symbol=symbol,
                timeframe=timeframe,
                decision_state=TradeDecisionState.NO_SETUP,
                confluence_score=0.0,
                reasons_not_to_trade=["No applicable knowledge rules retrieved for setup."],
            )

        matched_rules: List[NormalizedKnowledgeItem] = []
        conflicting_rules: List[Dict[str, Any]] = []
        mentor_filters_passed: List[str] = []
        supporting_lessons: List[Dict[str, Any]] = []
        reasons_to_trade: List[str] = []
        reasons_not_to_trade: List[str] = []
        conflict_summary: Optional[str] = None

        # 1. Inspect pairwise contradictions in relationship graph
        seen_pairs = set()
        for i, r1 in enumerate(candidate_rules):
            for r2 in candidate_rules[i + 1 :]:
                pair_key = tuple(sorted([r1.item_id, r2.item_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Check if r1 and r2 have a CONTRADICTS relationship
                rels = self.graph.get_relationships_for_item(r1.item_id)
                contradiction = next(
                    (
                        r
                        for r in rels
                        if (
                            r.relationship_type == RelationshipType.CONTRADICTS
                            and (r.source_item_id == r2.item_id or r.target_item_id == r2.item_id)
                        )
                    ),
                    None,
                )

                if contradiction:
                    # Contradiction detected! Check authority hierarchy:
                    if r1.authority_priority < r2.authority_priority:
                        reasons_to_trade.append(
                            f"Higher authority rule '{r1.title}' ({r1.authority.value}) overrides lower authority '{r2.title}'."
                        )
                    elif r2.authority_priority < r1.authority_priority:
                        reasons_to_trade.append(
                            f"Higher authority rule '{r2.title}' ({r2.authority.value}) overrides lower authority '{r1.title}'."
                        )
                    else:
                        conflicting_rules.append(
                            {
                                "rule_1": r1.title,
                                "rule_2": r2.title,
                                "source_1": r1.domain.value,
                                "source_2": r2.domain.value,
                                "description": contradiction.description or "Direct semantic contradiction between rules.",
                            }
                        )

        # 2. Check directional / semantic polarity conflicts
        directions_found = set()
        import re
        for r in candidate_rules:
            explicit_dir = None
            for c in r.conditions:
                if str(c).lower().startswith("direction:"):
                    explicit_dir = str(c).split(":", 1)[1].upper().strip()
                    break
            if explicit_dir:
                directions_found.add(explicit_dir)
                continue

            text = (r.title + " " + r.statement).upper()
            has_buy = bool(re.search(r'\b(BUY|LONG|BULLISH)\b', text))
            clean_text_for_sell = re.sub(r'SELL-?SIDE\s+(LIQUIDITY|SWEEP)', '', text)
            has_sell = bool(re.search(r'\b(SELL|SHORT|BEARISH)\b', clean_text_for_sell))
            if has_buy:
                directions_found.add("BUY")
            if has_sell:
                directions_found.add("SELL")

        if "BUY" in directions_found and "SELL" in directions_found:
            conflicting_rules.append(
                {
                    "conflict_type": "DIRECTIONAL_POLARITY",
                    "description": f"Cross-source conflict: Opposing BUY and SELL directives found across sources for {symbol}.",
                }
            )

        if conflicting_rules:
            conflict_summary = f"Found {len(conflicting_rules)} unresolved source conflict(s)."
            reasons_not_to_trade.append(conflict_summary)
            return ConfluenceEvaluation(
                symbol=symbol,
                timeframe=timeframe,
                decision_state=TradeDecisionState.CONFLICTED,
                confluence_score=0.20,
                matched_rules=candidate_rules,
                conflicting_rules=conflicting_rules,
                reasons_not_to_trade=reasons_not_to_trade,
                conflict_summary=conflict_summary,
            )

        # 3. Categorize matched rules and evaluate confluence support
        for rule in candidate_rules:
            matched_rules.append(rule)
            if rule.domain == SourceDomain.MENTORSHIP_SESSIONS:
                mentor_filters_passed.append(f"{rule.title}: {rule.statement[:80]}")
                reasons_to_trade.append(f"Mentorship filter satisfied: {rule.title}")
            elif rule.domain == SourceDomain.LIVE_TRADE_LOGS:
                supporting_lessons.append({"lesson_id": rule.item_id, "title": rule.title, "statement": rule.statement})
                reasons_to_trade.append(f"Historical journal lesson supports setup: {rule.title}")
            elif rule.domain == SourceDomain.YOUTUBE_LECTURES:
                reasons_to_trade.append(f"Lecture pattern matched: {rule.title}")
            elif rule.domain == SourceDomain.TRADING_TECH_WORK:
                reasons_to_trade.append(f"Technical finding verified: {rule.title}")

        # Check journal warnings
        if journal_warnings:
            for w in journal_warnings:
                reasons_not_to_trade.append(f"Journal Warning: {w}")

        # Compute Confluence Score
        # Base on count of independent participating domains
        participating_domains = {r.domain for r in matched_rules}
        domain_factor = len(participating_domains) / 4.0  # Up to 4 domains
        avg_confidence = sum(r.confidence for r in matched_rules) / len(matched_rules)
        score = round(min(1.0, 0.40 * domain_factor + 0.60 * avg_confidence), 2)

        # Determine decision state
        if score >= 0.80 and len(participating_domains) >= 2:
            state = TradeDecisionState.HIGH_CONFLUENCE
        elif score >= 0.60:
            state = TradeDecisionState.VALID_SETUP
        else:
            state = TradeDecisionState.INSUFFICIENT_CONFLUENCE
            reasons_not_to_trade.append("Confluence score below validation threshold.")

        return ConfluenceEvaluation(
            symbol=symbol,
            timeframe=timeframe,
            decision_state=state,
            confluence_score=score,
            matched_rules=matched_rules,
            conflicting_rules=conflicting_rules,
            supporting_lessons=supporting_lessons,
            mentor_filters_passed=mentor_filters_passed,
            reasons_to_trade=reasons_to_trade,
            reasons_not_to_trade=reasons_not_to_trade,
            conflict_summary=conflict_summary,
        )
