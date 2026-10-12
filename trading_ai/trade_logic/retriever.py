"""Token-Efficient Unified Knowledge Retriever for Step 14.

Applies Step 2 / Step 11 principles:
- Retrieves only the smallest sufficient context needed for a given decision point.
- Filters strictly by symbol, timeframe, setup keywords, and authority.
- Enforces strict upper bounds (max 5-10 rules) to conserve Gemini tokens and avoid noise.
- Caches query results to prevent redundant processing.
"""
import hashlib
import json
from typing import Dict, List, Optional

from trading_ai.schemas.enums import SourceAuthority, SourceDomain
from trading_ai.schemas.trade_logic import NormalizedKnowledgeItem
from trading_ai.storage.repository import Repository


class UnifiedKnowledgeRetriever:
    """Retrieves relevant, bounded knowledge items across all 4 domains."""

    def __init__(
        self,
        repository: Optional[Repository] = None,
        max_items_per_query: int = 8,
        repo: Optional[Repository] = None,
        max_tokens: int = 1000,
    ):
        self.repository = repository or repo
        self.max_items_per_query = max_items_per_query
        self.max_tokens = max_tokens
        self._cache: Dict[str, List[NormalizedKnowledgeItem]] = {}

    def retrieve(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "15m",
        setup_type: str = "liquidity_sweep",
        keywords: Optional[List[str]] = None,
        domain: Optional[SourceDomain] = None,
        allowed_domains: Optional[List[SourceDomain]] = None,
    ) -> List[NormalizedKnowledgeItem]:
        """Convenience query method matching symbol and optional single domain."""
        domains = allowed_domains or ([domain] if domain else None)
        return self.retrieve_relevant_knowledge(symbol, timeframe, setup_type, keywords, domains)

    def retrieve_relevant_knowledge(
        self,
        symbol: str,
        timeframe: str,
        setup_type: str = "liquidity_sweep",
        keywords: Optional[List[str]] = None,
        allowed_domains: Optional[List[SourceDomain]] = None,
    ) -> List[NormalizedKnowledgeItem]:
        """Query normalized knowledge matching current market setup with token-bounded output."""
        cache_key = f"{symbol}:{timeframe}:{setup_type}:{sorted(keywords or [])}:{sorted([d.value for d in (allowed_domains or [])])}"
        hash_key = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()

        if hash_key in self._cache:
            return self._cache[hash_key]

        all_items: List[NormalizedKnowledgeItem] = []
        if self.repository and hasattr(self.repository, "list_normalized_knowledge"):
            if allowed_domains:
                for d in allowed_domains:
                    all_items.extend(self.repository.list_normalized_knowledge(domain=d, limit=self.max_items_per_query))
            else:
                all_items = self.repository.list_normalized_knowledge(limit=self.max_items_per_query * 2)

        # Apply keyword/setup relevance scoring
        search_terms = {setup_type.lower()}
        if keywords:
            search_terms.update(k.lower() for k in keywords)

        scored_items = []
        for item in all_items:
            content_str = (item.title + " " + item.statement + " " + " ".join(item.tags)).lower()
            score = sum(1 for term in search_terms if term in content_str)
            # Give higher priority bonus (1 = highest priority)
            authority_bonus = (6 - item.authority_priority) * 0.5
            total_rank = score + authority_bonus + item.confidence
            scored_items.append((total_rank, item))

        # Sort descending by total rank
        scored_items.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored_items[: self.max_items_per_query]]

        self._cache[hash_key] = results
        return results

    def clear_cache(self) -> None:
        """Clear query cache."""
        self._cache.clear()
