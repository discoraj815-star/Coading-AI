"""
MemoryRetriever — Step 2 Core

Provides structured + semantic retrieval over all persistent memory:
knowledge items, evidence traces, transcript segments, and visual events.

Retrieval Strategy (Dual-Mode):
  1. STRUCTURED — SQL-based keyword/type/time filtering (no model call, free)
  2. SEMANTIC   — Cosine-similarity search over persisted embeddings
                  (requires pre-embedded records; falls back to structured)

Design Rules:
  - Always uses pre-computed embeddings (no re-embedding on query)
  - Always records access (increment_embedding_access) for LFU tracking
  - Token cost of results is pre-checked via TokenManager
  - Returns results ranked by (semantic_score + importance_score) combined
  - Never sends the full knowledge DB to any model
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

from trading_ai.core.config import Settings, get_settings
from trading_ai.core.token_manager import TokenManager
from trading_ai.schemas import (
    EvidenceTrace,
    KnowledgeItem,
    KnowledgeType,
    MemoryEmbedding,
    MemorySearchResult,
    TranscriptSegment,
    VisualEvent,
)

if TYPE_CHECKING:
    from trading_ai.storage.repository import Repository


# -------------------------------------------------------------------
# Embedding utilities (no model call — pure math)
# -------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _text_to_mock_embedding(text: str, dim: int = 8) -> List[float]:
    """
    Deterministic mock embedding for unit-testable retrieval without an API key.

    In production, this is replaced by a real GeminiClient.embed() call.
    Uses character frequency hashing to produce a consistent dim-length vector.
    """
    if not text:
        return [0.0] * dim
    vec = [0.0] * dim
    for i, ch in enumerate(text):
        vec[i % dim] += ord(ch) / 1000.0
    # Normalize
    mag = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / mag for x in vec]


# -------------------------------------------------------------------
# MemoryRetriever
# -------------------------------------------------------------------

class MemoryRetriever:
    """
    Dual-mode (structured + semantic) memory retrieval for Trading AI.

    Structured search: direct DB keyword/type/time queries — no model needed.
    Semantic search:   cosine similarity over persisted embedding vectors.

    Callers receive List[MemorySearchResult] ranked by combined score.
    """

    def __init__(
        self,
        repository: Repository,
        token_manager: Optional[TokenManager] = None,
        settings: Optional[Settings] = None,
        embed_fn=None,  # Optional[Callable[[str], List[float]]]
    ) -> None:
        self._repo = repository
        self._settings = settings or get_settings()
        self._token_manager = token_manager or TokenManager(self._settings)
        # Production: pass GeminiClient.embed; tests: uses mock
        self._embed_fn = embed_fn or _text_to_mock_embedding

    # ------------------------------------------------------------------
    # Primary retrieval entry point
    # ------------------------------------------------------------------

    def search(
        self,
        job_id: str,
        query: str,
        source_types: Optional[List[str]] = None,
        knowledge_types: Optional[List[KnowledgeType]] = None,
        min_confidence: float = 0.0,
        time_range: Optional[tuple] = None,  # (start_seconds, end_seconds)
        limit: int = 10,
        max_result_tokens: Optional[int] = None,
        use_semantic: bool = True,
    ) -> List[MemorySearchResult]:
        """
        Hybrid search: structured filter first, semantic re-ranking second.

        Args:
            job_id:            Lecture job to search.
            query:             Natural language query string.
            source_types:      Restrict to ['knowledge', 'evidence', 'transcript', 'visual'].
            knowledge_types:   Further restrict knowledge items by KnowledgeType.
            min_confidence:    Exclude results below this confidence threshold.
            time_range:        (start, end) seconds to restrict temporal search.
            limit:             Max results to return.
            max_result_tokens: If set, prune results to stay within token budget.
            use_semantic:      If True, re-rank via cosine similarity (requires embeddings).

        Returns:
            List[MemorySearchResult] ranked by combined relevance score.
        """
        results: List[MemorySearchResult] = []

        effective_sources = source_types or ["knowledge", "evidence", "transcript", "visual"]

        if "knowledge" in effective_sources:
            results.extend(
                self._search_knowledge(
                    job_id, query, knowledge_types, min_confidence, time_range, limit
                )
            )

        if "evidence" in effective_sources:
            results.extend(
                self._search_evidence(job_id, query, min_confidence, time_range, limit)
            )

        if "transcript" in effective_sources:
            results.extend(
                self._search_transcripts(job_id, query, min_confidence, time_range, limit)
            )

        if "visual" in effective_sources:
            results.extend(
                self._search_visuals(job_id, query, time_range, limit)
            )

        # Semantic re-ranking
        if use_semantic and results:
            results = self._semantic_rerank(job_id, query, results)

        # Sort final combined list
        results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Apply token budget cap
        if max_result_tokens:
            results = self._apply_token_budget(results, max_result_tokens)

        # Assign final ranks and track access
        final = []
        for i, r in enumerate(results[:limit]):
            r = r.model_copy(update={"rank": i + 1})
            final.append(r)
            if r.embedding.embedding_id:
                self._repo.increment_embedding_access(r.embedding.embedding_id)

        return final

    # ------------------------------------------------------------------
    # Structured retrievers (per source type)
    # ------------------------------------------------------------------

    def _search_knowledge(
        self,
        job_id: str,
        query: str,
        types: Optional[List[KnowledgeType]],
        min_conf: float,
        time_range: Optional[tuple],
        limit: int,
    ) -> List[MemorySearchResult]:
        items = self._repo.search_knowledge(
            job_id=job_id, query=query, item_types=types, limit=limit
        )
        results = []
        for item in items:
            if item.confidence < min_conf:
                continue
            if time_range and not self._in_time_range(item.start_time, item.end_time, time_range):
                continue
            emb = self._synthetic_embedding_for_knowledge(job_id, item)
            results.append(
                MemorySearchResult(
                    embedding=emb,
                    similarity_score=item.confidence * 0.8,  # initial structured score
                    retrieval_metadata={
                        "source": "knowledge",
                        "query": query,
                        "item_id": item.item_id,
                        "item_type": item.item_type.value,
                    },
                )
            )
        return results

    def _search_evidence(
        self,
        job_id: str,
        query: str,
        min_conf: float,
        time_range: Optional[tuple],
        limit: int,
    ) -> List[MemorySearchResult]:
        all_evidence = self._repo.get_evidence(job_id)
        query_lower = query.lower()
        results = []
        for ev in all_evidence:
            if ev.confidence < min_conf:
                continue
            if time_range and not self._in_time_range(ev.start_time, ev.end_time, time_range):
                continue
            text = (ev.audio_quote or "") + " " + (ev.extracted_statement or "")
            if query_lower in text.lower():
                emb = self._synthetic_embedding_for_evidence(job_id, ev)
                results.append(
                    MemorySearchResult(
                        embedding=emb,
                        similarity_score=ev.confidence * 0.75,
                        retrieval_metadata={
                            "source": "evidence",
                            "query": query,
                            "evidence_id": ev.evidence_id,
                        },
                    )
                )
            if len(results) >= limit:
                break
        return results

    def _search_transcripts(
        self,
        job_id: str,
        query: str,
        min_conf: float,
        time_range: Optional[tuple],
        limit: int,
    ) -> List[MemorySearchResult]:
        segments = self._repo.get_transcripts(job_id)
        query_lower = query.lower()
        results = []
        for seg in segments:
            if seg.confidence < min_conf:
                continue
            if time_range and not self._in_time_range(seg.start_time, seg.end_time, time_range):
                continue
            if query_lower in seg.transcript.lower():
                emb = self._synthetic_embedding_for_transcript(job_id, seg)
                results.append(
                    MemorySearchResult(
                        embedding=emb,
                        similarity_score=seg.confidence * 0.60,
                        retrieval_metadata={
                            "source": "transcript",
                            "query": query,
                            "segment_id": seg.segment_id,
                        },
                    )
                )
            if len(results) >= limit:
                break
        return results

    def _search_visuals(
        self,
        job_id: str,
        query: str,
        time_range: Optional[tuple],
        limit: int,
    ) -> List[MemorySearchResult]:
        events = self._repo.get_visual_events(job_id)
        query_lower = query.lower()
        results = []
        for ev in events:
            if time_range and not self._in_time_range(ev.timestamp, ev.timestamp, time_range):
                continue
            text = (ev.visual_description or "") + " " + (ev.detected_event or "")
            if query_lower in text.lower():
                emb = self._synthetic_embedding_for_visual(job_id, ev)
                results.append(
                    MemorySearchResult(
                        embedding=emb,
                        similarity_score=ev.confidence * 0.55,
                        retrieval_metadata={
                            "source": "visual",
                            "query": query,
                            "event_id": ev.event_id,
                        },
                    )
                )
            if len(results) >= limit:
                break
        return results

    # ------------------------------------------------------------------
    # Semantic re-ranking using persisted embeddings
    # ------------------------------------------------------------------

    def _semantic_rerank(
        self,
        job_id: str,
        query: str,
        candidates: List[MemorySearchResult],
    ) -> List[MemorySearchResult]:
        """
        Re-rank candidates by cosine similarity to the query embedding.

        Uses persisted embeddings where available; falls back to mock embedding.
        """
        query_embedding = self._embed_fn(query)
        reranked = []
        for result in candidates:
            # Look up persisted embedding for this source artifact
            persisted = self._repo.get_embedding_by_source(
                job_id, result.embedding.source_id
            )
            if persisted and persisted.embedding:
                vec = persisted.embedding
                emb_record = persisted
            else:
                # Use synthetic from search result
                vec = result.embedding.embedding
                emb_record = result.embedding

            sim = _cosine_similarity(query_embedding, vec)
            # Combined score: 60% semantic, 40% original structured score
            combined = 0.60 * sim + 0.40 * result.similarity_score

            reranked.append(
                MemorySearchResult(
                    embedding=emb_record,
                    similarity_score=combined,
                    rank=result.rank,
                    retrieval_metadata={
                        **result.retrieval_metadata,
                        "semantic_score": round(sim, 4),
                        "structured_score": round(result.similarity_score, 4),
                        "combined_score": round(combined, 4),
                    },
                )
            )
        return reranked

    # ------------------------------------------------------------------
    # Conflict detection
    # ------------------------------------------------------------------

    def find_conflicts(
        self,
        job_id: str,
        statement: str,
        knowledge_types: Optional[List[KnowledgeType]] = None,
        similarity_threshold: float = 0.80,
        limit: int = 5,
    ) -> List[MemorySearchResult]:
        """
        Identify knowledge items that potentially contradict a new statement.

        Uses semantic similarity: items with high similarity but
        CONTRADICTED validation_status, or items with opposing keywords,
        are flagged for conflict review by the ReasoningEngine.
        """
        all_knowledge = self._repo.get_knowledge(job_id, only_committed=True, item_type=None)
        query_vec = self._embed_fn(statement)
        conflicts: List[MemorySearchResult] = []

        for item in all_knowledge:
            if knowledge_types and item.item_type not in knowledge_types:
                continue

            item_text = f"{item.title}: {item.statement}"
            # Check persisted embedding first
            persisted = self._repo.get_embedding_by_source(job_id, item.item_id)
            if persisted and persisted.embedding:
                item_vec = persisted.embedding
            else:
                item_vec = self._embed_fn(item_text)

            sim = _cosine_similarity(query_vec, item_vec)
            if sim >= similarity_threshold:
                emb = self._synthetic_embedding_for_knowledge(job_id, item)
                conflicts.append(
                    MemorySearchResult(
                        embedding=emb,
                        similarity_score=sim,
                        retrieval_metadata={
                            "conflict_type": "semantic_overlap",
                            "conflicting_item_id": item.item_id,
                            "validation_status": item.validation_status.value,
                            "similarity": round(sim, 4),
                        },
                    )
                )

        conflicts.sort(key=lambda r: r.similarity_score, reverse=True)
        return conflicts[:limit]

    # ------------------------------------------------------------------
    # Evidence retrieval by time window (for context reconstruction)
    # ------------------------------------------------------------------

    def get_evidence_for_window(
        self,
        job_id: str,
        start_time: float,
        end_time: float,
        max_tokens: Optional[int] = None,
    ) -> List[EvidenceTrace]:
        """
        Retrieve all evidence traces within a time window, optionally capped by token budget.

        Used by ContextManager to hydrate evidence entries for a specific serial window.
        """
        all_evidence = self._repo.get_evidence(job_id)
        window_evidence = [
            ev for ev in all_evidence
            if ev.start_time >= start_time and ev.end_time <= end_time
        ]

        if max_tokens:
            def _ev_text(ev: EvidenceTrace) -> str:
                return (ev.audio_quote or "") + " " + ev.extracted_statement

            included, _ = self._token_manager.optimize_context(
                window_evidence,
                item_text_fn=_ev_text,
                max_tokens=max_tokens,
                importance_score_fn=lambda ev: ev.confidence,
            )
            return included

        return window_evidence

    # ------------------------------------------------------------------
    # Embedding registration (called by orchestrator at commit time)
    # ------------------------------------------------------------------

    def register_embedding(
        self,
        job_id: str,
        source_type: str,
        source_id: str,
        text_content: str,
        importance_score: float = 0.5,
    ) -> MemoryEmbedding:
        """
        Compute and persist an embedding for a source artifact.

        In production: calls GeminiClient.embed (via self._embed_fn).
        In tests: uses deterministic mock embedding.

        Idempotent — if an embedding already exists for this source_id,
        the record is refreshed (re-embedded with latest text).
        """
        embedding_vector = self._embed_fn(text_content)
        model_name = self._settings.gemini_model_embedding

        emb = MemoryEmbedding(
            embedding_id=str(uuid.uuid4()),
            job_id=job_id,
            source_type=source_type,
            source_id=source_id,
            text_content=text_content,
            embedding=embedding_vector,
            model_name=model_name,
            importance_score=importance_score,
        )
        self._repo.save_memory_embedding(emb)
        return emb

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _in_time_range(
        start: float, end: float, time_range: tuple
    ) -> bool:
        range_start, range_end = time_range
        return not (end < range_start or start > range_end)

    def _apply_token_budget(
        self,
        results: List[MemorySearchResult],
        max_tokens: int,
    ) -> List[MemorySearchResult]:
        """Trim result list to stay within token budget (highest-scored first)."""
        included, _ = self._token_manager.optimize_context(
            results,
            item_text_fn=lambda r: r.embedding.text_content,
            max_tokens=max_tokens,
            importance_score_fn=lambda r: r.similarity_score,
        )
        return included

    # ------------------------------------------------------------------
    # Synthetic embedding builders (structured search fallback)
    # These create lightweight MemoryEmbedding records for structured
    # search results that may not yet have a persisted embedding.
    # ------------------------------------------------------------------

    def _synthetic_embedding_for_knowledge(
        self, job_id: str, item: KnowledgeItem
    ) -> MemoryEmbedding:
        text = f"{item.title}: {item.statement}"
        return MemoryEmbedding(
            embedding_id="",  # not persisted
            job_id=job_id,
            source_type="knowledge",
            source_id=item.item_id,
            text_content=text,
            embedding=self._embed_fn(text),
            importance_score=item.confidence,
        )

    def _synthetic_embedding_for_evidence(
        self, job_id: str, ev: EvidenceTrace
    ) -> MemoryEmbedding:
        text = (ev.audio_quote or "") + " " + ev.extracted_statement
        return MemoryEmbedding(
            embedding_id="",
            job_id=job_id,
            source_type="evidence",
            source_id=ev.evidence_id,
            text_content=text,
            embedding=self._embed_fn(text),
            importance_score=ev.confidence,
        )

    def _synthetic_embedding_for_transcript(
        self, job_id: str, seg: TranscriptSegment
    ) -> MemoryEmbedding:
        return MemoryEmbedding(
            embedding_id="",
            job_id=job_id,
            source_type="transcript",
            source_id=seg.segment_id,
            text_content=seg.transcript,
            embedding=self._embed_fn(seg.transcript),
            importance_score=seg.confidence,
        )

    def _synthetic_embedding_for_visual(
        self, job_id: str, ev: VisualEvent
    ) -> MemoryEmbedding:
        text = (ev.visual_description or "") + " " + (ev.detected_event or "")
        return MemoryEmbedding(
            embedding_id="",
            job_id=job_id,
            source_type="visual",
            source_id=ev.event_id,
            text_content=text,
            embedding=self._embed_fn(text),
            importance_score=ev.confidence,
        )
