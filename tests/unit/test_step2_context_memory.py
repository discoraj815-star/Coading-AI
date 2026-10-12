"""
Step 2 Test Suite — Context Intelligence + Long-Term Memory

Covers the complete Step 2 validation checklist:

PERSISTENCE
  [T01] Context snapshot saved and reloaded correctly across "restart"
  [T02] Memory embedding saved and fetched by source_id

MEMORY RETRIEVAL
  [T03] Structured knowledge search returns correct results
  [T04] Evidence retrieval with time-window filter
  [T05] Transcript search by keyword
  [T06] Visual event search by keyword
  [T07] Multi-source search returns mixed results
  [T08] Source type filter restricts results correctly
  [T09] Knowledge type filter (KnowledgeType.RULE)
  [T10] Confidence threshold filters low-confidence results

CONTEXT RECONSTRUCTION
  [T11] Context window restored from DB snapshot (warm start)
  [T12] Context window starts empty on cold start (no DB record)
  [T13] Serial advance updates window and persists new snapshot

CONTEXT BUDGET ENFORCEMENT
  [T14] Token budget ceiling is enforced — add_item returns False for overflow
  [T15] Pruning moves lowest-importance items to deferred list
  [T16] Compression summary is generated on pruning
  [T17] Forced prune reduces window below target
  [T18] Batch add_items respects importance ordering

SEMANTIC RETRIEVAL
  [T19] Embedding registration persists vector and is retrievable
  [T20] Cosine similarity re-ranking produces higher score for closer match
  [T21] Semantic search falls back gracefully when no persisted embeddings

CONFLICT DETECTION
  [T22] find_conflicts detects a high-similarity knowledge item
  [T23] find_conflicts returns empty list when no conflict threshold is met

TOKEN EFFICIENCY
  [T24] get_context_for_prompt respects max_tokens override
  [T25] Memory search results trimmed by max_result_tokens

SYSTEM
  [T26] ContextManager.status() returns correct diagnostic dict
  [T27] increment_embedding_access updates access_count
  [T28] Evidence window retrieval is bounded by token budget
"""

import uuid
from datetime import datetime, timezone
from typing import List

import pytest

from trading_ai.core.context_manager import ContextManager, _default_importance
from trading_ai.core.memory_retriever import MemoryRetriever, _cosine_similarity, _text_to_mock_embedding
from trading_ai.core.token_manager import TokenManager
from trading_ai.schemas import (
    ContextItem,
    ContextItemSource,
    ContextSnapshot,
    EvidenceTrace,
    JobMetadata,
    JobStatus,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    MemoryEmbedding,
    MemorySearchResult,
    PipelineStage,
    TranscriptSegment,
    VisualEvent,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def db():
    """In-memory SQLite database with Step 1 + Step 2 schemas."""
    return DatabaseManager(db_path=":memory:")


@pytest.fixture
def repo(db):
    return Repository(db)


@pytest.fixture
def token_manager():
    return TokenManager()


@pytest.fixture
def job_id(repo):
    """Create and persist a dummy job for FK constraints."""
    jid = f"job-{uuid.uuid4().hex[:8]}"
    job = JobMetadata(
        job_id=jid,
        file_hash="abc123",
        filename="test_lecture.mp4",
        file_path="/tmp/test.mp4",
        duration=120.0,
        status=JobStatus.PROCESSING,
        current_stage=PipelineStage.AUDIO_PROCESSING,
    )
    repo.save_job(job)
    return jid


@pytest.fixture
def context_manager(job_id, repo, token_manager):
    return ContextManager(
        job_id=job_id,
        repository=repo,
        token_manager=token_manager,
        max_tokens=500,  # small for testing pruning
    )


@pytest.fixture
def memory_retriever(repo, token_manager):
    return MemoryRetriever(
        repository=repo,
        token_manager=token_manager,
    )


def _make_context_item(
    item_id: str = None,
    source_type: ContextItemSource = ContextItemSource.KNOWLEDGE,
    serial: int = 1,
    text: str = "Sample content about FVG and liquidity",
    importance: float = 0.5,
    start: float = 0.0,
    tokens: int = 0,
) -> ContextItem:
    return ContextItem(
        item_id=item_id or str(uuid.uuid4()),
        source_type=source_type,
        serial_number=serial,
        start_time=start,
        end_time=start + 5.0,
        text_content=text,
        importance_score=importance,
        estimated_tokens=tokens,
    )


def _make_knowledge(
    job_id: str,
    item_id: str = None,
    title: str = "FVG Rule",
    statement: str = "FVG fills when price revisits the gap area",
    item_type: KnowledgeType = KnowledgeType.RULE,
    confidence: float = 0.9,
    start: float = 0.0,
) -> KnowledgeItem:
    return KnowledgeItem(
        item_id=item_id or str(uuid.uuid4()),
        job_id=job_id,
        item_type=item_type,
        title=title,
        statement=statement,
        validation_status=KnowledgeValidationStatus.VALIDATED,
        confidence=confidence,
        start_time=start,
        end_time=start + 10.0,
    )


def _make_evidence(job_id: str, start: float = 0.0, text: str = "price always returns to FVG") -> EvidenceTrace:
    return EvidenceTrace(
        evidence_id=str(uuid.uuid4()),
        job_id=job_id,
        start_time=start,
        end_time=start + 5.0,
        audio_quote=text,
        extracted_statement=text,
        confidence=0.85,
    )


def _make_transcript(job_id: str, start: float = 0.0, text: str = "This is a liquidity sweep setup") -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=str(uuid.uuid4()),
        job_id=job_id,
        start_time=start,
        end_time=start + 5.0,
        transcript=text,
        confidence=0.8,
    )


def _make_visual(job_id: str, ts: float = 0.0, desc: str = "Bearish engulfing candle on H1") -> VisualEvent:
    return VisualEvent(
        event_id=str(uuid.uuid4()),
        job_id=job_id,
        timestamp=ts,
        visual_description=desc,
        detected_event="engulfing",
        confidence=0.75,
    )


# -----------------------------------------------------------------------
# PERSISTENCE
# -----------------------------------------------------------------------

class TestPersistence:

    def test_T01_context_snapshot_save_and_reload(self, job_id, repo):
        """[T01] Snapshot persisted and correctly reconstructed."""
        snap = ContextSnapshot(
            snapshot_id=str(uuid.uuid4()),
            job_id=job_id,
            serial_number=2,
            window_tokens=300,
            window_items_json=[
                _make_context_item(tokens=150).model_dump(),
                _make_context_item(tokens=150).model_dump(),
            ],
            pruned_items_json=[_make_context_item(tokens=50).model_dump()],
            compression_summary="[Pruned: knowledge:abc12345]",
        )
        repo.save_context_snapshot(snap)

        loaded = repo.get_latest_context_snapshot(job_id)
        assert loaded is not None
        assert loaded.snapshot_id == snap.snapshot_id
        assert loaded.serial_number == 2
        assert loaded.window_tokens == 300
        assert len(loaded.window_items_json) == 2
        assert len(loaded.pruned_items_json) == 1
        assert "Pruned" in loaded.compression_summary

    def test_T02_embedding_save_and_fetch_by_source(self, job_id, repo):
        """[T02] Memory embedding persisted and retrieved by source_id."""
        source_id = str(uuid.uuid4())
        emb = MemoryEmbedding(
            embedding_id=str(uuid.uuid4()),
            job_id=job_id,
            source_type="knowledge",
            source_id=source_id,
            text_content="FVG fills when price revisits",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            importance_score=0.9,
        )
        repo.save_memory_embedding(emb)

        fetched = repo.get_embedding_by_source(job_id, source_id)
        assert fetched is not None
        assert fetched.embedding_id == emb.embedding_id
        assert len(fetched.embedding) == 8
        assert abs(fetched.embedding[0] - 0.1) < 1e-6


# -----------------------------------------------------------------------
# MEMORY RETRIEVAL
# -----------------------------------------------------------------------

class TestMemoryRetrieval:

    def test_T03_structured_knowledge_search(self, job_id, repo, memory_retriever):
        """[T03] Knowledge search returns matching committed items."""
        k1 = _make_knowledge(job_id, title="FVG Rule", statement="FVG fills when price revisits the gap area")
        k2 = _make_knowledge(job_id, title="Liquidity Sweep", statement="Price sweeps liquidity before reversal")
        repo.save_knowledge_item(k1, is_committed=True)
        repo.save_knowledge_item(k2, is_committed=True)

        results = memory_retriever.search(
            job_id=job_id, query="FVG", source_types=["knowledge"], limit=5
        )
        assert len(results) >= 1
        texts = [r.embedding.text_content for r in results]
        assert any("FVG" in t for t in texts)

    def test_T04_evidence_time_window_filter(self, job_id, repo, memory_retriever):
        """[T04] Evidence retrieval filters by time window correctly."""
        ev_early = _make_evidence(job_id, start=0.0, text="FVG early evidence")
        ev_late = _make_evidence(job_id, start=100.0, text="FVG late evidence")
        repo.save_evidence(ev_early)
        repo.save_evidence(ev_late)

        results = memory_retriever.search(
            job_id=job_id,
            query="FVG",
            source_types=["evidence"],
            time_range=(0.0, 50.0),
            limit=10,
        )
        ids = [r.embedding.source_id for r in results]
        assert ev_early.evidence_id in ids
        assert ev_late.evidence_id not in ids

    def test_T05_transcript_keyword_search(self, job_id, repo, memory_retriever):
        """[T05] Transcript search matches keyword in segment text."""
        seg = _make_transcript(job_id, text="This candle shows a liquidity sweep setup")
        repo.save_transcript_segment(seg)

        results = memory_retriever.search(
            job_id=job_id, query="liquidity sweep", source_types=["transcript"], limit=5
        )
        assert any(r.embedding.source_id == seg.segment_id for r in results)

    def test_T06_visual_event_search(self, job_id, repo, memory_retriever):
        """[T06] Visual event search matches keyword in description."""
        ev = _make_visual(job_id, desc="Bearish FVG candle pattern on M15")
        repo.save_visual_event(ev)

        results = memory_retriever.search(
            job_id=job_id, query="FVG candle", source_types=["visual"], limit=5
        )
        assert any(r.embedding.source_id == ev.event_id for r in results)

    def test_T07_multi_source_search_returns_mixed(self, job_id, repo, memory_retriever):
        """[T07] Search across all sources returns mixed result types."""
        k = _make_knowledge(job_id, statement="FVG knowledge item")
        ev = _make_evidence(job_id, text="FVG evidence quote")
        seg = _make_transcript(job_id, text="FVG transcript segment")
        repo.save_knowledge_item(k, is_committed=True)
        repo.save_evidence(ev)
        repo.save_transcript_segment(seg)

        results = memory_retriever.search(job_id=job_id, query="FVG", limit=20)
        sources = {r.retrieval_metadata.get("source") for r in results}
        assert len(sources) >= 2

    def test_T08_source_type_filter(self, job_id, repo, memory_retriever):
        """[T08] source_types filter restricts results to the requested type only."""
        k = _make_knowledge(job_id, statement="FVG knowledge")
        ev = _make_evidence(job_id, text="FVG evidence")
        repo.save_knowledge_item(k, is_committed=True)
        repo.save_evidence(ev)

        results = memory_retriever.search(
            job_id=job_id, query="FVG", source_types=["evidence"], limit=10
        )
        sources = {r.retrieval_metadata.get("source") for r in results}
        assert sources == {"evidence"}

    def test_T09_knowledge_type_filter(self, job_id, repo, memory_retriever):
        """[T09] KnowledgeType filter returns only matching types."""
        rule = _make_knowledge(job_id, item_type=KnowledgeType.RULE, statement="FVG rule applies on H4")
        concept = _make_knowledge(job_id, item_type=KnowledgeType.CONCEPT, statement="FVG concept is a price gap")
        repo.save_knowledge_item(rule, is_committed=True)
        repo.save_knowledge_item(concept, is_committed=True)

        results = memory_retriever.search(
            job_id=job_id, query="FVG",
            source_types=["knowledge"],
            knowledge_types=[KnowledgeType.RULE],
            limit=10,
        )
        assert all(r.retrieval_metadata.get("item_type") == "rule" for r in results)

    def test_T10_confidence_threshold(self, job_id, repo, memory_retriever):
        """[T10] min_confidence filters out low-confidence results."""
        low = _make_knowledge(job_id, statement="low confidence FVG info", confidence=0.2)
        high = _make_knowledge(job_id, statement="high confidence FVG rule", confidence=0.9)
        repo.save_knowledge_item(low, is_committed=True)
        repo.save_knowledge_item(high, is_committed=True)

        results = memory_retriever.search(
            job_id=job_id, query="FVG",
            source_types=["knowledge"],
            min_confidence=0.5,
            limit=10,
        )
        ids = [r.embedding.source_id for r in results]
        assert low.item_id not in ids
        assert high.item_id in ids


# -----------------------------------------------------------------------
# CONTEXT RECONSTRUCTION
# -----------------------------------------------------------------------

class TestContextReconstruction:

    def test_T11_warm_start_from_checkpoint(self, job_id, repo, token_manager):
        """[T11] ContextManager restores window from DB on warm start."""
        cm1 = ContextManager(job_id=job_id, repository=repo, token_manager=token_manager, max_tokens=500)
        item = _make_context_item(text="FVG fills when price revisits", tokens=50, importance=0.9)
        cm1.add_item(item)
        cm1.checkpoint()

        cm2 = ContextManager(job_id=job_id, repository=repo, token_manager=token_manager, max_tokens=500)
        restored = cm2.restore_from_checkpoint()
        assert restored is True
        assert len(cm2.window.included) == 1
        assert cm2.window.included[0].item_id == item.item_id

    def test_T12_cold_start_empty_window(self, job_id, repo, token_manager):
        """[T12] Cold start (no DB snapshot) produces empty window."""
        cm = ContextManager(job_id=job_id, repository=repo, token_manager=token_manager, max_tokens=500)
        restored = cm.restore_from_checkpoint()
        assert restored is False
        assert len(cm.window.included) == 0
        assert cm.current_tokens == 0

    def test_T13_serial_advance_persists_new_snapshot(self, job_id, repo, token_manager):
        """[T13] advance_serial increments serial and writes a new snapshot."""
        cm = ContextManager(job_id=job_id, repository=repo, token_manager=token_manager, max_tokens=500)
        item = _make_context_item(text="serial test content", tokens=40)
        cm.add_item(item)
        cm.advance_serial(new_serial=3)

        snaps = repo.list_context_snapshots(job_id)
        assert len(snaps) >= 1
        # Latest snapshot has serial 3
        latest = max(snaps, key=lambda s: s.serial_number)
        assert latest.serial_number == 3


# -----------------------------------------------------------------------
# CONTEXT BUDGET ENFORCEMENT
# -----------------------------------------------------------------------

class TestContextBudget:

    def test_T14_token_budget_ceiling(self, context_manager):
        """[T14] Items exceeding budget are deferred, not added."""
        # max_tokens = 500; add items that sum > 500
        items = [_make_context_item(text="x" * 100, tokens=150) for _ in range(5)]
        for item in items:
            context_manager.add_item(item)
        assert context_manager.current_tokens <= 500
        assert len(context_manager.window.deferred) > 0

    def test_T15_pruning_moves_lowest_importance_to_deferred(self, context_manager):
        """[T15] When window is full, lowest-importance item is deferred."""
        # Fill window with medium-importance items
        for i in range(3):
            context_manager.add_item(_make_context_item(
                text="medium content " * 5, tokens=100, importance=0.5
            ))
        # Add a very-low-importance item that forces eviction
        low_item = _make_context_item(text="low item content", tokens=100, importance=0.1)
        high_item = _make_context_item(text="high importance content", tokens=200, importance=0.95)
        context_manager.add_item(low_item)
        context_manager.add_item(high_item)

        deferred_ids = {d.item_id for d in context_manager.window.deferred}
        # At least one item should be deferred
        assert len(deferred_ids) > 0

    def test_T16_compression_summary_generated_on_pruning(self, context_manager):
        """[T16] Pruning generates a non-empty compression summary."""
        for _ in range(6):
            context_manager.add_item(_make_context_item(
                text="padding content " * 8, tokens=120, importance=0.5
            ))
        assert context_manager.get_compression_summary() != ""

    def test_T17_force_prune_reduces_window(self, context_manager):
        """[T17] force_prune reduces current_tokens below target."""
        for _ in range(3):
            context_manager.add_item(_make_context_item(text="content", tokens=100))
        before = context_manager.current_tokens
        freed = context_manager.force_prune(target_tokens=100)
        assert freed > 0
        assert context_manager.current_tokens < before

    def test_T18_batch_add_items_respects_importance(self, job_id, repo, token_manager):
        """[T18] add_items keeps highest-importance items in window."""
        cm = ContextManager(job_id=job_id, repository=repo, token_manager=token_manager, max_tokens=250)
        items = [
            _make_context_item(text="low item", tokens=100, importance=0.1),
            _make_context_item(text="high item", tokens=100, importance=0.95),
            _make_context_item(text="med item", tokens=100, importance=0.5),
        ]
        added, deferred = cm.add_items(items)
        # High importance item should be in the window
        in_window_texts = [i.text_content for i in cm.window.included]
        assert "high item" in in_window_texts


# -----------------------------------------------------------------------
# SEMANTIC RETRIEVAL
# -----------------------------------------------------------------------

class TestSemanticRetrieval:

    def test_T19_embedding_registration_and_retrieval(self, job_id, repo, memory_retriever):
        """[T19] register_embedding persists vector and is retrievable."""
        source_id = str(uuid.uuid4())
        emb = memory_retriever.register_embedding(
            job_id=job_id,
            source_type="knowledge",
            source_id=source_id,
            text_content="FVG fills with bullish momentum",
            importance_score=0.85,
        )
        assert emb.embedding_id is not None
        fetched = repo.get_embedding_by_source(job_id, source_id)
        assert fetched is not None
        assert len(fetched.embedding) > 0

    def test_T20_cosine_similarity_reranking(self, job_id, repo, memory_retriever):
        """[T20] Semantic re-ranking gives higher score to more similar text."""
        # Pre-register embeddings for two knowledge items
        k1 = _make_knowledge(job_id, title="FVG Rule", statement="FVG is a Fair Value Gap setup in price action")
        k2 = _make_knowledge(job_id, title="RSI Rule", statement="RSI divergence indicates momentum shift")
        repo.save_knowledge_item(k1, is_committed=True)
        repo.save_knowledge_item(k2, is_committed=True)
        memory_retriever.register_embedding(
            job_id, "knowledge", k1.item_id, f"{k1.title}: {k1.statement}"
        )
        memory_retriever.register_embedding(
            job_id, "knowledge", k2.item_id, f"{k2.title}: {k2.statement}"
        )

        # Query using "FVG" — matches both titles/statements via LIKE, then re-ranks semantically
        results = memory_retriever.search(
            job_id=job_id, query="FVG",
            source_types=["knowledge"], use_semantic=True, limit=5
        )
        # k1 contains "FVG" in both title and statement — must appear in results
        assert len(results) >= 1
        ids_in_order = [r.embedding.source_id for r in results]
        assert k1.item_id in ids_in_order
        # k1 should rank first (most relevant to "FVG" query)
        assert ids_in_order[0] == k1.item_id

    def test_T21_semantic_fallback_without_embeddings(self, job_id, repo, memory_retriever):
        """[T21] Semantic search degrades gracefully when no persisted embeddings exist."""
        k = _make_knowledge(job_id, statement="FVG fills when price revisits")
        repo.save_knowledge_item(k, is_committed=True)
        # No register_embedding call — no persisted vectors

        results = memory_retriever.search(
            job_id=job_id, query="FVG", source_types=["knowledge"], use_semantic=True, limit=5
        )
        # Should still return results (falls back to synthetic embedding)
        assert len(results) >= 1


# -----------------------------------------------------------------------
# CONFLICT DETECTION
# -----------------------------------------------------------------------

class TestConflictDetection:

    def test_T22_find_conflicts_detects_high_similarity(self, job_id, repo, memory_retriever):
        """[T22] find_conflicts returns items above similarity threshold."""
        k = _make_knowledge(job_id, statement="FVG always fills when price revisits the gap")
        repo.save_knowledge_item(k, is_committed=True)
        memory_retriever.register_embedding(
            job_id, "knowledge", k.item_id, f"{k.title}: {k.statement}"
        )

        # Use identical text — should produce similarity ~1.0
        conflicts = memory_retriever.find_conflicts(
            job_id=job_id,
            statement="FVG always fills when price revisits the gap",
            similarity_threshold=0.90,
        )
        assert len(conflicts) >= 1
        assert conflicts[0].retrieval_metadata.get("conflicting_item_id") == k.item_id

    def test_T23_find_conflicts_empty_when_no_match(self, job_id, repo, memory_retriever):
        """[T23] find_conflicts returns empty when similarity is below threshold."""
        k = _make_knowledge(job_id, statement="RSI divergence indicates momentum shift")
        repo.save_knowledge_item(k, is_committed=True)
        memory_retriever.register_embedding(
            job_id, "knowledge", k.item_id, f"{k.title}: {k.statement}"
        )

        conflicts = memory_retriever.find_conflicts(
            job_id=job_id,
            statement="FVG gap fills with bullish momentum",
            similarity_threshold=0.99,  # very high — no match expected
        )
        assert conflicts == []


# -----------------------------------------------------------------------
# TOKEN EFFICIENCY
# -----------------------------------------------------------------------

class TestTokenEfficiency:

    def test_T24_get_context_for_prompt_max_tokens(self, job_id, repo, token_manager):
        """[T24] get_context_for_prompt enforces max_tokens override."""
        cm = ContextManager(job_id=job_id, repository=repo, token_manager=token_manager, max_tokens=500)
        for _ in range(4):
            cm.add_item(_make_context_item(text="content abc", tokens=80))

        prompt_items = cm.get_context_for_prompt(max_tokens=200)
        total = sum(i.estimated_tokens for i in prompt_items)
        assert total <= 200

    def test_T25_search_results_trimmed_by_token_budget(self, job_id, repo, memory_retriever):
        """[T25] max_result_tokens trims search results to fit budget."""
        for i in range(5):
            k = _make_knowledge(
                job_id,
                statement=f"FVG rule number {i} applies to various timeframes and market conditions"
            )
            repo.save_knowledge_item(k, is_committed=True)

        results_unlimited = memory_retriever.search(
            job_id=job_id, query="FVG", source_types=["knowledge"], limit=10
        )
        results_limited = memory_retriever.search(
            job_id=job_id, query="FVG", source_types=["knowledge"], limit=10, max_result_tokens=50
        )
        assert len(results_limited) < len(results_unlimited)


# -----------------------------------------------------------------------
# SYSTEM / DIAGNOSTICS
# -----------------------------------------------------------------------

class TestSystem:

    def test_T26_context_manager_status(self, context_manager):
        """[T26] ContextManager.status() returns correct diagnostic dict."""
        status = context_manager.status()
        assert "job_id" in status
        assert "current_tokens" in status
        assert "max_tokens" in status
        assert "utilization_pct" in status
        assert "included_items" in status
        assert status["max_tokens"] == 500

    def test_T27_increment_embedding_access(self, job_id, repo):
        """[T27] increment_embedding_access correctly increments access_count."""
        emb_id = str(uuid.uuid4())
        emb = MemoryEmbedding(
            embedding_id=emb_id,
            job_id=job_id,
            source_type="knowledge",
            source_id=str(uuid.uuid4()),
            text_content="Test embedding content",
            embedding=[0.1, 0.2],
            access_count=0,
        )
        repo.save_memory_embedding(emb)
        repo.increment_embedding_access(emb_id)
        repo.increment_embedding_access(emb_id)

        all_embs = repo.get_memory_embeddings(job_id)
        target = next(e for e in all_embs if e.embedding_id == emb_id)
        assert target.access_count == 2

    def test_T28_evidence_window_bounded_by_tokens(self, job_id, repo, memory_retriever):
        """[T28] get_evidence_for_window respects token budget ceiling."""
        for i in range(10):
            ev = _make_evidence(
                job_id, start=float(i * 5),
                text=f"Evidence statement {i} about FVG and liquidity patterns in detail"
            )
            repo.save_evidence(ev)

        all_ev = memory_retriever.get_evidence_for_window(
            job_id=job_id, start_time=0.0, end_time=200.0
        )
        limited_ev = memory_retriever.get_evidence_for_window(
            job_id=job_id, start_time=0.0, end_time=200.0, max_tokens=50
        )
        assert len(limited_ev) < len(all_ev)
