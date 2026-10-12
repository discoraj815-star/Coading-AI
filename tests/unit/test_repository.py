"""Unit tests for SQLite Repository CRUD and retrieval."""

import pytest
from trading_ai.core.token_manager import TokenUsageRecord
from trading_ai.schemas import (
    AlignmentRelationship,
    AudioVisualAlignment,
    ErrorSeverity,
    EvidenceTrace,
    JobMetadata,
    JobStatus,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    LectureNotes,
    PipelineStage,
    TimestampedNoteEntry,
    TranscriptSegment,
    VisualEvent,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def repo():
    # Use in-memory SQLite database for fast isolated tests
    db_manager = DatabaseManager(db_path=":memory:")
    return Repository(db_manager)


def test_job_crud(repo):
    job = JobMetadata(
        job_id="job_crud_1",
        file_hash="hash123",
        filename="lecture.mp4",
        file_path="data/uploads/lecture.mp4",
        duration=120.0,
    )
    repo.save_job(job)

    fetched = repo.get_job("job_crud_1")
    assert fetched is not None
    assert fetched.filename == "lecture.mp4"
    assert fetched.duration == 120.0

    # Update job
    fetched.status = JobStatus.COMPLETED
    fetched.current_stage = PipelineStage.COMPLETED
    repo.save_job(fetched)

    updated = repo.get_job("job_crud_1")
    assert updated.status == JobStatus.COMPLETED
    assert updated.current_stage == PipelineStage.COMPLETED


def test_transcripts_and_visual_events(repo):
    job = JobMetadata(
        job_id="job_tv",
        file_hash="hash_tv",
        filename="tv.mp4",
        file_path="data/uploads/tv.mp4",
    )
    repo.save_job(job)

    # Transcript
    t1 = TranscriptSegment(
        segment_id="t1",
        job_id="job_tv",
        start_time=0.0,
        end_time=30.0,
        transcript="Welcome to ICT liquidity lecture.",
        terminology=["liquidity"],
    )
    repo.save_transcript_segment(t1)
    transcripts = repo.get_transcripts("job_tv")
    assert len(transcripts) == 1
    assert transcripts[0].transcript == "Welcome to ICT liquidity lecture."

    # Visual Event
    v1 = VisualEvent(
        event_id="v1",
        job_id="job_tv",
        timestamp=15.0,
        visual_description="Chart opens on 4-hour timeframe.",
        detected_event="timeframe_change",
        timeframe="4h",
    )
    repo.save_visual_event(v1)
    visuals = repo.get_visual_events("job_tv")
    assert len(visuals) == 1
    assert visuals[0].timeframe == "4h"


def test_alignments_and_evidence(repo):
    job = JobMetadata(
        job_id="job_ae",
        file_hash="hash_ae",
        filename="ae.mp4",
        file_path="data/uploads/ae.mp4",
    )
    repo.save_job(job)

    # Alignment
    align = AudioVisualAlignment(
        alignment_id="a1",
        job_id="job_ae",
        start_time=0.0,
        end_time=30.0,
        relationship=AlignmentRelationship.SUPPORTING,
        reasoning="Teacher explains chart as timeframe changes.",
    )
    repo.save_alignment(align)
    alignments = repo.get_alignments("job_ae")
    assert len(alignments) == 1
    assert alignments[0].relationship == AlignmentRelationship.SUPPORTING

    # Evidence
    ev = EvidenceTrace(
        evidence_id="e1",
        job_id="job_ae",
        start_time=10.0,
        end_time=20.0,
        audio_quote="Mark this high as buy-side liquidity.",
        extracted_statement="Highs represent buy-side liquidity.",
    )
    repo.save_evidence(ev)
    ev_list = repo.get_evidence("job_ae")
    assert len(ev_list) == 1
    assert ev_list[0].extracted_statement == "Highs represent buy-side liquidity."


def test_knowledge_and_search(repo):
    job = JobMetadata(
        job_id="job_k",
        file_hash="hash_k",
        filename="k.mp4",
        file_path="data/uploads/k.mp4",
    )
    repo.save_job(job)

    k1 = KnowledgeItem(
        item_id="k1",
        job_id="job_k",
        item_type=KnowledgeType.RULE,
        title="Order Block Invalidation",
        statement="An order block is invalidated if price closes past the mean threshold.",
        validation_status=KnowledgeValidationStatus.VALIDATED,
        start_time=30.0,
        end_time=60.0,
        tags=["order_block", "invalidation"],
    )
    # Save as uncommitted initially
    repo.save_knowledge_item(k1, serial_number=1, is_committed=False)

    # Should not appear in committed knowledge list
    committed = repo.get_knowledge("job_k", only_committed=True)
    assert len(committed) == 0

    # Commit serial 1
    repo.commit_knowledge_for_serial("job_k", 1)
    committed = repo.get_knowledge("job_k", only_committed=True)
    assert len(committed) == 1
    assert committed[0].title == "Order Block Invalidation"

    # Test search_knowledge retrieval interface
    results = repo.search_knowledge("job_k", query="mean threshold")
    assert len(results) == 1
    assert results[0].item_id == "k1"

    empty_results = repo.search_knowledge("job_k", query="non_existent_topic")
    assert len(empty_results) == 0


def test_notes_and_observability(repo):
    job = JobMetadata(
        job_id="job_notes",
        file_hash="hash_notes",
        filename="notes.mp4",
        file_path="data/uploads/notes.mp4",
    )
    repo.save_job(job)

    notes = LectureNotes(
        job_id="job_notes",
        overview="High level summary of the entire session.",
        concepts=[
            TimestampedNoteEntry(
                title="Fair Value Gap",
                content="3-candle imbalance pattern.",
                start_time=10.0,
                end_time=40.0,
            )
        ],
    )
    repo.save_notes(notes)

    fetched_notes = repo.get_notes("job_notes")
    assert fetched_notes is not None
    assert len(fetched_notes.concepts) == 1
    assert fetched_notes.concepts[0].title == "Fair Value Gap"

    # Token usage logging
    repo.save_token_usage(
        TokenUsageRecord(
            job_id="job_notes",
            serial_number=1,
            stage="visual_processing",
            model="gemini-2.5-flash",
            input_tokens=1000,
            output_tokens=200,
            total_tokens=1200,
        )
    )

    # Error logging
    repo.log_error(
        job_id="job_notes",
        serial_number=2,
        stage="reasoning",
        error_message="Network glitch",
        severity=ErrorSeverity.RECOVERABLE,
        retry_count=1,
    )
