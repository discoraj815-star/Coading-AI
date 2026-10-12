"""End-to-End integration test for the full Step 1 Multimodal Lecture Intelligence pipeline."""

import os
import pytest
from trading_ai.core.config import Settings
from trading_ai.pipeline import LecturePipelineOrchestrator
from trading_ai.schemas import (
    AlignmentRelationship,
    JobStatus,
    KnowledgeValidationStatus,
    PipelineStage,
)
from trading_ai.storage import DatabaseManager, Repository


@pytest.fixture
def e2e_env(tmp_path):
    db_file = tmp_path / "e2e_trading_ai.db"
    upload_dir = tmp_path / "uploads"
    frames_dir = tmp_path / "frames"
    upload_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        db_path=str(db_file),
        upload_dir=str(upload_dir),
        frames_dir=str(frames_dir),
    )

    db_mgr = DatabaseManager(db_path=str(db_file))
    repo = Repository(db_mgr)
    orchestrator = LecturePipelineOrchestrator(repository=repo, settings=settings)

    # Create representative lecture media file
    media_file = tmp_path / "ict_liquidity_masterclass.mp4"
    media_file.write_bytes(b"Simulated trading video binary: EURUSD 15m liquidity sweep and fair value gap lecture.")

    return orchestrator, repo, str(media_file)


def test_full_pipeline_end_to_end(e2e_env):
    orchestrator, repo, media_file = e2e_env

    # Run the complete pipeline
    completed_job = orchestrator.run_pipeline(
        source_media_path=media_file,
        filename="ict_liquidity_masterclass.mp4",
        job_id="job_e2e_001",
        chunk_duration=30.0,
    )

    # 1. Job Validation
    assert completed_job.status == JobStatus.COMPLETED
    assert completed_job.current_stage == PipelineStage.COMPLETED
    assert completed_job.retry_count == 0
    assert len(completed_job.file_hash) == 64

    # 2. Transcripts Validation
    transcripts = repo.get_transcripts("job_e2e_001")
    assert len(transcripts) >= 1
    assert transcripts[0].start_time == 0.0
    assert len(transcripts[0].terminology) > 0
    assert "liquidity" in " ".join(transcripts[0].terminology).lower()

    # 3. Visual Events Validation
    visual_events = repo.get_visual_events("job_e2e_001")
    assert len(visual_events) >= 1
    assert visual_events[0].symbol == "EURUSD"
    assert visual_events[0].timeframe == "15m"
    assert visual_events[0].confidence >= 0.85

    # 4. Audio/Visual Alignment Validation
    alignments = repo.get_alignments("job_e2e_001")
    assert len(alignments) >= 1
    assert alignments[0].relationship in [
        AlignmentRelationship.SUPPORTING,
        AlignmentRelationship.PARTIALLY_SUPPORTING,
    ]
    assert len(alignments[0].reasoning) > 0

    # 5. Evidence Traces Validation
    evidence_traces = repo.get_evidence("job_e2e_001")
    assert len(evidence_traces) >= 1
    assert evidence_traces[0].extracted_statement is not None

    # 6. Committed Knowledge Validation (Commit Barrier Verified)
    committed_knowledge = repo.get_knowledge("job_e2e_001", only_committed=True)
    assert len(committed_knowledge) >= 1
    rule_items = [k for k in committed_knowledge if k.item_type.value in ["rule", "concept", "exception"]]
    assert len(rule_items) > 0
    assert committed_knowledge[0].validation_status in [
        KnowledgeValidationStatus.VALIDATED,
        KnowledgeValidationStatus.OBSERVED,
    ]

    # 7. Lecture Notes Validation (All 11 Sections Verified)
    notes = repo.get_notes("job_e2e_001")
    assert notes is not None
    assert len(notes.overview) > 0
    assert len(notes.concepts) > 0
    assert len(notes.rules) > 0
    assert len(notes.evidence_timestamps) > 0

    # 8. Token Observability Validation
    summary = orchestrator._token_manager.get_job_usage_summary("job_e2e_001")
    assert summary["total_calls"] >= 4
    assert summary["total_tokens"] > 0
