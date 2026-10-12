"""Unit tests for domain schemas and validation."""

import pytest
from pydantic import ValidationError
from trading_ai.schemas import (
    AlignmentRelationship,
    AudioVisualAlignment,
    CommitBarrierState,
    EvidenceTrace,
    JobMetadata,
    JobStatus,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    LectureNotes,
    PipelineStage,
    TaskUnit,
    TaskUnitStatus,
    TimestampedNoteEntry,
    TranscriptSegment,
    VisualEvent,
)


def test_job_metadata_validation():
    job = JobMetadata(
        job_id="job_001",
        file_hash="abc123sha",
        filename="lecture_01.mp4",
        file_path="data/uploads/lecture_01.mp4",
        duration=300.5,
    )
    assert job.status == JobStatus.PENDING
    assert job.current_stage == PipelineStage.INGESTION
    assert job.retry_count == 0


def test_transcript_segment():
    seg = TranscriptSegment(
        segment_id="seg_001",
        job_id="job_001",
        start_time=10.0,
        end_time=25.0,
        transcript="Here is the bullish order block on the 15-minute chart.",
        terminology=["order block", "15-minute"],
        rules=["Look for fair value gaps before the order block"],
        confidence=0.95,
    )
    assert seg.start_time == 10.0
    assert len(seg.terminology) == 2


def test_visual_event():
    ve = VisualEvent(
        event_id="ve_001",
        job_id="job_001",
        timestamp=12.5,
        visual_description="Chart displays a sharp green candle retesting a marked gray rectangle",
        detected_event="zone_retest",
        timeframe="15m",
        highlighted_zones=["demand_zone_rect"],
        confidence=0.92,
    )
    assert ve.detected_event == "zone_retest"
    assert ve.timeframe == "15m"


def test_audio_visual_alignment():
    seg = TranscriptSegment(
        segment_id="seg_001",
        job_id="job_001",
        start_time=10.0,
        end_time=20.0,
        transcript="Price breaks the previous high.",
    )
    ve = VisualEvent(
        event_id="ve_001",
        job_id="job_001",
        timestamp=15.0,
        visual_description="Bullish candle closes above previous swing high line",
        detected_event="swing_breakout",
    )
    align = AudioVisualAlignment(
        alignment_id="align_001",
        job_id="job_001",
        start_time=10.0,
        end_time=20.0,
        audio=seg,
        visual=[ve],
        relationship=AlignmentRelationship.SUPPORTING,
        confidence=0.94,
        reasoning="Teacher explains breakout while chart shows candle closing above high",
    )
    assert align.relationship == AlignmentRelationship.SUPPORTING
    assert len(align.visual) == 1


def test_knowledge_item_with_evidence():
    ev = EvidenceTrace(
        evidence_id="ev_001",
        job_id="job_001",
        start_time=12.0,
        end_time=18.0,
        audio_quote="Notice how price sweeps the liquidity pool before reversing.",
        visual_reference="ve_001",
        extracted_statement="Price must sweep liquidity prior to reversal confirmation.",
        confidence=0.96,
    )
    item = KnowledgeItem(
        item_id="k_001",
        job_id="job_001",
        item_type=KnowledgeType.RULE,
        title="Liquidity Sweep Requirement",
        statement="A liquidity sweep is mandatory before confirming a reversal setup.",
        reasoning="Spoken explicitly and confirmed by wick demonstration on chart.",
        validation_status=KnowledgeValidationStatus.VALIDATED,
        confidence=0.95,
        start_time=12.0,
        end_time=18.0,
        evidence=[ev],
    )
    assert item.validation_status == KnowledgeValidationStatus.VALIDATED
    assert item.evidence[0].evidence_id == "ev_001"


def test_lecture_notes_schema():
    notes = LectureNotes(
        job_id="job_001",
        overview="Introductory lecture covering ICT liquidity concepts.",
        concepts=[
            TimestampedNoteEntry(
                title="Liquidity Pool",
                content="Accumulation of stop orders above/below key levels.",
                start_time=5.0,
                end_time=30.0,
            )
        ],
        rules=[
            TimestampedNoteEntry(
                title="Sweep Confirmation",
                content="Wait for 5m candle close below swept level.",
                start_time=45.0,
                end_time=60.0,
            )
        ],
    )
    assert len(notes.concepts) == 1
    assert notes.overview.startswith("Introductory")


def test_task_unit_and_commit_barrier():
    unit = TaskUnit(
        unit_id="unit_1",
        job_id="job_001",
        serial_number=1,
        stage=PipelineStage.AUDIO_PROCESSING,
        start_time=0.0,
        end_time=60.0,
    )
    assert unit.serial_number == 1
    assert unit.status == TaskUnitStatus.PENDING

    barrier = CommitBarrierState(
        job_id="job_001",
        highest_committed_serial=4,
        buffered_units=[6, 7],
        blocked_on_serial=5,
    )
    assert barrier.highest_committed_serial == 4
    assert barrier.blocked_on_serial == 5
    assert barrier.buffered_units == [6, 7]
