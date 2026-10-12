"""Unit tests for media ingestion, video keyframe extraction, and AI multimodal processors."""

import os
import pytest
from trading_ai.ai import (
    AlignmentEngine,
    AudioProcessor,
    GeminiClient,
    VisualProcessor,
)
from trading_ai.core import ModelRole, ModelRouter, TokenManager
from trading_ai.media import MediaIngestionService, VideoProcessor
from trading_ai.schemas import AlignmentRelationship, JobMetadata
from trading_ai.storage import DatabaseManager, Repository


from trading_ai.core.config import Settings


@pytest.fixture
def test_env(tmp_path):
    db_file = tmp_path / "proc_test.db"
    db_mgr = DatabaseManager(db_path=str(db_file))
    repo = Repository(db_mgr)
    settings = Settings(db_path=str(db_file), gemini_api_key="")
    tm = TokenManager(settings)
    mr = ModelRouter(settings)
    ai_client = GeminiClient(
        settings=settings,
        model_router=mr,
        token_manager=tm,
        repository=repo,
    )

    # Create a small dummy video/media file
    media_file = tmp_path / "dummy_lecture.mp4"
    media_file.write_bytes(b"dummy video binary content representing 15m trading lecture")

    return repo, ai_client, str(media_file)


def test_media_ingestion(test_env):
    repo, ai_client, media_file = test_env
    ingestion = MediaIngestionService(repository=repo)

    job = ingestion.ingest_file(media_file, filename="dummy_lecture.mp4")
    assert job.job_id.startswith("job_")
    assert len(job.file_hash) == 64  # SHA-256
    assert job.filename == "dummy_lecture.mp4"

    persisted_job = repo.get_job(job.job_id)
    assert persisted_job is not None
    assert persisted_job.file_hash == job.file_hash


def test_video_keyframe_extraction(test_env):
    repo, ai_client, media_file = test_env
    vp = VideoProcessor()

    keyframes = vp.extract_keyframes(
        video_path=media_file,
        job_id="job_test_kf",
        sample_interval_seconds=10.0,
    )
    assert len(keyframes) >= 1
    assert os.path.exists(keyframes[0]["file_path"])
    assert keyframes[0]["width"] == 1280
    assert keyframes[0]["height"] == 720


def test_gemini_client_routing_and_budget(test_env):
    repo, ai_client, media_file = test_env

    # Generate content for visual analysis
    res = ai_client.generate_content(
        role=ModelRole.VISUAL_ANALYSIS,
        prompt="Analyze EURUSD chart keyframes.",
        job_id="job_gemini_test",
        stage="visual_processing",
        serial_number=1,
    )
    assert len(res) > 0
    assert "EURUSD" in res

    # Verify token manager recorded usage
    summary = ai_client._token_manager.get_job_usage_summary("job_gemini_test")
    assert summary["total_calls"] == 1
    assert summary["total_tokens"] > 0


def test_audio_and_visual_processors(test_env):
    repo, ai_client, media_file = test_env
    repo.save_job(
        JobMetadata(
            job_id="job_proc_test",
            file_hash="hash_proc",
            filename="dummy_lecture.mp4",
            file_path=media_file,
        )
    )
    audio_proc = AudioProcessor(gemini_client=ai_client, repository=repo)
    visual_proc = VisualProcessor(gemini_client=ai_client, repository=repo)

    vp = VideoProcessor()
    keyframes = vp.extract_keyframes(media_file, job_id="job_proc_test")

    # Audio
    transcripts = audio_proc.process_audio(
        job_id="job_proc_test",
        duration=30.0,
        serial_number=1,
    )
    assert len(transcripts) >= 1
    assert transcripts[0].start_time == 0.0
    assert len(transcripts[0].terminology) > 0

    # Visual
    visual_events = visual_proc.process_keyframes(
        job_id="job_proc_test",
        keyframes=keyframes,
        serial_number=1,
    )
    assert len(visual_events) >= 1
    assert visual_events[0].symbol == "EURUSD"
    assert visual_events[0].timeframe == "15m"


def test_alignment_engine(test_env):
    repo, ai_client, media_file = test_env
    repo.save_job(
        JobMetadata(
            job_id="job_align_test",
            file_hash="hash_align",
            filename="dummy_lecture.mp4",
            file_path=media_file,
        )
    )
    audio_proc = AudioProcessor(gemini_client=ai_client, repository=repo)
    visual_proc = VisualProcessor(gemini_client=ai_client, repository=repo)
    align_engine = AlignmentEngine(gemini_client=ai_client, repository=repo)

    vp = VideoProcessor()
    keyframes = vp.extract_keyframes(media_file, job_id="job_align_test")

    transcripts = audio_proc.process_audio(job_id="job_align_test", duration=30.0)
    visual_events = visual_proc.process_keyframes(job_id="job_align_test", keyframes=keyframes)

    alignments = align_engine.align_segments(
        job_id="job_align_test",
        transcripts=transcripts,
        visual_events=visual_events,
        serial_number=1,
    )

    assert len(alignments) == len(transcripts)
    assert alignments[0].relationship in [
        AlignmentRelationship.SUPPORTING,
        AlignmentRelationship.PARTIALLY_SUPPORTING,
    ]
    assert len(alignments[0].visual) > 0
    assert len(alignments[0].reasoning) > 0
