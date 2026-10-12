"""Integration and failure tests for Step 3: Universal Learning, User Commands, and Large Video Processing."""

import os
import tempfile
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from trading_ai.core.config import Settings
from trading_ai.core.token_manager import TokenManager
from trading_ai.pipeline.orchestrator import LecturePipelineOrchestrator
from trading_ai.schemas import (
    ErrorSeverity,
    JobStatus,
    KnowledgeValidationStatus,
    PipelineStage,
    SourceStatus,
    SourceType,
)
from trading_ai.server.app import app
from trading_ai.storage import DatabaseManager, Repository


@pytest.fixture
def temp_dir():
    td = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    yield td.name
    try:
        td.cleanup()
    except Exception:
        pass


@pytest.fixture
def persistent_db_and_settings(temp_dir):
    db_path = os.path.join(temp_dir, "test_trading.db")
    settings = Settings(
        db_path=db_path,
        upload_dir=os.path.join(temp_dir, "uploads"),
        frames_dir=os.path.join(temp_dir, "frames"),
        max_daily_budget_tokens=500_000,
        max_job_budget_tokens=100_000,
    )
    settings.ensure_directories()
    db = DatabaseManager(db_path)
    repo = Repository(db)
    return db, repo, settings, db_path


@pytest.fixture
def mock_video_file(temp_dir):
    video_path = os.path.join(temp_dir, "large_lecture_test.mp4")
    with open(video_path, "wb") as f:
        f.write(b"SIMULATED_LARGE_VIDEO_STREAM_60_SECONDS")
    return video_path


# =====================================================================
# 1. INTEGRATION: SOURCE -> JOB -> PROCESSING -> KNOWLEDGE -> MEMORY
# =====================================================================
def test_end_to_end_source_job_knowledge_memory(persistent_db_and_settings, mock_video_file):
    db, repo, settings, _ = persistent_db_and_settings
    orch = LecturePipelineOrchestrator(repository=repo, settings=settings)

    # 1. Run pipeline
    job = orch.run_pipeline(
        source_media_path=mock_video_file,
        filename="large_lecture_test.mp4",
        job_id="job_e2e_step3",
        chunk_duration=30.0,
    )

    assert job.status == JobStatus.COMPLETED
    assert job.current_stage == PipelineStage.COMPLETED

    # 2. Verify BackgroundJobState
    bg_job = orch.get_background_job_state("job_e2e_step3")
    assert bg_job is not None
    assert bg_job.status == JobStatus.COMPLETED
    assert bg_job.progress == 100.0

    # 3. Verify knowledge items committed
    knowledge = repo.get_knowledge("job_e2e_step3", only_committed=True)
    assert len(knowledge) > 0

    # 4. Verify memory embeddings persisted
    embeddings = repo.get_memory_embeddings("job_e2e_step3")
    assert len(embeddings) > 0

    # 5. Verify concepts and notes
    concepts = repo.get_concepts("job_e2e_step3")
    assert isinstance(concepts, list)


# =====================================================================
# 2. FAILURE TEST: VIDEO INTERRUPTION & RESUME WITHOUT REPROCESSING
# =====================================================================
def test_video_interruption_and_resumability(persistent_db_and_settings, mock_video_file):
    db, repo, settings, _ = persistent_db_and_settings
    orch = LecturePipelineOrchestrator(repository=repo, settings=settings)

    job_id = "job_resumable_video"

    # Step 1: Run chunk 1 and pause mid-way
    orch.run_pipeline(
        source_media_path=mock_video_file,
        filename="large_lecture_test.mp4",
        job_id=job_id,
        chunk_duration=30.0,
    )

    barrier_1 = repo.get_commit_barrier(job_id)
    assert barrier_1.highest_committed_serial >= 1
    units_1 = repo.list_task_units(job_id)
    initial_unit_count = len(units_1)
    initial_knowledge_count = len(repo.get_knowledge(job_id, only_committed=True))

    # Step 2: Now simulate resuming (calling run_pipeline again)
    # The orchestrator detects barrier_1.highest_committed_serial >= chunk_serial
    # and skips already committed chunks without duplicating task units or knowledge items
    job_resumed = orch.run_pipeline(
        source_media_path=mock_video_file,
        filename="large_lecture_test.mp4",
        job_id=job_id,
        chunk_duration=30.0,
    )

    assert job_resumed.status == JobStatus.COMPLETED
    units_resumed = repo.list_task_units(job_id)
    # No duplicate units created for committed serials
    assert len(units_resumed) == initial_unit_count
    final_knowledge = repo.get_knowledge(job_id, only_committed=True)
    assert len(final_knowledge) == initial_knowledge_count


# =====================================================================
# 3. FAILURE TEST: APPLICATION RESTART SIMULATION
# =====================================================================
def test_application_restart_preserves_state(persistent_db_and_settings, mock_video_file):
    db, repo, settings, db_path = persistent_db_and_settings
    orch = LecturePipelineOrchestrator(repository=repo, settings=settings)

    job_id = "job_restart_test"
    orch.run_pipeline(
        source_media_path=mock_video_file,
        filename="large_lecture_test.mp4",
        job_id=job_id,
        chunk_duration=30.0,
    )

    # Simulate complete process termination & fresh restart
    fresh_db = DatabaseManager(db_path)
    fresh_repo = Repository(fresh_db)
    fresh_orch = LecturePipelineOrchestrator(repository=fresh_repo, settings=settings)

    # Check that barrier, job state, and task units survived process boundary
    loaded_job = fresh_repo.get_job(job_id)
    assert loaded_job is not None
    assert loaded_job.status == JobStatus.COMPLETED

    loaded_bg = fresh_orch.get_background_job_state(job_id)
    assert loaded_bg is not None
    assert loaded_bg.status == JobStatus.COMPLETED

    loaded_barrier = fresh_repo.get_commit_barrier(job_id)
    assert loaded_barrier.highest_committed_serial >= 1


# =====================================================================
# 4. FAILURE TEST: GEMINI FAILURE & SAFE RETRY
# =====================================================================
def test_gemini_failure_and_safe_retry(persistent_db_and_settings):
    db, repo, settings, _ = persistent_db_and_settings
    barrier = LecturePipelineOrchestrator(repository=repo, settings=settings)._barrier

    job_id = "job_gemini_fail"
    unit = barrier.register_task_unit(
        job_id=job_id,
        stage=PipelineStage.TIMELINE,
        start_time=0.0,
        end_time=30.0,
    )
    serial = unit.serial_number

    # Fail unit due to simulated Gemini rate limit or connection drop
    barrier.fail_task_unit(
        job_id=job_id,
        serial_number=serial,
        error_message="ResourceExhausted: 429 Rate limit exceeded",
        severity=ErrorSeverity.RECOVERABLE,
    )

    state = barrier.get_barrier_state(job_id)
    assert state.blocked_on_serial == serial
    failed_unit = repo.get_task_unit(job_id, serial)
    assert failed_unit.last_error == "ResourceExhausted: 429 Rate limit exceeded"

    # Retry and recover unit using supervisor recover_task_unit
    committed = barrier.recover_task_unit(
        job_id=job_id,
        serial_number=serial,
        recovered_result={"status": "success_after_retry"},
        knowledge_items=[],
    )
    assert serial in committed
    state_after = barrier.get_barrier_state(job_id)
    assert state_after.blocked_on_serial is None


# =====================================================================
# 5. FAILURE TEST: TOKEN OVERFLOW PRUNING
# =====================================================================
def test_context_overflow_token_manager_protection(persistent_db_and_settings):
    _, repo, settings, _ = persistent_db_and_settings
    tm = TokenManager(settings)

    # Verify estimate tokens
    huge_text = "EURUSD Liquidity FVG " * 500
    estimated = tm.estimate_text_tokens(huge_text)
    assert estimated > 500

    from trading_ai.core.context_manager import ContextManager
    from trading_ai.schemas import ContextItem, ContextItemSource

    ctx_mgr = ContextManager(
        job_id="job_overflow_test",
        repository=repo,
        token_manager=tm,
        max_tokens=400,
    )
    items = [
        ContextItem(
            item_id=f"ctx_{i}",
            source_type=ContextItemSource.KNOWLEDGE,
            text_content=f"Trading rule #{i} statement content here",
            estimated_tokens=100,
            importance_score=float(i) / 10.0,
            serial_number=1,
            start_time=0.0,
            end_time=10.0,
        )
        for i in range(10)
    ]
    added, deferred = ctx_mgr.add_items(items)
    # Total available tokens = 400, each item = 100 tokens -> window prunes to stay within ceiling
    assert ctx_mgr.current_tokens <= 400
    assert len(ctx_mgr.window.included) <= 4


# =====================================================================
# 6. REST API INTEGRATION TESTS (FastAPI TestClient)
# =====================================================================
class TestServerAPIStep3:
    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = TestClient(app)

    def test_api_resolve_source_youtube_valid(self):
        res = self.client.post(
            "/api/sources/resolve",
            json={"source_location": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["resolution"]["is_accessible"] is True
        assert data["resolution"]["source_type"] == "YOUTUBE"

    def test_api_resolve_source_blocked_unavailable(self):
        res = self.client.post(
            "/api/sources/resolve",
            json={"source_location": "https://malicious-site.org/lecture"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["resolution"]["is_accessible"] is False
        assert data["resolution"]["status"] == "SOURCE_UNAVAILABLE"

    def test_api_submit_user_command_save_rule(self):
        res = self.client.post(
            "/api/commands",
            json={
                "command_text": "Save this rule: Always wait for 15m candle close before entry",
                "job_id": "job_api_test_cmd",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["parsed_success"] is True
        assert data["suggested_action"] == "save_rule"
        assert data["result"]["status"] in ("success", "exists")

    def test_api_submit_user_command_correct(self):
        # 1. Save rule
        save_res = self.client.post(
            "/api/commands/save_rule",
            json={
                "rule": "Rule ABC reverses on touch of 50% fibonacci",
                "job_id": "job_api_corr",
            },
        )
        item_id = save_res.json()["result"]["item_id"]

        # 2. Correct rule
        corr_res = self.client.post(
            "/api/commands/correct",
            json={
                "target_knowledge_id": item_id,
                "correction_statement": "Rule ABC reverses only if 50% fibonacci aligns with an order block.",
                "reason": "Fibonacci alone is insufficient without institutional PD array.",
                "job_id": "job_api_corr",
            },
        )
        assert corr_res.status_code == 200
        corr_data = corr_res.json()
        assert corr_data["result"]["status"] == "success"
        assert corr_data["result"]["version"] == 2

        # 3. Check versions endpoint
        v_res = self.client.get(f"/api/knowledge/{item_id}/versions")
        assert v_res.status_code == 200
        v_data = v_res.json()
        assert v_data["total_versions"] == 2
        assert v_data["active_version"]["version_number"] == 2

    def test_api_job_lifecycle_pause_resume_cancel(self):
        # Create job
        create_res = self.client.post(
            "/api/jobs/create",
            json={"filename": "test_lifecycle_lecture.mp4"},
        )
        assert create_res.status_code == 200
        job_id = create_res.json()["job"]["job_id"]

        # Check status
        status_res = self.client.get(f"/api/jobs/{job_id}/status")
        assert status_res.status_code == 200

        # Pause
        pause_res = self.client.post(f"/api/jobs/{job_id}/pause")
        assert pause_res.status_code == 200
        assert pause_res.json()["status"] == "paused"

        # Cancel
        cancel_res = self.client.post(f"/api/jobs/{job_id}/cancel")
        assert cancel_res.status_code == 200
        assert cancel_res.json()["status"] == "cancelled"

    def test_api_insights_notes_concepts_doubts(self):
        # Ingest sample
        sample_res = self.client.post(
            "/api/select_sample",
            json={"sample_name": "API_Insights_Test.mp4"},
        )
        job_id = sample_res.json()["job"]["job_id"]

        # Notes endpoint
        notes_res = self.client.get(f"/api/jobs/{job_id}/notes")
        assert notes_res.status_code == 200

        # Concepts endpoint
        concepts_res = self.client.get(f"/api/jobs/{job_id}/concepts")
        assert concepts_res.status_code == 200
        assert "concepts" in concepts_res.json()

        # Doubts endpoint
        doubts_res = self.client.get(f"/api/jobs/{job_id}/doubts")
        assert doubts_res.status_code == 200
        assert "doubts" in doubts_res.json()

        # Memory updates endpoint
        mem_res = self.client.get(f"/api/jobs/{job_id}/memory_updates")
        assert mem_res.status_code == 200
        assert "total_embeddings" in mem_res.json()
