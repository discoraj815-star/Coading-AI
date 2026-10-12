import pytest
import time
from trading_ai.benchmark import LectureBenchmarkEngine
from trading_ai.schemas.benchmark import BenchmarkStage, LectureBenchmarkRecord
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository


@pytest.fixture
def temp_repo(tmp_path):
    db_file = tmp_path / "test_benchmarks.db"
    db = DatabaseManager(str(db_file))
    return Repository(db)


def test_lecture_benchmark_lifecycle(temp_repo):
    engine = LectureBenchmarkEngine(temp_repo)
    lecture_id = "test_lecture_101"

    rec = engine.start_benchmark(
        lecture_id=lecture_id,
        file_name="Lesson1_SMC.mp4",
        file_hash="hash123456",
        file_size_bytes=50_000_000,
        duration_seconds=900.0,
        resolution="1920x1080",
        fps=30.0,
    )

    assert rec.lecture_id == lecture_id
    assert rec.current_stage == BenchmarkStage.QUEUED
    assert len(rec.timeline) >= 1

    # Ingestion stage
    with engine.benchmark_stage(lecture_id, BenchmarkStage.INGESTION):
        time.sleep(0.02)

    # Audio stage
    with engine.benchmark_stage(lecture_id, BenchmarkStage.AUDIO):
        time.sleep(0.03)

    # Visual stage
    with engine.benchmark_stage(lecture_id, BenchmarkStage.VISUAL):
        time.sleep(0.05)

    # Update counts and metrics
    engine.update_counts(
        lecture_id=lecture_id,
        frames_sampled=60,
        frames_processed=15,
        audio_segments=20,
        visual_segments=15,
        aligned_segments=12,
        gemini_calls=3,
        tokens_input=5000,
        tokens_output=800,
    )

    engine.update_knowledge_metrics(
        lecture_id=lecture_id,
        concepts=4,
        rules=2,
        evidence=8,
        exceptions=1,
        examples=3,
    )

    final = engine.finalize_benchmark(lecture_id, is_complete=True)

    assert final.is_complete is True
    assert final.current_stage == BenchmarkStage.COMPLETE
    assert final.metrics["total_processing_time"] > 0.08
    assert final.metrics["visual_processing_time"] > 0.04
    assert final.counts["tokens_input"] == 5000
    assert final.counts["tokens_output"] == 800
    assert final.estimated_cost_usd > 0.0
    assert final.knowledge_metrics["rules_extracted"] == 2

    # Bottleneck verification: visual analysis should be largest contributor
    assert "VISUAL_ANALYSIS" in final.bottlenecks
    assert final.bottlenecks["VISUAL_ANALYSIS"] > 0.0

    # Persistence verification: retrieve from SQLite database
    stored = temp_repo.get_lecture_benchmark(lecture_id)
    assert stored is not None
    assert stored["lecture_id"] == lecture_id
    assert len(stored["timeline"]) >= 4
    assert stored["is_complete"] is True
