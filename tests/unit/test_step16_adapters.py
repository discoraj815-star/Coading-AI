"""Unit tests for Step 16 C2 Vector Store Adapter and C3 Task Queue Adapter."""
import tempfile
import pytest
from datetime import datetime, timezone

from trading_ai.schemas.cloud import TaskQueueJob
from trading_ai.schemas.media import JobMetadata, JobStatus, PipelineStage
from trading_ai.schemas.memory import MemoryEmbedding
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.storage.task_queue import LocalPersistentTaskQueue
from trading_ai.storage.vector_store import SQLiteVectorStore
from trading_ai.supervisor.commit_barrier import CommitBarrier


@pytest.fixture
def repo():
    db = DatabaseManager(":memory:")
    return Repository(db)


def test_vector_store_adapter_sqlite_lifecycle(repo):
    vector_store = SQLiteVectorStore(repo)

    # Health check
    health = vector_store.health()
    assert health["backend"] == "SQLiteVectorStore"
    assert health["status"] == "HEALTHY"

    # Add embedding
    # Ensure parent job exists
    repo.save_job(
        JobMetadata(
            job_id="job_vtest",
            file_hash="hash_vtest",
            filename="test_lec",
            file_path="test.mp4",
            status=JobStatus.COMPLETED,
            current_stage=PipelineStage.STORAGE,
        )
    )

    emb = MemoryEmbedding(
        embedding_id="emb_001",
        job_id="job_vtest",
        source_type="transcript",
        source_id="seg_001",
        text_content="Bullish Order Block entry",
        embedding=[0.1, 0.2, 0.3, 0.4],
    )
    ok = vector_store.add_embedding(emb)
    assert ok is True
    assert vector_store.count() == 1

    # Search
    results = vector_store.search(query_vector=[0.1, 0.2, 0.3, 0.4], limit=5)
    assert len(results) == 1
    found_emb, score = results[0]
    assert found_emb.embedding_id == "emb_001"
    assert score > 0.99

    # Snapshot and Restore
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        snap_path = tf.name
    try:
        assert vector_store.snapshot(snap_path) is True
        # Clear & restore
        vector_store.delete("emb_001")
        assert vector_store.count() == 0
        assert vector_store.restore(snap_path) is True
        assert vector_store.count() == 1
    finally:
        import os
        if os.path.exists(snap_path):
            os.remove(snap_path)


def test_task_queue_adapter_lifecycle_and_barrier(repo):
    barrier = CommitBarrier(repo)
    queue = LocalPersistentTaskQueue(repo, commit_barrier=barrier)

    # Health check
    h = queue.health()
    assert h["backend"] == "LocalPersistentTaskQueue"
    assert h["status"] == "HEALTHY"

    # Enqueue task
    job = queue.enqueue(
        task_type="VIDEO_CHUNK_PROCESSING",
        payload={"chunk_index": 0},
        priority=1,
    )
    assert job.status == "PENDING"

    # Dequeue task
    dequeued = queue.dequeue(limit=1)
    assert len(dequeued) == 1
    djob = dequeued[0]
    assert djob.job_id == job.job_id

    # Acknowledge
    ack = queue.acknowledge(djob.job_id)
    assert ack is True
    st = queue.get_status(djob.job_id)
    assert st.status == "COMPLETED"


def test_task_queue_retry_and_cancellation(repo):
    queue = LocalPersistentTaskQueue(repo)

    job = queue.enqueue("TRANSCRIPTION", {"audio": "test.wav"}, priority=2)
    # Retry once
    queue.retry(job.job_id, "Gemini API Timeout")
    st = queue.get_status(job.job_id)
    assert st.status == "RETRYING"
    assert st.retry_count == 1

    # Cancel
    queue.cancel(job.job_id)
    st = queue.get_status(job.job_id)
    assert st.status == "CANCELLED"
