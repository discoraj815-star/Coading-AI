"""Unit tests for Supervisor/Worker Commit Barrier, ordered commitment, and restart recovery."""

import pytest
from trading_ai.schemas import (
    ErrorSeverity,
    JobMetadata,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    PipelineStage,
    TaskUnitStatus,
)
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier


@pytest.fixture
def barrier_setup(tmp_path):
    # Use a file-based temporary SQLite database to test persistence across restarts
    db_file = tmp_path / "barrier_test.db"
    db_mgr = DatabaseManager(db_path=str(db_file))
    repo = Repository(db_mgr)

    # Initialize a test job
    job = JobMetadata(
        job_id="job_barrier_test",
        file_hash="hash_barrier",
        filename="test_lecture.mp4",
        file_path=str(tmp_path / "test_lecture.mp4"),
    )
    repo.save_job(job)

    barrier = CommitBarrier(repo)
    return barrier, repo, str(db_file), "job_barrier_test"


def test_serial_commit_and_recovery_cascade(barrier_setup):
    barrier, repo, db_file, job_id = barrier_setup

    # Register 7 units: 1, 2, 3, 4, 5, 6, 7
    units = []
    for i in range(1, 8):
        u = barrier.register_task_unit(
            job_id=job_id,
            stage=PipelineStage.GEMINI_REASONING,
            start_time=(i - 1) * 30.0,
            end_time=i * 30.0,
            payload={"window": i},
        )
        assert u.serial_number == i
        units.append(u)

    # 1 ✓ -> COMMITTED immediately
    c1 = barrier.complete_task_unit(job_id, serial_number=1, result={"ok": 1})
    assert c1 == [1]

    # 2 ✓ -> COMMITTED
    c2 = barrier.complete_task_unit(job_id, serial_number=2, result={"ok": 2})
    assert c2 == [2]

    # 3 ✓ -> COMMITTED
    c3 = barrier.complete_task_unit(job_id, serial_number=3, result={"ok": 3})
    assert c3 == [3]

    # 4 ✓ -> COMMITTED
    c4 = barrier.complete_task_unit(job_id, serial_number=4, result={"ok": 4})
    assert c4 == [4]

    state = barrier.get_barrier_state(job_id)
    assert state.highest_committed_serial == 4
    assert state.blocked_on_serial is None
    assert state.buffered_units == []

    # 5 ✗ -> FAILS!
    barrier.fail_task_unit(
        job_id=job_id,
        serial_number=5,
        error_message="Model rate limit or network timeout",
        severity=ErrorSeverity.RECOVERABLE,
    )
    unit_5 = repo.get_task_unit(job_id, 5)
    assert unit_5.status == TaskUnitStatus.FAILED
    assert unit_5.retry_count == 1

    state = barrier.get_barrier_state(job_id)
    assert state.blocked_on_serial == 5

    # 6 ✓ -> MUST REMAIN BUFFERED!
    k6 = KnowledgeItem(
        item_id="k6",
        job_id=job_id,
        item_type=KnowledgeType.RULE,
        title="Rule from unit 6",
        statement="Unit 6 statement.",
        validation_status=KnowledgeValidationStatus.VALIDATED,
        start_time=150.0,
        end_time=180.0,
    )
    c6 = barrier.complete_task_unit(
        job_id, serial_number=6, result={"ok": 6}, knowledge_items=[k6]
    )
    assert c6 == []  # Not committed!
    unit_6 = repo.get_task_unit(job_id, 6)
    assert unit_6.status == TaskUnitStatus.BUFFERED

    # 7 ✓ -> MUST REMAIN BUFFERED!
    k7 = KnowledgeItem(
        item_id="k7",
        job_id=job_id,
        item_type=KnowledgeType.RULE,
        title="Rule from unit 7",
        statement="Unit 7 statement.",
        validation_status=KnowledgeValidationStatus.VALIDATED,
        start_time=180.0,
        end_time=210.0,
    )
    c7 = barrier.complete_task_unit(
        job_id, serial_number=7, result={"ok": 7}, knowledge_items=[k7]
    )
    assert c7 == []  # Not committed!
    unit_7 = repo.get_task_unit(job_id, 7)
    assert unit_7.status == TaskUnitStatus.BUFFERED

    state = barrier.get_barrier_state(job_id)
    assert state.highest_committed_serial == 4
    assert state.buffered_units == [6, 7]
    assert state.blocked_on_serial == 5

    # Verify knowledge items for 6 and 7 are uncommitted
    committed_k = repo.get_knowledge(job_id, only_committed=True)
    assert len(committed_k) == 0

    # -------------------------------------------------------------
    # SIMULATE APPLICATION RESTART
    # -------------------------------------------------------------
    restarted_db_mgr = DatabaseManager(db_path=db_file)
    restarted_repo = Repository(restarted_db_mgr)
    restarted_barrier = CommitBarrier(restarted_repo)

    restarted_state = restarted_barrier.get_barrier_state(job_id)
    assert restarted_state.highest_committed_serial == 4
    assert restarted_state.buffered_units == [6, 7]
    assert restarted_state.blocked_on_serial == 5

    # -------------------------------------------------------------
    # RECOVER UNIT 5 -> CASCADE COMMITS 5, 6, 7!
    # -------------------------------------------------------------
    k5 = KnowledgeItem(
        item_id="k5",
        job_id=job_id,
        item_type=KnowledgeType.RULE,
        title="Recovered Rule 5",
        statement="Unit 5 statement recovered.",
        validation_status=KnowledgeValidationStatus.VALIDATED,
        start_time=120.0,
        end_time=150.0,
    )

    committed_cascade = restarted_barrier.recover_task_unit(
        job_id=job_id,
        serial_number=5,
        recovered_result={"ok": 5, "recovered": True},
        knowledge_items=[k5],
    )

    # 5, 6, 7 should all be committed in order!
    assert committed_cascade == [5, 6, 7]

    final_state = restarted_barrier.get_barrier_state(job_id)
    assert final_state.highest_committed_serial == 7
    assert final_state.buffered_units == []
    assert final_state.blocked_on_serial is None

    # Verify all task units are now COMMITTED
    for s in [5, 6, 7]:
        unit = restarted_repo.get_task_unit(job_id, s)
        assert unit.status == TaskUnitStatus.COMMITTED

    # Verify all knowledge items (k5, k6, k7) are now in committed knowledge
    committed_knowledge = restarted_repo.get_knowledge(job_id, only_committed=True)
    assert len(committed_knowledge) == 3
    committed_ids = [k.item_id for k in committed_knowledge]
    assert "k5" in committed_ids
    assert "k6" in committed_ids
    assert "k7" in committed_ids
