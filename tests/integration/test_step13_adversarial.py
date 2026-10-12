"""Adversarial and Stress Resilience Tests for Step 13."""
import os
import shutil
import tempfile
import time
import pytest

from trading_ai.schemas.enums import (
    KnowledgeType,
    PipelineStage,
    TaskWatchdogState,
    ThreatLevel,
)
from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.resilience.cyber_wall import CyberWallSandbox
from trading_ai.resilience.threat_detector import ThreatDetector, ThreatScannerAdapter
from trading_ai.resilience.backup_manager import BackupManager
from trading_ai.resilience.watchdog import AntiHangWatchdog
from trading_ai.storage.db import DatabaseManager
from trading_ai.storage.repository import Repository
from trading_ai.supervisor.commit_barrier import CommitBarrier


def test_adversarial_path_traversal_and_sandbox_escape():
    tmp_workspace = tempfile.mkdtemp()
    try:
        cw = CyberWallSandbox(workspace_root=tmp_workspace)

        traversal_attempts = [
            "../../windows/system32/cmd.exe",
            "..\\..\\windows\\system32\\cmd.exe",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "data/../../.env",
            "logs/../../../Program Files",
        ]

        for attempt in traversal_attempts:
            res = cw.validate_path(attempt, action="READ")
            assert res.allowed is False, f"Attempt '{attempt}' must be blocked by Cyber Wall."
            assert res.threat_level in (ThreatLevel.BLOCKED, ThreatLevel.CRITICAL)
    finally:
        shutil.rmtree(tmp_workspace, ignore_errors=True)


def test_adversarial_malicious_script_extension_blocked():
    tmp_workspace = tempfile.mkdtemp()
    try:
        cw = CyberWallSandbox(workspace_root=tmp_workspace)

        malicious_files = [
            os.path.join(tmp_workspace, "data", "exploit.bat"),
            os.path.join(tmp_workspace, "data", "dropper.ps1"),
            os.path.join(tmp_workspace, "data", "payload.exe"),
            os.path.join(tmp_workspace, "data", "malicious.vbs"),
            os.path.join(tmp_workspace, "data", "shell.sh"),
        ]

        for mf in malicious_files:
            res = cw.validate_path(mf, action="WRITE")
            assert res.allowed is False, f"Writing dangerous executable '{mf}' must be blocked."
            assert res.threat_level == ThreatLevel.CRITICAL
    finally:
        shutil.rmtree(tmp_workspace, ignore_errors=True)


def test_adversarial_tampered_backup_detected_and_rejected():
    tmp_dir = tempfile.mkdtemp()
    db = DatabaseManager(":memory:")
    repo = Repository(db)
    try:
        bm = BackupManager(backup_dir=tmp_dir, repository=repo)

        items = [
            KnowledgeItem(
                item_id="k_safe_1",
                job_id="job_adv_01",
                title="Safe Rule",
                item_type=KnowledgeType.RULE,
                statement="Valid trading setup",
                confidence=1.0,
            )
        ]
        meta = bm.create_backup(items=items)

        # Adversary tampers with backup file directly on disk!
        with open(meta.backup_path, "w", encoding="utf-8") as f:
            f.write('[{"item_id": "k_safe_1", "content": "TAMPERED BY ATTACKER"}]')

        # Restore must detect checksum mismatch and reject!
        res = bm.restore_backup(meta.backup_id)
        assert res.success is False
        assert "mismatch" in res.error_message.lower() or "corrupted" in res.error_message.lower()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_adversarial_hung_task_graceful_cancellation():
    wd = AntiHangWatchdog()
    flag = {"cancelled": False}

    def cancel_task():
        flag["cancelled"] = True

    wd.register_task(
        task_id="stuck_worker_01",
        task_name="Hanging Network Job",
        timeout_seconds=0.08,
        cancellation_cb=cancel_task,
    )

    time.sleep(0.12)
    hung = wd.check_for_hangs()
    assert len(hung) == 1
    assert hung[0].task_id == "stuck_worker_01"
    assert hung[0].state == TaskWatchdogState.GRACEFUL_CANCEL
    assert flag["cancelled"] is True


def test_adversarial_crash_recovery_prevents_out_of_order_commit():
    db = DatabaseManager(":memory:")
    repo = Repository(db)
    barrier = CommitBarrier(repo)

    job_id = "job_crash_test_99"

    # Register serial 1, 2, 3
    barrier.register_task_unit(job_id, PipelineStage.AUDIO_PROCESSING, 0.0, 1.0)
    barrier.register_task_unit(job_id, PipelineStage.VISUAL_PROCESSING, 1.0, 2.0)
    barrier.register_task_unit(job_id, PipelineStage.ALIGNMENT, 2.0, 3.0)

    # Unit 1 commits
    barrier.complete_task_unit(job_id, 1, {"status": "ok"})

    # Worker 2 crashes
    barrier.fail_task_unit(job_id, 2, "Hardware crash")

    # Worker 3 finishes while 2 is crashed
    barrier.complete_task_unit(job_id, 3, {"status": "ok"})

    # Serial invariant check: highest committed MUST remain 1!
    b_state = barrier.get_barrier_state(job_id)
    assert b_state.highest_committed_serial == 1
    assert b_state.blocked_on_serial == 2

    # Attempting to directly complete unit 3 again must not bypass barrier
    u3 = repo.get_task_unit(job_id, 3)
    assert u3.status.value == "buffered"
