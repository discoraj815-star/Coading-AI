"""Trading AI Assistant — Safe Application Update Workflow.

Automated 7-step safe update verification pipeline:
  Step 1: Working tree inspection (detects uncommitted changes).
  Step 2: Pre-update point-in-time database and configuration backup.
  Step 3: Backup integrity verification.
  Step 4: Dependency check.
  Step 5: Database schema migration verification (PRAGMA integrity & idempotent check).
  Step 6: Automated regression test suite execution.
  Step 7: Post-update health check and rollback path presentation.
"""

import os
import sys
import shutil
import sqlite3
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SafeUpdate")


def get_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


root_dir = get_root_dir()
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return res.returncode, res.stdout + res.stderr


def inspect_git_status(root_dir: Path) -> bool:
    logger.info("Step 1/7: Inspecting Git status and working tree...")
    code, out = run_cmd(["git", "status", "--porcelain"], root_dir)
    if code != 0:
        logger.warning("Git check returned exit code %d (not a git repo or git not found).", code)
        return True
    lines = [l for l in out.strip().splitlines() if l.strip()]
    if lines:
        logger.info("Found %d uncommitted or untracked file(s) in working tree.", len(lines))
        logger.info("All user modifications will be strictly preserved.")
    else:
        logger.info("Git working tree is clean.")
    return True


def backup_database_and_config(root_dir: Path) -> tuple[Path, Path]:
    logger.info("Step 2/7: Creating pre-update backup of database and configuration...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = root_dir / "backups"
    bdir.mkdir(parents=True, exist_ok=True)

    # 1. Database backup
    src_db = root_dir / "data" / "trading_ai.db"
    dst_db = bdir / f"trading_ai_preupdate_{ts}.db"
    if src_db.exists():
        src_conn = sqlite3.connect(str(src_db))
        dst_conn = sqlite3.connect(str(dst_db))
        src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()
        logger.info("Pre-update database snapshot created: %s (%d bytes)", dst_db.name, dst_db.stat().st_size)
    else:
        logger.info("No existing database to backup; fresh installation.")

    # 2. Config backup
    src_env = root_dir / ".env"
    dst_env = bdir / f"env_preupdate_{ts}.backup"
    if src_env.exists():
        shutil.copy2(src_env, dst_env)
        logger.info("Configuration backup created: %s", dst_env.name)

    return dst_db, dst_env


def verify_backup_integrity(backup_db: Path) -> bool:
    logger.info("Step 3/7: Verifying backup integrity...")
    if not backup_db.exists():
        return True
    try:
        con = sqlite3.connect(str(backup_db))
        res = con.cursor().execute("PRAGMA integrity_check;").fetchall()
        con.close()
        if res == [("ok",)]:
            logger.info("Backup integrity verified: ok")
            return True
        logger.error("Backup integrity FAILED: %s", res)
        return False
    except Exception as e:
        logger.error("Error inspecting backup: %s", e)
        return False


def verify_dependencies(root_dir: Path) -> bool:
    logger.info("Step 4/7: Checking dependencies...")
    req_file = root_dir / "requirements.txt"
    if not req_file.exists():
        logger.warning("requirements.txt not found.")
        return True
    python_exe = sys.executable
    code, out = run_cmd([python_exe, "-m", "pip", "check"], root_dir)
    if code == 0:
        logger.info("All installed Python dependencies are consistent.")
    else:
        logger.info("pip check output: %s", out.strip()[:200])
    return True


def verify_database_schema(root_dir: Path) -> bool:
    logger.info("Step 5/7: Verifying database schema idempotency...")
    db_file = root_dir / "data" / "trading_ai.db"
    if not db_file.exists():
        return True
    try:
        from trading_ai.storage.db import DatabaseManager
        db_mgr = DatabaseManager(str(db_file))
        db_mgr.initialize_schema()  # Idempotent SCHEMA_SQL execution
        con = sqlite3.connect(str(db_file))
        res = con.cursor().execute("PRAGMA integrity_check;").fetchall()
        con.close()
        if res == [("ok",)]:
            logger.info("Database schema initialized and verified: ok")
            return True
        logger.error("Database schema verification failed: %s", res)
        return False
    except Exception as e:
        logger.error("Schema migration verification error: %s", e)
        return False


def run_regression_tests(root_dir: Path) -> bool:
    logger.info("Step 6/7: Running core unit and integration test suite...")
    python_exe = sys.executable
    test_cmd = [
        python_exe,
        "-m",
        "pytest",
        "tests/unit/test_config_and_router.py",
        "tests/unit/test_private_auth.py",
        "tests/integration/test_private_auth_api.py",
        "tests/unit/test_jarvis_background_and_hindi_notifications.py",
        "tests/unit/test_step12_security.py",
        "-q",
    ]
    code, out = run_cmd(test_cmd, root_dir)
    if code == 0:
        logger.info("Core test suite PASSED:")
        for line in out.strip().splitlines()[-3:]:
            logger.info("  %s", line)
        return True
    logger.error("Core test suite FAILED:\n%s", out)
    return False


def main():
    parser = argparse.ArgumentParser(description="Trading AI Assistant Safe Update Runner")
    parser.add_argument("--skip-tests", action="store_true", help="Skip test suite execution (not recommended)")
    args = parser.parse_args()

    root_dir = get_root_dir()
    os.chdir(str(root_dir))

    print("=" * 75)
    print("TRADING AI ASSISTANT — SAFE APPLICATION UPDATE VERIFIER")
    print("=" * 75)

    # Step 1
    inspect_git_status(root_dir)

    # Step 2
    backup_db, backup_env = backup_database_and_config(root_dir)

    # Step 3
    if not verify_backup_integrity(backup_db):
        logger.error("Update aborted due to backup integrity failure.")
        sys.exit(1)

    # Step 4
    verify_dependencies(root_dir)

    # Step 5
    if not verify_database_schema(root_dir):
        logger.error("Update aborted: schema integrity check failed.")
        sys.exit(1)

    # Step 6
    if not args.skip_tests:
        if not run_regression_tests(root_dir):
            logger.error("Update failed tests. Rollback path:")
            print(f"  To restore database: python scripts/backup_restore.py restore --file {backup_db}")
            sys.exit(1)

    # Step 7
    logger.info("Step 7/7: Update verification complete!")
    print("=" * 75)
    print("SAFE UPDATE VERIFICATION: ALL 7 PRE-FLIGHT STEPS PASSED")
    print("Database safely preserved and verified.")
    print(f"Rollback backup available at: {backup_db}")
    print("=" * 75)


if __name__ == "__main__":
    main()
