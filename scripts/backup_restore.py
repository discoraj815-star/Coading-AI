"""Trading AI Assistant — Database Backup and Safe Point-in-Time Recovery Utility.

Commands:
  python scripts/backup_restore.py backup [--label NAME]
  python scripts/backup_restore.py verify [--file PATH]
  python scripts/backup_restore.py restore --file PATH
  python scripts/backup_restore.py list
"""

import os
import sys
import shutil
import sqlite3
import logging
import argparse
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BackupRestore")


def get_root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def get_db_path() -> Path:
    return get_root_dir() / "data" / "trading_ai.db"


def get_backups_dir() -> Path:
    bdir = get_root_dir() / "backups"
    bdir.mkdir(parents=True, exist_ok=True)
    return bdir


def verify_integrity(db_file: Path) -> bool:
    if not db_file.exists():
        logger.error("File does not exist: %s", db_file)
        return False
    try:
        con = sqlite3.connect(str(db_file))
        res = con.cursor().execute("PRAGMA integrity_check;").fetchall()
        con.close()
        is_ok = res == [("ok",)]
        if is_ok:
            logger.info("Integrity check OK for %s", db_file.name)
        else:
            logger.error("Integrity check FAILED for %s: %s", db_file.name, res)
        return is_ok
    except Exception as e:
        logger.error("Error reading database %s: %s", db_file.name, e)
        return False


def create_backup(label: str = "manual") -> Path:
    src_path = get_db_path()
    if not src_path.exists():
        logger.error("Source database %s does not exist.", src_path)
        sys.exit(1)

    if not verify_integrity(src_path):
        logger.error("Cannot back up corrupted database.")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = get_backups_dir() / f"trading_ai_{label}_{ts}.db"

    logger.info("Performing online atomic backup to %s...", backup_file.name)
    src_conn = sqlite3.connect(str(src_path))
    dst_conn = sqlite3.connect(str(backup_file))
    src_conn.backup(dst_conn)
    dst_conn.close()
    src_conn.close()

    if not verify_integrity(backup_file):
        logger.error("Created backup failed integrity verification!")
        backup_file.unlink(missing_ok=True)
        sys.exit(1)

    logger.info("Backup successfully created and verified: %s (%d bytes)", backup_file, backup_file.stat().st_size)
    return backup_file


def restore_backup(backup_file: Path) -> None:
    target_path = get_db_path()

    if not backup_file.exists():
        logger.error("Backup file not found: %s", backup_file)
        sys.exit(1)

    logger.info("Verifying integrity of backup file before restore...")
    if not verify_integrity(backup_file):
        logger.error("Selected backup is corrupted. Restoration aborted.")
        sys.exit(1)

    # Pre-restore safety backup of active database
    if target_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safety_path = get_backups_dir() / f"trading_ai_prerestore_safety_{ts}.db"
        logger.info("Creating safety copy of current database: %s", safety_path.name)
        shutil.copy2(target_path, safety_path)

    # Perform restoration
    logger.info("Restoring database from %s to %s...", backup_file.name, target_path)
    src_conn = sqlite3.connect(str(backup_file))
    dst_conn = sqlite3.connect(str(target_path))
    src_conn.backup(dst_conn)
    dst_conn.close()
    src_conn.close()

    # Post-restore integrity check
    if not verify_integrity(target_path):
        logger.error("Post-restore verification failed! Restoring safety backup...")
        shutil.copy2(safety_path, target_path)
        sys.exit(1)

    logger.info("Database successfully restored and verified from %s.", backup_file.name)


def list_backups() -> None:
    bdir = get_backups_dir()
    backups = sorted(bdir.glob("*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        print("No backups found in backups/ directory.")
        return

    print("=" * 75)
    print("AVAILABLE DATABASE BACKUPS")
    print("=" * 75)
    print(f"{'Filename':<42} {'Size (KB)':<12} {'Created At'}")
    print("-" * 75)
    for b in backups:
        sz = b.stat().st_size // 1024
        dt = datetime.fromtimestamp(b.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{b.name:<42} {sz:<12} {dt}")
    print("-" * 75)


def main():
    parser = argparse.ArgumentParser(description="Trading AI Assistant Backup and Recovery")
    subparsers = parser.add_subparsers(dest="action")

    # Backup command
    bk_parser = subparsers.add_parser("backup", help="Create an atomic point-in-time backup")
    bk_parser.add_argument("--label", default="manual", help="Label prefix for the backup filename")

    # Verify command
    vf_parser = subparsers.add_parser("verify", help="Verify database integrity")
    vf_parser.add_argument("--file", help="Path to database or backup to verify (defaults to active db)")

    # Restore command
    rs_parser = subparsers.add_parser("restore", help="Restore database from an existing verified backup")
    rs_parser.add_argument("--file", required=True, help="Path to backup file to restore")

    # List command
    subparsers.add_parser("list", help="List all available backups")

    args = parser.parse_args()

    if args.action == "backup":
        create_backup(label=args.label)
    elif args.action == "verify":
        target = Path(args.file) if args.file else get_db_path()
        ok = verify_integrity(target)
        sys.exit(0 if ok else 1)
    elif args.action == "restore":
        restore_backup(Path(args.file))
    elif args.action == "list":
        list_backups()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
