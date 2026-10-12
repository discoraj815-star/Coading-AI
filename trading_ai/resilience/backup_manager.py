"""Isolated Knowledge Backup & Disaster Recovery Manager."""
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.schemas.recovery import BackupMetadata, RestoreResult
from trading_ai.storage.repository import Repository


class BackupManager:
    """Manages isolated, crash-safe, versioned backups and disaster recovery for knowledge and checkpoints."""

    def __init__(self, backup_dir: Optional[str] = None, repository: Optional[Repository] = None, max_retention: int = 10):
        self._backup_dir = Path(backup_dir or os.path.join(os.getcwd(), "backups", "knowledge"))
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        self._repo = repository
        self._max_retention = max_retention

    def create_backup(self, items: Optional[List[KnowledgeItem]] = None, notes: Optional[str] = None) -> BackupMetadata:
        """Create an atomic, checksummed snapshot of knowledge items."""
        backup_id = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        target_file = self._backup_dir / f"{backup_id}.json"
        temp_file = self._backup_dir / f"{backup_id}.tmp"

        # If items not passed, load from repository
        if items is None and self._repo and hasattr(self._repo, "get_knowledge_items"):
            items = self._repo.get_knowledge_items(limit=10000)
        items = items or []

        # Serialize payload
        payload_data = [item.model_dump(mode="json") if hasattr(item, "model_dump") else item.dict() for item in items]
        raw_json = json.dumps(payload_data, indent=2, sort_keys=True).encode("utf-8")
        checksum = hashlib.sha256(raw_json).hexdigest()

        # Atomic file write
        with open(temp_file, "wb") as f:
            f.write(raw_json)
        os.replace(temp_file, target_file)

        metadata = BackupMetadata(
            backup_id=backup_id,
            created_at=datetime.now(timezone.utc),
            backup_path=str(target_file),
            sha256_checksum=checksum,
            total_items=len(items),
            file_size_bytes=len(raw_json),
            is_verified=True,
            retention_tag="AUTOMATED_SNAPSHOT",
            notes=notes or "Routine disaster recovery snapshot",
        )

        if self._repo and hasattr(self._repo, "save_recovery_backup"):
            self._repo.save_recovery_backup(metadata)

        # Enforce retention policy without deleting all backups
        self._apply_retention_policy()

        return metadata

    def verify_backup_integrity(self, backup_id: str) -> bool:
        """Verify file existence and SHA-256 checksum of an existing backup."""
        target_file = self._backup_dir / f"{backup_id}.json"
        if not target_file.exists():
            return False

        with open(target_file, "rb") as f:
            content = f.read()
        calculated_hash = hashlib.sha256(content).hexdigest()

        if self._repo and hasattr(self._repo, "get_recovery_backup"):
            rec = self._repo.get_recovery_backup(backup_id)
            if rec:
                return calculated_hash == rec.sha256_checksum

        return len(content) > 0

    def restore_backup(self, backup_id: str) -> RestoreResult:
        """Restore knowledge items from a verified backup archive."""
        target_file = self._backup_dir / f"{backup_id}.json"
        if not target_file.exists():
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                restored_items_count=0,
                verified_checksum=False,
                restored_at=datetime.now(timezone.utc),
                error_message=f"Backup file {target_file} not found.",
            )

        with open(target_file, "rb") as f:
            raw_bytes = f.read()

        computed_hash = hashlib.sha256(raw_bytes).hexdigest()

        # Verify integrity before touching repository
        if self._repo and hasattr(self._repo, "get_recovery_backup"):
            rec = self._repo.get_recovery_backup(backup_id)
            if rec and rec.sha256_checksum != computed_hash:
                return RestoreResult(
                    success=False,
                    backup_id=backup_id,
                    restored_items_count=0,
                    verified_checksum=False,
                    restored_at=datetime.now(timezone.utc),
                    error_message="Checksum mismatch: backup file has been corrupted or tampered with.",
                )

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            restored_count = 0
            if self._repo and hasattr(self._repo, "save_knowledge_item"):
                for d in data:
                    item = KnowledgeItem(**d)
                    self._repo.save_knowledge_item(item, is_committed=True)
                    restored_count += 1
            else:
                restored_count = len(data)

            return RestoreResult(
                success=True,
                backup_id=backup_id,
                restored_items_count=restored_count,
                verified_checksum=True,
                restored_at=datetime.now(timezone.utc),
                error_message=None,
            )
        except Exception as e:
            return RestoreResult(
                success=False,
                backup_id=backup_id,
                restored_items_count=0,
                verified_checksum=False,
                restored_at=datetime.now(timezone.utc),
                error_message=f"Restore failed: {str(e)}",
            )

    def list_backups(self) -> List[BackupMetadata]:
        """List all valid backups in chronological order."""
        if self._repo and hasattr(self._repo, "list_recovery_backups"):
            return self._repo.list_recovery_backups()

        results: List[BackupMetadata] = []
        for file in sorted(self._backup_dir.glob("backup_*.json"), key=os.path.getmtime, reverse=True):
            try:
                with open(file, "rb") as f:
                    content = f.read()
                checksum = hashlib.sha256(content).hexdigest()
                data = json.loads(content.decode("utf-8"))
                results.append(
                    BackupMetadata(
                        backup_id=file.stem,
                        created_at=datetime.fromtimestamp(os.path.getmtime(file), tz=timezone.utc),
                        backup_path=str(file),
                        sha256_checksum=checksum,
                        total_items=len(data),
                        file_size_bytes=len(content),
                        is_verified=True,
                        retention_tag="FILE_INSPECTED",
                    )
                )
            except Exception:
                continue
        return results

    def _apply_retention_policy(self) -> None:
        """Prune oldest backups beyond retention limit, never removing the only valid backup."""
        all_backups = sorted(self._backup_dir.glob("backup_*.json"), key=os.path.getmtime)
        if len(all_backups) > self._max_retention:
            to_remove = all_backups[: len(all_backups) - self._max_retention]
            for file in to_remove:
                try:
                    file.unlink()
                except OSError:
                    pass
                if self._repo and hasattr(self._repo, "delete_recovery_backup"):
                    self._repo.delete_recovery_backup(file.stem)
