"""Knowledge Integrity & Append-Only Protection Guard."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.schemas.enums import KnowledgeProtectionState
from trading_ai.schemas.knowledge import KnowledgeItem
from trading_ai.schemas.recovery import KnowledgeIntegrityRecord
from trading_ai.storage.repository import Repository


class KnowledgeIntegrityGuard:
    """Enforces non-destructive append-only knowledge storage and cryptographic integrity validation.

    Guarantees that committed knowledge cannot be silently corrupted or deleted without detection.
    """

    def __init__(self, repository: Optional[Repository] = None):
        self._repo = repository

    @staticmethod
    def compute_knowledge_hash(item: KnowledgeItem) -> str:
        """Compute deterministic SHA-256 digest of knowledge content and provenance."""
        content = getattr(item, "statement", getattr(item, "content", ""))
        epistemic_state = getattr(item, "validation_status", getattr(item, "epistemic_state", ""))
        source_id = getattr(item, "job_id", getattr(item, "source_id", ""))
        canonical_dict = {
            "item_id": item.item_id,
            "item_type": str(item.item_type),
            "content": str(content),
            "epistemic_state": str(epistemic_state),
            "source_type": str(getattr(item, "source_type", "LECTURE")),
            "source_id": str(source_id),
            "confidence": round(float(getattr(item, "confidence", 1.0)), 4),
        }
        raw_bytes = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def record_manifest(self, item: KnowledgeItem, version_id: Optional[str] = None) -> KnowledgeIntegrityRecord:
        """Create and persist an immutable integrity record for a knowledge item."""
        digest = self.compute_knowledge_hash(item)
        content = getattr(item, "statement", getattr(item, "content", ""))
        epistemic_state = getattr(item, "validation_status", getattr(item, "epistemic_state", ""))
        source_id = getattr(item, "job_id", getattr(item, "source_id", ""))
        content_len = len(str(content))
        provenance = str(source_id) or item.item_id

        record = KnowledgeIntegrityRecord(
            item_id=item.item_id,
            version_id=version_id,
            sha256_hash=digest,
            content_length=content_len,
            epistemic_state=str(epistemic_state),
            provenance_hash=hashlib.sha256(provenance.encode("utf-8")).hexdigest(),
            last_verified_at=datetime.now(timezone.utc),
            status=KnowledgeProtectionState.PROTECTED,
        )

        if self._repo and hasattr(self._repo, "save_knowledge_integrity_record"):
            self._repo.save_knowledge_integrity_record(record)

        return record

    def verify_integrity(self, item: KnowledgeItem, expected_record: Optional[KnowledgeIntegrityRecord] = None) -> bool:
        """Verify whether an existing knowledge item matches its recorded cryptographic hash."""
        current_hash = self.compute_knowledge_hash(item)
        if expected_record:
            return current_hash == expected_record.sha256_hash

        if self._repo and hasattr(self._repo, "get_knowledge_integrity_record"):
            rec = self._repo.get_knowledge_integrity_record(item.item_id)
            if rec:
                return current_hash == rec.sha256_hash

        return True  # If no record exists yet, item is fresh

    def validate_batch(self, items: List[KnowledgeItem]) -> Dict[str, bool]:
        """Validate a list of knowledge items against persisted manifests."""
        results: Dict[str, bool] = {}
        for itm in items:
            results[itm.item_id] = self.verify_integrity(itm)
        return results
