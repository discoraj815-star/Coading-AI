"""Step 16 / C2: Vector Store Adapter and SQLite Reference Implementation."""
from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.memory import MemoryEmbedding
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStoreAdapter(ABC):
    """
    Abstract Vector Store interface (C2 foundation).
    Decouples MemoryRetriever and memory systems from concrete databases.
    Enables future plug-and-play migration to Qdrant, Milvus, Chroma, etc.
    """

    @abstractmethod
    def add_embedding(self, embedding: MemoryEmbedding) -> bool:
        """Add a single embedding record."""
        pass

    @abstractmethod
    def add_batch(self, embeddings: List[MemoryEmbedding]) -> int:
        """Add a batch of embeddings."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        min_score: float = 0.0,
        job_id: Optional[str] = None,
    ) -> List[Tuple[MemoryEmbedding, float]]:
        """Search nearest embeddings by cosine similarity."""
        pass

    @abstractmethod
    def delete(self, embedding_id: str) -> bool:
        """Delete an embedding by ID."""
        pass

    @abstractmethod
    def update(self, embedding: MemoryEmbedding) -> bool:
        """Update an existing embedding."""
        pass

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Return backend health and connection statistics."""
        pass

    @abstractmethod
    def count(self, job_id: Optional[str] = None) -> int:
        """Return total count of indexed embeddings."""
        pass

    @abstractmethod
    def snapshot(self, target_path: str) -> bool:
        """Export snapshot of vector store state."""
        pass

    @abstractmethod
    def restore(self, snapshot_path: str) -> bool:
        """Restore vector store state from snapshot."""
        pass


class SQLiteVectorStore(VectorStoreAdapter):
    """
    Production-ready SQLite implementation of VectorStoreAdapter.
    Stores dense vectors in SQLite JSON columns and executes fast in-process cosine re-ranking.
    Authoritative default backend that guarantees zero external dependency overhead.
    """

    def __init__(self, repository: Repository):
        self._repo = repository

    def add_embedding(self, embedding: MemoryEmbedding) -> bool:
        try:
            self._repo.save_memory_embedding(embedding)
            return True
        except Exception as e:
            logger.error("Failed to add embedding to SQLiteVectorStore: %s", e)
            return False

    def add_batch(self, embeddings: List[MemoryEmbedding]) -> int:
        count = 0
        for emb in embeddings:
            if self.add_embedding(emb):
                count += 1
        return count

    def search(
        self,
        query_vector: List[float],
        limit: int = 10,
        min_score: float = 0.0,
        job_id: Optional[str] = None,
    ) -> List[Tuple[MemoryEmbedding, float]]:
        all_embeddings: List[MemoryEmbedding] = []
        if job_id:
            all_embeddings = self._repo.get_memory_embeddings(job_id=job_id, limit=limit * 5)
        else:
            with self._repo._db.connection() as conn:
                rows = conn.execute("SELECT * FROM memory_embeddings ORDER BY importance_score DESC LIMIT ?", (limit * 10,)).fetchall()
                for row in rows:
                    all_embeddings.append(
                        MemoryEmbedding(
                            embedding_id=row["embedding_id"],
                            job_id=row["job_id"],
                            source_type=row["source_type"],
                            source_id=row["source_id"],
                            text_content=row["text_content"],
                            embedding=json.loads(row["embedding_json"]),
                            model_name=row["model_name"],
                            importance_score=row["importance_score"],
                            access_count=row["access_count"],
                            created_at=datetime.fromisoformat(row["created_at"]),
                        )
                    )

        scored: List[Tuple[MemoryEmbedding, float]] = []
        for emb in all_embeddings:
            if not emb.embedding:
                continue
            sim = _cosine_similarity(query_vector, emb.embedding)
            if sim >= min_score:
                scored.append((emb, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def delete(self, embedding_id: str) -> bool:
        try:
            with self._repo._db.connection() as conn:
                conn.execute("DELETE FROM memory_embeddings WHERE embedding_id = ?", (embedding_id,))
                return True
        except Exception as e:
            logger.error("Failed to delete embedding %s: %s", embedding_id, e)
            return False

    def update(self, embedding: MemoryEmbedding) -> bool:
        return self.add_embedding(embedding)

    def health(self) -> Dict[str, Any]:
        total = self.count()
        return {
            "backend": "SQLiteVectorStore",
            "status": "HEALTHY",
            "total_embeddings": total,
            "dimension": 1536,
            "external_dependency": "none",
        }

    def count(self, job_id: Optional[str] = None) -> int:
        with self._repo._db.connection() as conn:
            if job_id:
                row = conn.execute("SELECT COUNT(*) FROM memory_embeddings WHERE job_id = ?", (job_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()
            return row[0] if row else 0

    def snapshot(self, target_path: str) -> bool:
        try:
            with self._repo._db.connection() as conn:
                rows = conn.execute("SELECT * FROM memory_embeddings").fetchall()
                data = [
                    {
                        "embedding_id": r["embedding_id"],
                        "job_id": r["job_id"],
                        "source_type": r["source_type"],
                        "source_id": r["source_id"],
                        "text_content": r["text_content"],
                        "embedding": json.loads(r["embedding_json"]),
                        "model_name": r["model_name"],
                        "importance_score": r["importance_score"],
                        "access_count": r["access_count"],
                        "created_at": r["created_at"],
                    }
                    for r in rows
                ]
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error("Snapshot error: %s", e)
            return False

    def restore(self, snapshot_path: str) -> bool:
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                emb = MemoryEmbedding(
                    embedding_id=item["embedding_id"],
                    job_id=item["job_id"],
                    source_type=item["source_type"],
                    source_id=item["source_id"],
                    text_content=item["text_content"],
                    embedding=item["embedding"],
                    model_name=item["model_name"],
                    importance_score=item.get("importance_score", 1.0),
                    access_count=item.get("access_count", 0),
                    created_at=datetime.fromisoformat(item["created_at"]),
                )
                self.add_embedding(emb)
            return True
        except Exception as e:
            logger.error("Restore error: %s", e)
            return False
