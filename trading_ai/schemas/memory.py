"""Step 2 schemas: MemoryEmbedding and MemorySearchResult."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryEmbedding(BaseModel):
    """
    Persisted embedding record for a knowledge item, evidence trace, or note entry.

    Stored in `memory_embeddings` table. Enables semantic (cosine-similarity)
    search across all job knowledge without re-embedding on every query.
    """

    embedding_id: str = Field(description="Unique embedding record identifier")
    job_id: str
    source_type: str = Field(
        description="Logical type: 'knowledge', 'evidence', 'transcript', 'visual'"
    )
    source_id: str = Field(description="Primary key of the source artifact in its own table")
    text_content: str = Field(description="Original text that was embedded")
    embedding: List[float] = Field(
        default_factory=list,
        description="Dense embedding vector from gemini-embedding-2 (natively multimodal)"
    )
    model_name: str = Field(
        default="gemini-embedding-2",
        description="Embedding model used (from ModelRouter EMBEDDING role)"
    )
    importance_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Priority hint for pruning and context inclusion decisions"
    )
    access_count: int = Field(
        default=0,
        description="Times this embedding has been returned in a retrieval result (for LRU/LFU)"
    )
    last_accessed_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"json_encoders": {}}


class MemorySearchResult(BaseModel):
    """
    Single result from MemoryRetriever search.

    Carries the embedding record plus a similarity score for ranking.
    """

    embedding: MemoryEmbedding
    similarity_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Cosine similarity to the query embedding (1.0 = perfect match)"
    )
    rank: int = Field(default=1, description="Position in the result list (1-indexed)")
    retrieval_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Debug / traceability data (query text, filter applied, search strategy)"
    )
