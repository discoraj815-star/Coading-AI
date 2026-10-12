"""
Continuous Token-Resilient Journal Processing for Step 10.
Manages bounded batch processing, token limits, checkpoints, crash recovery, and restart resumption.
Guarantees the system never halts permanently due to Gemini context limits or process crashes.
"""
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime, timezone
import json
import logging

from trading_ai.schemas.journal import (
    JournalProcessingCheckpoint,
    TradeJournalEntry,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class JournalResilienceManager:
    """
    Orchestrates batch journal processing with checkpoint persistence, token budgeting,
    and automatic recovery across restarts.
    """

    def __init__(self, repository: Repository, token_budget_per_batch: int = 16000):
        self.repository = repository
        self.token_budget_per_batch = token_budget_per_batch

    def process_journal_batch(
        self,
        job_id: str,
        entries: List[TradeJournalEntry],
        processor_fn: Callable[[TradeJournalEntry], Dict[str, Any]],
    ) -> JournalProcessingCheckpoint:
        """
        Processes a list of journal entries incrementally, updating checkpoint after each entry.
        Can be safely interrupted and resumed from the last successfully processed entry.
        """
        # 1. Check for existing checkpoint
        checkpoint = self.repository.get_journal_checkpoint(job_id)
        start_index = 0
        cumulative_tokens = 0

        if checkpoint:
            start_index = checkpoint.processed_count
            cumulative_tokens = checkpoint.tokens_consumed
            logger.info(f"Resuming journal job {job_id} from index {start_index} (checkpointed at {checkpoint.last_checkpoint_at})")
        else:
            checkpoint = JournalProcessingCheckpoint(
                job_id=job_id,
                total_entries=len(entries),
                processed_count=0,
                last_processed_journal_id=None,
                tokens_consumed=0,
                status="RUNNING",
                last_checkpoint_at=datetime.now(timezone.utc),
                metadata={},
            )
            self.repository.save_journal_checkpoint(checkpoint)

        # 2. Iterate from start_index
        for idx in range(start_index, len(entries)):
            entry = entries[idx]

            # Bounded context token estimation (~400 tokens per journal entry summary)
            estimated_tokens = 400
            if cumulative_tokens + estimated_tokens > self.token_budget_per_batch:
                # Rollover context boundary: save checkpoint and mark paused for rollover
                checkpoint.status = "PAUSED_CONTEXT_ROLLOVER"
                checkpoint.last_checkpoint_at = datetime.now(timezone.utc)
                self.repository.save_journal_checkpoint(checkpoint)
                logger.info(f"Context budget reached for job {job_id}. Rolling over at entry {entry.journal_id}")
                return checkpoint

            try:
                # Execute single entry processing
                result = processor_fn(entry)
                cumulative_tokens += estimated_tokens

                # Advance checkpoint
                checkpoint.processed_count = idx + 1
                checkpoint.last_processed_journal_id = entry.journal_id
                checkpoint.tokens_consumed = cumulative_tokens
                checkpoint.last_checkpoint_at = datetime.now(timezone.utc)
                self.repository.save_journal_checkpoint(checkpoint)

            except Exception as e:
                logger.error(f"Error processing journal entry {entry.journal_id}: {e}")
                checkpoint.status = "FAILED"
                checkpoint.metadata["error"] = str(e)
                checkpoint.last_checkpoint_at = datetime.now(timezone.utc)
                self.repository.save_journal_checkpoint(checkpoint)
                raise e

        # 3. Mark completed
        checkpoint.status = "COMPLETED"
        checkpoint.last_checkpoint_at = datetime.now(timezone.utc)
        self.repository.save_journal_checkpoint(checkpoint)
        return checkpoint

    def get_job_status(self, job_id: str) -> Optional[JournalProcessingCheckpoint]:
        return self.repository.get_journal_checkpoint(job_id)
