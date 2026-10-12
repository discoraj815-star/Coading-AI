"""
ContextManager — Step 2 Core

Maintains the active working context window for the reasoning engine.
Enforces token budgets, prunes low-priority items, generates compression
summaries, and persists/restores state across restarts via context_snapshots.

Design rules:
  - Uses smallest sufficient context: only items relevant to current serial
  - Compression summaries replace evicted items (not discarded silently)
  - Persisted to DB after every serial commit (crash recovery)
  - Token arithmetic is always pre-checked via TokenManager
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from trading_ai.core.config import Settings, get_settings
from trading_ai.core.token_manager import TokenManager
from trading_ai.schemas import (
    ContextItem,
    ContextItemSource,
    ContextSnapshot,
    ContextWindow,
)

if TYPE_CHECKING:
    from trading_ai.storage.repository import Repository


# ---------------------------------------------------------------------
# Importance scoring helpers
# ---------------------------------------------------------------------

def _default_importance(item: ContextItem) -> float:
    """Default importance scorer: evidence and knowledge score higher."""
    _PRIORITY = {
        ContextItemSource.KNOWLEDGE: 0.90,
        ContextItemSource.EVIDENCE: 0.85,
        ContextItemSource.ALIGNMENT: 0.75,
        ContextItemSource.NOTES: 0.70,
        ContextItemSource.TRANSCRIPT: 0.55,
        ContextItemSource.VISUAL: 0.50,
        ContextItemSource.SUMMARY: 0.40,
    }
    base = _PRIORITY.get(item.source_type, 0.50)
    # Recency bonus: higher serial == more recent == more important
    recency_bonus = min(0.10, item.serial_number * 0.01)
    return min(1.0, base + item.importance_score * 0.10 + recency_bonus)


# ---------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------

class ContextManager:
    """
    Token-budget-aware active context window manager.

    Responsibilities:
      1. Accept new ContextItems from any pipeline stage
      2. Enforce max_tokens hard ceiling (prune by importance)
      3. Maintain a compression_summary of pruned content
      4. Checkpoint to DB on every serial commit
      5. Restore last checkpoint on warm start

    Token Budget Design:
      max_tokens defaults to 80% of TokenManager effective_input_limit so
      ContextManager's window never crowds out the reasoning prompt itself.
    """

    CONTEXT_BUDGET_RATIO: float = 0.80  # fraction of effective_limit for context window

    def __init__(
        self,
        job_id: str,
        repository: Repository,
        token_manager: Optional[TokenManager] = None,
        settings: Optional[Settings] = None,
        max_tokens: Optional[int] = None,
        importance_fn: Optional[Callable[[ContextItem], float]] = None,
    ) -> None:
        self._job_id = job_id
        self._repo = repository
        self._settings = settings or get_settings()
        self._token_manager = token_manager or TokenManager(self._settings)
        self._importance_fn = importance_fn or _default_importance

        # Compute window ceiling
        if max_tokens is not None:
            self._max_tokens = max_tokens
        else:
            effective_limit = self._token_manager.default_budget.effective_input_limit
            self._max_tokens = int(effective_limit * self.CONTEXT_BUDGET_RATIO)

        # Active window (loaded fresh or reconstructed from checkpoint)
        self._window: ContextWindow = ContextWindow(
            job_id=job_id,
            serial_number=1,
            max_tokens=self._max_tokens,
        )

    # ------------------------------------------------------------------
    # Public: Window access
    # ------------------------------------------------------------------

    @property
    def window(self) -> ContextWindow:
        """Return the current active context window (read-only view)."""
        return self._window

    @property
    def current_tokens(self) -> int:
        return self._window.current_tokens

    @property
    def headroom_tokens(self) -> int:
        return self._window.headroom_tokens

    # ------------------------------------------------------------------
    # Public: Add items
    # ------------------------------------------------------------------

    def add_item(self, item: ContextItem) -> bool:
        """
        Attempt to add a ContextItem to the active window.

        Computes token cost if not pre-computed.  If the item fits within
        budget, it is added.  If not, the window is pruned first; the item
        is then added if budget permits, or deferred if it still doesn't fit.

        Returns True if the item was added to the window, False if deferred.
        """
        if item.estimated_tokens == 0:
            item = item.model_copy(
                update={"estimated_tokens": self._token_manager.estimate_text_tokens(item.text_content)}
            )

        if item.estimated_tokens > self._max_tokens:
            # Item is larger than the entire window — always defer
            self._window.deferred.append(item)
            return False

        # Try to fit with current budget
        if self._window.current_tokens + item.estimated_tokens <= self._max_tokens:
            self._window.included.append(item)
            self._window.current_tokens += item.estimated_tokens
            return True

        # Need to prune first
        self._prune_to_fit(item.estimated_tokens)

        if self._window.current_tokens + item.estimated_tokens <= self._max_tokens:
            self._window.included.append(item)
            self._window.current_tokens += item.estimated_tokens
            return True

        self._window.deferred.append(item)
        return False

    def add_items(self, items: List[ContextItem]) -> Tuple[int, int]:
        """
        Batch-add items sorted by importance (highest first).
        Returns (added_count, deferred_count).
        """
        sorted_items = sorted(items, key=self._importance_fn, reverse=True)
        added, deferred = 0, 0
        for item in sorted_items:
            if self.add_item(item):
                added += 1
            else:
                deferred += 1
        return added, deferred

    # ------------------------------------------------------------------
    # Public: Retrieve context for prompt assembly
    # ------------------------------------------------------------------

    def get_context_for_prompt(
        self,
        max_tokens: Optional[int] = None,
        source_types: Optional[List[ContextItemSource]] = None,
        min_importance: float = 0.0,
    ) -> List[ContextItem]:
        """
        Return the subset of included items suitable for injecting into a prompt.

        Optionally filtered by source type or minimum importance score.
        Sorted chronologically (earliest first).
        """
        budget = max_tokens or self._max_tokens
        candidates = self._window.included

        if source_types:
            candidates = [c for c in candidates if c.source_type in source_types]
        if min_importance > 0.0:
            candidates = [c for c in candidates if c.importance_score >= min_importance]

        # Respect budget for this specific call
        result: List[ContextItem] = []
        token_sum = 0
        for item in sorted(candidates, key=lambda x: x.start_time):
            if token_sum + item.estimated_tokens <= budget:
                result.append(item)
                token_sum += item.estimated_tokens
            else:
                break
        return result

    def get_compression_summary(self) -> str:
        """Return the running digest of pruned (evicted) context content."""
        return self._window.compression_summary

    # ------------------------------------------------------------------
    # Public: Serial commit / checkpoint
    # ------------------------------------------------------------------

    def advance_serial(self, new_serial: int) -> None:
        """
        Advance the active window to a new serial number and checkpoint.

        Called by the orchestrator after each CommitBarrier commit.
        Items from previous serials may be downscored for future pruning.
        """
        self._window = self._window.model_copy(update={"serial_number": new_serial})
        self._checkpoint()

    def checkpoint(self) -> ContextSnapshot:
        """Manually trigger a persistence checkpoint and return the snapshot."""
        return self._checkpoint()

    # ------------------------------------------------------------------
    # Public: Restart recovery
    # ------------------------------------------------------------------

    def restore_from_checkpoint(self) -> bool:
        """
        Load the most recent DB snapshot and reconstruct the active window.

        Returns True if a checkpoint was found and restored, False for fresh start.
        """
        snap = self._repo.get_latest_context_snapshot(self._job_id)
        if snap is None:
            return False

        included_items = [
            ContextItem(**d) for d in snap.window_items_json
        ]
        deferred_items = [
            ContextItem(**d) for d in snap.pruned_items_json
        ]
        total_tokens = sum(i.estimated_tokens for i in included_items)

        self._window = ContextWindow(
            job_id=self._job_id,
            serial_number=snap.serial_number,
            max_tokens=self._max_tokens,
            current_tokens=total_tokens,
            included=included_items,
            deferred=deferred_items,
            compression_summary=snap.compression_summary,
        )
        return True

    # ------------------------------------------------------------------
    # Public: Prune / compress
    # ------------------------------------------------------------------

    def force_prune(self, target_tokens: Optional[int] = None) -> int:
        """
        Force pruning of the window to target_tokens (or max_tokens * 0.6).

        Returns the number of tokens freed.
        """
        target = target_tokens or int(self._max_tokens * 0.60)
        before = self._window.current_tokens
        self._prune_to_fit(needed_tokens=self._max_tokens - target)
        return before - self._window.current_tokens

    def set_compression_summary(self, summary: str) -> None:
        """
        Externally set (or append) a compression summary generated by the LLM.

        The LLM-generated digest replaces the pruned content's semantic footprint
        in the context window with minimal token cost.
        """
        if self._window.compression_summary:
            self._window.compression_summary += "\n\n" + summary
        else:
            self._window.compression_summary = summary

        # Update the window token count: add summary tokens, subtract nothing
        # (summary token cost is tracked but deferred items were already removed)
        summary_tokens = self._token_manager.estimate_text_tokens(summary)
        self._window.current_tokens = min(
            self._max_tokens,
            self._window.current_tokens + summary_tokens,
        )

    # ------------------------------------------------------------------
    # Internal: Pruning logic
    # ------------------------------------------------------------------

    def _prune_to_fit(self, needed_tokens: int) -> None:
        """
        Remove lowest-importance items from the window until needed_tokens
        of headroom are available.  Pruned items move to deferred list.
        """
        if not self._window.included:
            return

        # Score all included items; sort ascending (lowest importance first)
        scored = sorted(
            self._window.included,
            key=lambda item: self._importance_fn(item),
        )

        freed = 0
        evicted: List[ContextItem] = []
        while freed < needed_tokens and scored:
            victim = scored.pop(0)
            freed += victim.estimated_tokens
            evicted.append(victim)
            self._window.included.remove(victim)
            self._window.deferred.append(victim)

        self._window.current_tokens = max(0, self._window.current_tokens - freed)

        # Append a mini-digest of what was evicted (without calling LLM)
        if evicted:
            eviction_note = "[Pruned: " + "; ".join(
                f"{e.source_type.value}:{e.item_id[:8]}" for e in evicted[:5]
            )
            if len(evicted) > 5:
                eviction_note += f" +{len(evicted) - 5} more"
            eviction_note += "]"
            if self._window.compression_summary:
                self._window.compression_summary += " " + eviction_note
            else:
                self._window.compression_summary = eviction_note

    # ------------------------------------------------------------------
    # Internal: Checkpoint
    # ------------------------------------------------------------------

    def _checkpoint(self) -> ContextSnapshot:
        """Serialize current window state to a ContextSnapshot and persist it."""
        snap = ContextSnapshot(
            snapshot_id=str(uuid.uuid4()),
            job_id=self._job_id,
            serial_number=self._window.serial_number,
            window_tokens=self._window.current_tokens,
            window_items_json=[item.model_dump() for item in self._window.included],
            pruned_items_json=[item.model_dump() for item in self._window.deferred],
            compression_summary=self._window.compression_summary,
        )
        self._repo.save_context_snapshot(snap)
        return snap

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        """Return a human-readable status dict for logging / API exposure."""
        return {
            "job_id": self._job_id,
            "serial_number": self._window.serial_number,
            "current_tokens": self._window.current_tokens,
            "max_tokens": self._max_tokens,
            "headroom_tokens": self._window.headroom_tokens,
            "utilization_pct": self._window.utilization_pct,
            "included_items": len(self._window.included),
            "deferred_items": len(self._window.deferred),
            "has_compression_summary": bool(self._window.compression_summary),
        }
