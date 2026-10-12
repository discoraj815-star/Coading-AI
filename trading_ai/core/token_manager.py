"""TokenManager for token estimation, safety margin enforcement, context optimization, dynamic budgeting, and usage tracking."""

from enum import Enum, IntEnum
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar
from pydantic import BaseModel, Field
from trading_ai.core.config import Settings, get_settings

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Model Capacity Registry
# ---------------------------------------------------------------------------
MODEL_CONTEXT_CAPACITIES: Dict[str, int] = {
    # Gemini 3.8 Flash (Flagship fast multimodal, 1M context capacity)
    "gemini-3.8-flash": 1_048_576,
    # Gemini 3.1 Pro Preview (Deep reasoning, 1M context capacity)
    "gemini-3.1-pro-preview": 1_048_576,
    # Gemini Embedding 2
    "gemini-embedding-2": 8_192,
    # Legacy / Alternative Fallbacks
    "gemini-1.5-pro": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
}
DEFAULT_SAFE_MODEL_CAPACITY: int = 32_768


class TokenBudgetTier(str, Enum):
    """Strongly typed dynamic token budget tiers."""

    SMALL = "SMALL"        # Simple UI inquiries, app guide lookup, greetings (~4k input, ~2k output)
    MEDIUM = "MEDIUM"      # Market status, general chat, knowledge query (~15k input, ~3k output)
    LARGE = "LARGE"        # Deep MTF analysis, multi-rule evaluation (~30k input, ~4k output)
    MAX_SAFE = "MAX_SAFE"  # Lecture analysis, deep multimodal synthesis, complex conflict resolution (~60k input, ~8k output)


class RequestComplexity(str, Enum):
    """Deterministic classification of user request complexity."""

    SIMPLE_UI = "SIMPLE_UI"
    APPLICATION_GUIDANCE = "APPLICATION_GUIDANCE"
    GENERAL_CHAT = "GENERAL_CHAT"
    MARKET_STATUS = "MARKET_STATUS"
    MARKET_ANALYSIS = "MARKET_ANALYSIS"
    KNOWLEDGE_QUERY = "KNOWLEDGE_QUERY"
    LECTURE_ANALYSIS = "LECTURE_ANALYSIS"
    MULTIMODAL_ANALYSIS = "MULTIMODAL_ANALYSIS"
    DEEP_REASONING = "DEEP_REASONING"


class EscalationReason(str, Enum):
    """Reasons why a token budget tier was escalated."""

    CONTEXT_SIZE = "CONTEXT_SIZE"
    RETRIEVED_EVIDENCE = "RETRIEVED_EVIDENCE"
    COMPLEX_REASONING = "COMPLEX_REASONING"
    MULTIMODAL_DATA = "MULTIMODAL_DATA"
    OUTPUT_RESERVE_INSUFFICIENT = "OUTPUT_RESERVE_INSUFFICIENT"


class ContextPriority(IntEnum):
    """Relative priority of context pieces for packing and knapsack selection."""

    CRITICAL = 100   # User prompt, active symbol, user correction, explicit rule, active setup
    HIGH = 75       # Market current situation, MDE confluence, key trading rules
    MEDIUM = 50     # Recent conversation, relevant knowledge items, AppGuide entries
    LOW = 25        # Old conversation turns, stale market notes, duplicate entries


class ContextItem(BaseModel):
    """A prioritized piece of context with token cost and metadata."""

    item_id: str
    text: str
    priority: ContextPriority = ContextPriority.MEDIUM
    importance: float = 1.0  # 0.0 to 1.0
    recency: float = 1.0     # 0.0 to 1.0
    token_cost: int = 0
    source: str = "general"


class TokenBudget(BaseModel):
    """Budget limits and safety margins for model interactions."""

    tier: Optional[TokenBudgetTier] = None
    input_budget: int = Field(default=30000, description="Maximum total input token budget")
    output_reserve: int = Field(default=4000, description="Tokens reserved for model response generation")
    safety_margin_percent: float = Field(
        default=0.20, ge=0.05, le=0.50, description="Safety headroom percentage (e.g. 0.20 = 20% reserved)"
    )

    @property
    def effective_input_limit(self) -> int:
        """The maximum allowable input tokens after deducting safety margin."""
        margin = int(self.input_budget * self.safety_margin_percent)
        return max(100, self.input_budget - margin)


class BudgetCheckResult(BaseModel):
    """Result of pre-flight token budget evaluation."""

    is_sufficient: bool
    estimated_tokens: int
    effective_limit: int
    headroom: int
    overage: int = 0
    message: str
    tier: Optional[TokenBudgetTier] = None


class TokenUsageRecord(BaseModel):
    """Granular log record of token consumption."""

    job_id: str
    serial_number: int = 1
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cached_tokens: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TokenDiagnostics(BaseModel):
    """Safe, secret-free diagnostic summary of token evaluation."""

    model: str
    tier: TokenBudgetTier
    model_capacity: int
    estimated_input_tokens: int
    output_reserve: int
    safety_margin_percent: float
    effective_input_limit: int
    headroom: int
    compression_performed: bool = False
    dropped_items_count: int = 0
    escalation_count: int = 0


# ---------------------------------------------------------------------------
# Structured Conversation Compressor
# ---------------------------------------------------------------------------
class ConversationCompressor:
    """
    Compresses long conversation history threads into token-efficient structured summaries.
    Preserves critical invariants: active symbols, active setups, user rules, and corrections.
    """

    @staticmethod
    def should_compress(turn_count: int, estimated_tokens: int, max_turns_threshold: int = 4, token_threshold: int = 1500) -> bool:
        """Determine if history should be compressed."""
        return turn_count > max_turns_threshold or estimated_tokens > token_threshold

    @staticmethod
    def extract_structured_summary(
        turns: List[Dict[str, Any]],
        active_symbol: str = "EUR/USD",
        active_timeframe: str = "15m",
    ) -> str:
        """
        Produce a structured deterministic summary of older turns without losing critical facts.
        """
        if not turns:
            return "No previous conversation context."

        user_goals: List[str] = []
        rules_saved: List[str] = []
        user_corrections: List[str] = []
        key_decisions: List[str] = []
        unresolved_queries: List[str] = []

        for turn in turns:
            role = turn.get("role", "user")
            text = turn.get("text", "") or turn.get("user_message", "") or turn.get("ai_message", "")
            t_lower = text.lower()

            if role == "user":
                if any(k in t_lower for k in ["rule", "save rule", "remember"]):
                    rules_saved.append(text[:120])
                elif any(k in t_lower for k in ["wrong", "correction", "correct this", "not correct"]):
                    user_corrections.append(text[:120])
                elif any(k in t_lower for k in ["analyze", "look at", "check", "what about", "how"]):
                    user_goals.append(text[:100])
                elif "?" in text:
                    unresolved_queries.append(text[:100])
            else:
                if any(k in t_lower for k in ["confluence", "bullish", "bearish", "decision", "invalid"]):
                    # Capture high level assistant finding
                    snippet = text.split("\n")[0][:120]
                    if snippet and snippet not in key_decisions:
                        key_decisions.append(snippet)

        lines = [
            "=== STRUCTURED CONVERSATION SUMMARY ===",
            f"• Active Symbol & Timeframe: {active_symbol} ({active_timeframe})",
        ]
        if user_goals:
            lines.append(f"• User Focus: {'; '.join(user_goals[-2:])}")
        if rules_saved:
            lines.append(f"• Active Rules Highlighted: {'; '.join(rules_saved[-2:])}")
        if user_corrections:
            lines.append(f"• User Corrections Preserved: {'; '.join(user_corrections[-2:])}")
        if key_decisions:
            lines.append(f"• Key Findings: {'; '.join(key_decisions[-2:])}")
        if unresolved_queries:
            lines.append(f"• Pending Points: {'; '.join(unresolved_queries[-2:])}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TokenManager Core
# ---------------------------------------------------------------------------
class TokenManager:
    """Manages token budgeting, estimation, context pruning, dynamic tiers, and consumption observability."""

    # Default Budget Tiers Definitions
    TIER_DEFAULTS: Dict[TokenBudgetTier, Dict[str, Any]] = {
        TokenBudgetTier.SMALL: {
            "input_budget": 4_000,
            "output_reserve": 2_000,
            "safety_margin_percent": 0.20,
        },
        TokenBudgetTier.MEDIUM: {
            "input_budget": 15_000,
            "output_reserve": 3_000,
            "safety_margin_percent": 0.20,
        },
        TokenBudgetTier.LARGE: {
            "input_budget": 30_000,
            "output_reserve": 4_000,
            "safety_margin_percent": 0.20,
        },
        TokenBudgetTier.MAX_SAFE: {
            "input_budget": 60_000,
            "output_reserve": 8_000,
            "safety_margin_percent": 0.20,
        },
    }

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._default_budget = TokenBudget(
            tier=TokenBudgetTier.LARGE,
            input_budget=self._settings.token_default_input_budget,
            output_reserve=self._settings.token_default_output_reserve,
            safety_margin_percent=self._settings.token_safety_margin_percent,
        )
        self._usage_history: List[TokenUsageRecord] = []
        self._compressor = ConversationCompressor()

    @property
    def default_budget(self) -> TokenBudget:
        return self._default_budget

    @property
    def compressor(self) -> ConversationCompressor:
        return self._compressor

    # -----------------------------------------------------------------------
    # Capacity & Budget Tiers
    # -----------------------------------------------------------------------
    def get_model_capacity(self, model_identifier: str) -> int:
        """Resolve actual or configured context capacity for a model ID or role."""
        if not model_identifier:
            return DEFAULT_SAFE_MODEL_CAPACITY

        m_clean = model_identifier.lower().strip()
        for known_model, cap in MODEL_CONTEXT_CAPACITIES.items():
            if known_model in m_clean or m_clean in known_model:
                return cap

        return DEFAULT_SAFE_MODEL_CAPACITY

    def get_budget_for_tier(
        self,
        tier: TokenBudgetTier,
        model_capacity: Optional[int] = None,
        custom_safety_margin: Optional[float] = None,
    ) -> TokenBudget:
        """Construct a TokenBudget for a specific tier, bounded safely by model capacity."""
        defaults = self.TIER_DEFAULTS.get(tier, self.TIER_DEFAULTS[TokenBudgetTier.MEDIUM])
        safety_margin = custom_safety_margin or defaults["safety_margin_percent"]

        input_budget = defaults["input_budget"]
        output_reserve = defaults["output_reserve"]

        # Ensure tier input budget never exceeds model capacity minus reserve
        if model_capacity is not None and model_capacity > 0:
            usable = int(model_capacity * (1.0 - safety_margin))
            output_reserve = min(output_reserve, max(100, usable // 4))
            max_allowable_input = usable - output_reserve
            input_budget = min(input_budget, max(100, max_allowable_input))

        return TokenBudget(
            tier=tier,
            input_budget=input_budget,
            output_reserve=output_reserve,
            safety_margin_percent=safety_margin,
        )

    def calculate_available_input(
        self,
        model_capacity: int,
        system_instruction_tokens: int = 0,
        tool_overhead_tokens: int = 0,
        output_reserve: int = 2000,
        safety_margin_percent: float = 0.20,
    ) -> int:
        """
        Calculate available input tokens:
        MODEL_CAPACITY * (1 - SAFETY_MARGIN) - SYSTEM_PROMPT - TOOL_OVERHEAD - OUTPUT_RESERVE
        """
        effective_capacity = int(model_capacity * (1.0 - safety_margin_percent))
        available = effective_capacity - system_instruction_tokens - tool_overhead_tokens - output_reserve
        return max(0, available)

    def classify_request_complexity(
        self,
        query: str,
        has_multimodal: bool = False,
        is_lecture: bool = False,
    ) -> RequestComplexity:
        """Deterministically classify request complexity to assign initial budget tier."""
        q_lower = query.lower().strip()

        if is_lecture:
            return RequestComplexity.LECTURE_ANALYSIS

        if has_multimodal:
            return RequestComplexity.MULTIMODAL_ANALYSIS

        # Simple UI or App Guide Questions
        if any(w in q_lower for w in ["where is", "where are", "how to find", "tour", "walkthrough", "guide", "overview of the app"]):
            return RequestComplexity.APPLICATION_GUIDANCE

        if len(q_lower.split()) <= 4 and any(w in q_lower for w in ["hi", "hello", "hey", "help", "status", "ready"]):
            return RequestComplexity.SIMPLE_UI

        # Deep Reasoning / Conflict Resolution
        if any(w in q_lower for w in ["conflict", "deep analysis", "why not trade", "evaluate setup", "full analysis", "invalidation reason"]):
            return RequestComplexity.DEEP_REASONING

        # Knowledge / Rule query (e.g. "What is the rule for...", "Remember concept")
        if any(w in q_lower for w in ["rule", "concept", "remember", "learned", "mentorship"]):
            return RequestComplexity.KNOWLEDGE_QUERY

        # Market Analysis
        if any(w in q_lower for w in ["analyze", "structure", "mtf", "fvg", "order block", "liquidity sweep", "choch", "bos"]):
            return RequestComplexity.MARKET_ANALYSIS

        # Market Status
        if any(w in q_lower for w in ["kya chal raha", "current status", "price", "market doing", "overview"]):
            return RequestComplexity.MARKET_STATUS

        return RequestComplexity.GENERAL_CHAT

    def select_initial_tier(self, complexity: RequestComplexity) -> TokenBudgetTier:
        """Map request complexity to initial budget tier."""
        mapping = {
            RequestComplexity.SIMPLE_UI: TokenBudgetTier.SMALL,
            RequestComplexity.APPLICATION_GUIDANCE: TokenBudgetTier.SMALL,
            RequestComplexity.GENERAL_CHAT: TokenBudgetTier.SMALL,
            RequestComplexity.MARKET_STATUS: TokenBudgetTier.MEDIUM,
            RequestComplexity.KNOWLEDGE_QUERY: TokenBudgetTier.MEDIUM,
            RequestComplexity.MARKET_ANALYSIS: TokenBudgetTier.LARGE,
            RequestComplexity.LECTURE_ANALYSIS: TokenBudgetTier.LARGE,
            RequestComplexity.MULTIMODAL_ANALYSIS: TokenBudgetTier.MAX_SAFE,
            RequestComplexity.DEEP_REASONING: TokenBudgetTier.MAX_SAFE,
        }
        return mapping.get(complexity, TokenBudgetTier.MEDIUM)

    def escalate_tier(
        self,
        current_tier: TokenBudgetTier,
        reason: EscalationReason = EscalationReason.CONTEXT_SIZE,
    ) -> Tuple[TokenBudgetTier, bool]:
        """
        Escalate to the next higher budget tier if justified.
        Returns: (next_tier, was_escalated)
        """
        progression = {
            TokenBudgetTier.SMALL: TokenBudgetTier.MEDIUM,
            TokenBudgetTier.MEDIUM: TokenBudgetTier.LARGE,
            TokenBudgetTier.LARGE: TokenBudgetTier.MAX_SAFE,
            TokenBudgetTier.MAX_SAFE: TokenBudgetTier.MAX_SAFE,
        }
        next_tier = progression.get(current_tier, current_tier)
        was_escalated = next_tier != current_tier
        return next_tier, was_escalated

    # -----------------------------------------------------------------------
    # Token Estimation
    # -----------------------------------------------------------------------
    def estimate_text_tokens(self, text: str) -> int:
        """Estimate token count for a text string using Gemini heuristic (~3.6 chars per token)."""
        if not text:
            return 0
        char_count = len(text)
        tokens = math.ceil(char_count / 3.6)
        return max(1, tokens)

    def estimate_image_tokens(self, width: int = 1280, height: int = 720) -> int:
        """Estimate Gemini multimodal image tokens based on standard resolution tiling."""
        tiles_x = math.ceil(width / 768)
        tiles_y = math.ceil(height / 768)
        total_tiles = max(1, tiles_x * tiles_y)
        return total_tiles * 258

    def estimate_request_tokens(
        self,
        prompt: str,
        image_count: int = 0,
        image_dimensions: Tuple[int, int] = (1280, 720),
    ) -> int:
        """Calculate total estimated input tokens for a prompt plus multimodal images."""
        text_tokens = self.estimate_text_tokens(prompt)
        image_tokens = 0
        if image_count > 0:
            per_image = self.estimate_image_tokens(image_dimensions[0], image_dimensions[1])
            image_tokens = per_image * image_count
        return text_tokens + image_tokens

    # -----------------------------------------------------------------------
    # Budget Evaluation & Headroom Enforcement
    # -----------------------------------------------------------------------
    def check_budget(
        self, estimated_tokens: int, budget: Optional[TokenBudget] = None
    ) -> BudgetCheckResult:
        """Enforce pre-flight token budget check with 20% safety margin."""
        active_budget = budget or self._default_budget
        limit = active_budget.effective_input_limit

        if estimated_tokens <= limit:
            headroom = limit - estimated_tokens
            return BudgetCheckResult(
                is_sufficient=True,
                estimated_tokens=estimated_tokens,
                effective_limit=limit,
                headroom=headroom,
                overage=0,
                message=f"Budget sufficient: {estimated_tokens}/{limit} tokens ({headroom} headroom, {int(active_budget.safety_margin_percent * 100)}% safety reserve).",
                tier=active_budget.tier,
            )
        else:
            overage = estimated_tokens - limit
            return BudgetCheckResult(
                is_sufficient=False,
                estimated_tokens=estimated_tokens,
                effective_limit=limit,
                headroom=0,
                overage=overage,
                message=f"Budget exceeded by {overage} tokens: {estimated_tokens} requested vs {limit} effective limit.",
                tier=active_budget.tier,
            )

    # -----------------------------------------------------------------------
    # Context Prioritization & Packing
    # -----------------------------------------------------------------------
    def pack_context_items(
        self,
        items: List[ContextItem],
        max_tokens: int,
    ) -> Tuple[List[ContextItem], List[ContextItem]]:
        """
        Token-aware priority packing (knapsack style).
        Selects highest priority and importance items that fit strictly within max_tokens budget.
        Returns: (included_items, dropped_items)
        """
        if not items:
            return [], []

        # Calculate token cost for each item if not set
        for item in items:
            if item.token_cost <= 0:
                item.token_cost = self.estimate_text_tokens(item.text)

        # Sort by: 1. Priority descending, 2. Importance descending, 3. Recency descending
        sorted_items = sorted(
            items,
            key=lambda x: (x.priority.value, x.importance, x.recency),
            reverse=True,
        )

        included: List[ContextItem] = []
        dropped: List[ContextItem] = []
        accumulated_tokens = 0

        for item in sorted_items:
            if accumulated_tokens + item.token_cost <= max_tokens:
                included.append(item)
                accumulated_tokens += item.token_cost
            else:
                dropped.append(item)

        return included, dropped

    def optimize_context(
        self,
        items: List[T],
        item_text_fn: Callable[[T], str],
        max_tokens: int,
        importance_score_fn: Optional[Callable[[T], float]] = None,
    ) -> Tuple[List[T], List[T]]:
        """
        Prune context items so total tokens stay within max_tokens budget.
        Returns: (included_items, deferred_or_omitted_items)
        Preserves original chronological ordering among selected items.
        """
        if not items:
            return [], []

        indexed_items = list(enumerate(items))
        if importance_score_fn:
            indexed_items.sort(
                key=lambda x: importance_score_fn(x[1]), reverse=True
            )

        included: List[Tuple[int, T]] = []
        deferred: List[Tuple[int, T]] = []
        accumulated_tokens = 0

        for original_idx, item in indexed_items:
            item_text = item_text_fn(item)
            item_tokens = self.estimate_text_tokens(item_text)
            if accumulated_tokens + item_tokens <= max_tokens:
                included.append((original_idx, item))
                accumulated_tokens += item_tokens
            else:
                deferred.append((original_idx, item))

        included.sort(key=lambda x: x[0])
        deferred.sort(key=lambda x: x[0])

        return [item for _, item in included], [item for _, item in deferred]

    def split_text_into_budgeted_chunks(
        self, text: str, max_tokens_per_chunk: int
    ) -> List[str]:
        """Split a long text block into coherent chunks each fitting strictly within max_tokens."""
        if not text:
            return []

        total_est = self.estimate_text_tokens(text)
        if total_est <= max_tokens_per_chunk:
            return [text]

        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk_parts: List[str] = []
        current_chunk_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = self.estimate_text_tokens(para)

            if para_tokens > max_tokens_per_chunk:
                lines = para.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    line_tokens = self.estimate_text_tokens(line)
                    if current_chunk_tokens + line_tokens > max_tokens_per_chunk:
                        if current_chunk_parts:
                            chunks.append("\n".join(current_chunk_parts))
                            current_chunk_parts = []
                            current_chunk_tokens = 0
                    current_chunk_parts.append(line)
                    current_chunk_tokens += line_tokens
                continue

            if current_chunk_tokens + para_tokens > max_tokens_per_chunk:
                if current_chunk_parts:
                    chunks.append("\n\n".join(current_chunk_parts))
                    current_chunk_parts = []
                    current_chunk_tokens = 0

            current_chunk_parts.append(para)
            current_chunk_tokens += para_tokens

        if current_chunk_parts:
            chunks.append("\n\n".join(current_chunk_parts))

        return chunks

    # -----------------------------------------------------------------------
    # Relevance-Based Memory Ranking & Retrieval
    # -----------------------------------------------------------------------
    def rank_and_select_memories(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
        max_tokens: int,
        min_score: float = 0.20,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Rank candidate knowledge items by relevance, importance, and recency,
        packing up to max_tokens budget.
        """
        if not candidates:
            return [], []

        q_terms = set(re.findall(r"\w+", query.lower()))

        scored_candidates: List[Tuple[float, Dict[str, Any], int]] = []
        for cand in candidates:
            text = cand.get("text", "") or cand.get("statement", "") or cand.get("title", "")
            tokens = self.estimate_text_tokens(text)

            # Relevance score via term overlap
            c_terms = set(re.findall(r"\w+", text.lower()))
            overlap = len(q_terms.intersection(c_terms))
            relevance = min(1.0, overlap / max(1, len(q_terms)))

            # Source priority boost
            source_type = cand.get("source", "").lower()
            source_multiplier = 1.0
            if "correction" in source_type:
                source_multiplier = 1.5
            elif "rule" in source_type:
                source_multiplier = 1.3
            elif "verified" in source_type:
                source_multiplier = 1.1

            importance = cand.get("importance", 0.8)
            recency = cand.get("recency", 0.7)
            confidence = cand.get("confidence", 0.9)

            composite = (
                (relevance * 0.45)
                + (importance * 0.25)
                + (recency * 0.20)
                + (confidence * 0.10)
            ) * source_multiplier

            if composite >= min_score or overlap > 0:
                scored_candidates.append((composite, cand, tokens))

        # Sort descending by composite score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        selected: List[Dict[str, Any]] = []
        omitted: List[Dict[str, Any]] = []
        accumulated_tokens = 0

        for score, cand, cost in scored_candidates:
            if accumulated_tokens + cost <= max_tokens:
                selected.append(cand)
                accumulated_tokens += cost
            else:
                omitted.append(cand)

        return selected, omitted

    # -----------------------------------------------------------------------
    # Observability & Usage Logging
    # -----------------------------------------------------------------------
    def record_usage(self, record: TokenUsageRecord) -> None:
        """Record token usage for audit, cost monitoring, and observability."""
        self._usage_history.append(record)

    def get_job_usage_summary(self, job_id: str) -> Dict[str, Any]:
        """Aggregate total token usage metrics for a specific job."""
        job_records = [r for r in self._usage_history if r.job_id == job_id]
        total_input = sum(r.input_tokens for r in job_records)
        total_output = sum(r.output_tokens for r in job_records)
        total_cached = sum(r.cached_tokens for r in job_records)
        calls_by_stage: Dict[str, int] = {}
        for r in job_records:
            calls_by_stage[r.stage] = calls_by_stage.get(r.stage, 0) + 1

        return {
            "job_id": job_id,
            "total_calls": len(job_records),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cached_tokens": total_cached,
            "calls_by_stage": calls_by_stage,
        }
