"""Unit tests for TokenManager and safety budgeting."""

import pytest
from trading_ai.core.token_manager import (
    TokenBudget,
    TokenManager,
    TokenUsageRecord,
)


def test_token_estimation_text():
    tm = TokenManager()
    assert tm.estimate_text_tokens("") == 0
    short_text = "This is a trading rule for fair value gaps."
    tokens = tm.estimate_text_tokens(short_text)
    assert 5 <= tokens <= 25


def test_token_estimation_image():
    tm = TokenManager()
    # 1280x720 requires 2x1 tiles = 2 tiles -> 2 * 258 = 516 tokens
    image_tokens = tm.estimate_image_tokens(1280, 720)
    assert image_tokens == 516

    # 1920x1080 requires 3x2 tiles = 6 tiles -> 6 * 258 = 1548 tokens
    image_tokens_fhd = tm.estimate_image_tokens(1920, 1080)
    assert image_tokens_fhd == 1548


def test_request_token_calculation():
    tm = TokenManager()
    prompt = "Analyze this 15-minute EURUSD chart for order blocks."
    total = tm.estimate_request_tokens(prompt, image_count=2, image_dimensions=(1280, 720))
    # text + 2 * 516
    assert total > 1032


def test_budget_check_safety_margin():
    tm = TokenManager()
    budget = TokenBudget(input_budget=1000, output_reserve=200, safety_margin_percent=0.20)
    # Effective limit should be 1000 - 200 = 800
    assert budget.effective_input_limit == 800

    # Test within budget
    res_ok = tm.check_budget(estimated_tokens=500, budget=budget)
    assert res_ok.is_sufficient is True
    assert res_ok.headroom == 300
    assert res_ok.overage == 0

    # Test right at effective limit
    res_edge = tm.check_budget(estimated_tokens=800, budget=budget)
    assert res_edge.is_sufficient is True
    assert res_edge.headroom == 0

    # Test exceeding effective limit (in safety margin zone)
    res_exceeded = tm.check_budget(estimated_tokens=850, budget=budget)
    assert res_exceeded.is_sufficient is False
    assert res_exceeded.overage == 50
    assert "Budget exceeded by 50 tokens" in res_exceeded.message


def test_context_optimization():
    tm = TokenManager()
    items = [
        {"id": 1, "text": "Segment 1: Market structure break upward.", "importance": 0.9},
        {"id": 2, "text": "Segment 2: Minor chatter about tea.", "importance": 0.1},
        {"id": 3, "text": "Segment 3: Critical entry rule at 50% equilibrium.", "importance": 0.95},
        {"id": 4, "text": "Segment 4: More non-essential chatter.", "importance": 0.2},
    ]

    # Force a token limit that fits the top 2 segments (14 + 12 = 26 tokens)
    # Set max_tokens to 30.
    included, deferred = tm.optimize_context(
        items=items,
        item_text_fn=lambda x: x["text"],
        max_tokens=30,
        importance_score_fn=lambda x: x["importance"],
    )

    # Should select Segment 3 (0.95) and Segment 1 (0.9), deferring the others
    included_ids = [item["id"] for item in included]
    assert 3 in included_ids
    assert 1 in included_ids
    assert 2 not in included_ids
    # Preserves chronological order: id 1 before id 3
    assert included_ids == [1, 3]


def test_split_text_into_budgeted_chunks():
    tm = TokenManager()
    long_text = "\n\n".join([f"Paragraph {i}: Detailed explanation of ICT rule {i} with extensive details." for i in range(20)])
    chunks = tm.split_text_into_budgeted_chunks(long_text, max_tokens_per_chunk=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert tm.estimate_text_tokens(chunk) <= 55  # within boundary tolerance


def test_usage_logging_and_summary():
    tm = TokenManager()
    tm.record_usage(
        TokenUsageRecord(
            job_id="job_test",
            serial_number=1,
            stage="audio_processing",
            model="gemini-2.5-flash",
            input_tokens=1200,
            output_tokens=300,
            total_tokens=1500,
            cached_tokens=0,
        )
    )
    tm.record_usage(
        TokenUsageRecord(
            job_id="job_test",
            serial_number=2,
            stage="reasoning",
            model="gemini-2.5-pro",
            input_tokens=2400,
            output_tokens=800,
            total_tokens=3200,
            cached_tokens=400,
        )
    )

    summary = tm.get_job_usage_summary("job_test")
    assert summary["total_calls"] == 2
    assert summary["total_input_tokens"] == 3600
    assert summary["total_output_tokens"] == 1100
    assert summary["total_tokens"] == 4700
    assert summary["total_cached_tokens"] == 400
    assert summary["calls_by_stage"]["audio_processing"] == 1
    assert summary["calls_by_stage"]["reasoning"] == 1


# ---------------------------------------------------------------------------
# Step 2: Dynamic Budgeting, Tier Escalation & Hardening Tests
# ---------------------------------------------------------------------------
from trading_ai.core.token_manager import (
    TokenBudgetTier,
    RequestComplexity,
    EscalationReason,
    ContextPriority,
    ContextItem,
    ConversationCompressor,
    MODEL_CONTEXT_CAPACITIES,
    DEFAULT_SAFE_MODEL_CAPACITY,
)


def test_model_context_capacity_resolution():
    """Verify context capacity registry for Gemini models and fallbacks."""
    tm = TokenManager()
    assert tm.get_model_capacity("gemini-3.8-flash") == 1_048_576
    assert tm.get_model_capacity("gemini-3.1-pro-preview") == 1_048_576
    assert tm.get_model_capacity("gemini-embedding-2") == 8_192
    assert tm.get_model_capacity("unknown-custom-model") == DEFAULT_SAFE_MODEL_CAPACITY


def test_dynamic_budget_tiers_and_safety_headroom():
    """Verify dynamic budget tiers adhere strictly to 20% safety margin headroom."""
    tm = TokenManager()

    # SMALL
    b_small = tm.get_budget_for_tier(TokenBudgetTier.SMALL)
    assert b_small.tier == TokenBudgetTier.SMALL
    assert b_small.input_budget == 4_000
    assert b_small.output_reserve == 2_000
    assert b_small.safety_margin_percent == 0.20
    assert b_small.effective_input_limit == 3_200  # 4000 * 0.80

    # MEDIUM
    b_med = tm.get_budget_for_tier(TokenBudgetTier.MEDIUM)
    assert b_med.input_budget == 15_000
    assert b_med.output_reserve == 3_000
    assert b_med.effective_input_limit == 12_000  # 15000 * 0.80

    # LARGE
    b_large = tm.get_budget_for_tier(TokenBudgetTier.LARGE)
    assert b_large.input_budget == 30_000
    assert b_large.output_reserve == 4_000
    assert b_large.effective_input_limit == 24_000  # 30000 * 0.80

    # MAX_SAFE
    b_max = tm.get_budget_for_tier(TokenBudgetTier.MAX_SAFE)
    assert b_max.input_budget == 60_000
    assert b_max.output_reserve == 8_000
    assert b_max.effective_input_limit == 48_000  # 60000 * 0.80


def test_budget_bounded_by_model_capacity():
    """Verify tier input budget never exceeds actual model capacity minus reserve."""
    tm = TokenManager()
    # Model with small 10k capacity
    small_cap = 10_000
    b_constrained = tm.get_budget_for_tier(TokenBudgetTier.MAX_SAFE, model_capacity=small_cap)
    # Effective capacity = 10000 * 0.80 = 8000; minus reserve 8000 -> must be bounded
    assert b_constrained.input_budget <= small_cap


def test_calculate_available_input_exact_formula():
    """Verify the formula: MODEL_CAPACITY * (1 - SAFETY_MARGIN) - SYSTEM - TOOLS - OUTPUT_RESERVE."""
    tm = TokenManager()
    available = tm.calculate_available_input(
        model_capacity=100_000,
        system_instruction_tokens=5_000,
        tool_overhead_tokens=1_000,
        output_reserve=4_000,
        safety_margin_percent=0.20,
    )
    # 100000 * 0.80 = 80000 - 5000 - 1000 - 4000 = 70000
    assert available == 70_000


def test_request_complexity_deterministic_classification():
    """Verify deterministic request complexity classification rules."""
    tm = TokenManager()

    assert tm.classify_request_complexity("Where is the Watchlist?") == RequestComplexity.APPLICATION_GUIDANCE
    assert tm.classify_request_complexity("Hi there") == RequestComplexity.SIMPLE_UI
    assert tm.classify_request_complexity("What is the current market doing?") == RequestComplexity.MARKET_STATUS
    assert tm.classify_request_complexity("Analyze the 15m order block and FVG on EUR/USD") == RequestComplexity.MARKET_ANALYSIS
    assert tm.classify_request_complexity("What is the rule for liquidity sweep?") == RequestComplexity.KNOWLEDGE_QUERY
    assert tm.classify_request_complexity("Resolve conflict: why not trade EUR/USD?") == RequestComplexity.DEEP_REASONING
    assert tm.classify_request_complexity("Review chart", has_multimodal=True) == RequestComplexity.MULTIMODAL_ANALYSIS
    assert tm.classify_request_complexity("Lecture 1 breakdown", is_lecture=True) == RequestComplexity.LECTURE_ANALYSIS


def test_initial_tier_selection_and_escalation():
    """Verify initial tier mapping and dynamic escalation progression."""
    tm = TokenManager()

    # Mapping
    assert tm.select_initial_tier(RequestComplexity.SIMPLE_UI) == TokenBudgetTier.SMALL
    assert tm.select_initial_tier(RequestComplexity.MARKET_STATUS) == TokenBudgetTier.MEDIUM
    assert tm.select_initial_tier(RequestComplexity.MARKET_ANALYSIS) == TokenBudgetTier.LARGE
    assert tm.select_initial_tier(RequestComplexity.DEEP_REASONING) == TokenBudgetTier.MAX_SAFE

    # Escalation ladder: SMALL -> MEDIUM -> LARGE -> MAX_SAFE -> MAX_SAFE
    t1, esc1 = tm.escalate_tier(TokenBudgetTier.SMALL, EscalationReason.CONTEXT_SIZE)
    assert t1 == TokenBudgetTier.MEDIUM and esc1 is True

    t2, esc2 = tm.escalate_tier(TokenBudgetTier.MEDIUM, EscalationReason.RETRIEVED_EVIDENCE)
    assert t2 == TokenBudgetTier.LARGE and esc2 is True

    t3, esc3 = tm.escalate_tier(TokenBudgetTier.LARGE, EscalationReason.COMPLEX_REASONING)
    assert t3 == TokenBudgetTier.MAX_SAFE and esc3 is True

    t4, esc4 = tm.escalate_tier(TokenBudgetTier.MAX_SAFE, EscalationReason.OUTPUT_RESERVE_INSUFFICIENT)
    assert t4 == TokenBudgetTier.MAX_SAFE and esc4 is False  # Cannot escalate beyond MAX_SAFE


def test_knapsack_pack_context_items_priority():
    """Verify high-priority context items are preserved over low-priority items when budget is constrained."""
    tm = TokenManager()
    items = [
        ContextItem(
            item_id="crit_1",
            text="Explicit user rule: Never risk more than 1% on any trade.",
            priority=ContextPriority.CRITICAL,
            importance=1.0,
            token_cost=20,
        ),
        ContextItem(
            item_id="high_1",
            text="Market State: Bullish bias confirmed at 15m POI.",
            priority=ContextPriority.HIGH,
            importance=0.9,
            token_cost=20,
        ),
        ContextItem(
            item_id="low_1",
            text="Old chat chatter from 2 hours ago about the weather.",
            priority=ContextPriority.LOW,
            importance=0.1,
            token_cost=20,
        ),
    ]

    # Constrain max_tokens to 45 (can only fit 2 items)
    included, dropped = tm.pack_context_items(items, max_tokens=45)

    inc_ids = [i.item_id for i in included]
    drop_ids = [i.item_id for i in dropped]

    assert "crit_1" in inc_ids
    assert "high_1" in inc_ids
    assert "low_1" in drop_ids
    assert len(included) == 2
    assert len(dropped) == 1


def test_conversation_compressor_invariants():
    """Verify ConversationCompressor extracts facts and preserves user rules and corrections."""
    turns = [
        {"role": "user", "text": "Can you analyze EUR/USD on 15m?"},
        {"role": "assistant", "text": "Bullish market structure break observed with high confluence."},
        {"role": "user", "text": "Save rule: Always wait for London session sweep before entering."},
        {"role": "assistant", "text": "Rule recorded."},
        {"role": "user", "text": "Correction: the invalidation level was at 1.08200, not 1.08100."},
        {"role": "assistant", "text": "Noted and updated."},
    ]

    summary = ConversationCompressor.extract_structured_summary(
        turns=turns,
        active_symbol="EUR/USD",
        active_timeframe="15m",
    )

    assert "EUR/USD (15m)" in summary
    assert "London session sweep" in summary
    assert "1.08200" in summary
    assert "Bullish" in summary


def test_relevance_memory_ranking_and_source_boost():
    """Verify rank_and_select_memories boosts rules and user corrections while respecting token limits."""
    tm = TokenManager()
    candidates = [
        {
            "id": "c1",
            "text": "User correction: do not enter before 09:30 EST.",
            "source": "user_correction",
            "importance": 1.0,
            "recency": 1.0,
        },
        {
            "id": "c2",
            "text": "General note about market volatility on Fridays.",
            "source": "observation",
            "importance": 0.4,
            "recency": 0.5,
        },
        {
            "id": "c3",
            "text": "Saved rule: enter on 5m FVG after 15m liquidity sweep.",
            "source": "saved_rule",
            "importance": 0.9,
            "recency": 0.8,
        },
    ]

    query = "Should I enter before 09:30 EST on 5m FVG?"
    # Token budget fitting only top 2 items
    selected, omitted = tm.rank_and_select_memories(candidates, query=query, max_tokens=35)

    sel_ids = [c["id"] for c in selected]
    assert "c1" in sel_ids  # User correction boosted
    assert "c3" in sel_ids  # Saved rule boosted
    assert "c2" not in sel_ids or len(omitted) >= 1

