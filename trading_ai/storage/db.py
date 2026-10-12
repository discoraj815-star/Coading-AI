"""Database connection and schema initialization for SQLite."""

import json
import sqlite3
from pathlib import Path
from typing import Optional


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    duration REAL NOT NULL DEFAULT 0.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    current_stage TEXT NOT NULL,
    serial_number INTEGER NOT NULL DEFAULT 1,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transcripts (
    segment_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    transcript TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    speaker TEXT,
    important_statements_json TEXT NOT NULL DEFAULT '[]',
    terminology_json TEXT NOT NULL DEFAULT '[]',
    rules_json TEXT NOT NULL DEFAULT '[]',
    examples_json TEXT NOT NULL DEFAULT '[]',
    uncertainty_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcripts_job_time ON transcripts(job_id, start_time);

CREATE TABLE IF NOT EXISTS visual_events (
    event_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    frame_ref TEXT,
    visual_description TEXT NOT NULL,
    detected_event TEXT NOT NULL,
    symbol TEXT,
    timeframe TEXT,
    candles_json TEXT NOT NULL DEFAULT '[]',
    drawings_json TEXT NOT NULL DEFAULT '[]',
    zones_json TEXT NOT NULL DEFAULT '[]',
    arrows_json TEXT NOT NULL DEFAULT '[]',
    labels_json TEXT NOT NULL DEFAULT '[]',
    screen_text_json TEXT NOT NULL DEFAULT '[]',
    cursor_interaction TEXT,
    chart_changes_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    regions_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_visual_events_job_time ON visual_events(job_id, timestamp);

CREATE TABLE IF NOT EXISTS alignments (
    alignment_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    audio_json TEXT,
    visual_json TEXT NOT NULL DEFAULT '[]',
    relationship TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    reasoning TEXT NOT NULL DEFAULT '',
    conflicts_json TEXT NOT NULL DEFAULT '[]',
    deictic_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_alignments_job_time ON alignments(job_id, start_time);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    audio_quote TEXT,
    visual_reference TEXT,
    extracted_statement TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    notes TEXT,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_evidence_job_time ON evidence(job_id, start_time);

CREATE TABLE IF NOT EXISTS knowledge (
    item_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    title TEXT NOT NULL,
    statement TEXT NOT NULL,
    conditions_json TEXT NOT NULL DEFAULT '[]',
    exceptions_json TEXT NOT NULL DEFAULT '[]',
    reasoning TEXT NOT NULL DEFAULT '',
    validation_status TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    parent_id TEXT,
    tags_json TEXT NOT NULL DEFAULT '[]',
    serial_number INTEGER NOT NULL DEFAULT 1,
    is_committed INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_knowledge_job_committed ON knowledge(job_id, is_committed);
CREATE INDEX IF NOT EXISTS idx_knowledge_type ON knowledge(job_id, item_type);

CREATE TABLE IF NOT EXISTS notes (
    job_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    overview TEXT NOT NULL,
    concepts_json TEXT NOT NULL DEFAULT '[]',
    definitions_json TEXT NOT NULL DEFAULT '[]',
    rules_json TEXT NOT NULL DEFAULT '[]',
    examples_json TEXT NOT NULL DEFAULT '[]',
    exceptions_json TEXT NOT NULL DEFAULT '[]',
    visual_explanations_json TEXT NOT NULL DEFAULT '[]',
    important_statements_json TEXT NOT NULL DEFAULT '[]',
    relationships_json TEXT NOT NULL DEFAULT '[]',
    uncertainty_json TEXT NOT NULL DEFAULT '[]',
    evidence_timestamps_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS task_units (
    unit_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    serial_number INTEGER NOT NULL,
    stage TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    last_error TEXT,
    error_severity TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
    UNIQUE(job_id, serial_number)
);

CREATE INDEX IF NOT EXISTS idx_task_units_job_serial ON task_units(job_id, serial_number);

CREATE TABLE IF NOT EXISTS commit_barriers (
    job_id TEXT PRIMARY KEY,
    highest_committed_serial INTEGER NOT NULL DEFAULT 0,
    buffered_units_json TEXT NOT NULL DEFAULT '[]',
    blocked_on_serial INTEGER,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS token_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    serial_number INTEGER NOT NULL DEFAULT 1,
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_usage_job ON token_usage(job_id);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    serial_number INTEGER NOT NULL DEFAULT 1,
    stage TEXT NOT NULL,
    error_message TEXT NOT NULL,
    error_severity TEXT NOT NULL,
    is_recovered INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_errors_job ON errors(job_id);
"""

# ------------------------------------------------------------------ #
# Step 2 — Context Intelligence + Long-Term Memory schema additions    #
# ------------------------------------------------------------------ #

STEP2_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS context_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    serial_number INTEGER NOT NULL DEFAULT 1,
    window_tokens INTEGER NOT NULL DEFAULT 0,
    window_items_json TEXT NOT NULL DEFAULT '[]',
    pruned_items_json TEXT NOT NULL DEFAULT '[]',
    compression_summary TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_context_snapshots_job ON context_snapshots(job_id, serial_number);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    embedding_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    text_content TEXT NOT NULL,
    embedding_json TEXT NOT NULL DEFAULT '[]',
    model_name TEXT NOT NULL DEFAULT 'gemini-embedding-2',
    importance_score REAL NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_job_source ON memory_embeddings(job_id, source_type);
CREATE INDEX IF NOT EXISTS idx_embeddings_source_id ON memory_embeddings(source_id);
"""

# ------------------------------------------------------------------ #
# Step 3 — Universal Sources, Knowledge Versions, Background Jobs    #
# ------------------------------------------------------------------ #

STEP3_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_location TEXT NOT NULL,
    title TEXT,
    language TEXT NOT NULL DEFAULT 'en',
    normalized_language TEXT NOT NULL DEFAULT 'en',
    permissions_json TEXT NOT NULL DEFAULT '{}',
    acquisition_json TEXT NOT NULL DEFAULT '{}',
    processing_status TEXT NOT NULL,
    error_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(processing_status);

CREATE TABLE IF NOT EXISTS knowledge_versions (
    version_id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    version_number INTEGER NOT NULL,
    job_id TEXT NOT NULL,
    title TEXT NOT NULL,
    statement TEXT NOT NULL,
    conditions_json TEXT NOT NULL DEFAULT '[]',
    exceptions_json TEXT NOT NULL DEFAULT '[]',
    reasoning TEXT NOT NULL DEFAULT '',
    validation_status TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    correction_notes TEXT,
    corrected_by TEXT NOT NULL DEFAULT 'system',
    corrected_at TEXT NOT NULL,
    superseded_by_version INTEGER,
    is_active INTEGER NOT NULL DEFAULT 1,
    provenance_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_versions_item_num ON knowledge_versions(item_id, version_number);
CREATE INDEX IF NOT EXISTS idx_versions_job_active ON knowledge_versions(job_id, is_active);

CREATE TABLE IF NOT EXISTS background_jobs (
    job_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0.0,
    current_stage TEXT NOT NULL,
    current_serial INTEGER NOT NULL DEFAULT 0,
    total_units INTEGER NOT NULL DEFAULT 1,
    completed_units INTEGER NOT NULL DEFAULT 0,
    failed_units INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_bg_jobs_status ON background_jobs(status);
CREATE INDEX IF NOT EXISTS idx_bg_jobs_source ON background_jobs(source_id);

CREATE TABLE IF NOT EXISTS user_commands (
    command_id TEXT PRIMARY KEY,
    raw_text TEXT NOT NULL,
    intent TEXT NOT NULL,
    target_knowledge_id TEXT,
    parameters_json TEXT NOT NULL DEFAULT '{}',
    source TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_user_commands_intent ON user_commands(intent);
"""

STEP4_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS market_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    latest_price REAL NOT NULL,
    candles_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    timestamp REAL NOT NULL,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_tf ON market_snapshots(symbol, timeframe);

CREATE TABLE IF NOT EXISTS market_levels (
    level_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    price REAL NOT NULL,
    level_type TEXT NOT NULL,
    rule_id TEXT,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 1.0,
    validity TEXT NOT NULL DEFAULT 'VALID',
    invalidation_condition TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_levels_symbol_tf ON market_levels(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_levels_type ON market_levels(level_type);

CREATE TABLE IF NOT EXISTS chart_markings (
    marking_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    marking_type TEXT NOT NULL,
    coordinates_json TEXT NOT NULL DEFAULT '{}',
    rule_id TEXT,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    validity TEXT NOT NULL DEFAULT 'VALID',
    confidence REAL NOT NULL DEFAULT 1.0,
    invalidation_condition TEXT,
    priority INTEGER NOT NULL DEFAULT 1,
    concept TEXT,
    price REAL,
    price_range_json TEXT NOT NULL DEFAULT '{}',
    direction TEXT,
    source TEXT NOT NULL DEFAULT 'SUPERVISOR',
    rule_reference TEXT,
    rule_version TEXT NOT NULL DEFAULT '1.0',
    status TEXT NOT NULL DEFAULT 'VALIDATED',
    visibility TEXT NOT NULL DEFAULT 'VISIBLE',
    created_by TEXT NOT NULL DEFAULT 'MARKET_SUPERVISOR',
    broker_id TEXT,
    market_context_json TEXT NOT NULL DEFAULT '{}',
    backtest_run_id TEXT,
    timestamp REAL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_markings_symbol_tf ON chart_markings(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_markings_validity ON chart_markings(validity);

CREATE TABLE IF NOT EXISTS market_setups (
    setup_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    setup_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    status TEXT NOT NULL,
    applicable_rule_id TEXT,
    conditions_satisfied_json TEXT NOT NULL DEFAULT '[]',
    exceptions_checked_json TEXT NOT NULL DEFAULT '[]',
    invalidation_checked_json TEXT NOT NULL DEFAULT '[]',
    supporting_evidence_json TEXT NOT NULL DEFAULT '[]',
    contradicting_evidence_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_setups_symbol_tf ON market_setups(symbol, timeframe);
CREATE INDEX IF NOT EXISTS idx_setups_status ON market_setups(status);

CREATE TABLE IF NOT EXISTS market_analysis (
    analysis_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    htf_json TEXT,
    mtf_json TEXT,
    ltf_json TEXT,
    overall_bias TEXT NOT NULL,
    conflict_notes_json TEXT NOT NULL DEFAULT '[]',
    timestamp REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analysis_symbol ON market_analysis(symbol);

CREATE TABLE IF NOT EXISTS watchlist_monitor_state (
    symbol TEXT PRIMARY KEY,
    timeframe TEXT NOT NULL DEFAULT '15m',
    last_processed_timestamp REAL NOT NULL DEFAULT 0.0,
    latest_status TEXT NOT NULL,
    worker_state TEXT NOT NULL DEFAULT 'IDLE',
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS screen_permissions (
    permission_id TEXT PRIMARY KEY,
    app_name TEXT NOT NULL,
    window_title TEXT,
    region_box_json TEXT,
    status TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    granted_at TEXT,
    revoked_at TEXT
);
"""

# ------------------------------------------------------------------ #
# Step 5 — Intelligent Selective Visualization & Current Situation   #
# ------------------------------------------------------------------ #

STEP5_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS current_situation_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    current_price REAL NOT NULL,
    overall_bias TEXT NOT NULL,
    bias_summary TEXT NOT NULL,
    nearest_major_liquidity_json TEXT,
    current_poi_json TEXT,
    active_confirmed_setup_json TEXT,
    setup_confirmation_state TEXT NOT NULL DEFAULT 'NO_ACTIVE_SETUP',
    invalidation_level REAL,
    confirmation_level REAL,
    timeframe_conflicts_json TEXT NOT NULL DEFAULT '[]',
    recent_changes_json TEXT NOT NULL DEFAULT '[]',
    visible_markings_json TEXT NOT NULL DEFAULT '[]',
    background_tracking_count INTEGER NOT NULL DEFAULT 0,
    timestamp REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_situation_symbol ON current_situation_snapshots(symbol, timestamp);
"""

# ------------------------------------------------------------------ #
# Step 6 — Deep Professional Technical & Fundamental Intelligence    #
# ------------------------------------------------------------------ #

STEP6_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS fundamental_events (
    event_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    currency TEXT NOT NULL,
    event_time TEXT NOT NULL,
    impact TEXT NOT NULL,
    previous TEXT,
    forecast TEXT,
    actual TEXT,
    status TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source_url TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_fund_currency_time ON fundamental_events(currency, event_time);
CREATE INDEX IF NOT EXISTS idx_fund_impact ON fundamental_events(impact);
CREATE INDEX IF NOT EXISTS idx_fund_source ON fundamental_events(source);

CREATE TABLE IF NOT EXISTS fundamental_provider_state (
    provider_name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    last_successful_fetch TEXT,
    stale_after_seconds INTEGER NOT NULL DEFAULT 86400,
    is_stale INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    unavailable_reason TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tech_fund_correlations (
    analysis_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    relationship TEXT NOT NULL,
    technical_bias TEXT NOT NULL,
    news_risk TEXT NOT NULL,
    timing_context_json TEXT NOT NULL DEFAULT '{}',
    conflict_json TEXT NOT NULL DEFAULT '{}',
    action_context TEXT NOT NULL,
    explanation TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    timestamp REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_correlations_symbol ON tech_fund_correlations(symbol, timestamp);
"""

STEP7_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS alert_events (
    alert_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    priority TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL,
    decision TEXT NOT NULL,
    quality TEXT NOT NULL,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    is_acknowledged INTEGER NOT NULL DEFAULT 0,
    acknowledged_at TEXT,
    dedup_key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alert_events(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_dedup ON alert_events(dedup_key, lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alert_events(priority, is_acknowledged);

CREATE TABLE IF NOT EXISTS alert_state (
    dedup_key TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    last_alerted_at REAL NOT NULL,
    last_lifecycle_state TEXT NOT NULL,
    alert_count INTEGER NOT NULL DEFAULT 1,
    suppression_reason TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alert_state_sym ON alert_state(symbol);

CREATE TABLE IF NOT EXISTS discipline_state (
    user_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL DEFAULT '{}',
    trades_today_count INTEGER NOT NULL DEFAULT 0,
    daily_realized_loss REAL NOT NULL DEFAULT 0.0,
    last_trade_timestamp REAL,
    cooldown_active_until TEXT,
    recent_trades_timestamps_json TEXT NOT NULL DEFAULT '[]',
    active_violations_json TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scenario_analyses (
    scenario_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    level REAL NOT NULL,
    direction TEXT NOT NULL,
    if_condition TEXT NOT NULL,
    then_outcome TEXT NOT NULL,
    unless_condition TEXT NOT NULL,
    invalidated_when TEXT NOT NULL,
    next_poi_or_liquidity REAL,
    confidence_quality TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scenarios_symbol ON scenario_analyses(symbol, created_at);

CREATE TABLE IF NOT EXISTS companion_interactions (
    interaction_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    user_query TEXT NOT NULL,
    intent TEXT NOT NULL,
    answer TEXT NOT NULL,
    context_summary TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '[]',
    scenarios_json TEXT NOT NULL DEFAULT '[]',
    timestamp REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_companion_interactions_sym ON companion_interactions(symbol, timestamp);
"""

STEP8_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trade_proposals (
    proposal_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    market_regime TEXT NOT NULL,
    htf_context TEXT NOT NULL DEFAULT '',
    mtf_context TEXT NOT NULL DEFAULT '',
    overall_bias TEXT NOT NULL,
    liquidity_context TEXT NOT NULL DEFAULT '',
    poi_context TEXT NOT NULL DEFAULT '',
    lecture_rule_evidence_json TEXT NOT NULL DEFAULT '[]',
    confluences_json TEXT NOT NULL DEFAULT '[]',
    entry_trigger TEXT NOT NULL,
    entry_price REAL NOT NULL,
    hard_stop_loss REAL NOT NULL,
    tp1 REAL NOT NULL,
    tp2 REAL,
    final_target REAL NOT NULL,
    invalidation_condition TEXT NOT NULL,
    risk_assessment_json TEXT,
    proposed_quantity REAL NOT NULL DEFAULT 0.0,
    engagement_mode TEXT NOT NULL,
    execution_environment TEXT NOT NULL DEFAULT 'PAPER',
    news_event_context TEXT NOT NULL DEFAULT '',
    trade_thesis_json TEXT,
    quality TEXT NOT NULL,
    lifecycle_state TEXT NOT NULL,
    user_approval_required INTEGER NOT NULL DEFAULT 1,
    is_approved INTEGER NOT NULL DEFAULT 0,
    rejection_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trade_proposals_sym ON trade_proposals(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_trade_proposals_state ON trade_proposals(lifecycle_state);

CREATE TABLE IF NOT EXISTS execution_policies (
    user_id TEXT PRIMARY KEY,
    execution_mode TEXT NOT NULL DEFAULT 'MANUAL',
    execution_environment TEXT NOT NULL DEFAULT 'PAPER',
    live_execution_enabled INTEGER NOT NULL DEFAULT 0,
    auto_execution_enabled INTEGER NOT NULL DEFAULT 0,
    max_auto_risk_pct REAL NOT NULL DEFAULT 1.0,
    min_rr_ratio REAL NOT NULL DEFAULT 2.0,
    preferred_rr_ratio REAL NOT NULL DEFAULT 3.0,
    default_tp1_r_multiple REAL NOT NULL DEFAULT 1.5,
    default_tp1_close_pct REAL NOT NULL DEFAULT 0.5,
    move_sl_to_entry_at_tp1 INTEGER NOT NULL DEFAULT 1,
    min_confluence_count INTEGER NOT NULL DEFAULT 2,
    allowed_symbols_json TEXT NOT NULL DEFAULT '[]',
    defensive_size_pct REAL NOT NULL DEFAULT 0.25,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS execution_intents (
    intent_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    execution_mode TEXT NOT NULL,
    execution_environment TEXT NOT NULL DEFAULT 'PAPER',
    order_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity REAL NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    take_profit REAL NOT NULL,
    tp1 REAL NOT NULL,
    status TEXT NOT NULL,
    anti_duplicate_token TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    FOREIGN KEY (proposal_id) REFERENCES trade_proposals(proposal_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_exec_intents_token ON execution_intents(anti_duplicate_token);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    broker_order_id TEXT,
    intent_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    execution_environment TEXT NOT NULL DEFAULT 'PAPER',
    is_simulated INTEGER NOT NULL DEFAULT 1,
    order_type TEXT NOT NULL,
    direction TEXT NOT NULL,
    quantity REAL NOT NULL,
    limit_price REAL,
    stop_price REAL,
    status TEXT NOT NULL,
    filled_quantity REAL NOT NULL DEFAULT 0.0,
    average_fill_price REAL,
    broker_response_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (proposal_id) REFERENCES trade_proposals(proposal_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    execution_environment TEXT NOT NULL DEFAULT 'PAPER',
    entry_price REAL NOT NULL,
    current_quantity REAL NOT NULL,
    initial_quantity REAL NOT NULL,
    initial_sl REAL NOT NULL,
    current_sl REAL NOT NULL,
    tp1 REAL NOT NULL,
    tp2 REAL,
    final_target REAL NOT NULL,
    status TEXT NOT NULL,
    unrealized_pnl REAL NOT NULL DEFAULT 0.0,
    realized_pnl REAL NOT NULL DEFAULT 0.0,
    tp1_reached INTEGER NOT NULL DEFAULT 0,
    sl_moved_to_entry INTEGER NOT NULL DEFAULT 0,
    engagement_mode TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    closed_at TEXT,
    FOREIGN KEY (proposal_id) REFERENCES trade_proposals(proposal_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol_status ON positions(symbol, status);

CREATE TABLE IF NOT EXISTS position_events (
    event_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    action TEXT NOT NULL,
    price REAL NOT NULL,
    quantity_affected REAL NOT NULL,
    realized_pnl_delta REAL NOT NULL DEFAULT 0.0,
    notes TEXT NOT NULL DEFAULT '',
    timestamp REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (position_id) REFERENCES positions(position_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_position_events_pos ON position_events(position_id, timestamp);
"""

STEP9_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rule_versions (
    rule_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    title TEXT NOT NULL,
    rule_text TEXT NOT NULL,
    conditions_json TEXT NOT NULL DEFAULT '[]',
    exceptions_json TEXT NOT NULL DEFAULT '[]',
    source_lecture_id TEXT,
    source_timestamp REAL,
    provenance TEXT NOT NULL DEFAULT 'user_lecture',
    state TEXT NOT NULL DEFAULT 'LEARNED',
    tags_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (rule_id, version)
);

CREATE INDEX IF NOT EXISTS idx_rule_versions_id ON rule_versions(rule_id);
CREATE INDEX IF NOT EXISTS idx_rule_versions_state ON rule_versions(state);

CREATE TABLE IF NOT EXISTS strategy_versions (
    strategy_id TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    rule_version_ids_json TEXT NOT NULL DEFAULT '[]',
    asset_classes_json TEXT NOT NULL DEFAULT '["FOREX"]',
    symbols_json TEXT NOT NULL DEFAULT '["EUR/USD"]',
    timeframes_json TEXT NOT NULL DEFAULT '["15m"]',
    created_at TEXT NOT NULL,
    PRIMARY KEY (strategy_id, version)
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    backtest_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    strategy_version_id TEXT,
    rule_version_id TEXT,
    config_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'PENDING',
    progress_pct REAL NOT NULL DEFAULT 0.0,
    current_candle_index INTEGER NOT NULL DEFAULT 0,
    total_candles INTEGER NOT NULL DEFAULT 0,
    metrics_json TEXT,
    is_reproducible INTEGER NOT NULL DEFAULT 1,
    dataset_identifier TEXT NOT NULL DEFAULT 'simulated_historical_ohlcv',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_backtest_runs_status ON backtest_runs(status);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_asset ON backtest_runs(asset_class, symbol);

CREATE TABLE IF NOT EXISTS backtest_trades (
    trade_id TEXT PRIMARY KEY,
    backtest_id TEXT NOT NULL REFERENCES backtest_runs(backtest_id) ON DELETE CASCADE,
    symbol TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_time REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_time REAL NOT NULL,
    exit_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    target_price REAL NOT NULL,
    quantity REAL NOT NULL,
    pnl REAL NOT NULL,
    r_multiple REAL NOT NULL,
    result TEXT NOT NULL,
    rule_version_id TEXT,
    source TEXT NOT NULL DEFAULT 'AI',
    confluence_summary TEXT NOT NULL DEFAULT '',
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run ON backtest_trades(backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol ON backtest_trades(symbol, result);

CREATE TABLE IF NOT EXISTS rule_promotions (
    promotion_id TEXT PRIMARY KEY,
    rule_version_id TEXT NOT NULL,
    current_state TEXT NOT NULL DEFAULT 'LEARNED',
    backtest_id TEXT,
    user_backtest_approved INTEGER NOT NULL DEFAULT 0,
    paper_validation_approved INTEGER NOT NULL DEFAULT 0,
    user_live_approved INTEGER NOT NULL DEFAULT 0,
    paper_trade_count INTEGER NOT NULL DEFAULT 0,
    paper_pnl REAL NOT NULL DEFAULT 0.0,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rule_promotions_rule ON rule_promotions(rule_version_id);
CREATE INDEX IF NOT EXISTS idx_rule_promotions_state ON rule_promotions(current_state);
"""

STEP10_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trade_journal_entries (
    journal_id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    environment TEXT NOT NULL,
    backtest_type TEXT NOT NULL DEFAULT 'NOT_APPLICABLE',
    asset_class TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    direction TEXT NOT NULL,
    rule_version_id TEXT,
    strategy_version_id TEXT,
    lecture_source_id TEXT,
    setup_name TEXT NOT NULL,
    market_regime TEXT NOT NULL DEFAULT 'BALANCED',
    htf_context TEXT NOT NULL DEFAULT '',
    mtf_context TEXT NOT NULL DEFAULT '',
    ltf_context TEXT NOT NULL DEFAULT '',
    news_context TEXT NOT NULL DEFAULT 'NO_IMMINENT_NEWS',
    confluences_json TEXT NOT NULL DEFAULT '[]',
    engagement_mode TEXT NOT NULL DEFAULT 'DEFENSIVE',
    trade_thesis TEXT NOT NULL DEFAULT '',
    planned_json TEXT NOT NULL,
    actual_json TEXT NOT NULL,
    rule_adherence_score REAL NOT NULL DEFAULT 1.0,
    rule_violations_json TEXT NOT NULL DEFAULT '[]',
    behaviors_json TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '',
    evidence_references_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_journal_env_symbol ON trade_journal_entries(environment, symbol);
CREATE INDEX IF NOT EXISTS idx_journal_trade_id ON trade_journal_entries(trade_id);
CREATE INDEX IF NOT EXISTS idx_journal_created_at ON trade_journal_entries(created_at);

CREATE TABLE IF NOT EXISTS journal_lessons (
    lesson_id TEXT PRIMARY KEY,
    source_journal_ids_json TEXT NOT NULL DEFAULT '[]',
    symbol TEXT,
    asset_class TEXT,
    affected_rule_version_id TEXT,
    title TEXT NOT NULL,
    observation TEXT NOT NULL,
    explanation TEXT NOT NULL,
    proposed_action TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    state TEXT NOT NULL DEFAULT 'OBSERVED',
    reviewer_notes TEXT,
    candidate_rule_version_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_journal_lessons_state ON journal_lessons(state);
CREATE INDEX IF NOT EXISTS idx_journal_lessons_symbol ON journal_lessons(symbol);

CREATE TABLE IF NOT EXISTS journal_reviews (
    review_id TEXT PRIMARY KEY,
    proposal_id TEXT,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL,
    plan_is_complete INTEGER NOT NULL DEFAULT 1,
    missing_plan_elements_json TEXT NOT NULL DEFAULT '[]',
    historical_similar_trades_count INTEGER NOT NULL DEFAULT 0,
    historical_win_rate REAL NOT NULL DEFAULT 0.0,
    relevant_lessons_json TEXT NOT NULL DEFAULT '[]',
    relevant_mistakes_json TEXT NOT NULL DEFAULT '[]',
    discipline_warnings_json TEXT NOT NULL DEFAULT '[]',
    risk_reward_ratio REAL NOT NULL DEFAULT 0.0,
    recommendation TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_journal_reviews_proposal ON journal_reviews(proposal_id);

CREATE TABLE IF NOT EXISTS journal_checkpoints (
    job_id TEXT PRIMARY KEY,
    last_processed_trade_id TEXT,
    processed_count INTEGER NOT NULL DEFAULT 0,
    total_count INTEGER NOT NULL DEFAULT 0,
    context_token_estimate INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'COMPLETED',
    updated_at TEXT NOT NULL
);
"""

STEP11_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS component_upgrades (
    upgrade_id TEXT PRIMARY KEY,
    component_name TEXT NOT NULL,
    current_version TEXT NOT NULL,
    candidate_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'CANDIDATE',
    benchmark_accuracy REAL,
    benchmark_latency_ms REAL,
    regression_pass_rate REAL,
    user_approved INTEGER NOT NULL DEFAULT 0,
    approved_by TEXT,
    notes TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_upgrades_status ON component_upgrades(status);

CREATE TABLE IF NOT EXISTS performance_telemetry (
    telemetry_id TEXT PRIMARY KEY,
    symbol TEXT,
    ingestion_latency_ms REAL NOT NULL DEFAULT 0.0,
    normalization_latency_ms REAL NOT NULL DEFAULT 0.0,
    technical_analysis_ms REAL NOT NULL DEFAULT 0.0,
    mtf_synthesis_ms REAL NOT NULL DEFAULT 0.0,
    gemini_reasoning_ms REAL NOT NULL DEFAULT 0.0,
    end_to_end_decision_ms REAL NOT NULL DEFAULT 0.0,
    queue_wait_ms REAL NOT NULL DEFAULT 0.0,
    measured_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_measured_at ON performance_telemetry(measured_at);

CREATE TABLE IF NOT EXISTS trusted_devices (
    device_id TEXT PRIMARY KEY,
    device_name TEXT NOT NULL,
    device_type TEXT NOT NULL DEFAULT 'phone',
    status TEXT NOT NULL,
    public_key_or_token_hash TEXT NOT NULL,
    paired_at TEXT,
    last_seen TEXT,
    session_token_hash TEXT,
    revoked_at TEXT,
    revocation_reason TEXT
);

CREATE TABLE IF NOT EXISTS capability_permissions (
    capability TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'DENIED',
    requires_user_approval INTEGER NOT NULL DEFAULT 1,
    is_dangerous INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL,
    approved_by TEXT,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS security_audit_logs (
    audit_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    capability TEXT,
    action TEXT NOT NULL,
    requesting_component TEXT NOT NULL,
    decision TEXT NOT NULL,
    is_user_authorized INTEGER NOT NULL DEFAULT 0,
    authorized_by TEXT,
    device_id TEXT,
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_security_audit_time ON security_audit_logs(timestamp);

CREATE TABLE IF NOT EXISTS cross_device_notifications (
    notification_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'MEDIUM',
    category TEXT NOT NULL DEFAULT 'system',
    timestamp TEXT NOT NULL,
    read INTEGER NOT NULL DEFAULT 0,
    delivered_devices_json TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_notifications_time ON cross_device_notifications(timestamp);
"""

# ------------------------------------------------------------------ #
# Step 13 — Data Security, Cyber Wall, Watchdog & Auto-Recovery       #
# ------------------------------------------------------------------ #

STEP13_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_integrity_manifest (
    item_id TEXT PRIMARY KEY,
    version_id TEXT,
    sha256_hash TEXT NOT NULL,
    content_length INTEGER NOT NULL,
    epistemic_state TEXT NOT NULL,
    provenance_hash TEXT,
    last_verified_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PROTECTED'
);

CREATE INDEX IF NOT EXISTS idx_manifest_status ON knowledge_integrity_manifest(status);

CREATE TABLE IF NOT EXISTS recovery_backups (
    backup_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    backup_path TEXT NOT NULL,
    sha256_checksum TEXT NOT NULL,
    total_items INTEGER NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    is_verified INTEGER NOT NULL DEFAULT 1,
    retention_tag TEXT NOT NULL DEFAULT 'AUTOMATED_SNAPSHOT',
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_backups_created ON recovery_backups(created_at);

CREATE TABLE IF NOT EXISTS recovery_checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    job_id TEXT NOT NULL,
    serial_number INTEGER,
    state_payload_json TEXT NOT NULL,
    sha256_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    classification TEXT NOT NULL DEFAULT 'AUTO_RECOVERABLE',
    requires_user_approval INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_recovery_chk_job ON recovery_checkpoints(job_id, created_at);

CREATE TABLE IF NOT EXISTS threat_incidents (
    incident_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    threat_level TEXT NOT NULL,
    source TEXT NOT NULL,
    target_path_or_resource TEXT,
    description TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    action_taken TEXT NOT NULL DEFAULT 'BLOCKED',
    is_mitigated INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_threats_time ON threat_incidents(timestamp);
CREATE INDEX IF NOT EXISTS idx_threats_level ON threat_incidents(threat_level);
"""

STEP14_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS normalized_knowledge (
    item_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_document_id TEXT,
    source_session_id TEXT,
    source_location TEXT NOT NULL,
    concept_type TEXT NOT NULL DEFAULT 'Rule',
    title TEXT NOT NULL,
    statement TEXT NOT NULL,
    conditions_json TEXT NOT NULL DEFAULT '[]',
    exceptions_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 1.0,
    authority TEXT NOT NULL DEFAULT 'LECTURE_RULE',
    authority_priority INTEGER NOT NULL DEFAULT 4,
    version INTEGER NOT NULL DEFAULT 1,
    evidence_references_json TEXT NOT NULL DEFAULT '[]',
    tags_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_norm_know_domain ON normalized_knowledge(domain);
CREATE INDEX IF NOT EXISTS idx_norm_know_concept ON normalized_knowledge(concept_type);
CREATE INDEX IF NOT EXISTS idx_norm_know_authority ON normalized_knowledge(authority_priority);
CREATE INDEX IF NOT EXISTS idx_norm_know_source ON normalized_knowledge(source_id);

CREATE TABLE IF NOT EXISTS knowledge_relationships (
    relationship_id TEXT PRIMARY KEY,
    source_item_id TEXT NOT NULL,
    target_item_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_know_rel_source ON knowledge_relationships(source_item_id);
CREATE INDEX IF NOT EXISTS idx_know_rel_target ON knowledge_relationships(target_item_id);
CREATE INDEX IF NOT EXISTS idx_know_rel_type ON knowledge_relationships(relationship_type);

CREATE TABLE IF NOT EXISTS cross_source_confluences (
    confluence_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    decision_state TEXT NOT NULL,
    confluence_score REAL NOT NULL,
    matched_rule_ids_json TEXT NOT NULL DEFAULT '[]',
    conflicting_rule_ids_json TEXT NOT NULL DEFAULT '[]',
    supporting_lesson_ids_json TEXT NOT NULL DEFAULT '[]',
    reasons_to_trade_json TEXT NOT NULL DEFAULT '[]',
    reasons_not_to_trade_json TEXT NOT NULL DEFAULT '[]',
    proposal_id TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_confluence_sym_time ON cross_source_confluences(symbol, created_at);
"""

# ------------------------------------------------------------------ #
# Step 15 — AI Core, Large Video Processing & Plugin Architecture    #
# ------------------------------------------------------------------ #

STEP15_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS plugin_registry (
    plugin_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    category TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT '',
    capabilities_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'UNLOADED',
    is_enabled INTEGER NOT NULL DEFAULT 1,
    permissions_json TEXT NOT NULL DEFAULT '[]',
    config_json TEXT NOT NULL DEFAULT '{}',
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    health_status TEXT NOT NULL DEFAULT 'HEALTHY',
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plugins_category ON plugin_registry(category);
CREATE INDEX IF NOT EXISTS idx_plugins_status ON plugin_registry(status);

CREATE TABLE IF NOT EXISTS plugin_audit_logs (
    log_id TEXT PRIMARY KEY,
    plugin_id TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_plugin_audit_time ON plugin_audit_logs(plugin_id, created_at);

CREATE TABLE IF NOT EXISTS video_processing_jobs (
    job_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    chunk_duration REAL NOT NULL DEFAULT 30.0,
    total_chunks INTEGER NOT NULL DEFAULT 1,
    completed_chunks INTEGER NOT NULL DEFAULT 0,
    current_chunk INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'PENDING',
    checkpoint_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_video_jobs_status ON video_processing_jobs(status);

CREATE TABLE IF NOT EXISTS comprehension_reviews (
    review_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    concept_id TEXT NOT NULL,
    iteration INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 0.0,
    epistemic_state TEXT NOT NULL DEFAULT 'NEEDS_REVIEW',
    review_notes TEXT NOT NULL DEFAULT '',
    is_resolved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_comprehension_reviews ON comprehension_reviews(job_id, concept_id);

CREATE TABLE IF NOT EXISTS live_decision_scenarios (
    scenario_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    active_concept TEXT NOT NULL,
    learned_rule_id TEXT,
    scenarios_json TEXT NOT NULL DEFAULT '[]',
    confirmation_conditions_json TEXT NOT NULL DEFAULT '[]',
    invalidation_conditions_json TEXT NOT NULL DEFAULT '[]',
    decision_state TEXT NOT NULL DEFAULT 'WAIT',
    hindi_explanation TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_live_scenarios_symbol ON live_decision_scenarios(symbol, created_at);
"""

STEP16_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cloud_runtime_state (
    runtime_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    active_workers INTEGER NOT NULL DEFAULT 4,
    uptime_seconds REAL NOT NULL DEFAULT 0.0,
    last_heartbeat TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS event_stream_log (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    priority TEXT NOT NULL,
    symbol TEXT,
    source TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    payload_json TEXT NOT NULL DEFAULT '{}',
    deduplication_key TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    delivery_status TEXT NOT NULL DEFAULT 'QUEUED',
    timestamp TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_event_stream_timestamp ON event_stream_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_event_stream_priority ON event_stream_log(priority);
CREATE INDEX IF NOT EXISTS idx_event_stream_dedup ON event_stream_log(deduplication_key);

CREATE TABLE IF NOT EXISTS notification_records (
    notification_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    priority TEXT NOT NULL,
    title TEXT NOT NULL,
    message_hindi TEXT NOT NULL,
    message_english TEXT NOT NULL,
    symbol TEXT,
    channel TEXT NOT NULL,
    delivery_status TEXT NOT NULL,
    privacy_level TEXT NOT NULL DEFAULT 'NORMAL',
    deduplication_key TEXT NOT NULL,
    target_device_id TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    delivered_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_notif_created_at ON notification_records(created_at);
CREATE INDEX IF NOT EXISTS idx_notif_dedup ON notification_records(deduplication_key);
CREATE INDEX IF NOT EXISTS idx_notif_status ON notification_records(delivery_status);

CREATE TABLE IF NOT EXISTS user_notification_preferences (
    user_id TEXT PRIMARY KEY,
    preferences_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS device_push_tokens (
    device_id TEXT PRIMARY KEY,
    push_token TEXT NOT NULL,
    platform TEXT NOT NULL,
    last_updated TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS persistent_task_queue (
    job_id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 2,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    serial_number INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_task_queue_status_prio ON persistent_task_queue(status, priority, created_at);
"""

STEP17_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS decision_audit_log (
    decision_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp REAL NOT NULL,
    final_state TEXT NOT NULL,
    direction TEXT NOT NULL,
    confluence_score REAL NOT NULL DEFAULT 0.0,
    threshold REAL NOT NULL DEFAULT 8.0,
    concept_results_json TEXT NOT NULL DEFAULT '[]',
    evidence_references_json TEXT NOT NULL DEFAULT '[]',
    rule_ids_json TEXT NOT NULL DEFAULT '[]',
    source_provenance_json TEXT NOT NULL DEFAULT '[]',
    epistemic_state_json TEXT NOT NULL DEFAULT '{}',
    conflicts_json TEXT NOT NULL DEFAULT '[]',
    missing_evidence_json TEXT NOT NULL DEFAULT '[]',
    risk_result_json TEXT NOT NULL DEFAULT '{}',
    entry_zone TEXT,
    invalidation_level TEXT,
    target_level TEXT,
    primary_reason TEXT,
    user_summary_en TEXT,
    user_summary_hi TEXT,
    notification_event_id TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decision_audit_sym_time ON decision_audit_log(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_decision_audit_state ON decision_audit_log(final_state);

CREATE TABLE IF NOT EXISTS concept_modules_registry (
    module_id TEXT PRIMARY KEY,
    concept_name TEXT NOT NULL,
    concept_type TEXT NOT NULL,
    authority_priority INTEGER NOT NULL DEFAULT 4,
    is_enabled INTEGER NOT NULL DEFAULT 1,
    is_approved INTEGER NOT NULL DEFAULT 1,
    data_requirements_json TEXT NOT NULL DEFAULT '[]',
    config_json TEXT NOT NULL DEFAULT '{}',
    version TEXT NOT NULL DEFAULT '1.0.0',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

STEP18_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS universal_providers (
    provider_id TEXT PRIMARY KEY,
    provider_name TEXT NOT NULL,
    provider_type TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1.0.0',
    status TEXT NOT NULL DEFAULT 'DISCOVERED',
    trading_state TEXT NOT NULL DEFAULT 'PAPER_ONLY',
    capabilities_json TEXT NOT NULL DEFAULT '[]',
    supported_symbols_json TEXT NOT NULL DEFAULT '[]',
    supported_markets_json TEXT NOT NULL DEFAULT '[]',
    supported_timeframes_json TEXT NOT NULL DEFAULT '[]',
    authentication_type TEXT NOT NULL DEFAULT 'NONE',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    approved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_universal_providers_type ON universal_providers(provider_type);
CREATE INDEX IF NOT EXISTS idx_universal_providers_status ON universal_providers(status);

CREATE TABLE IF NOT EXISTS provider_connection_configs (
    provider_id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL DEFAULT '{}',
    is_encrypted INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lecture_benchmarks (
    lecture_id TEXT PRIMARY KEY,
    file_name TEXT NOT NULL DEFAULT '',
    file_hash TEXT NOT NULL DEFAULT '',
    file_size_bytes INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    current_stage TEXT NOT NULL DEFAULT 'QUEUED',
    is_complete INTEGER NOT NULL DEFAULT 0,
    benchmark_data_json TEXT NOT NULL DEFAULT '{}',
    timeline_json TEXT NOT NULL DEFAULT '[]',
    bottlenecks_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lecture_benchmarks_created ON lecture_benchmarks(created_at);
"""

from contextlib import contextmanager


class DatabaseManager:
    """Manages SQLite connections and schema migrations."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self._db_path = db_path
        else:
            from trading_ai.core.config import get_settings
            settings = get_settings()
            self._db_path = settings.db_path

        self._memory_conn: Optional[sqlite3.Connection] = None
        if self._db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_conn.row_factory = sqlite3.Row
            self._memory_conn.execute("PRAGMA foreign_keys = ON;")
        else:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)

        self.initialize_schema()

    @property
    def db_path(self) -> str:
        return self._db_path

    def get_connection(self) -> sqlite3.Connection:
        """Create and configure a SQLite connection."""
        if self._memory_conn is not None:
            return self._memory_conn
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def connection(self):
        """Context manager for managing transactions and connection closing."""
        conn = self.get_connection()
        try:
            with conn:
                yield conn
        finally:
            if self._memory_conn is None:
                conn.close()

    def initialize_schema(self) -> None:
        """Apply Step 1 through Step 17 table definitions and migrations."""
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.executescript(STEP2_SCHEMA_SQL)
            conn.executescript(STEP3_SCHEMA_SQL)
            conn.executescript(STEP4_SCHEMA_SQL)
            conn.executescript(STEP5_SCHEMA_SQL)
            conn.executescript(STEP6_SCHEMA_SQL)
            conn.executescript(STEP7_SCHEMA_SQL)
            conn.executescript(STEP8_SCHEMA_SQL)
            conn.executescript(STEP9_SCHEMA_SQL)
            conn.executescript(STEP10_SCHEMA_SQL)
            conn.executescript(STEP11_SCHEMA_SQL)
            conn.executescript(STEP13_SCHEMA_SQL)
            conn.executescript(STEP14_SCHEMA_SQL)
            conn.executescript(STEP15_SCHEMA_SQL)
            conn.executescript(STEP16_SCHEMA_SQL)
            conn.executescript(STEP17_SCHEMA_SQL)
            conn.executescript(STEP18_SCHEMA_SQL)

            # Safe additive migrations for knowledge table
            try:
                conn.execute("ALTER TABLE knowledge ADD COLUMN conditions_json TEXT NOT NULL DEFAULT '[]';")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE knowledge ADD COLUMN exceptions_json TEXT NOT NULL DEFAULT '[]';")
            except Exception:
                pass
            # Safe additive migrations for market_setups table (Step 5 lifecycle)
            setup_additions = [
                ("detection_timestamp", "REAL NOT NULL DEFAULT 0.0"),
                ("confirmation_timestamp", "REAL"),
                ("invalidation_timestamp", "REAL"),
                ("invalidation_condition", "TEXT"),
                ("confirmation_evidence_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("previous_status", "TEXT"),
                ("status_transition_reason", "TEXT"),
                ("is_displayed", "INTEGER NOT NULL DEFAULT 0"),
                ("updated_at", "TEXT"),
            ]
            for col_name, col_def in setup_additions:
                try:
                    conn.execute(f"ALTER TABLE market_setups ADD COLUMN {col_name} {col_def};")
                except Exception:
                    pass

            # Safe additive migrations for current_situation_snapshots table (Step 6 fundamentals)
            situation_additions = [
                ("fundamental_context", "TEXT"),
                ("relevant_news_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("next_high_impact_event_json", "TEXT"),
                ("news_time_distance_minutes", "REAL"),
                ("news_risk", "TEXT NOT NULL DEFAULT 'NO_RISK'"),
                ("technical_fundamental_relationship", "TEXT NOT NULL DEFAULT 'TECHNICALLY_VALID'"),
                ("technical_fundamental_conflicts_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("recommended_action_context", "TEXT NOT NULL DEFAULT 'PROCEED_WITH_STANDARD_CONFIRMATION'"),
            ]
            for col_name, col_def in situation_additions:
                try:
                    conn.execute(f"ALTER TABLE current_situation_snapshots ADD COLUMN {col_name} {col_def};")
                except Exception:
                    pass

            # Safe additive migrations for Step 8 Paper/Live environment isolation
            step8_additions = [
                ("trade_proposals", "execution_environment", "TEXT NOT NULL DEFAULT 'PAPER'"),
                ("execution_policies", "execution_environment", "TEXT NOT NULL DEFAULT 'PAPER'"),
                ("execution_policies", "live_execution_enabled", "INTEGER NOT NULL DEFAULT 0"),
                ("execution_intents", "execution_environment", "TEXT NOT NULL DEFAULT 'PAPER'"),
                ("orders", "execution_environment", "TEXT NOT NULL DEFAULT 'PAPER'"),
                ("orders", "is_simulated", "INTEGER NOT NULL DEFAULT 1"),
                ("positions", "execution_environment", "TEXT NOT NULL DEFAULT 'PAPER'"),
            ]
            for tbl, col, col_def in step8_additions:
                try:
                    conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_def};")
                except Exception:
                    pass

            # Safe additive migrations for Step 1 Multimodal Hardening
            step1_hardening_additions = [
                ("visual_events", "regions_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("alignments", "conflicts_json", "TEXT NOT NULL DEFAULT '[]'"),
                ("alignments", "deictic_json", "TEXT NOT NULL DEFAULT '[]'"),
            ]
            for tbl, col, col_def in step1_hardening_additions:
                try:
                    conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_def};")
                except Exception:
                    pass

            # Safe additive migrations for Live Broker AI Marking & Backtest Integration
            chart_marking_additions = [
                ("concept", "TEXT"),
                ("price", "REAL"),
                ("price_range_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("direction", "TEXT"),
                ("source", "TEXT NOT NULL DEFAULT 'SUPERVISOR'"),
                ("rule_reference", "TEXT"),
                ("rule_version", "TEXT NOT NULL DEFAULT '1.0'"),
                ("status", "TEXT NOT NULL DEFAULT 'VALIDATED'"),
                ("visibility", "TEXT NOT NULL DEFAULT 'VISIBLE'"),
                ("created_by", "TEXT NOT NULL DEFAULT 'MARKET_SUPERVISOR'"),
                ("broker_id", "TEXT"),
                ("market_context_json", "TEXT NOT NULL DEFAULT '{}'"),
                ("backtest_run_id", "TEXT"),
                ("timestamp", "REAL"),
            ]
            for col_name, col_def in chart_marking_additions:
                try:
                    conn.execute(f"ALTER TABLE chart_markings ADD COLUMN {col_name} {col_def};")
                except Exception:
                    pass
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_markings_source ON chart_markings(source);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_markings_backtest ON chart_markings(backtest_run_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_markings_broker ON chart_markings(broker_id);")
            except Exception:
                pass

def init_db(db_path: Optional[str] = None) -> DatabaseManager:
    """Initialize database and ensure all schema migrations are applied."""
    return DatabaseManager(db_path)
