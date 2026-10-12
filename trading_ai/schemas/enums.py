"""Enumerations for pipeline stages, statuses, relationships, and knowledge types."""

from enum import Enum


class JobStatus(str, Enum):
    """Lifecycle status of a lecture processing job."""

    PENDING = "pending"
    QUEUED = "queued"
    INGESTING = "ingesting"
    PROCESSING = "processing"
    PAUSED = "paused"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(str, Enum):
    """Discrete pipeline stages from ingestion to final notes and storage."""

    SOURCE = "source"
    INGESTION = "ingestion"
    MEDIA_ANALYSIS = "media_analysis"
    AUDIO = "audio"
    AUDIO_PROCESSING = "audio_processing"
    VISUAL = "visual"
    VISUAL_PROCESSING = "visual_processing"
    TIMELINE = "timeline"
    ALIGNMENT = "alignment"
    AUDIO_VISUAL_ALIGNMENT = "audio_visual_alignment"
    EVIDENCE = "evidence"
    EVIDENCE_EXTRACTION = "evidence_extraction"
    REASONING = "reasoning"
    GEMINI_REASONING = "gemini_reasoning"
    KNOWLEDGE = "knowledge"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    MEMORY = "memory"
    STORAGE = "storage"
    NOTES = "notes"
    NOTES_GENERATION = "notes_generation"
    COMPLETED = "completed"


class AlignmentRelationship(str, Enum):
    """Relationship between what was said and what was shown."""

    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    UNRELATED = "unrelated"
    PARTIALLY_SUPPORTING = "partially_supporting"
    AMBIGUOUS = "ambiguous"


class KnowledgeType(str, Enum):
    """Categorization of extracted knowledge items."""

    CONCEPT = "concept"
    DEFINITION = "definition"
    RULE = "rule"
    SUB_RULE = "sub_rule"
    EXAMPLE = "example"
    COUNTEREXAMPLE = "counterexample"
    EXCEPTION = "exception"
    CONDITION = "condition"
    RELATIONSHIP = "relationship"
    WARNING = "warning"
    TERMINOLOGY = "terminology"


class KnowledgeValidationStatus(str, Enum):
    """Evidence-first epistemic validation status."""

    CONFIRMED = "CONFIRMED"
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    UNCERTAIN = "UNCERTAIN"
    CONTRADICTED = "CONTRADICTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    VALIDATED = "VALIDATED"
    VERIFIED = "VERIFIED"


class SourceType(str, Enum):
    """Universal learning source categories."""

    LOCAL_VIDEO = "LOCAL_VIDEO"
    LOCAL_AUDIO = "LOCAL_AUDIO"
    YOUTUBE = "YOUTUBE"
    WEB_PAGE = "WEB_PAGE"
    WEB_DOCUMENT = "WEB_DOCUMENT"
    PDF = "PDF"
    IMAGE = "IMAGE"
    USER_TEXT = "USER_TEXT"
    FUTURE_CONNECTOR = "FUTURE_CONNECTOR"


class SourceStatus(str, Enum):
    """Availability and acquisition status of learning sources."""

    AVAILABLE = "AVAILABLE"
    PENDING = "PENDING"
    ACQUIRING = "ACQUIRING"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    RESTRICTED = "RESTRICTED"
    FAILED = "FAILED"


class CommandIntent(str, Enum):
    """User command intent categories."""

    SAVE_RULE = "SAVE_RULE"
    CORRECT = "CORRECT"
    REMEMBER = "REMEMBER"
    FORGET = "FORGET"
    REVIEW = "REVIEW"
    EXPLAIN = "EXPLAIN"
    MARK_EXCEPTION = "MARK_EXCEPTION"
    USE_RULE = "USE_RULE"
    ADD_KNOWLEDGE = "ADD_KNOWLEDGE"
    UNKNOWN = "UNKNOWN"
    ANALYZE_SYMBOL = "ANALYZE_SYMBOL"
    ANALYZE_CHART = "ANALYZE_CHART"
    ANALYZE_WATCHLIST = "ANALYZE_WATCHLIST"
    CHECK_HTF = "CHECK_HTF"
    FILTER_VALID_SETUPS = "FILTER_VALID_SETUPS"
    USE_LECTURE_RULES_ONLY = "USE_LECTURE_RULES_ONLY"
    EXPLAIN_SETUP = "EXPLAIN_SETUP"
    MARK_PATTERN = "MARK_PATTERN"
    STOP_MONITORING = "STOP_MONITORING"
    RESUME_MONITORING = "RESUME_MONITORING"
    # Step 5 Command Intents
    SHOW_ALL_LIQUIDITY = "SHOW_ALL_LIQUIDITY"
    SHOW_HIDDEN_SETUPS = "SHOW_HIDDEN_SETUPS"
    SHOW_FVGS = "SHOW_FVGS"
    SHOW_ORDER_BLOCKS = "SHOW_ORDER_BLOCKS"
    SHOW_UNCONFIRMED_SETUPS = "SHOW_UNCONFIRMED_SETUPS"
    SHOW_CONFIRMATION_EVIDENCE = "SHOW_CONFIRMATION_EVIDENCE"
    SHOW_INVALIDATION_CONDITION = "SHOW_INVALIDATION_CONDITION"
    HIDE_BACKGROUND_MARKINGS = "HIDE_BACKGROUND_MARKINGS"
    FOCUS_CURRENT_SETUP = "FOCUS_CURRENT_SETUP"
    SHOW_HTF_CONTEXT = "SHOW_HTF_CONTEXT"


class TaskUnitStatus(str, Enum):
    """Status of an individual serial processing unit under supervisor control."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    BUFFERED = "buffered"
    COMMITTED = "committed"
    FAILED = "failed"


class ErrorSeverity(str, Enum):
    """Classification of errors for recovery."""

    RECOVERABLE = "recoverable"
    PERMANENT = "permanent"


# =====================================================================
# STEP 4: Market Intelligence, Multi-Timeframe, Markings & Screen Enums
# =====================================================================

class MarketDataStatus(str, Enum):
    """Health and freshness status of market data."""

    DATA_AVAILABLE = "DATA_AVAILABLE"
    DATA_DELAYED = "DATA_DELAYED"
    DATA_MISSING = "DATA_MISSING"
    DATA_STALE = "DATA_STALE"
    PROVIDER_ERROR = "PROVIDER_ERROR"


class TimeframeHierarchy(str, Enum):
    """Role of a timeframe in the HTF -> MTF -> LTF decision tree."""

    HTF = "HTF"
    MTF = "MTF"
    LTF = "LTF"


class MarketDirection(str, Enum):
    """Directional trend of market structure."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"


class MarketBias(str, Enum):
    """Synthesized directional bias across timeframes."""

    BULLISH_BIAS = "BULLISH_BIAS"
    BEARISH_BIAS = "BEARISH_BIAS"
    NEUTRAL_BIAS = "NEUTRAL_BIAS"
    MULTI_TIMEFRAME_CONFLICT = "MULTI_TIMEFRAME_CONFLICT"


class MarketStructureType(str, Enum):
    """Identified structural points and breaks in price action."""

    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"
    RANGE = "RANGE"
    CONSOLIDATION = "CONSOLIDATION"


class LevelType(str, Enum):
    """Classification of critical lecture-approved price levels."""

    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"
    PREVIOUS_HIGH = "PREVIOUS_HIGH"
    PREVIOUS_LOW = "PREVIOUS_LOW"
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"
    LIQUIDITY_BUY_SIDE = "LIQUIDITY_BUY_SIDE"
    LIQUIDITY_SELL_SIDE = "LIQUIDITY_SELL_SIDE"
    FVG_TOP = "FVG_TOP"
    FVG_BOTTOM = "FVG_BOTTOM"
    OB_HIGH = "OB_HIGH"
    OB_LOW = "OB_LOW"
    SESSION_HIGH = "SESSION_HIGH"
    SESSION_LOW = "SESSION_LOW"
    DAILY_OPEN = "DAILY_OPEN"
    DAILY_HIGH = "DAILY_HIGH"
    DAILY_LOW = "DAILY_LOW"


class SetupStatus(str, Enum):
    """Epistemic validation and lifecycle status of a trading setup."""

    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    RULE_MATCHED = "RULE_MATCHED"
    CONTEXT_VALID = "CONTEXT_VALID"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    ACTIVE = "ACTIVE"
    OBSERVED = "OBSERVED"
    POSSIBLE = "POSSIBLE"
    VALID = "VALID"
    INVALID = "INVALID"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    UNCERTAIN = "UNCERTAIN"
    CONFLICTED = "CONFLICTED"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class MarkingType(str, Enum):
    """Categories of visual chart markings."""

    FVG = "FVG"
    FAIR_VALUE_GAP = "FVG"
    ORDER_BLOCK = "ORDER_BLOCK"
    LIQUIDITY = "LIQUIDITY"
    BOS = "BOS"
    BREAK_OF_STRUCTURE = "BOS"
    CHOCH = "CHOCH"
    CHANGE_OF_CHARACTER = "CHOCH"
    MARKET_STRUCTURE = "MARKET_STRUCTURE"
    BIAS = "BIAS"
    IMPORTANT_LEVEL = "IMPORTANT_LEVEL"
    SETUP_ZONE = "SETUP_ZONE"
    INVALIDATION_ZONE = "INVALIDATION_ZONE"
    SWEEP = "SWEEP"
    SWING_HIGH = "SWING_HIGH"
    SWING_LOW = "SWING_LOW"
    POI = "POI"
    SUPPORT = "SUPPORT"
    RESISTANCE = "RESISTANCE"
    ENTRY = "ENTRY"
    STOP = "STOP"
    TARGET = "TARGET"
    CONFIRMATION = "CONFIRMATION"
    INVALIDATED_SETUP = "INVALIDATED_SETUP"


class MarkingValidity(str, Enum):
    """Validity and lifecycle status of a plotted chart marking."""

    VALID = "VALID"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    SUPPRESSED = "SUPPRESSED"
    DETECTED = "DETECTED"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    UNCERTAIN = "UNCERTAIN"


class ScreenAccessStatus(str, Enum):
    """Explicit privacy permission status for screen access."""

    SCREEN_ACCESS_DISABLED = "SCREEN_ACCESS_DISABLED"
    SCREEN_ACCESS_REQUESTED = "SCREEN_ACCESS_REQUESTED"
    SCREEN_ACCESS_GRANTED = "SCREEN_ACCESS_GRANTED"
    SCREEN_ACCESS_REVOKED = "SCREEN_ACCESS_REVOKED"


class WatchlistMonitorStatus(str, Enum):
    """Status of the background watchlist monitoring engine."""

    IDLE = "IDLE"
    MONITORING = "MONITORING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


# --- Step 6 Deep Technical & Fundamental Enums ---

class SwingClassification(str, Enum):
    """Detailed classification of swing highs and lows in market structure."""

    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EQH"
    EQUAL_LOW = "EQL"


class DealingRangeZone(str, Enum):
    """Dealing range equilibrium categorization."""

    PREMIUM = "PREMIUM"
    DISCOUNT = "DISCOUNT"
    EQUILIBRIUM = "EQUILIBRIUM"


class EventImpact(str, Enum):
    """Forex Factory economic event impact classification."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NON_ECONOMIC = "NON_ECONOMIC"


class EventStatus(str, Enum):
    """Lifecycle status of a fundamental news event."""

    SCHEDULED = "SCHEDULED"
    ACTIVE = "ACTIVE"
    RELEASED = "RELEASED"
    CANCELLED = "CANCELLED"


class NewsTimingCategory(str, Enum):
    """Event timing classification relative to current market time."""

    UPCOMING_EVENT = "UPCOMING_EVENT"
    EVENT_IMMINENT = "EVENT_IMMINENT"
    EVENT_HAPPENING_NOW = "EVENT_HAPPENING_NOW"
    RECENTLY_RELEASED = "RECENTLY_RELEASED"
    HISTORICAL = "HISTORICAL"
    NO_EVENT = "NO_EVENT"


class NewsRiskLevel(str, Enum):
    """Assessed market volatility risk stemming from fundamental events."""

    NO_RISK = "NO_RISK"
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"
    EXTREME_RISK = "EXTREME_RISK"


class TechFundRelationship(str, Enum):
    """Correlated technical and fundamental situation classification."""

    TECHNICALLY_VALID = "TECHNICALLY_VALID"
    TECHNICALLY_INVALID = "TECHNICALLY_INVALID"
    FUNDAMENTAL_SUPPORT = "FUNDAMENTAL_SUPPORT"
    FUNDAMENTAL_CONFLICT = "FUNDAMENTAL_CONFLICT"
    NEWS_RISK = "NEWS_RISK"
    NEWS_IMMINENT = "NEWS_IMMINENT"
    NEWS_ACTIVE = "NEWS_ACTIVE"
    POST_NEWS_REACTION = "POST_NEWS_REACTION"
    HIGH_VOLATILITY_CONTEXT = "HIGH_VOLATILITY_CONTEXT"
    WAIT_FOR_CONFIRMATION = "WAIT_FOR_CONFIRMATION"
    CONTEXT_ALIGNED = "CONTEXT_ALIGNED"
    CONTEXT_CONFLICTED = "CONTEXT_CONFLICTED"
    UNCERTAIN = "UNCERTAIN"
    DATA_STALE = "DATA_STALE"
    FUNDAMENTAL_UNAVAILABLE = "FUNDAMENTAL_UNAVAILABLE"


class ProviderStatus(str, Enum):
    """Health and connection status of a fundamental data provider."""

    OPERATIONAL = "OPERATIONAL"
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    STALE = "STALE"
    RATE_LIMITED = "RATE_LIMITED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    UNAVAILABLE = "UNAVAILABLE"


# --- Step 7 AI Trading Companion Enums ---

class AlertDecision(str, Enum):
    """Low-noise alert engine decision."""

    SUPPRESS = "SUPPRESS"
    TRACK = "TRACK"
    ALERT = "ALERT"


class AlertPriority(str, Enum):
    """Hierarchical alert priority levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AlertLifecycleState(str, Enum):
    """Persistent lifecycle states for an alert candidate."""

    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    QUALIFIED = "QUALIFIED"
    ALERTED = "ALERTED"
    UPDATED = "UPDATED"
    RESOLVED = "RESOLVED"
    SUPPRESSED = "SUPPRESSED"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class AlertType(str, Enum):
    """Specific contextual event category for trading alerts."""

    SETUP_HIGH_PROBABILITY = "SETUP_HIGH_PROBABILITY"
    SETUP_LOW_PROBABILITY = "SETUP_LOW_PROBABILITY"
    KEY_CONFIRMATION = "KEY_CONFIRMATION"
    MARKET_BIAS_CHANGE = "MARKET_BIAS_CHANGE"
    LIQUIDITY_EVENT = "LIQUIDITY_EVENT"
    POI_REACTION = "POI_REACTION"
    NEWS_IMMINENT = "NEWS_IMMINENT"
    NEWS_RELEASED = "NEWS_RELEASED"
    POST_NEWS_REACTION = "POST_NEWS_REACTION"
    FUNDAMENTAL_CONFLICT = "FUNDAMENTAL_CONFLICT"
    SETUP_INVALIDATED = "SETUP_INVALIDATED"
    RISK_RULE_WARNING = "RISK_RULE_WARNING"
    OVERTRADING_WARNING = "OVERTRADING_WARNING"
    DAILY_LOSS_LIMIT_WARNING = "DAILY_LOSS_LIMIT_WARNING"
    MAX_TRADES_WARNING = "MAX_TRADES_WARNING"
    STOP_LOSS_REQUIRED = "STOP_LOSS_REQUIRED"
    COOLDOWN_REQUIRED = "COOLDOWN_REQUIRED"


class AlertQuality(str, Enum):
    """Qualitative confidence classification (no fabricated statistical claims)."""

    LOW_QUALITY = "LOW_QUALITY"
    MEDIUM_QUALITY = "MEDIUM_QUALITY"
    HIGH_QUALITY = "HIGH_QUALITY"


class VoiceIntent(str, Enum):
    """Supported interactive voice and companion conversational intents."""

    MARKET_STATUS = "MARKET_STATUS"
    EXPLAIN_BIAS = "EXPLAIN_BIAS"
    EXPLAIN_SETUP = "EXPLAIN_SETUP"
    NEWS_QUERY = "NEWS_QUERY"
    SCENARIO_ANALYSIS = "SCENARIO_ANALYSIS"
    COMPANION_SCENARIO = "COMPANION_SCENARIO"
    SETUP_VALIDATION = "SETUP_VALIDATION"
    MARKET_HISTORY = "MARKET_HISTORY"
    RISK_STATUS = "RISK_STATUS"
    DISCIPLINE_STATUS = "DISCIPLINE_STATUS"
    ACTIVE_ALERTS = "ACTIVE_ALERTS"
    POSITIONS_STATUS = "POSITIONS_STATUS"
    PROPOSALS_STATUS = "PROPOSALS_STATUS"
    EXECUTION_STATUS = "EXECUTION_STATUS"
    WHY_ALERT = "WHY_ALERT"
    GENERAL_MARKET_QA = "GENERAL_MARKET_QA"


class DisciplineViolationType(str, Enum):
    """Classified trading discipline breaches."""

    RULE_NOT_CONFIGURED = "RULE_NOT_CONFIGURED"
    STOP_LOSS_MISSING = "STOP_LOSS_MISSING"
    DAILY_LOSS_EXCEEDED = "DAILY_LOSS_EXCEEDED"
    MAX_TRADES_EXCEEDED = "MAX_TRADES_EXCEEDED"
    RISK_LIMIT_EXCEEDED = "RISK_LIMIT_EXCEEDED"
    OVERTRADING_TRIGGERED = "OVERTRADING_TRIGGERED"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    SESSION_OUT_OF_BOUNDS = "SESSION_OUT_OF_BOUNDS"


# ==========================================
# STEP 8: HEDGE FUND EXECUTION & RISK ENUMS
# ==========================================


class EngagementMode(str, Enum):
    """Capital allocation and risk scaling mode based on market regime."""

    DEFENSIVE = "DEFENSIVE"  # Max 25% base size for choppy/ranging conditions
    RISK_OFF = "RISK_OFF"  # 0% / Wait during major event risk or high uncertainty
    CALCULATED_AGGRESSION = "CALCULATED_AGGRESSION"  # 100% base size for A+ setups
    PYRAMIDING = "PYRAMIDING"  # Add to winning position with protected SL only


class ProposalLifecycleState(str, Enum):
    """Strict lifecycle transitions for institutional trade proposals."""

    PROPOSED = "PROPOSED"
    PRE_TRADE_CHECKING = "PRE_TRADE_CHECKING"
    CONFLUENCE_VALIDATED = "CONFLUENCE_VALIDATED"
    RISK_VALIDATED = "RISK_VALIDATED"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    SUBMITTING = "SUBMITTING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    MANAGED = "MANAGED"
    CLOSED = "CLOSED"
    REJECTED = "REJECTED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    BROKER_ERROR = "BROKER_ERROR"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    UNKNOWN = "UNKNOWN"


class ExecutionMode(str, Enum):
    """Engagement execution authorization mode."""

    MANUAL = "MANUAL"  # Explicit human approval mandatory before execution
    AUTO = "AUTO"  # Automated execution governed strictly by RiskEngine & ExecutionGate


class ExecutionEnvironment(str, Enum):
    """Trading execution environment."""

    PAPER = "PAPER"  # Deterministic paper simulation environment (Default)
    LIVE = "LIVE"    # Real broker live execution environment (Strict activation only)


class UnifiedExecutionTestingMode(str, Enum):
    """Single-click operational mode selector strictly enforced by ExecutionGate."""

    BACKTEST = "BACKTEST"        # Historical replay simulation only; zero broker interaction
    PAPER = "PAPER"              # Simulated execution with live or replay market feeds
    MANUAL_LIVE = "MANUAL_LIVE"  # Live broker market data + AI marking; trades require manual confirmation
    AUTO_LIVE = "AUTO_LIVE"      # Explicitly authorized automated live execution governed by RiskEngine


class OrderType(str, Enum):
    """Supported order routing types."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    """Execution status returned by BrokerAdapter."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    BROKER_ERROR = "BROKER_ERROR"
    UNKNOWN = "UNKNOWN"


class PositionStatus(str, Enum):
    """Open status for managed live positions."""

    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"


class ConfluenceCategory(str, Enum):
    """Independent technical and fundamental confirmation categories."""

    MARKET_STRUCTURE = "MARKET_STRUCTURE"
    LIQUIDITY = "LIQUIDITY"
    POI = "POI"
    CONFIRMATION_TRIGGER = "CONFIRMATION_TRIGGER"
    VOLUME = "VOLUME"
    FUNDAMENTAL = "FUNDAMENTAL"
    NEWS = "NEWS"
    MOMENTUM = "MOMENTUM"
    LECTURE_RULE = "LECTURE_RULE"


class PositionManagementAction(str, Enum):
    """Auditable actions taken during position lifecycle."""

    INITIAL_SL_SET = "INITIAL_SL_SET"
    TP1_REACHED_PARTIAL_BOOKED = "TP1_REACHED_PARTIAL_BOOKED"
    SL_MOVED_TO_ENTRY = "SL_MOVED_TO_ENTRY"
    TP2_REACHED = "TP2_REACHED"
    FINAL_TARGET_REACHED = "FINAL_TARGET_REACHED"
    TRAILING_SL_UPDATED = "TRAILING_SL_UPDATED"
    POSITION_CLOSED = "POSITION_CLOSED"
    PYRAMIDED = "PYRAMIDED"


# ==============================================================================
# STEP 9: BACKTESTING, TRADING-BRAIN VALIDATION & LIVE-MARKET PROMOTION ENUMS
# ==============================================================================

class AssetClass(str, Enum):
    """Asset class categorization for strict performance and testing isolation."""

    FOREX = "FOREX"
    CRYPTO = "CRYPTO"
    COMMODITY = "COMMODITY"


class BacktestMode(str, Enum):
    """Supported backtesting execution modes."""

    MANUAL_REPLAY = "MANUAL_REPLAY"
    AI_BACKTEST = "AI_BACKTEST"
    RECENT_LECTURE = "RECENT_LECTURE"
    HISTORICAL_RULE = "HISTORICAL_RULE"
    COMBINED_RULE = "COMBINED_RULE"


class BacktestStatus(str, Enum):
    """Execution status for historical backtest jobs."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RuleLifecycleState(str, Enum):
    """
    Strict rule qualification and promotion lifecycle states.

    Progression:
      LEARNED -> CANDIDATE -> BACKTESTING -> BACKTEST_COMPLETE ->
      USER_REVIEW -> APPROVED_FOR_PAPER -> PAPER_VALIDATION ->
      USER_LIVE_APPROVAL -> LIVE_ELIGIBLE -> LIVE_ACTIVE
    """

    LEARNED = "LEARNED"
    CANDIDATE = "CANDIDATE"
    BACKTESTING = "BACKTESTING"
    BACKTEST_COMPLETE = "BACKTEST_COMPLETE"
    USER_REVIEW = "USER_REVIEW"
    APPROVED_FOR_PAPER = "APPROVED_FOR_PAPER"
    PAPER_VALIDATION = "PAPER_VALIDATION"
    USER_LIVE_APPROVAL = "USER_LIVE_APPROVAL"
    LIVE_ELIGIBLE = "LIVE_ELIGIBLE"
    LIVE_ACTIVE = "LIVE_ACTIVE"
    REJECTED = "REJECTED"
    INVALIDATED = "INVALIDATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    CONFLICTED = "CONFLICTED"
    EXPIRED = "EXPIRED"


class TradeSource(str, Enum):
    """Distinguishes user-recorded manual replay trades from AI-executed trades."""

    USER = "USER"
    AI = "AI"


# ==============================================================================
# STEP 1 HARDENING: MULTIMODAL PERCEPTION, VISUAL GROUNDING & EPISTEMIC ENUMS
# ==============================================================================

class VisualRegionType(str, Enum):
    """Fine-grained classification of grounded visual bounding boxes."""

    CHART = "CHART"
    CANDLE = "CANDLE"
    PRICE_AXIS = "PRICE_AXIS"
    TIME_AXIS = "TIME_AXIS"
    INDICATOR = "INDICATOR"
    DRAWING = "DRAWING"
    ORDER_BLOCK = "ORDER_BLOCK"
    FVG = "FVG"
    LIQUIDITY_LEVEL = "LIQUIDITY_LEVEL"
    SUPPORT_RESISTANCE = "SUPPORT_RESISTANCE"
    TEXT = "TEXT"
    ANNOTATION = "ANNOTATION"
    CURSOR = "CURSOR"
    HIGHLIGHTED_ZONE = "HIGHLIGHTED_ZONE"
    SPEAKER = "SPEAKER"
    SLIDE = "SLIDE"
    UI = "UI"
    UNKNOWN = "UNKNOWN"


class EpistemicStatementType(str, Enum):
    """Epistemic classification distinguishing rules, examples, hypotheses, and caveats."""

    RULE = "RULE"
    EXAMPLE = "EXAMPLE"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    EXCEPTION = "EXCEPTION"
    WARNING = "WARNING"
    PERSONAL_PREFERENCE = "PERSONAL_PREFERENCE"
    HYPOTHESIS = "HYPOTHESIS"
    OBSERVATION = "OBSERVATION"
    QUESTION = "QUESTION"
    UNCERTAIN_STATEMENT = "UNCERTAIN_STATEMENT"
    DEICTIC_REFERENCE = "DEICTIC_REFERENCE"


class CrossModalConflictType(str, Enum):
    """Categories of mismatch between spoken claim and visual chart evidence."""

    NONE = "NONE"
    DIRECTIONAL_CONTRADICTION = "DIRECTIONAL_CONTRADICTION"
    TIMEFRAME_MISMATCH = "TIMEFRAME_MISMATCH"
    LEVEL_DISCREPANCY = "LEVEL_DISCREPANCY"
    UNRESOLVED_REFERENCE = "UNRESOLVED_REFERENCE"
    UNVERIFIED_VISUAL_CLAIM = "UNVERIFIED_VISUAL_CLAIM"


class ReferenceResolutionStatus(str, Enum):
    """Epistemic status of deictic references ('this candle', 'here', 'that zone')."""

    RESOLVED = "RESOLVED"
    PARTIALLY_RESOLVED = "PARTIALLY_RESOLVED"
    REFERENCE_UNRESOLVED = "REFERENCE_UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class MultimodalConfidence(str, Enum):
    """Confidence levels for multi-modal perception and grounding."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ==============================================================================
# STEP 10: TRADING JOURNAL, BEHAVIORAL OBSERVATION, AND LESSON ENGINE ENUMS
# ==============================================================================

class JournalEnvironment(str, Enum):
    """Strictly isolated execution environments for journal records."""

    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


class JournalBacktestType(str, Enum):
    """Distinguishes manual replay from AI automated backtests in the journal."""

    MANUAL_BACKTEST = "MANUAL_BACKTEST"
    AI_BACKTEST = "AI_BACKTEST"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class TradeOutcome(str, Enum):
    """Final outcome classification of a journaled trade."""

    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    OPEN = "OPEN"
    CANCELLED = "CANCELLED"


class ExitReasonCategory(str, Enum):
    """Root-cause attribution for why a trade won or lost, anchored in deterministic evidence."""

    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    THESIS_CORRECT = "THESIS_CORRECT"
    THESIS_WRONG = "THESIS_WRONG"
    ENTRY_TIMING = "ENTRY_TIMING"
    INVALIDATION = "INVALIDATION"
    NEWS_EVENT = "NEWS_EVENT"
    SPREAD_SLIPPAGE = "SPREAD_SLIPPAGE"
    EXECUTION_ISSUE = "EXECUTION_ISSUE"
    PREMATURE_EXIT = "PREMATURE_EXIT"
    LATE_ENTRY = "LATE_ENTRY"
    RULE_VIOLATION = "RULE_VIOLATION"
    INSUFFICIENT_CONFIRMATION = "INSUFFICIENT_CONFIRMATION"
    MTF_CONFLICT = "MTF_CONFLICT"
    OTHER = "OTHER"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNKNOWN = "UNKNOWN"


class ObservedBehaviorType(str, Enum):
    """Observable human and execution behaviors anchored strictly in timestamps and order events."""

    FOMO = "FOMO"
    REVENGE_TRADING = "REVENGE_TRADING"
    OVERTRADING = "OVERTRADING"
    HESITATION = "HESITATION"
    PREMATURE_EXIT = "PREMATURE_EXIT"
    MOVING_SL = "MOVING_SL"
    MOVING_TP_IRRATIONALLY = "MOVING_TP_IRRATIONALLY"
    ENTERING_WITHOUT_CONFIRMATION = "ENTERING_WITHOUT_CONFIRMATION"
    BREAKING_DAILY_LOSS_LIMIT = "BREAKING_DAILY_LOSS_LIMIT"
    BREAKING_MAX_TRADE_RULE = "BREAKING_MAX_TRADE_RULE"
    IGNORING_JOURNAL_WARNING = "IGNORING_JOURNAL_WARNING"
    REPEATED_RULE_VIOLATION = "REPEATED_RULE_VIOLATION"
    CHASING_PRICE = "CHASING_PRICE"
    IMPULSIVE_ENTRY = "IMPULSIVE_ENTRY"
    DISCIPLINED_EXECUTION = "DISCIPLINED_EXECUTION"


class LessonLifecycleState(str, Enum):
    """Strict promotion lifecycle for journal-extracted lessons."""

    OBSERVED = "OBSERVED"
    ANALYZED = "ANALYZED"
    LESSON_CANDIDATE = "LESSON_CANDIDATE"
    USER_REVIEW = "USER_REVIEW"
    VALIDATED = "VALIDATED"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    CONFLICTED = "CONFLICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class PreTradeReviewStatus(str, Enum):
    """Assessment recommendation from the Pre-Trade Journal Review."""

    APPROVED = "APPROVED"
    WARNING = "WARNING"
    REJECTED = "REJECTED"
    INSUFFICIENT_PLAN = "INSUFFICIENT_PLAN"


class EventPriority(str, Enum):
    """Priority classifications for market and system events in Step 11."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    BACKGROUND = "BACKGROUND"


class SystemEventType(str, Enum):
    """Standardized event types for high-performance reactive event bus."""

    PRICE_UPDATE = "PRICE_UPDATE"
    CANDLE_CLOSED = "CANDLE_CLOSED"
    STRUCTURE_CHANGED = "STRUCTURE_CHANGED"
    BOS_DETECTED = "BOS_DETECTED"
    CHOCH_DETECTED = "CHOCH_DETECTED"
    LIQUIDITY_CHANGED = "LIQUIDITY_CHANGED"
    POI_REACTION = "POI_REACTION"
    SETUP_CHANGED = "SETUP_CHANGED"
    SETUP_CONFIRMED = "SETUP_CONFIRMED"
    SETUP_INVALIDATED = "SETUP_INVALIDATED"
    NEWS_EVENT = "NEWS_EVENT"
    NEWS_RELEASED = "NEWS_RELEASED"
    BIAS_CHANGED = "BIAS_CHANGED"
    RISK_CHANGED = "RISK_CHANGED"
    DISCIPLINE_WARNING = "DISCIPLINE_WARNING"
    TRADE_PROPOSED = "TRADE_PROPOSED"
    ORDER_UPDATE = "ORDER_UPDATE"
    POSITION_UPDATE = "POSITION_UPDATE"
    JOURNAL_UPDATE = "JOURNAL_UPDATE"
    SECURITY_ALERT = "SECURITY_ALERT"
    DEVICE_CONNECTED = "DEVICE_CONNECTED"
    DEVICE_REVOKED = "DEVICE_REVOKED"
    NOTIFICATION_DISPATCHED = "NOTIFICATION_DISPATCHED"
    CAPABILITY_REQUESTED = "CAPABILITY_REQUESTED"
    CAPABILITY_APPROVED = "CAPABILITY_APPROVED"
    CAPABILITY_DENIED = "CAPABILITY_DENIED"
    THREAT_DETECTED = "THREAT_DETECTED"
    SECURITY_AUDIT_LOGGED = "SECURITY_AUDIT_LOGGED"
    TASK_HANG_DETECTED = "TASK_HANG_DETECTED"
    PERFORMANCE_DEGRADED = "PERFORMANCE_DEGRADED"
    RECOVERY_TRIGGERED = "RECOVERY_TRIGGERED"


class MemoryTier(str, Enum):
    """Hot, Warm, and Cold tiered memory states."""

    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"


class UpgradeStatus(str, Enum):
    """Lifecycle status for controlled component and model upgrades."""

    CANDIDATE = "CANDIDATE"
    SANDBOX_TESTING = "SANDBOX_TESTING"
    BENCHMARK_PASSED = "BENCHMARK_PASSED"
    PAPER_VALIDATED = "PAPER_VALIDATED"
    AWAITING_USER_APPROVAL = "AWAITING_USER_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"


class DataPrivacyClassification(str, Enum):
    """Privacy classifications for data processing location and exposure."""

    LOCAL_ONLY = "LOCAL_ONLY"
    USER_APPROVED_CLOUD = "USER_APPROVED_CLOUD"
    NON_SENSITIVE_EXTERNAL = "NON_SENSITIVE_EXTERNAL"
    BLOCKED_EXTERNAL = "BLOCKED_EXTERNAL"


class PCCapability(str, Enum):
    """Granular, capability-based PC permissions (Default DENY)."""

    SCREEN_READ = "SCREEN_READ"
    SCREEN_ANALYZE = "SCREEN_ANALYZE"
    ALERT_TRIGGER = "ALERT_TRIGGER"
    VOICE_OUTPUT = "VOICE_OUTPUT"
    NOTIFICATION_SEND = "NOTIFICATION_SEND"
    APP_READ = "APP_READ"
    APP_INTERACT = "APP_INTERACT"
    FILE_READ = "FILE_READ"
    FILE_WRITE = "FILE_WRITE"
    TERMINAL_EXECUTE = "TERMINAL_EXECUTE"
    SYSTEM_SETTINGS = "SYSTEM_SETTINGS"


class CapabilityState(str, Enum):
    """State of capability authorization."""

    DENIED = "DENIED"
    REQUESTED = "REQUESTED"
    APPROVED_ONCE = "APPROVED_ONCE"
    APPROVED_ALWAYS = "APPROVED_ALWAYS"
    REVOKED = "REVOKED"


class DeviceStatus(str, Enum):
    """Cross-device pairing and connection lifecycle states."""

    UNPAIRED = "UNPAIRED"
    PAIRING_REQUESTED = "PAIRING_REQUESTED"
    USER_APPROVAL = "USER_APPROVAL"
    AUTHENTICATED = "AUTHENTICATED"
    TRUSTED = "TRUSTED"
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    DISCONNECTED = "DISCONNECTED"


class NotificationPriority(str, Enum):
    """Priority levels for cross-device push notifications."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class VoiceProcessingState(str, Enum):
    """State machine for two-way voice processing."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    RESPONDING = "RESPONDING"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


# ============================================================================
# Step 13: Data Security, Cyber Wall, Watchdog & Auto-Recovery Enums
# ============================================================================

class ThreatLevel(str, Enum):
    """Cyber Wall threat assessment level."""

    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    BLOCKED = "BLOCKED"
    CRITICAL = "CRITICAL"


class HealthState(str, Enum):
    """Subsystem and service operational health state."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    RECOVERING = "RECOVERING"


class RecoveryClassification(str, Enum):
    """Diagnostic categorization of error recovery pathways."""

    AUTO_RECOVERABLE = "AUTO_RECOVERABLE"
    USER_APPROVAL_REQUIRED = "USER_APPROVAL_REQUIRED"
    NON_RECOVERABLE = "NON_RECOVERABLE"
    SECURITY_BLOCKED = "SECURITY_BLOCKED"


class TaskWatchdogState(str, Enum):
    """Anti-hang task watchdog states."""

    RUNNING = "RUNNING"
    SUSPECTED_HANG = "SUSPECTED_HANG"
    GRACEFUL_CANCEL = "GRACEFUL_CANCEL"
    FORCE_STOPPING = "FORCE_STOPPING"
    TERMINATED = "TERMINATED"
    RECOVERED = "RECOVERED"


class AudioHardwareMode(str, Enum):
    """Voice audio peripheral hardware mode."""

    REAL_HARDWARE = "REAL_HARDWARE"
    TEST_FALLBACK = "TEST_FALLBACK"


class KnowledgeProtectionState(str, Enum):
    """Knowledge store integrity and protection states."""

    VALID = "VALID"
    CORRUPTED = "CORRUPTED"
    RESTORED = "RESTORED"
    PROTECTED = "PROTECTED"


# =====================================================================
# STEP 14: Separate File Architecture & Multi-Source Trade Logic Enums
# =====================================================================
class SourceDomain(str, Enum):
    """Four isolated physical knowledge storage domains."""

    YOUTUBE_LECTURES = "01_YouTube_Lectures"
    MENTORSHIP_SESSIONS = "02_Mentorship_Sessions"
    LIVE_TRADE_LOGS = "03_Live_Trade_Logs"
    TRADING_TECH_WORK = "04_Trading_Tech_Work"


class SourceAuthority(str, Enum):
    """Source authority and priority hierarchy for rule precedence."""

    USER_OVERRIDE = "USER_OVERRIDE"         # Priority 1: User Direct Command / Correction
    MENTORSHIP_RULE = "MENTORSHIP_RULE"     # Priority 2: Approved Mentorship Rule
    JOURNAL_LESSON = "JOURNAL_LESSON"       # Priority 3: Validated Personal Trading / Journal Learning
    LECTURE_RULE = "LECTURE_RULE"           # Priority 4: Approved Lecture Rule
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE" # Priority 5: General / Low-Confidence Knowledge

    # Canonical aliases
    APPROVED_MENTOR = "MENTORSHIP_RULE"
    APPROVED_LECTURE = "LECTURE_RULE"
    VALIDATED_JOURNAL = "JOURNAL_LESSON"



class RelationshipType(str, Enum):
    """Cross-source semantic and epistemic relationship types."""

    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    REFINES = "REFINES"
    OVERRIDES = "OVERRIDES"
    EXCEPTION_TO = "EXCEPTION_TO"
    DERIVED_FROM = "DERIVED_FROM"
    VALIDATED_BY = "VALIDATED_BY"
    FAILED_BY = "FAILED_BY"
    RELATED_TO = "RELATED_TO"


class TradeDecisionState(str, Enum):
    """Core Trade Logic and Master Decision Engine states with zero ungrounded confidence."""

    NO_SETUP = "NO_SETUP"
    INSUFFICIENT_CONFLUENCE = "INSUFFICIENT_CONFLUENCE"
    CONFLICTED = "CONFLICTED"
    VALID_SETUP = "VALID_SETUP"
    HIGH_CONFLUENCE = "HIGH_CONFLUENCE"
    CONFIRMED_SETUP = "CONFIRMED_SETUP"
    ACTIVE_SETUP = "ACTIVE_SETUP"
    UNCERTAIN = "UNCERTAIN"
    INVALIDATED = "INVALIDATED"
    NO_TRADE = "NO_TRADE"
    WATCH = "WATCH"
    SETUP_FORMING = "SETUP_FORMING"
    BLOCKED = "BLOCKED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


class ConceptDirection(str, Enum):
    """Deterministic directional polarity of an analytical concept."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


# ============================================================================
# Step 16: 24×7 Cloud-Resident Architecture & Advanced Event Enums
# ============================================================================

class CloudRuntimeState(str, Enum):
    """Execution state of the 24×7 cloud runtime."""

    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    PAUSED = "PAUSED"
    ERROR = "ERROR"
    RECOVERING = "RECOVERING"


class EventPriorityLevel(str, Enum):
    """Step 16 strict event and notification priority levels."""

    P0_SYSTEM_CRITICAL = "P0_SYSTEM_CRITICAL"
    P1_SECURITY = "P1_SECURITY"
    P2_ACTIVE_POSITION_RISK = "P2_ACTIVE_POSITION_RISK"
    P3_CONFIRMED_SETUP = "P3_CONFIRMED_SETUP"
    P4_HIGH_VALUE_MARKET_EVENT = "P4_HIGH_VALUE_MARKET_EVENT"
    P5_INFORMATIONAL = "P5_INFORMATIONAL"


class NotificationChannel(str, Enum):
    """Multi-channel notification delivery channels."""

    WEBSOCKET = "WEBSOCKET"
    PUSH = "PUSH"
    SMS = "SMS"
    VOICE = "VOICE"
    EMAIL = "EMAIL"
    IN_APP = "IN_APP"


class NotificationDeliveryStatus(str, Enum):
    """Standardized delivery status across all external and internal providers."""

    SUCCESS = "SUCCESS"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    UNAVAILABLE = "UNAVAILABLE"
    AUTH_FAILED = "AUTH_FAILED"
    NETWORK_ERROR = "NETWORK_ERROR"
    UNKNOWN = "UNKNOWN"
    QUEUED = "QUEUED"
    FILTERED = "FILTERED"


class NotificationPrivacyLevel(str, Enum):
    """User privacy settings for notification content visibility."""

    NORMAL = "NORMAL"
    SECURE = "SECURE"


class SetupNotificationState(str, Enum):
    """Step 16 setup lifecycle notification state machine."""

    DETECTED = "DETECTED"
    TRACKING = "TRACKING"
    RULE_MATCHED = "RULE_MATCHED"
    CONTEXT_VALID = "CONTEXT_VALID"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    NOTIFICATION_SENT = "NOTIFICATION_SENT"
    ACTIVE = "ACTIVE"
    INVALIDATED = "INVALIDATED"


class FeatureImplementationStatus(str, Enum):
    """Authoritative status of an application or UI feature."""

    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    BROKEN = "BROKEN"
    DISCONNECTED = "DISCONNECTED"
    DUPLICATED = "DUPLICATED"
    UNUSED = "UNUSED"
    UNTESTED = "UNTESTED"
    UNKNOWN = "UNKNOWN"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"











