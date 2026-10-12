"""Application Component Registry & Authoritative Guide.

Provides factual, grounded knowledge of all dashboard tabs, panels, features,
and routes to prevent AI hallucination regarding application structure.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel
from trading_ai.schemas.chat import AppFeatureItem, AppGuideResponse, ChatNavigationAction


class AppTabInfo(BaseModel):
    tab_id: str
    number: int
    name: str


APP_FEATURES_REGISTRY: List[AppFeatureItem] = [
    # Top Bar & Global
    AppFeatureItem(
        feature_id="ai_chat",
        name="AI Chat & Assistant Drawer",
        description="Global conversational assistant, application guide, and market reasoning panel.",
        tab_id="global-drawer",
        tab_name="Global Drawer",
        panel_id="ai-chat-drawer",
        status="AVAILABLE",
        how_to_access="Click '💬 AI CHAT & GUIDE' in the top header bar or the floating launcher on bottom-right.",
        related_features=["app_tour", "companion", "live_market"],
    ),
    AppFeatureItem(
        feature_id="model_pills",
        name="AI Model Status Indicators",
        description="Live status indicators for Fast, Reasoning, Embedding, and Supervisor models.",
        tab_id="top-header",
        tab_name="Top Header",
        panel_id="model-pills",
        status="AVAILABLE",
        how_to_access="Visible in the top right of the dashboard header.",
        related_features=["token_safety"],
    ),
    # Left Column
    AppFeatureItem(
        feature_id="lecture_ingestion",
        name="Lecture & Video Source Ingestion",
        description="Dropzone to upload trading video/audio files (MP4, MKV, WAV) or paste YouTube/Web URLs.",
        tab_id="left-column",
        tab_name="Left Panel",
        panel_id="dropzone",
        status="AVAILABLE",
        how_to_access="Located in the top left card of the dashboard.",
        related_features=["supervisor_status", "transcripts"],
    ),
    AppFeatureItem(
        feature_id="supervisor_status",
        name="Supervisor & Background Job Status",
        description="Multimodal pipeline progress bar, stage monitor, pause/resume/cancel controls, and Commit Barrier.",
        tab_id="left-column",
        tab_name="Left Panel",
        panel_id="barrier-display",
        status="AVAILABLE",
        how_to_access="Located in the second card of the left column.",
        related_features=["lecture_ingestion"],
    ),
    AppFeatureItem(
        feature_id="command_console",
        name="User Command Console",
        description="Direct instruction console to save trading rules, correct mistakes, or add exceptions.",
        tab_id="left-column",
        tab_name="Left Panel",
        panel_id="user-command-input",
        status="AVAILABLE",
        how_to_access="Located in the third card of the left column.",
        related_features=["knowledge_rules"],
    ),
    AppFeatureItem(
        feature_id="token_safety",
        name="Token Safety Observability",
        description="Real-time tracking of AI input/output token usage with 20% safety headroom reserve enforcement.",
        tab_id="left-column",
        tab_name="Left Panel",
        panel_id="tokens-calls",
        status="AVAILABLE",
        how_to_access="Located at the bottom of the left column.",
        related_features=["model_pills"],
    ),
    # Tabs 1 to 8: Knowledge & Extraction
    AppFeatureItem(
        feature_id="notes",
        name="Lecture Notes",
        description="Comprehensive structured markdown notes synthesized from processed lectures.",
        tab_id="tab-notes",
        tab_name="1. Notes",
        panel_id="notes-content",
        status="AVAILABLE",
        how_to_access="Click Tab '1. Notes' in the right workspace container.",
        related_features=["knowledge_rules", "transcripts"],
    ),
    AppFeatureItem(
        feature_id="knowledge_rules",
        name="Knowledge & Trading Rules",
        description="Formal verified trading rules, core concepts, and market exceptions extracted from lectures.",
        tab_id="tab-knowledge",
        tab_name="2. Knowledge",
        panel_id="knowledge-content",
        status="AVAILABLE",
        how_to_access="Click Tab '2. Knowledge'. Filter by Rules, Concepts, or Exceptions.",
        related_features=["command_console", "notes"],
    ),
    AppFeatureItem(
        feature_id="concepts",
        name="Core Concepts",
        description="Trading concept taxonomy, definitions, and terminology.",
        tab_id="tab-concepts",
        tab_name="3. Concepts",
        panel_id="concepts-content",
        status="AVAILABLE",
        how_to_access="Click Tab '3. Concepts'.",
        related_features=["knowledge_rules"],
    ),
    AppFeatureItem(
        feature_id="doubts",
        name="Doubts & Uncertainties",
        description="Detected contradictions, caveats, and unverified statements requiring review.",
        tab_id="tab-doubts",
        tab_name="4. Doubts",
        panel_id="doubts-content",
        status="AVAILABLE",
        how_to_access="Click Tab '4. Doubts'.",
        related_features=["knowledge_rules"],
    ),
    AppFeatureItem(
        feature_id="alignment",
        name="Audio-Visual Alignment",
        description="Cross-modal correlation timestamps connecting what the speaker said to what was on screen.",
        tab_id="tab-alignment",
        tab_name="5. Alignment",
        panel_id="alignment-content",
        status="AVAILABLE",
        how_to_access="Click Tab '5. Alignment'.",
        related_features=["transcripts", "visuals"],
    ),
    AppFeatureItem(
        feature_id="transcripts",
        name="Audio Transcripts",
        description="Segment-by-segment speech transcript with exact timestamps.",
        tab_id="tab-transcript",
        tab_name="6. Transcripts",
        panel_id="transcript-content",
        status="AVAILABLE",
        how_to_access="Click Tab '6. Transcripts'.",
        related_features=["alignment"],
    ),
    AppFeatureItem(
        feature_id="visuals",
        name="Visual Chart Events",
        description="Detected chart bounding boxes, candle structures, drawings, and indicator displays.",
        tab_id="tab-visual",
        tab_name="7. Visuals",
        panel_id="visual-content",
        status="AVAILABLE",
        how_to_access="Click Tab '7. Visuals'.",
        related_features=["alignment"],
    ),
    AppFeatureItem(
        feature_id="memory",
        name="Memory & Revisions",
        description="Dual-layer vector embeddings, context snapshots, and knowledge revision lineage.",
        tab_id="tab-memory",
        tab_name="8. Memory",
        panel_id="memory-content",
        status="AVAILABLE",
        how_to_access="Click Tab '8. Memory'.",
        related_features=["knowledge_rules"],
    ),
    # Tab 9: Live Market
    AppFeatureItem(
        feature_id="watchlist",
        name="Watchlist & Screen Privacy Guard",
        description="Monitored currency and crypto symbols (BTC/USD, ETH/USD, SOL/USD, EUR/USD, XAU/USD) with screen boundary access controls.",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="market-watchlist-buttons",
        status="AVAILABLE",
        how_to_access="Click Tab '9. Live Market'. Watchlist buttons are in the top header card.",
        related_features=["mtf_analysis", "live_market"],
    ),
    AppFeatureItem(
        feature_id="mtf_analysis",
        name="Multi-Timeframe Analysis Grid",
        description="Simultaneous HTF (1d), MTF (1h), and LTF (15m) bias and market structure alignment.",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="market-active-sym-title",
        status="AVAILABLE",
        how_to_access="Click Tab '9. Live Market' -> Scroll below Watchlist.",
        related_features=["watchlist", "chart_markings"],
    ),
    AppFeatureItem(
        feature_id="chart_markings",
        name="Chart Markings & Price Levels",
        description="Identified support/resistance, order blocks, fair value gaps (FVG), and invalidation levels.",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="market-markings-body",
        status="AVAILABLE",
        how_to_access="Click Tab '9. Live Market' -> Section 2.",
        related_features=["mtf_analysis"],
    ),
    AppFeatureItem(
        feature_id="fundamental_news",
        name="Fundamental Intelligence & Calendar",
        description="Economic calendar events, time-to-event alerts, and technical-fundamental conflict detection.",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="fund-next-event",
        status="AVAILABLE",
        how_to_access="Click Tab '9. Live Market' -> Fundamental Intelligence card.",
        related_features=["companion"],
    ),
    AppFeatureItem(
        feature_id="companion",
        name="Trading Companion (Voice/Text)",
        description="Embedded trading companion panel with discipline ratios, cooldown alerts, and quick query prompts.",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="trading-companion-panel",
        status="AVAILABLE",
        how_to_access="Click Tab '9. Live Market' -> Scroll to TRADING COMPANION section.",
        related_features=["ai_chat", "execution"],
    ),
    AppFeatureItem(
        feature_id="execution",
        name="Hedge Fund Execution & Risk Management",
        description="Trade proposal cards, manual approval gate, position sizing (1% cap), and open positions table.",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="hedge-fund-execution-panel",
        status="AVAILABLE",
        how_to_access="Click Tab '9. Live Market' -> Scroll to HEDGE FUND EXECUTION section.",
        related_features=["companion", "journal"],
    ),
    AppFeatureItem(
        feature_id="backtest_engine",
        name="Deterministic Backtesting Engine",
        description="Historical replay evaluation, lookahead bias protection, and rule promotion gate.",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="step9-backtesting-panel",
        status="AVAILABLE",
        how_to_access="Click Tab '9. Live Market' -> Scroll to bottom section.",
        related_features=["execution", "replay_bar"],
    ),
    # Tab 10: Journal
    AppFeatureItem(
        feature_id="journal",
        name="Trade Journal & Analytics",
        description="Complete record of Paper, Live, and Backtest trades with win rate, profit factor, drawdown, and learned lessons.",
        tab_id="tab-journal",
        tab_name="10. Journal",
        panel_id="tab-journal",
        status="AVAILABLE",
        how_to_access="Click Tab '10. Journal'.",
        related_features=["execution"],
    ),
    # Tab 11: Performance
    AppFeatureItem(
        feature_id="performance",
        name="System Performance & Latency",
        description="Engine latency benchmarks, hardware utilization, caching efficiency, and token optimization.",
        tab_id="tab-performance",
        tab_name="11. Performance",
        panel_id="tab-performance",
        status="AVAILABLE",
        how_to_access="Click Tab '11. Performance'.",
        related_features=["security"],
    ),
    # Tab 12: Security
    AppFeatureItem(
        feature_id="security",
        name="Security, HCI & Device Isolation",
        description="Air-gapped operation controls, biometric authorization status, registered mobile devices, and audit log.",
        tab_id="tab-security",
        tab_name="12. Security & HCI",
        panel_id="tab-security",
        status="AVAILABLE",
        how_to_access="Click Tab '12. Security & HCI'.",
        related_features=["cyber_wall"],
    ),
    # Tab 13: Cyber Wall
    AppFeatureItem(
        feature_id="cyber_wall",
        name="Cyber Wall & System Resilience",
        description="Active threat detection, immutable snapshot backups, watchdog tasks, and self-healing recovery.",
        tab_id="tab-resilience",
        tab_name="13. Cyber Wall & Recovery",
        panel_id="tab-resilience",
        status="AVAILABLE",
        how_to_access="Click Tab '13. Cyber Wall & Recovery'.",
        related_features=["security"],
    ),
    # Tab 14: Core Trade Logic
    AppFeatureItem(
        feature_id="core_trade_logic",
        name="Multi-Source Confluence & Trade Logic",
        description="Deterministic SMC confluence scoring, multi-source conflict resolution, and Why-Not-Trade checks.",
        tab_id="tab-trade-logic",
        tab_name="14. Multi-Source & Core Logic",
        panel_id="tab-trade-logic",
        status="AVAILABLE",
        how_to_access="Click Tab '14. Multi-Source & Core Logic'.",
        related_features=["master_decision"],
    ),
    # Tab 15: AI Core & Plugins
    AppFeatureItem(
        feature_id="ai_core_plugins",
        name="AI Core, Plugins & Live Decision",
        description="Dynamic plugin loader, live market forward-scenario forecasts, and Hindi speech synthesis.",
        tab_id="tab-step15",
        tab_name="15. AI Core, Plugins & Live Decision",
        panel_id="tab-step15",
        status="AVAILABLE",
        how_to_access="Click Tab '15. AI Core, Plugins & Live Decision'.",
        related_features=["master_decision"],
    ),
    # Tab 16: Cloud & Events
    AppFeatureItem(
        feature_id="cloud_events",
        name="24×7 Cloud Runtime & Event Stream",
        description="Cloud runtime health, notification delivery preferences (Email, Push, Voice), and WebSocket stream replay.",
        tab_id="tab-step16",
        tab_name="16. 24×7 Cloud & Event Stream",
        panel_id="tab-step16",
        status="AVAILABLE",
        how_to_access="Click Tab '16. 24×7 Cloud & Event Stream'.",
        related_features=["universal_providers"],
    ),
    # Tab 17: Master Decision Engine
    AppFeatureItem(
        feature_id="master_decision",
        name="Master Decision Engine",
        description="10-point deterministic confluence matrix combining SMC, orderflow, volume profile, HTF trend, and news risk.",
        tab_id="tab-step17",
        tab_name="17. Master Decision Engine",
        panel_id="tab-step17",
        status="AVAILABLE",
        how_to_access="Click Tab '17. Master Decision Engine'. Set decision threshold here.",
        related_features=["core_trade_logic", "execution"],
    ),
    # Tab 18: Providers & Replay Bar
    AppFeatureItem(
        feature_id="universal_providers",
        name="Universal Providers Matrix",
        description="Connection manager for TradingView (Market Data), TradingLite (Backtest), Delta Exchange (Crypto), MT5 & XM (Forex).",
        tab_id="tab-step18",
        tab_name="18. Universal Providers & Benchmark",
        panel_id="step18-providers-container",
        status="AVAILABLE",
        how_to_access="Click Tab '18. Universal Providers & Benchmark'.",
        related_features=["replay_bar", "lecture_benchmark"],
    ),
    AppFeatureItem(
        feature_id="replay_bar",
        name="Historical Replay Bar Controller",
        description="User-intended bar-by-bar historical replay with variable speeds (1x, 2x, 5x, 10x, 25x, 50x) and zero-lookahead slicing.",
        tab_id="tab-step18",
        tab_name="18. Universal Providers & Benchmark",
        panel_id="replay-bar-container",
        status="AVAILABLE",
        how_to_access="Click Tab '18. Universal Providers & Benchmark' -> Replay Bar Controller card.",
        related_features=["universal_providers", "backtest_engine"],
    ),
    AppFeatureItem(
        feature_id="lecture_benchmark",
        name="Lecture Performance Benchmarking",
        description="Deterministic pipeline timing instrumentation, token costing, and bottleneck breakdown.",
        tab_id="tab-step18",
        tab_name="18. Universal Providers & Benchmark",
        panel_id="bm-total-time",
        status="AVAILABLE",
        how_to_access="Click Tab '18. Universal Providers & Benchmark' -> Bottom section.",
        related_features=["lecture_ingestion", "universal_providers"],
    ),
    # Implemented Step 5 Core Enhancements
    AppFeatureItem(
        feature_id="settings_page",
        name="Dedicated Unified Settings Hub",
        description="Centralized configuration panel for AI models, Voice, Trading & Risk guardrails, Broker credentials, Notifications, and System runtime.",
        tab_id="settings-modal",
        tab_name="Settings Hub",
        panel_id="settings-modal-content",
        status="AVAILABLE",
        how_to_access="Click the Settings button (⚙️) in the top navigation header or trigger via Chat/JARVIS voice ('open settings').",
        related_features=["universal_providers", "cloud_events", "token_safety", "execution"],
    ),
    AppFeatureItem(
        feature_id="interactive_chart",
        name="Interactive Candlestick Chart Canvas",
        description="Interactive visual candlestick chart with timeframe selector, dynamic OHLC candles, and lecture-grounded ICT/SMC overlays (FVG, Order Blocks, Liquidity levels).",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="interactive-chart-container",
        status="AVAILABLE",
        how_to_access="Located in Tab 9 (Live Market) above the Multi-Timeframe Analysis grid, with symbol and timeframe selectors.",
        related_features=["mtf_analysis", "chart_markings"],
    ),
    AppFeatureItem(
        feature_id="auto_execution_toggle",
        name="Dashboard Execution Mode Toggle",
        description="Interactive control switch on header and Tab 9 to switch between Manual Approval and Automated Execution with safety confirmation modal.",
        tab_id="tab-market",
        tab_name="9. Live Market",
        panel_id="btn-toggle-execution-mode",
        status="AVAILABLE",
        how_to_access="Click the Execution Mode button in the top navigation bar or Tab 9 header, protected by safety confirmation modal.",
        related_features=["execution"],
    ),
]


class ApplicationGuide:
    """Authoritative Guide and Component Registry for the Trading AI Assistant."""

    def __init__(self):
        self._features = APP_FEATURES_REGISTRY

    def list_features(self) -> List[AppFeatureItem]:
        """Return all registered application features and their locations."""
        return self._features

    def find_feature(self, query: str) -> Optional[AppFeatureItem]:
        """Match user query against feature names, descriptions, or IDs."""
        q = query.lower().strip()
        # Direct keyword aliases
        aliases = {
            "watchlist": "watchlist",
            "pairs": "watchlist",
            "symbols": "watchlist",
            "chart": "interactive_chart",
            "candlestick": "interactive_chart",
            "settings": "settings_page",
            "preferences": "settings_page",
            "config": "settings_page",
            "master decision": "master_decision",
            "confluence": "master_decision",
            "decision threshold": "master_decision",
            "providers": "universal_providers",
            "broker": "universal_providers",
            "delta": "universal_providers",
            "xm": "universal_providers",
            "mt5": "universal_providers",
            "tradingview": "universal_providers",
            "tradinglite": "universal_providers",
            "replay": "replay_bar",
            "replay bar": "replay_bar",
            "auto execution": "auto_execution_toggle",
            "execution mode": "auto_execution_toggle",
            "mode toggle": "auto_execution_toggle",
            "auto gate": "auto_execution_toggle",
            "execution": "execution",
            "positions": "execution",
            "orders": "execution",
            "proposal": "execution",
            "journal": "journal",
            "trades": "journal",
            "pnl": "journal",
            "lecture": "lecture_ingestion",
            "upload": "lecture_ingestion",
            "rules": "knowledge_rules",
            "knowledge": "knowledge_rules",
            "supervisor": "supervisor_status",
            "progress": "supervisor_status",
            "token": "token_safety",
            "cloud": "cloud_events",
            "notifications": "cloud_events",
            "news": "fundamental_news",
            "economic calendar": "fundamental_news",
            "chat": "ai_chat",
            "assistant": "ai_chat",
        }

        for alias, feat_id in aliases.items():
            if alias in q:
                for f in self._features:
                    if f.feature_id == feat_id:
                        return f

        # General search in name and description
        for f in self._features:
            if f.name.lower() in q or f.feature_id in q:
                return f
        for f in self._features:
            if any(word in f.description.lower() for word in q.split() if len(word) > 3):
                return f
        return None

    def get_navigation_action(self, feature: AppFeatureItem) -> Optional[ChatNavigationAction]:
        """Generate a clickable navigation action for UI if feature is accessible."""
        if feature.tab_id and feature.tab_id not in ("none", "global-drawer", "top-header", "left-column"):
            return ChatNavigationAction(
                tab_id=feature.tab_id,
                label=f"Open {feature.tab_name}",
                element_selector=feature.panel_id,
            )
        return None

    def get_all_tabs(self) -> List[AppTabInfo]:
        """Return all 18 tabs with their IDs and titles."""
        return [
            AppTabInfo(tab_id="tab-notes", number=1, name="Notes"),
            AppTabInfo(tab_id="tab-knowledge", number=2, name="Knowledge"),
            AppTabInfo(tab_id="tab-concepts", number=3, name="Concepts"),
            AppTabInfo(tab_id="tab-doubts", number=4, name="Doubts"),
            AppTabInfo(tab_id="tab-alignment", number=5, name="Alignment"),
            AppTabInfo(tab_id="tab-transcript", number=6, name="Transcripts"),
            AppTabInfo(tab_id="tab-visual", number=7, name="Visuals"),
            AppTabInfo(tab_id="tab-memory", number=8, name="Memory"),
            AppTabInfo(tab_id="tab-market", number=9, name="Live Market"),
            AppTabInfo(tab_id="tab-journal", number=10, name="Journal"),
            AppTabInfo(tab_id="tab-performance", number=11, name="Performance"),
            AppTabInfo(tab_id="tab-security", number=12, name="Security & HCI"),
            AppTabInfo(tab_id="tab-resilience", number=13, name="Cyber Wall & Recovery"),
            AppTabInfo(tab_id="tab-trade-logic", number=14, name="Multi-Source & Core Logic"),
            AppTabInfo(tab_id="tab-step15", number=15, name="AI Core, Plugins & Live Decision"),
            AppTabInfo(tab_id="tab-step16", number=16, name="24×7 Cloud & Event Stream"),
            AppTabInfo(tab_id="tab-step17", number=17, name="Master Decision Engine"),
            AppTabInfo(tab_id="tab-step18", number=18, name="Universal Providers & Benchmark"),
        ]

    def generate_tour(self) -> str:
        """Generate a complete, structured tour of the entire application."""
        return (
            "### 🗺️ TRADING AI ASSISTANT — APPLICATION TOUR\n\n"
            "The application is organized into a **Left Control Panel** and an **18-Tab Workspace**:\n\n"
            "**1. Ingestion & Controls (Left Column):**\n"
            "- **Lecture Ingestion**: Upload trading masterclass videos/audio or resolve YouTube URLs.\n"
            "- **Supervisor & Barrier**: Real-time progress bar and commit barrier for extracted units.\n"
            "- **User Command Console**: Direct the AI to save rules, make corrections, or note exceptions.\n"
            "- **Token Safety**: Real-time token budget monitor with 20% safety headroom.\n\n"
            "**2. Knowledge & Mentorship (Tabs 1–8):**\n"
            "- **Tab 1: Notes & Tab 2: Knowledge**: View synthesized markdown notes and extracted trading rules.\n"
            "- **3. Concepts & 4. Doubts**: Domain terminology, detected ambiguities, and caveats.\n"
            "- **5. Alignment, 6. Transcripts, 7. Visuals**: Audio speech timestamps and on-screen chart markers.\n"
            "- **8. Memory**: Vector embeddings and knowledge revision history.\n\n"
            "**3. Trading & Execution (Tabs 9–10):**\n"
            "- **Tab 9: Live Market**: Watchlist (BTC/USD, ETH/USD, EUR/USD, etc.), Multi-Timeframe Analysis (1d, 1h, 15m), Chart Markings, Fundamental News, Hedge Fund Execution (Trade Proposals, 1% Risk Sizing, Open Positions), and Backtesting.\n"
            "- **10. Journal**: Performance analytics, win rate, profit factor, and trade review lessons.\n\n"
            "**4. System Architecture & Intelligence (Tabs 11–18):**\n"
            "- **11. Performance & 12. Security**: Latency benchmarks and air-gapped device security.\n"
            "- **13. Cyber Wall**: Threat monitoring and snapshot backups.\n"
            "- **14. Multi-Source Core Logic & 15. AI Core**: SMC confluence validator, Why-Not-Trade checks, and plugins.\n"
            "- **16. 24×7 Cloud**: Runtime status and notification preferences (Email, Push, Voice).\n"
            "- **Tab 17: Master Decision Engine**: 10-point confluence matrix and decision threshold control.\n"
            "- **Tab 18: Universal Providers & Benchmark**: Connect TradingView, Delta Exchange, MT5, XM, and use the Historical Replay Bar (1x–50x speeds)."
        )

    def get_system_prompt_summary(self) -> str:
        """Compact token-efficient application map for system prompt injection."""
        return (
            "STRICT INSTRUCTION: Never invent tabs, panels, or buttons. All locations must be factually grounded.\n\n"
            "APPLICATION MAP (Trading AI Assistant Layout):\n"
            "- Left Panel: Lecture Video Upload, Supervisor Progress, User Command Console (save rule), Token Safety.\n"
            "- Tab 1 (Notes): Synthesized lecture notes.\n"
            "- Tab 2 (Knowledge): Verified rules, concepts, exceptions.\n"
            "- Tab 5-7: Alignment, Transcripts, Visuals.\n"
            "- tab-market / Tab 9 (Live Market): Watchlist (BTC/USD, ETH/USD, EUR/USD, XAU/USD), MTF (1d/1h/15m), Chart Markings, Fundamental News, Hedge Fund Execution (Proposals, Open Positions, 1% Risk Sizing), Backtesting.\n"
            "- Tab 10 (Journal): Trade logs, win rate, performance.\n"
            "- Tab 11-13: Performance, Security, Cyber Wall.\n"
            "- Tab 14-15: Multi-Source SMC Logic, Why-Not-Trade, Plugins.\n"
            "- Tab 16 (Cloud): Runtime status, Notification preferences.\n"
            "- Tab 17 (Master Decision Engine): 10-point Confluence Matrix, Decision Threshold.\n"
            "- Dedicated Settings Hub: OPERATIONAL (Click ⚙️ in header or open settings modal).\n"
            "- Interactive Chart Canvas: OPERATIONAL in Tab 9 (Live Candlestick visualizer with FVG and OB zones).\n"
            "- Execution Mode Toggle: OPERATIONAL (Header button switches between MANUAL GATE and AUTOMATIC GATE with safety confirmation)."
        )
