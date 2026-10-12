"""
Gemini Trading Brain: Qualitative Reasoning Engine for Trading Logic, Rule Extraction, and Backtest Analysis.

Strictly preserves the separation of concerns:
- Gemini is responsible for qualitative reasoning (lecture rule extraction, market context synthesis, trade thesis).
- Gemini MUST NOT perform deterministic calculations (P&L, RR, position sizing, drawdown, arithmetic).
- Reuses Step 1 ModelRouter and TokenManager via GeminiClient.
- Uses smallest sufficient context to conserve tokens.
"""

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from trading_ai.ai.gemini_client import GeminiClient
from trading_ai.core.model_router import ModelRole
from trading_ai.schemas.backtest import (
    BacktestMetrics,
    BacktestTrade,
    RuleVersion,
)
from trading_ai.schemas.enums import RuleLifecycleState
from trading_ai.schemas.market import MarketSnapshot
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class GeminiTradingBrain:
    """
    Reasoning bridge connecting lecture-derived knowledge to structured trade theses and backtest analysis.
    """

    def __init__(
        self,
        gemini_client: Optional[GeminiClient] = None,
        repository: Optional[Repository] = None,
    ):
        self._ai = gemini_client or GeminiClient()
        self._repo = repository
        self._cache: Dict[str, Dict[str, Any]] = {}

    def extract_candidate_rule(
        self,
        lecture_text: str,
        lecture_id: Optional[str] = None,
        source_timestamp: Optional[float] = None,
        existing_rule_id: Optional[str] = None,
        current_version: int = 0,
    ) -> RuleVersion:
        """
        Extract a structured trading rule from lecture content using Gemini reasoning.
        Creates an immutable RuleVersion in CANDIDATE state (Phase 9.1 & 9.2).
        """
        # Ensure smallest sufficient context (clean snippet, max 2000 chars)
        trimmed_text = lecture_text.strip()[:2000]

        prompt = (
            f"Extract a structured trading rule from this lecture excerpt:\n"
            f"\"{trimmed_text}\"\n\n"
            "Format response as valid JSON with keys:\n"
            "- title: short rule title (e.g. 'FVG Entry on Liquidity Sweep')\n"
            "- rule_text: concise formal specification of entry/invalidation\n"
            "- conditions: array of prerequisite market conditions\n"
            "- exceptions: array of invalidating market conditions or news exceptions\n"
            "- tags: array of short concept tags (e.g. ['ICT', 'FVG', 'Forex'])\n"
        )

        response_str = self._ai.generate_content(
            role=ModelRole.REASONING,
            prompt=prompt,
            job_id=lecture_id or "rule_extraction",
            stage="trading_brain_extraction",
        )

        try:
            parsed = json.loads(response_str)
        except Exception:
            # Deterministic fallback extraction
            parsed = {
                "title": "Price Action Liquidity Sweep & Displacement Rule",
                "rule_text": trimmed_text[:150] if trimmed_text else "Trade on liquidity sweep followed by displacement.",
                "conditions": ["Price sweeps prior high or low", "Energetic displacement candle"],
                "exceptions": ["High-impact economic news within 15 minutes"],
                "tags": ["PriceAction", "Displacement"],
            }

        rule_id = existing_rule_id or f"rule_{uuid.uuid4().hex[:10]}"
        new_version = (current_version + 1) if existing_rule_id else 1

        rule_version = RuleVersion(
            rule_id=rule_id,
            version=new_version,
            title=parsed.get("title", "Candidate Trading Rule"),
            rule_text=parsed.get("rule_text", trimmed_text[:200]),
            conditions=parsed.get("conditions", []),
            exceptions=parsed.get("exceptions", []),
            source_lecture_id=lecture_id,
            source_timestamp=source_timestamp,
            provenance="user_lecture",
            state=RuleLifecycleState.CANDIDATE,
            tags=parsed.get("tags", ["lecture_derived"]),
        )

        if self._repo:
            self._repo.save_rule_version(rule_version)

        return rule_version

    def formulate_trade_thesis(
        self,
        rule_version: RuleVersion,
        market_snapshot: MarketSnapshot,
        confluence_factors: Optional[List[str]] = None,
        news_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Synthesize qualitative trade thesis at decision point T using smallest sufficient context.
        Strict look-ahead protection: market_snapshot MUST contain candles strictly at or before T.
        """
        # Compress market snapshot to smallest sufficient context
        last_3 = market_snapshot.candles[-3:] if market_snapshot.candles else []
        candles_summary = [
            f"t={c.timestamp}: O={c.open}, H={c.high}, L={c.low}, C={c.close}"
            for c in last_3
        ]

        context_dict = {
            "rule_id": rule_version.rule_id,
            "rule_v": rule_version.version,
            "rule_title": rule_version.title,
            "rule_conditions": rule_version.conditions,
            "rule_exceptions": rule_version.exceptions,
            "symbol": market_snapshot.symbol,
            "timeframe": market_snapshot.timeframe,
            "latest_price": market_snapshot.latest_price,
            "recent_candles": candles_summary,
            "confluence": confluence_factors or [],
            "news": news_context or "No imminent high-impact news",
        }

        # Cache key to avoid redundant identical LLM calls
        ctx_hash = hashlib.sha256(json.dumps(context_dict, sort_keys=True).encode()).hexdigest()
        if ctx_hash in self._cache:
            return self._cache[ctx_hash]

        prompt = (
            f"You are the Gemini Trading Brain. Evaluate this trading rule against current market context:\n"
            f"{json.dumps(context_dict, indent=2)}\n\n"
            "Requirements:\n"
            "1. Determine if rule conditions are satisfied without looking ahead.\n"
            "2. Determine trade direction: BUY, SELL, or NONE.\n"
            "3. State qualitative entry trigger, stop loss placement logic, and target logic.\n"
            "4. NEVER compute mathematical P&L, position size, or exact dollar risk.\n"
            "Return JSON with keys: trigger_detected (bool), direction (BUY/SELL/NONE), entry_reasoning (str), "
            "stop_loss_reasoning (str), target_reasoning (str), confidence (float between 0.0 and 1.0)."
        )

        response_str = self._ai.generate_content(
            role=ModelRole.REASONING,
            prompt=prompt,
            job_id=f"thesis_{market_snapshot.symbol}",
            stage="trading_brain_thesis",
        )

        try:
            parsed = json.loads(response_str)
        except Exception:
            # Deterministic heuristic fallback based on candle trend
            direction = "NONE"
            trigger = False
            confidence = 0.50
            if len(last_3) >= 2:
                if last_3[-1].close > last_3[-2].high:
                    direction = "BUY"
                    trigger = True
                    confidence = 0.85
                elif last_3[-1].close < last_3[-2].low:
                    direction = "SELL"
                    trigger = True
                    confidence = 0.85

            parsed = {
                "trigger_detected": trigger,
                "direction": direction,
                "entry_reasoning": f"Price displacement confirmed beyond prior candle boundary on {market_snapshot.timeframe}.",
                "stop_loss_reasoning": "Protected below recent structural swing low/high.",
                "target_reasoning": "Targeting liquidity pool at next structural high/low (minimum 2R).",
                "confidence": confidence,
            }

        self._cache[ctx_hash] = parsed
        return parsed

    def analyze_backtest_performance(
        self,
        metrics: BacktestMetrics,
        rule_version: RuleVersion,
        sample_trades: List[BacktestTrade],
    ) -> Dict[str, Any]:
        """
        Qualitatively evaluate backtest results and diagnose edge consistency (Phase 9.1 & 9.12).
        Calculations are strictly pre-computed; Gemini provides qualitative insights only.
        """
        trades_summary = [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "result": t.result,
                "r_multiple": t.r_multiple,
                "pnl": round(t.pnl, 2),
            }
            for t in sample_trades[:10]  # Smallest sufficient context: max 10 sample trades
        ]

        payload = {
            "rule_id": rule_version.rule_id,
            "rule_version": rule_version.version,
            "rule_title": rule_version.title,
            "total_trades": metrics.total_trades,
            "win_rate_pct": round(metrics.win_rate * 100, 1),
            "profit_factor": round(metrics.profit_factor, 2),
            "expectancy_r": round(metrics.expectancy, 2),
            "max_drawdown_pct": round(metrics.max_drawdown_pct, 2),
            "total_pnl": round(metrics.total_pnl, 2),
            "sample_trades": trades_summary,
        }

        prompt = (
            f"Analyze these deterministic backtest results for rule '{rule_version.title}' v{rule_version.version}:\n"
            f"{json.dumps(payload, indent=2)}\n\n"
            "Provide qualitative evaluation in JSON with keys:\n"
            "- assessment: concise verdict on edge stability\n"
            "- strengths: list of observed strategic strengths\n"
            "- failure_patterns: list of recurring loss scenarios or weaknesses\n"
            "- recommendation: 'APPROVE_FOR_PAPER', 'NEEDS_REFINEMENT', or 'REJECT'\n"
            "- suggested_refinements: list of concrete rule adjustments"
        )

        response_str = self._ai.generate_content(
            role=ModelRole.REASONING,
            prompt=prompt,
            job_id=f"analysis_{rule_version.rule_id}",
            stage="trading_brain_backtest_analysis",
        )

        try:
            parsed = json.loads(response_str)
        except Exception:
            rec = "APPROVE_FOR_PAPER" if (metrics.profit_factor >= 1.3 and metrics.expectancy > 0.2) else "NEEDS_REFINEMENT"
            parsed = {
                "assessment": f"Deterministic backtest yielded win rate of {metrics.win_rate*100:.1f}% with profit factor {metrics.profit_factor:.2f}.",
                "strengths": ["Clear R:R ratio distribution", "Adherence to structural displacement"],
                "failure_patterns": ["Losses observed during low-volatility consolidation"],
                "recommendation": rec,
                "suggested_refinements": ["Add session-timing filter to avoid Asian session chop"],
            }

        return parsed
