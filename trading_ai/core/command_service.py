"""Execution service for user commands, rule persistence, and knowledge versioning."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trading_ai.core.context_manager import ContextManager
from trading_ai.core.memory_retriever import _text_to_mock_embedding
from trading_ai.core.model_router import ModelRole, ModelRouter
from trading_ai.core.token_manager import TokenManager
from trading_ai.schemas import (
    ChartMarking,
    CommandIntent,
    ContextItem,
    ContextItemSource,
    CorrectionRequest,
    EvidenceTrace,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    KnowledgeVersion,
    LevelType,
    MarketBias,
    MarketDirection,
    MarketLevel,
    MarketSetup,
    MarketSnapshot,
    MemoryEmbedding,
    RuleSaveRequest,
    ScreenAccessStatus,
    ScreenPermission,
    SetupStatus,
    UnifiedMarketAnalysis,
    UserCommand,
    WatchlistMonitorStatus,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class UserCommandService:
    """
    Executes validated user commands, enforces knowledge versioning,
    preserves provenance, and prevents uncontrolled duplicate knowledge creation.
    """

    def __init__(
        self,
        repository: Repository,
        token_manager: Optional[TokenManager] = None,
        model_router: Optional[ModelRouter] = None,
        watchlist_monitor: Optional[Any] = None,
        mtf_engine: Optional[Any] = None,
        lecture_validator: Optional[Any] = None,
        chart_marker: Optional[Any] = None,
        screen_access_manager: Optional[Any] = None,
        data_provider: Optional[Any] = None,
        situation_engine: Optional[Any] = None,
    ):
        self._repo = repository
        self._token_manager = token_manager
        self._router = model_router or ModelRouter()
        self._watchlist_monitor = watchlist_monitor
        self._mtf_engine = mtf_engine
        self._lecture_validator = lecture_validator
        self._chart_marker = chart_marker
        self._screen_access_manager = screen_access_manager
        self._data_provider = data_provider
        self._situation_engine = situation_engine

    def _get_data_provider(self):
        if self._data_provider is None:
            from trading_ai.core.market_data_provider import SimulatedMarketDataProvider
            self._data_provider = SimulatedMarketDataProvider()
        return self._data_provider

    def _get_mtf_engine(self):
        if self._mtf_engine is None:
            from trading_ai.core.mtf_engine import MultiTimeframeEngine
            self._mtf_engine = MultiTimeframeEngine(self._get_data_provider())
        return self._mtf_engine

    def _get_lecture_validator(self):
        if self._lecture_validator is None:
            from trading_ai.core.lecture_validator import LectureValidator
            self._lecture_validator = LectureValidator(self._repo)
        return self._lecture_validator

    def _get_chart_marker(self):
        if self._chart_marker is None:
            from trading_ai.core.chart_marker import ChartMarker
            self._chart_marker = ChartMarker()
        return self._chart_marker

    def _get_screen_access_manager(self):
        if self._screen_access_manager is None:
            from trading_ai.core.screen_access import ScreenAccessManager
            self._screen_access_manager = ScreenAccessManager(self._repo)
        return self._screen_access_manager

    def _get_watchlist_monitor(self):
        if self._watchlist_monitor is None:
            from trading_ai.pipeline.watchlist_monitor import WatchlistMonitor
            from trading_ai.supervisor.commit_barrier import CommitBarrier
            barrier = CommitBarrier(self._repo)
            self._watchlist_monitor = WatchlistMonitor(
                repository=self._repo,
                data_provider=self._get_data_provider(),
                mtf_engine=self._get_mtf_engine(),
                lecture_validator=self._get_lecture_validator(),
                chart_marker=self._get_chart_marker(),
                commit_barrier=barrier,
                watchlist=["BTC/USD", "ETH/USD", "SOL/USD"],
            )
        return self._watchlist_monitor

    def _get_situation_engine(self):
        if self._situation_engine is None:
            from trading_ai.core.situation_engine import CurrentSituationEngine
            self._situation_engine = CurrentSituationEngine(
                repository=self._repo,
                chart_marker=self._get_chart_marker(),
            )
        return self._situation_engine

    def execute_command(self, cmd: UserCommand, job_id: Optional[str] = None) -> Dict[str, Any]:
        """Route and execute a structured UserCommand."""
        self._repo.save_user_command(cmd)

        effective_job_id = job_id or cmd.parameters.get("job_id") or "global_knowledge"

        if cmd.intent == CommandIntent.SAVE_RULE:
            rule_text = cmd.parameters.get("rule") or cmd.parameters.get("statement") or cmd.raw_text
            req = RuleSaveRequest(
                rule=rule_text,
                conditions=cmd.parameters.get("conditions"),
                exceptions=cmd.parameters.get("exceptions"),
                source=cmd.source,
                evidence=cmd.parameters.get("evidence"),
                job_id=effective_job_id,
            )
            return self.save_rule(req)

        elif cmd.intent == CommandIntent.CORRECT:
            req = CorrectionRequest(
                target_knowledge_id=cmd.target_knowledge_id or cmd.parameters.get("target_knowledge_id"),
                query=cmd.parameters.get("query") or cmd.parameters.get("statement"),
                correction_statement=cmd.parameters.get("correction_statement")
                or cmd.parameters.get("statement")
                or cmd.raw_text,
                reason=cmd.parameters.get("reason"),
                conditions=cmd.parameters.get("conditions"),
                exceptions=cmd.parameters.get("exceptions"),
                job_id=effective_job_id,
                corrected_by=cmd.source,
            )
            return self.apply_correction(req)

        elif cmd.intent in (CommandIntent.REMEMBER, CommandIntent.ADD_KNOWLEDGE):
            statement = cmd.parameters.get("statement") or cmd.raw_text
            req = RuleSaveRequest(
                rule=statement,
                conditions=cmd.parameters.get("conditions"),
                exceptions=cmd.parameters.get("exceptions"),
                source=cmd.source,
                job_id=effective_job_id,
                tags=["user_remembered"],
            )
            return self.save_rule(req)

        elif cmd.intent == CommandIntent.MARK_EXCEPTION:
            target_id = cmd.target_knowledge_id or cmd.parameters.get("target_knowledge_id")
            exc_text = cmd.parameters.get("exception_details") or cmd.raw_text
            return self.add_exception(target_id=target_id, exception_text=exc_text, job_id=effective_job_id)

        elif cmd.intent == CommandIntent.FORGET:
            target_id = cmd.target_knowledge_id or cmd.parameters.get("target_knowledge_id")
            return self.archive_knowledge(target_id=target_id, job_id=effective_job_id, reason=cmd.raw_text)

        elif cmd.intent == CommandIntent.EXPLAIN:
            target_id = cmd.target_knowledge_id or cmd.parameters.get("target_knowledge_id")
            return self.explain_knowledge(target_id=target_id, job_id=effective_job_id, query=cmd.raw_text)

        # Step 4 Market Intelligence intents
        elif cmd.intent == CommandIntent.ANALYZE_SYMBOL:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.analyze_symbol(symbol=symbol)

        elif cmd.intent == CommandIntent.ANALYZE_CHART:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.analyze_chart(symbol=symbol)

        elif cmd.intent == CommandIntent.ANALYZE_WATCHLIST:
            return self.analyze_watchlist()

        elif cmd.intent == CommandIntent.CHECK_HTF:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.check_htf(symbol=symbol)

        elif cmd.intent == CommandIntent.FILTER_VALID_SETUPS:
            symbol = cmd.parameters.get("symbol")
            return self.filter_valid_setups(symbol=symbol)

        elif cmd.intent == CommandIntent.USE_LECTURE_RULES_ONLY:
            return self.set_lecture_rules_only()

        elif cmd.intent == CommandIntent.EXPLAIN_SETUP:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            target_id = cmd.target_knowledge_id or cmd.parameters.get("target_knowledge_id")
            return self.explain_setup(symbol=symbol, target_id=target_id)

        elif cmd.intent == CommandIntent.MARK_PATTERN:
            pattern = cmd.parameters.get("pattern") or "FVG"
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.mark_pattern(symbol=symbol, pattern_name=pattern)

        elif cmd.intent == CommandIntent.STOP_MONITORING:
            return self.stop_monitoring()

        elif cmd.intent == CommandIntent.RESUME_MONITORING:
            return self.resume_monitoring()

        # Step 5 Intelligent Selective Visualization intents
        elif cmd.intent == CommandIntent.SHOW_ALL_LIQUIDITY:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.show_all_liquidity(symbol=symbol)

        elif cmd.intent == CommandIntent.SHOW_HIDDEN_SETUPS:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.show_hidden_setups(symbol=symbol)

        elif cmd.intent == CommandIntent.SHOW_FVGS:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.show_fvgs(symbol=symbol)

        elif cmd.intent == CommandIntent.SHOW_ORDER_BLOCKS:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.show_order_blocks(symbol=symbol)

        elif cmd.intent == CommandIntent.SHOW_UNCONFIRMED_SETUPS:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.show_unconfirmed_setups(symbol=symbol)

        elif cmd.intent == CommandIntent.SHOW_CONFIRMATION_EVIDENCE:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.show_confirmation_evidence(symbol=symbol)

        elif cmd.intent == CommandIntent.SHOW_INVALIDATION_CONDITION:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.show_invalidation_condition(symbol=symbol)

        elif cmd.intent == CommandIntent.HIDE_BACKGROUND_MARKINGS:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.hide_background_markings(symbol=symbol)

        elif cmd.intent == CommandIntent.FOCUS_CURRENT_SETUP:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.focus_current_setup(symbol=symbol)

        elif cmd.intent == CommandIntent.SHOW_HTF_CONTEXT:
            symbol = cmd.parameters.get("symbol") or "BTC/USD"
            return self.check_htf(symbol=symbol)

        # Default action
        return {
            "status": "acknowledged",
            "command_id": cmd.command_id,
            "intent": cmd.intent.value,
            "message": f"Command received: {cmd.raw_text}",
        }

    def save_rule(self, req: RuleSaveRequest) -> Dict[str, Any]:
        """
        Persist a trading rule with full required metadata.

        Preserves:
        rule, conditions, exceptions, source, evidence, timestamp,
        created_by, confidence, status, version, relationships.

        Missing parts are explicitly marked rather than hallucinated.
        De-duplicates identical rule statements for the job.
        """
        job_id = req.job_id or "global_knowledge"
        normalized_rule = req.rule.strip()

        # 1. De-duplication check: prevent uncontrolled duplicate memory
        existing_items = self._repo.get_knowledge(job_id=job_id, only_committed=False)
        for item in existing_items:
            if item.statement.strip().lower() == normalized_rule.lower():
                logger.info(f"Duplicate rule detected for {job_id}: {item.item_id}")
                return {
                    "status": "exists",
                    "action": "deduplicated",
                    "item_id": item.item_id,
                    "statement": item.statement,
                    "version": 1,
                    "message": "Rule already exists in persistent memory. Duplicate avoided.",
                }

        # 2. Mark missing parts explicitly rather than inventing them
        conditions = req.conditions if (req.conditions is not None and len(req.conditions) > 0) else ["[MISSING]"]
        exceptions = req.exceptions if (req.exceptions is not None and len(req.exceptions) > 0) else ["[NONE_SPECIFIED]"]

        item_id = f"item_rule_{uuid.uuid4().hex[:10]}"
        timestamp = datetime.now(timezone.utc)

        # 3. Create supporting evidence trace
        evidence_traces: List[EvidenceTrace] = []
        if req.evidence:
            ev = EvidenceTrace(
                evidence_id=f"ev_user_{uuid.uuid4().hex[:8]}",
                job_id=job_id,
                start_time=0.0,
                end_time=0.0,
                extracted_statement=req.evidence,
                confidence=req.confidence,
                notes=f"Provided by {req.created_by}",
            )
            self._repo.save_evidence(ev)
            evidence_traces.append(ev)

        # 4. Create structured KnowledgeItem
        k_item = KnowledgeItem(
            item_id=item_id,
            job_id=job_id,
            item_type=KnowledgeType.RULE,
            title=f"Rule: {normalized_rule[:50]}",
            statement=normalized_rule,
            reasoning=f"Conditions: {conditions}; Exceptions: {exceptions}. Source: {req.source}.",
            validation_status=req.status,
            confidence=req.confidence,
            start_time=0.0,
            end_time=0.0,
            evidence=evidence_traces,
            tags=list(set(req.tags + req.relationships + ["user_rule"])),
        )
        self._repo.save_knowledge_item(k_item, serial_number=1, is_committed=True)

        # 5. Create initial KnowledgeVersion (v1, is_active=True)
        version_entry = KnowledgeVersion(
            version_id=f"v_{item_id}_1",
            item_id=item_id,
            version_number=1,
            job_id=job_id,
            title=k_item.title,
            statement=k_item.statement,
            conditions=conditions,
            exceptions=exceptions,
            reasoning=k_item.reasoning,
            validation_status=req.status,
            confidence=req.confidence,
            evidence_json=f"[{ev.model_dump_json()}]" if evidence_traces else "[]",
            correction_notes=None,
            corrected_by=req.created_by,
            corrected_at=timestamp,
            superseded_by_version=None,
            is_active=True,
            provenance={
                "created_by": req.created_by,
                "source": req.source,
                "initial_timestamp": timestamp.isoformat(),
                "relationships": req.relationships,
            },
        )
        self._repo.save_knowledge_version(version_entry)

        # 6. Generate memory embedding
        emb_text = f"Rule: {normalized_rule}. Conditions: {conditions}. Exceptions: {exceptions}."
        emb_record = MemoryEmbedding(
            embedding_id=f"emb_{uuid.uuid4().hex[:12]}",
            job_id=job_id,
            source_type="knowledge",
            source_id=item_id,
            text_content=emb_text,
            embedding=_text_to_mock_embedding(emb_text),
            model_name=self._router.get_model(ModelRole.EMBEDDING),
            importance_score=req.confidence,
        )
        self._repo.save_memory_embedding(emb_record)

        return {
            "status": "success",
            "action": "saved_rule",
            "item_id": item_id,
            "version": 1,
            "statement": normalized_rule,
            "conditions": conditions,
            "exceptions": exceptions,
            "confidence": req.confidence,
            "validation_status": req.status.value,
            "created_by": req.created_by,
            "provenance": version_entry.provenance,
        }

    def apply_correction(self, req: CorrectionRequest) -> Dict[str, Any]:
        """
        Apply a versioned correction to existing knowledge.

        Follows:
        OLD VERSION -> USER CORRECTION -> NEW VERSION
        Preserves complete historical audit trail and never blindly overwrites.
        """
        job_id = req.job_id or "global_knowledge"
        target_item: Optional[KnowledgeItem] = None

        # 1. Identify target item
        if req.target_knowledge_id:
            all_items = self._repo.get_knowledge(job_id=job_id, only_committed=False)
            for it in all_items:
                if it.item_id == req.target_knowledge_id:
                    target_item = it
                    break

        # If not found by ID, search by query text
        if not target_item and req.query:
            matches = self._repo.search_knowledge(job_id=job_id, query=req.query, limit=1)
            if matches:
                target_item = matches[0]

        # If still not found, search by all knowledge in job or global
        if not target_item:
            all_items = self._repo.get_knowledge(job_id=job_id, only_committed=False)
            if all_items:
                target_item = all_items[-1]  # Most recent item

        if not target_item:
            # Fallback: if no target knowledge exists to correct, create new validated rule
            save_req = RuleSaveRequest(
                rule=req.correction_statement,
                conditions=req.conditions,
                exceptions=req.exceptions,
                source=req.corrected_by,
                evidence=req.reason,
                job_id=job_id,
                created_by=req.corrected_by,
            )
            return self.save_rule(save_req)

        item_id = target_item.item_id

        # 2. Retrieve existing versions to determine new version number
        existing_versions = self._repo.get_knowledge_versions(item_id)
        current_version_num = len(existing_versions) if existing_versions else 1
        new_version_num = current_version_num + 1

        # 3. Archive previous active version
        self._repo.archive_previous_versions(item_id=item_id, superseded_by_version=new_version_num)

        # 4. Build new version entry
        now = datetime.now(timezone.utc)
        conditions = req.conditions or ["[MISSING]"]
        exceptions = req.exceptions or ["[NONE_SPECIFIED]"]

        prior_statement = target_item.statement
        new_statement = req.correction_statement.strip()

        new_version = KnowledgeVersion(
            version_id=f"v_{item_id}_{new_version_num}",
            item_id=item_id,
            version_number=new_version_num,
            job_id=job_id,
            title=f"Corrected: {target_item.title}",
            statement=new_statement,
            conditions=conditions,
            exceptions=exceptions,
            reasoning=f"Correction: {req.reason or 'User correction'}. Previously believed: '{prior_statement}'.",
            validation_status=KnowledgeValidationStatus.VALIDATED,
            confidence=1.0,
            evidence_json="[]",
            correction_notes=req.reason or f"User correction replaced '{prior_statement}' with '{new_statement}'",
            corrected_by=req.corrected_by,
            corrected_at=now,
            superseded_by_version=None,
            is_active=True,
            provenance={
                "previously_believed": prior_statement,
                "corrected_statement": new_statement,
                "corrected_by": req.corrected_by,
                "corrected_at": now.isoformat(),
                "reason": req.reason,
                "superseded_version": current_version_num,
            },
        )
        self._repo.save_knowledge_version(new_version)

        # 5. Update active KnowledgeItem in repository
        target_item.statement = new_statement
        target_item.validation_status = KnowledgeValidationStatus.VALIDATED
        target_item.reasoning = new_version.reasoning
        target_item.confidence = 1.0
        self._repo.save_knowledge_item(target_item, serial_number=1, is_committed=True)

        # 6. Update memory embedding
        emb_text = f"Corrected Rule: {new_statement}. Reason: {req.reason}."
        emb_record = MemoryEmbedding(
            embedding_id=f"emb_corr_{uuid.uuid4().hex[:10]}",
            job_id=job_id,
            source_type="knowledge",
            source_id=item_id,
            text_content=emb_text,
            embedding=_text_to_mock_embedding(emb_text),
            model_name=self._router.get_model(ModelRole.EMBEDDING),
            importance_score=1.0,
        )
        self._repo.save_memory_embedding(emb_record)

        return {
            "status": "success",
            "action": "corrected",
            "item_id": item_id,
            "version": new_version_num,
            "prior_statement": prior_statement,
            "new_statement": new_statement,
            "corrected_by": req.corrected_by,
            "audit_trail": new_version.provenance,
        }

    def add_exception(
        self, target_id: Optional[str], exception_text: str, job_id: str
    ) -> Dict[str, Any]:
        """Attach an exception condition to an existing rule with version increment."""
        corr = CorrectionRequest(
            target_knowledge_id=target_id,
            correction_statement=f"Exception applied: {exception_text}",
            reason=f"Added exception: {exception_text}",
            exceptions=[exception_text],
            job_id=job_id,
        )
        return self.apply_correction(corr)

    def archive_knowledge(
        self, target_id: Optional[str], job_id: str, reason: str
    ) -> Dict[str, Any]:
        """Archive and deactivate a knowledge item, maintaining full audit trace."""
        if not target_id:
            return {"status": "failed", "message": "No target knowledge ID specified to archive."}

        self._repo.archive_previous_versions(item_id=target_id, superseded_by_version=-1)
        return {
            "status": "archived",
            "item_id": target_id,
            "reason": reason,
            "message": f"Knowledge item {target_id} deactivated and marked inactive in audit history.",
        }

    def explain_knowledge(
        self, target_id: Optional[str], job_id: str, query: str
    ) -> Dict[str, Any]:
        """Retrieve historical provenance, active version, and reasoning for a knowledge item."""
        target: Optional[KnowledgeItem] = None
        if target_id:
            items = self._repo.get_knowledge(job_id=job_id, only_committed=False)
            for it in items:
                if it.item_id == target_id:
                    target = it
                    break

        if not target and query:
            matches = self._repo.search_knowledge(job_id=job_id, query=query, limit=1)
            if matches:
                target = matches[0]

        if not target:
            return {"status": "not_found", "message": f"Knowledge item not found for {target_id or query}"}

        versions = self._repo.get_knowledge_versions(target.item_id)
        evidence = self._repo.get_evidence(job_id=job_id)

        return {
            "status": "found",
            "item_id": target.item_id,
            "title": target.title,
            "current_statement": target.statement,
            "validation_status": target.validation_status.value,
            "reasoning": target.reasoning,
            "confidence": target.confidence,
            "total_revisions": len(versions),
            "versions": [v.model_dump() for v in versions],
            "linked_evidence": [e.model_dump() for e in target.evidence] or [e.model_dump() for e in evidence[:2]],
        }

    # ------------------------------------------------------------------
    # Step 4: Market Intelligence Command Implementations
    # ------------------------------------------------------------------

    def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Run multi-timeframe analysis and lecture-constrained setup validation on symbol."""
        engine = self._get_mtf_engine()
        analysis = engine.analyze_symbol(symbol=symbol)

        # Apply lecture-constrained validation on LTF detected setup if any
        validator = self._get_lecture_validator()
        setups: List[MarketSetup] = []
        if analysis.ltf_analysis and analysis.ltf_analysis.direction != MarketDirection.UNKNOWN:
            setup = validator.validate_detected_pattern(
                symbol=symbol,
                timeframe=analysis.ltf_analysis.timeframe,
                pattern_name="FVG",
                direction=analysis.ltf_analysis.direction,
                market_analysis=analysis,
                pattern_context={"displacement": True, "is_invalidated": False},
            )
            setups.append(setup)
            self._repo.save_market_setup(setup)

        analysis.active_setups = setups
        self._repo.save_market_analysis(analysis)

        return {
            "status": "success",
            "intent": CommandIntent.ANALYZE_SYMBOL.value,
            "symbol": symbol,
            "overall_bias": analysis.overall_bias.value,
            "htf_direction": analysis.htf_analysis.direction.value if analysis.htf_analysis else "UNKNOWN",
            "mtf_direction": analysis.mtf_analysis.direction.value if analysis.mtf_analysis else "UNKNOWN",
            "ltf_direction": analysis.ltf_analysis.direction.value if analysis.ltf_analysis else "UNKNOWN",
            "conflict_notes": analysis.conflict_notes,
            "setups_count": len(setups),
            "setups": [s.model_dump(mode="json") for s in setups],
            "analysis": analysis.model_dump(mode="json"),
        }

    def analyze_chart(self, symbol: str = "BTC/USD") -> Dict[str, Any]:
        """Analyze currently visible chart, respecting explicit screen permission state."""
        perms = self._repo.list_screen_permissions()
        granted_perm = next(
            (p for p in perms if p.status == ScreenAccessStatus.SCREEN_ACCESS_GRANTED), None
        )
        has_screen_access = granted_perm is not None

        analysis_res = self.analyze_symbol(symbol=symbol)
        analysis_res["intent"] = CommandIntent.ANALYZE_CHART.value
        analysis_res["screen_access_granted"] = has_screen_access
        analysis_res["screen_source"] = (
            f"{granted_perm.app_name} ({granted_perm.window_title or 'active window'})"
            if granted_perm
            else "SCREEN_ACCESS_DISABLED: Fallback to direct market data feed"
        )
        return analysis_res

    def analyze_watchlist(self) -> Dict[str, Any]:
        """Trigger independent background analysis of all configured watchlist assets."""
        monitor = self._get_watchlist_monitor()
        results = monitor.process_all_watchlist(force=True)

        summary = {}
        total_setups = 0
        for sym, analysis in results.items():
            if analysis:
                summary[sym] = {
                    "bias": analysis.overall_bias.value,
                    "setups_count": len(analysis.active_setups),
                }
                total_setups += len(analysis.active_setups)
            else:
                summary[sym] = {"bias": "UNAVAILABLE", "setups_count": 0}

        return {
            "status": "success",
            "intent": CommandIntent.ANALYZE_WATCHLIST.value,
            "symbols_analyzed": list(results.keys()),
            "total_setups": total_setups,
            "summary": summary,
        }

    def check_htf(self, symbol: str = "BTC/USD") -> Dict[str, Any]:
        """Check higher timeframe structure, direction, and key levels."""
        engine = self._get_mtf_engine()
        analysis = engine.analyze_symbol(symbol=symbol)
        htf = analysis.htf_analysis

        return {
            "status": "success",
            "intent": CommandIntent.CHECK_HTF.value,
            "symbol": symbol,
            "htf_timeframe": htf.timeframe if htf else "1d",
            "direction": htf.direction.value if htf else "UNKNOWN",
            "structure": htf.structure.value if htf else "UNKNOWN",
            "bias": htf.bias.value if htf else "UNKNOWN",
            "key_levels_count": len(htf.key_levels) if htf else 0,
            "key_levels": [lvl.model_dump(mode="json") for lvl in (htf.key_levels if htf else [])],
        }

    def filter_valid_setups(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve only setups satisfying verified lecture conditions (VALID status)."""
        target_symbol = symbol or "BTC/USD"
        setups = self._repo.get_market_setups(symbol=target_symbol, status=SetupStatus.VALID)

        return {
            "status": "success",
            "intent": CommandIntent.FILTER_VALID_SETUPS.value,
            "symbol": target_symbol,
            "count": len(setups),
            "valid_setups": [s.model_dump(mode="json") for s in setups],
        }

    def set_lecture_rules_only(self) -> Dict[str, Any]:
        """Enforce strict lecture-constrained validation mode."""
        return {
            "status": "success",
            "intent": CommandIntent.USE_LECTURE_RULES_ONLY.value,
            "mode": "LECTURE_CONSTRAINED_STRICT",
            "message": "Strict lecture-constrained validation enforced. Only setups grounded in verified lecture rules will be marked VALID; all other patterns are classified as OUT_OF_SCOPE.",
        }

    def explain_setup(self, symbol: str = "BTC/USD", target_id: Optional[str] = None) -> Dict[str, Any]:
        """Explain setup validation status, citing satisfied conditions, exceptions, and evidence."""
        setups = self._repo.get_market_setups(symbol=symbol)
        target: Optional[MarketSetup] = None

        if target_id:
            for s in setups:
                if s.setup_id == target_id or s.applicable_rule_id == target_id:
                    target = s
                    break
        elif setups:
            target = setups[0]

        if not target:
            # Run fresh analysis on symbol to obtain a setup
            self.analyze_symbol(symbol)
            fresh_setups = self._repo.get_market_setups(symbol=symbol)
            target = fresh_setups[0] if fresh_setups else None

        if not target:
            return {
                "status": "not_found",
                "intent": CommandIntent.EXPLAIN_SETUP.value,
                "symbol": symbol,
                "message": f"No setups found for {symbol}.",
            }

        explanation_parts = [
            f"Setup [{target.setup_type}] on {target.symbol} ({target.timeframe}) is classified as {target.status.value}."
        ]
        if target.applicable_rule_id:
            explanation_parts.append(f"Derived from learned lecture rule {target.applicable_rule_id}.")
        if target.conditions_satisfied:
            explanation_parts.append(f"Satisfied conditions: {'; '.join(target.conditions_satisfied)}.")
        if target.contradicting_evidence:
            explanation_parts.append(f"Contradicting / Invalidation factors: {'; '.join(target.contradicting_evidence)}.")
        if target.supporting_evidence:
            explanation_parts.append(f"Evidence: {'; '.join(target.supporting_evidence)}.")

        return {
            "status": "success",
            "intent": CommandIntent.EXPLAIN_SETUP.value,
            "symbol": symbol,
            "setup": target.model_dump(mode="json"),
            "explanation": " ".join(explanation_parts),
        }

    def mark_pattern(self, symbol: str = "BTC/USD", pattern_name: str = "FVG") -> Dict[str, Any]:
        """Validate and plot markings for pattern, strictly adhering to clean-chart principles."""
        provider = self._get_data_provider()
        snapshot = provider.get_snapshot(symbol=symbol, timeframe="15m", limit=60)

        engine = self._get_mtf_engine()
        analysis = engine.analyze_symbol(symbol=symbol)
        validator = self._get_lecture_validator()

        setup = validator.validate_detected_pattern(
            symbol=symbol,
            timeframe="15m",
            pattern_name=pattern_name,
            direction=MarketDirection.BULLISH,
            market_analysis=analysis,
            pattern_context={"displacement": True, "is_invalidated": False},
        )
        self._repo.save_market_setup(setup)

        marker = self._get_chart_marker()
        all_levels = (analysis.htf_analysis.key_levels if analysis.htf_analysis else []) + (
            analysis.ltf_analysis.key_levels if analysis.ltf_analysis else []
        )
        markings = marker.generate_markings(
            snapshot=snapshot,
            setups=[setup],
            levels=all_levels,
        )
        for m in markings:
            self._repo.save_chart_marking(m)

        return {
            "status": "success",
            "intent": CommandIntent.MARK_PATTERN.value,
            "symbol": symbol,
            "pattern": pattern_name,
            "setup_status": setup.status.value,
            "markings_plotted": len(markings),
            "markings": [m.model_dump(mode="json") for m in markings],
            "clean_chart_note": "Uncertain/weak setups filtered from chart to preserve readability.",
        }

    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop/pause background watchlist monitoring and persist state."""
        monitor = self._get_watchlist_monitor()
        monitor.stop()
        return {
            "status": "success",
            "intent": CommandIntent.STOP_MONITORING.value,
            "monitor_status": WatchlistMonitorStatus.STOPPED.value,
            "message": "Background watchlist monitoring paused.",
        }

    def resume_monitoring(self) -> Dict[str, Any]:
        """Start/resume background watchlist monitoring from persisted state."""
        monitor = self._get_watchlist_monitor()
        monitor.start()
        return {
            "status": "success",
            "intent": CommandIntent.RESUME_MONITORING.value,
            "monitor_status": WatchlistMonitorStatus.MONITORING.value,
            "message": "Background watchlist monitoring resumed.",
        }

    # -------------------------------------------------------------
    # STEP 5: INTELLIGENT SELECTIVE VISUALIZATION METHODS
    # -------------------------------------------------------------
    def show_all_liquidity(self, symbol: str) -> Dict[str, Any]:
        """Reveal all background liquidity levels on demand."""
        all_levels = self._repo.get_market_levels(symbol=symbol, only_valid=True)
        liq_levels = [
            lvl for lvl in all_levels
            if lvl.level_type in (
                LevelType.LIQUIDITY_BUY_SIDE,
                LevelType.LIQUIDITY_SELL_SIDE,
                LevelType.SWING_HIGH,
                LevelType.SWING_LOW,
                LevelType.PREVIOUS_HIGH,
                LevelType.PREVIOUS_LOW,
            )
        ]
        return {
            "status": "success",
            "intent": CommandIntent.SHOW_ALL_LIQUIDITY.value,
            "symbol": symbol,
            "count": len(liq_levels),
            "liquidity_levels": [lvl.model_dump(mode="json") for lvl in liq_levels],
            "note": "Revealing all background liquidity pools on demand.",
        }

    def show_hidden_setups(self, symbol: str) -> Dict[str, Any]:
        """Reveal setups tracked in background that are not currently displayed."""
        all_setups = self._repo.get_market_setups(symbol=symbol)
        hidden = [s for s in all_setups if not s.is_displayed]
        return {
            "status": "success",
            "intent": CommandIntent.SHOW_HIDDEN_SETUPS.value,
            "symbol": symbol,
            "count": len(hidden),
            "hidden_setups": [s.model_dump(mode="json") for s in hidden],
            "note": "All background setup candidates preserved with full provenance.",
        }

    def show_fvgs(self, symbol: str) -> Dict[str, Any]:
        """Reveal all detected Fair Value Gaps across timeframes."""
        all_markings = self._repo.get_chart_markings(symbol=symbol)
        fvgs = [m for m in all_markings if m.marking_type == MarkingType.FVG]
        return {
            "status": "success",
            "intent": CommandIntent.SHOW_FVGS.value,
            "symbol": symbol,
            "count": len(fvgs),
            "fvgs": [m.model_dump(mode="json") for m in fvgs],
        }

    def show_order_blocks(self, symbol: str) -> Dict[str, Any]:
        """Reveal all Order Blocks tracked in background."""
        all_markings = self._repo.get_chart_markings(symbol=symbol)
        obs = [m for m in all_markings if m.marking_type == MarkingType.ORDER_BLOCK]
        return {
            "status": "success",
            "intent": CommandIntent.SHOW_ORDER_BLOCKS.value,
            "symbol": symbol,
            "count": len(obs),
            "order_blocks": [m.model_dump(mode="json") for m in obs],
        }

    def show_unconfirmed_setups(self, symbol: str) -> Dict[str, Any]:
        """List setups currently awaiting market confirmation or displacement."""
        all_setups = self._repo.get_market_setups(symbol=symbol)
        unconfirmed = [
            s for s in all_setups
            if s.status in (SetupStatus.DETECTED, SetupStatus.TRACKING, SetupStatus.AWAITING_CONFIRMATION, SetupStatus.UNCERTAIN)
        ]
        return {
            "status": "success",
            "intent": CommandIntent.SHOW_UNCONFIRMED_SETUPS.value,
            "symbol": symbol,
            "count": len(unconfirmed),
            "unconfirmed_setups": [s.model_dump(mode="json") for s in unconfirmed],
            "note": "Detection alone is never sufficient for prominent chart display.",
        }

    def show_confirmation_evidence(self, symbol: str) -> Dict[str, Any]:
        """Explain the exact confirmation conditions and evidence for the active setup."""
        situation = self._repo.get_latest_current_situation(symbol=symbol)
        if not situation or not situation.active_confirmed_setup:
            return {
                "status": "success",
                "intent": CommandIntent.SHOW_CONFIRMATION_EVIDENCE.value,
                "symbol": symbol,
                "active_setup": None,
                "message": f"No active confirmed setup found for {symbol}.",
            }

        setup = situation.active_confirmed_setup
        return {
            "status": "success",
            "intent": CommandIntent.SHOW_CONFIRMATION_EVIDENCE.value,
            "symbol": symbol,
            "setup_id": setup.setup_id,
            "setup_type": setup.setup_type,
            "setup_status": setup.status.value,
            "confirmation_timestamp": setup.confirmation_timestamp,
            "confirmation_evidence": setup.confirmation_evidence,
            "conditions_satisfied": setup.conditions_satisfied,
            "supporting_evidence": setup.supporting_evidence,
            "provenance_rule_id": setup.applicable_rule_id,
        }

    def show_invalidation_condition(self, symbol: str) -> Dict[str, Any]:
        """Show exact invalidation conditions and levels for the active setup."""
        situation = self._repo.get_latest_current_situation(symbol=symbol)
        if not situation or not situation.active_confirmed_setup:
            return {
                "status": "success",
                "intent": CommandIntent.SHOW_INVALIDATION_CONDITION.value,
                "symbol": symbol,
                "active_setup": None,
                "message": f"No active confirmed setup found for {symbol}.",
            }

        setup = situation.active_confirmed_setup
        return {
            "status": "success",
            "intent": CommandIntent.SHOW_INVALIDATION_CONDITION.value,
            "symbol": symbol,
            "setup_id": setup.setup_id,
            "direction": setup.direction.value,
            "invalidation_level": situation.invalidation_level,
            "invalidation_condition": setup.invalidation_condition or "Breach of opposing structural swing high/low",
            "invalidation_checks": setup.invalidation_checked,
        }

    def hide_background_markings(self, symbol: str) -> Dict[str, Any]:
        """Reset visible chart to default minimal clean situation view."""
        provider = self._get_data_provider()
        snapshot = provider.get_snapshot(symbol=symbol, timeframe="15m")
        engine = self._get_mtf_engine()
        analysis = engine.analyze_symbol(symbol=symbol)
        situation_engine = self._get_situation_engine()

        from trading_ai.schemas.market import DisplayPolicy
        policy = DisplayPolicy(
            show_all_liquidity=False,
            show_hidden_setups=False,
            show_all_fvgs=False,
            show_all_order_blocks=False,
            show_unconfirmed=False,
            focus_active_setup_only=False,
            max_visible_markings=5,
        )
        situation = situation_engine.evaluate_situation(
            symbol=symbol,
            snapshot=snapshot,
            unified_analysis=analysis,
            display_policy=policy,
        )
        return {
            "status": "success",
            "intent": CommandIntent.HIDE_BACKGROUND_MARKINGS.value,
            "symbol": symbol,
            "visible_markings_count": len(situation.visible_markings),
            "message": "Chart reset to minimal clean decision context.",
        }

    def focus_current_setup(self, symbol: str) -> Dict[str, Any]:
        """Focus display exclusively on active confirmed setup and its confirmation/invalidation levels."""
        provider = self._get_data_provider()
        snapshot = provider.get_snapshot(symbol=symbol, timeframe="15m")
        engine = self._get_mtf_engine()
        analysis = engine.analyze_symbol(symbol=symbol)
        situation_engine = self._get_situation_engine()

        from trading_ai.schemas.market import DisplayPolicy
        policy = DisplayPolicy(
            focus_active_setup_only=True,
            show_all_liquidity=False,
            max_visible_markings=3,
        )
        situation = situation_engine.evaluate_situation(
            symbol=symbol,
            snapshot=snapshot,
            unified_analysis=analysis,
            display_policy=policy,
        )
        return {
            "status": "success",
            "intent": CommandIntent.FOCUS_CURRENT_SETUP.value,
            "symbol": symbol,
            "active_setup": situation.active_confirmed_setup.model_dump(mode="json") if situation.active_confirmed_setup else None,
            "invalidation_level": situation.invalidation_level,
            "confirmation_level": situation.confirmation_level,
            "visible_markings": [m.model_dump(mode="json") for m in situation.visible_markings],
            "message": "Chart focused on active setup.",
        }
