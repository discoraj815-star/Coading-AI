"""Repository abstraction and SQLite implementation for all pipeline artifacts."""

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from trading_ai.core.token_manager import TokenUsageRecord
from trading_ai.schemas import (
    AlignmentRelationship,
    AudioVisualAlignment,
    CommitBarrierState,
    ContextItem,
    ContextItemSource,
    ContextSnapshot,
    ErrorSeverity,
    EvidenceTrace,
    JobMetadata,
    JobStatus,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    LectureNotes,
    MemoryEmbedding,
    PipelineStage,
    TaskUnit,
    TaskUnitStatus,
    TimestampedNoteEntry,
    TranscriptSegment,
    VisualEvent,
    VisualRegion,
    CrossModalConflict,
    DeicticResolution,
    # Step 3
    BackgroundJobState,
    CommandIntent,
    KnowledgeVersion,
    SourceMetadata,
    SourceStatus,
    SourceType,
    UserCommand,
    # Step 4
    Candle,
    ChartMarking,
    LevelType,
    MarketBias,
    MarketDataStatus,
    MarketDirection,
    MarketLevel,
    MarketSetup,
    MarketSnapshot,
    MarketStructureType,
    MarkingType,
    MarkingValidity,
    ScreenAccessStatus,
    ScreenPermission,
    SetupStatus,
    TimeframeAnalysis,
    TimeframeHierarchy,
    UnifiedMarketAnalysis,
    WatchlistMonitorState,
    WatchlistMonitorStatus,
    CurrentSituationSnapshot,
    # Step 6
    EventImpact,
    EventStatus,
    NewsRiskLevel,
    NewsTimingCategory,
    ProviderStatus,
    TechFundRelationship,
    FundamentalEvent,
    FundamentalProviderState,
    NewsTimingContext,
    TechFundConflict,
    TechnicalFundamentalAnalysis,
    # Step 7
    AlertDecision,
    AlertPriority,
    AlertLifecycleState,
    AlertType,
    AlertQuality,
    VoiceIntent,
    DisciplineViolationType,
    AlertEvent,
    AlertStateRecord,
    TradingCompanionContext,
    ScenarioCondition,
    DisciplineConfig,
    DisciplineState,
    TradeIntentRequest,
    TradeConfirmationGateResult,
    CompanionQueryRequest,
    CompanionQueryResponse,
    CompanionInteractionRecord,
    # Step 8
    EngagementMode,
    ProposalLifecycleState,
    ExecutionMode,
    ExecutionEnvironment,
    OrderType,
    OrderStatus,
    PositionStatus,
    ConfluenceCategory,
    PositionManagementAction,
    TradeConfluence,
    TradeThesis,
    RiskAssessment,
    TradeProposal,
    ExecutionPolicy,
    ExecutionIntent,
    Order,
    Position,
    PositionEvent,
    # Step 9
    AssetClass,
    BacktestMode,
    BacktestStatus,
    RuleLifecycleState,
    TradeSource,
    RuleVersion,
    StrategyVersion,
    BacktestConfig,
    BacktestTrade,
    BacktestMetrics,
    BacktestRun,
    RulePromotionRecord,
    # Step 10
    JournalEnvironment,
    JournalBacktestType,
    TradeOutcome,
    ExitReasonCategory,
    ObservedBehaviorType,
    LessonLifecycleState,
    PreTradeReviewStatus,
    PlannedTradeValues,
    ActualTradeValues,
    ObservedBehavior,
    TradeJournalEntry,
    JournalLesson,
    PreTradeReviewResult,
    JournalProcessingCheckpoint,
    # Step 11
    ComponentUpgradeRecord,
    UpgradeStatus,
    LatencyTelemetry,
    # Step 12
    DeviceRecord,
    DeviceStatus,
    PCCapabilityRule,
    PCCapability,
    CapabilityState,
    SecurityAuditEntry,
    NotificationItem,
    NotificationPriority,
    # Step 13
    ThreatLevel,
    KnowledgeProtectionState,
    RecoveryClassification,
    ThreatIncident,
    KnowledgeIntegrityRecord,
    BackupMetadata,
    RecoveryCheckpoint,
)
from trading_ai.storage.db import DatabaseManager


class BaseKnowledgeRetriever(ABC):
    """Abstract retrieval interface enabling future RAG and vector database backends."""

    @abstractmethod
    def search_knowledge(
        self,
        job_id: str,
        query: str,
        item_types: Optional[List[KnowledgeType]] = None,
        limit: int = 10,
    ) -> List[KnowledgeItem]:
        """Search committed knowledge items by query and optional filters."""
        pass


class Repository(BaseKnowledgeRetriever):
    """Full-featured SQLite repository for Trading AI Assistant artifacts."""

    def __init__(self, db_manager: Optional[Any] = None):
        if isinstance(db_manager, str):
            self._db = DatabaseManager(db_manager)
        else:
            self._db = db_manager or DatabaseManager()

    # -------------------------------------------------------------
    # JOBS
    # -------------------------------------------------------------
    def save_job(self, job: JobMetadata) -> None:
        """Insert or update a lecture job record."""
        sql = """
        INSERT INTO jobs (
            job_id, file_hash, filename, file_path, duration, metadata_json,
            status, current_stage, serial_number, retry_count, last_error,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            status = excluded.status,
            current_stage = excluded.current_stage,
            serial_number = excluded.serial_number,
            retry_count = excluded.retry_count,
            last_error = excluded.last_error,
            duration = excluded.duration,
            metadata_json = excluded.metadata_json,
            updated_at = excluded.updated_at;
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    job.job_id,
                    job.file_hash,
                    job.filename,
                    job.file_path,
                    job.duration,
                    json.dumps(job.metadata),
                    job.status.value,
                    job.current_stage.value,
                    job.serial_number,
                    job.retry_count,
                    job.last_error,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                ),
            )

    def get_job(self, job_id: str) -> Optional[JobMetadata]:
        """Retrieve a lecture job by its unique ID."""
        sql = "SELECT * FROM jobs WHERE job_id = ?;"
        with self._db.connection() as conn:
            row = conn.execute(sql, (job_id,)).fetchone()
            if not row:
                return None
            return JobMetadata(
                job_id=row["job_id"],
                file_hash=row["file_hash"],
                filename=row["filename"],
                file_path=row["file_path"],
                duration=row["duration"],
                metadata=json.loads(row["metadata_json"]),
                status=JobStatus(row["status"]),
                current_stage=PipelineStage(row["current_stage"]),
                serial_number=row["serial_number"],
                retry_count=row["retry_count"],
                last_error=row["last_error"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def _ensure_job_exists(self, job_id: str) -> None:
        """Ensure parent job record exists to satisfy foreign key constraints."""
        job = self.get_job(job_id)
        if not job:
            now = datetime.now(timezone.utc)
            placeholder = JobMetadata(
                job_id=job_id,
                file_hash=job_id,
                filename=f"{job_id}.mp4",
                file_path=f"storage://{job_id}",
                duration=30.0,
                status=JobStatus.PENDING,
                current_stage=PipelineStage.SOURCE,
                created_at=now,
                updated_at=now,
            )
            self.save_job(placeholder)

    def list_jobs(self) -> List[JobMetadata]:
        """List all lecture jobs in chronological order."""
        sql = "SELECT * FROM jobs ORDER BY created_at DESC;"
        with self._db.connection() as conn:
            rows = conn.execute(sql).fetchall()
            jobs = []
            for row in rows:
                jobs.append(
                    JobMetadata(
                        job_id=row["job_id"],
                        file_hash=row["file_hash"],
                        filename=row["filename"],
                        file_path=row["file_path"],
                        duration=row["duration"],
                        metadata=json.loads(row["metadata_json"]),
                        status=JobStatus(row["status"]),
                        current_stage=PipelineStage(row["current_stage"]),
                        serial_number=row["serial_number"],
                        retry_count=row["retry_count"],
                        last_error=row["last_error"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )
            return jobs

    # -------------------------------------------------------------
    # TRANSCRIPTS
    # -------------------------------------------------------------
    def save_transcript_segment(self, segment: TranscriptSegment) -> None:
        """Insert or replace a timestamped transcript segment."""
        sql = """
        INSERT OR REPLACE INTO transcripts (
            segment_id, job_id, start_time, end_time, transcript, language,
            speaker, important_statements_json, terminology_json, rules_json,
            examples_json, uncertainty_json, confidence
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    segment.segment_id,
                    segment.job_id,
                    segment.start_time,
                    segment.end_time,
                    segment.transcript,
                    segment.language,
                    segment.speaker,
                    json.dumps(segment.important_statements),
                    json.dumps(segment.terminology),
                    json.dumps(segment.rules),
                    json.dumps(segment.examples),
                    json.dumps(segment.uncertainty),
                    segment.confidence,
                ),
            )

    def get_transcripts(self, job_id: str) -> List[TranscriptSegment]:
        """Fetch all transcript segments for a job in chronological order."""
        sql = "SELECT * FROM transcripts WHERE job_id = ? ORDER BY start_time ASC;"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (job_id,)).fetchall()
            return [
                TranscriptSegment(
                    segment_id=row["segment_id"],
                    job_id=row["job_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    transcript=row["transcript"],
                    language=row["language"],
                    speaker=row["speaker"],
                    important_statements=json.loads(row["important_statements_json"]),
                    terminology=json.loads(row["terminology_json"]),
                    rules=json.loads(row["rules_json"]),
                    examples=json.loads(row["examples_json"]),
                    uncertainty=json.loads(row["uncertainty_json"]),
                    confidence=row["confidence"],
                )
                for row in rows
            ]

    # -------------------------------------------------------------
    # VISUAL EVENTS
    # -------------------------------------------------------------
    def save_visual_event(self, event: VisualEvent) -> None:
        """Insert or replace a detected visual event."""
        regions_json = json.dumps([r.model_dump() for r in event.regions])
        sql = """
        INSERT OR REPLACE INTO visual_events (
            event_id, job_id, timestamp, frame_ref, visual_description,
            detected_event, symbol, timeframe, candles_json, drawings_json,
            zones_json, arrows_json, labels_json, screen_text_json,
            cursor_interaction, chart_changes_json, confidence, regions_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    event.event_id,
                    event.job_id,
                    event.timestamp,
                    event.frame_ref,
                    event.visual_description,
                    event.detected_event,
                    event.symbol,
                    event.timeframe,
                    json.dumps(event.candles),
                    json.dumps(event.drawings),
                    json.dumps(event.highlighted_zones),
                    json.dumps(event.arrows),
                    json.dumps(event.labels),
                    json.dumps(event.screen_text),
                    event.cursor_interaction,
                    json.dumps(event.chart_changes),
                    event.confidence,
                    regions_json,
                ),
            )

    def get_visual_events(self, job_id: str) -> List[VisualEvent]:
        """Fetch all visual events for a job chronologically."""
        sql = "SELECT * FROM visual_events WHERE job_id = ? ORDER BY timestamp ASC;"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (job_id,)).fetchall()
            events = []
            for row in rows:
                regions_raw = row["regions_json"] if "regions_json" in row.keys() and row["regions_json"] else "[]"
                regions = [VisualRegion(**r) for r in json.loads(regions_raw)]
                events.append(
                    VisualEvent(
                        event_id=row["event_id"],
                        job_id=row["job_id"],
                        timestamp=row["timestamp"],
                        frame_ref=row["frame_ref"],
                        visual_description=row["visual_description"],
                        detected_event=row["detected_event"],
                        symbol=row["symbol"],
                        timeframe=row["timeframe"],
                        candles=json.loads(row["candles_json"]),
                        drawings=json.loads(row["drawings_json"]),
                        highlighted_zones=json.loads(row["zones_json"]),
                        arrows=json.loads(row["arrows_json"]),
                        labels=json.loads(row["labels_json"]),
                        screen_text=json.loads(row["screen_text_json"]),
                        cursor_interaction=row["cursor_interaction"],
                        chart_changes=json.loads(row["chart_changes_json"]),
                        confidence=row["confidence"],
                        regions=regions,
                    )
                )
            return events

    # -------------------------------------------------------------
    # ALIGNMENTS
    # -------------------------------------------------------------
    def save_alignment(self, alignment: AudioVisualAlignment) -> None:
        """Insert or replace an audio-visual alignment record."""
        conflicts_json = json.dumps([c.model_dump() for c in alignment.conflicts])
        deictic_json = json.dumps([d.model_dump() for d in alignment.deictic_resolutions])
        sql = """
        INSERT OR REPLACE INTO alignments (
            alignment_id, job_id, start_time, end_time, audio_json,
            visual_json, relationship, confidence, reasoning,
            conflicts_json, deictic_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    alignment.alignment_id,
                    alignment.job_id,
                    alignment.start_time,
                    alignment.end_time,
                    json.dumps(alignment.audio.model_dump()) if alignment.audio else None,
                    json.dumps([v.model_dump() for v in alignment.visual]),
                    alignment.relationship.value,
                    alignment.confidence,
                    alignment.reasoning,
                    conflicts_json,
                    deictic_json,
                ),
            )

    def get_alignments(self, job_id: str) -> List[AudioVisualAlignment]:
        """Fetch all audio-visual alignments for a job."""
        sql = "SELECT * FROM alignments WHERE job_id = ? ORDER BY start_time ASC;"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (job_id,)).fetchall()
            alignments = []
            for row in rows:
                audio_data = json.loads(row["audio_json"]) if row["audio_json"] else None
                audio = TranscriptSegment(**audio_data) if audio_data else None
                visual_data = json.loads(row["visual_json"])
                visual = [VisualEvent(**v) for v in visual_data]
                conflicts_raw = row["conflicts_json"] if "conflicts_json" in row.keys() and row["conflicts_json"] else "[]"
                conflicts = [CrossModalConflict(**c) for c in json.loads(conflicts_raw)]
                deictic_raw = row["deictic_json"] if "deictic_json" in row.keys() and row["deictic_json"] else "[]"
                deictic_resolutions = [DeicticResolution(**d) for d in json.loads(deictic_raw)]
                alignments.append(
                    AudioVisualAlignment(
                        alignment_id=row["alignment_id"],
                        job_id=row["job_id"],
                        start_time=row["start_time"],
                        end_time=row["end_time"],
                        audio=audio,
                        visual=visual,
                        relationship=AlignmentRelationship(row["relationship"]),
                        confidence=row["confidence"],
                        reasoning=row["reasoning"],
                        conflicts=conflicts,
                        deictic_resolutions=deictic_resolutions,
                    )
                )
            return alignments

    # -------------------------------------------------------------
    # EVIDENCE
    # -------------------------------------------------------------
    def save_evidence(self, ev: EvidenceTrace) -> None:
        """Insert or replace an immutable evidence trace."""
        self._ensure_job_exists(ev.job_id)
        sql = """
        INSERT OR REPLACE INTO evidence (
            evidence_id, job_id, start_time, end_time, audio_quote,
            visual_reference, extracted_statement, confidence, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    ev.evidence_id,
                    ev.job_id,
                    ev.start_time,
                    ev.end_time,
                    ev.audio_quote,
                    ev.visual_reference,
                    ev.extracted_statement,
                    ev.confidence,
                    ev.notes,
                ),
            )

    def get_evidence(self, job_id: str) -> List[EvidenceTrace]:
        """Retrieve all evidence traces for a job."""
        sql = "SELECT * FROM evidence WHERE job_id = ? ORDER BY start_time ASC;"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (job_id,)).fetchall()
            return [
                EvidenceTrace(
                    evidence_id=row["evidence_id"],
                    job_id=row["job_id"],
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    audio_quote=row["audio_quote"],
                    visual_reference=row["visual_reference"],
                    extracted_statement=row["extracted_statement"],
                    confidence=row["confidence"],
                    notes=row["notes"],
                )
                for row in rows
            ]

    # -------------------------------------------------------------
    # KNOWLEDGE & COMMIT BARRIER
    # -------------------------------------------------------------
    def save_knowledge_item(
        self, item: KnowledgeItem, serial_number: int = 1, is_committed: bool = False
    ) -> None:
        """Save a knowledge item with its associated serial number and commit status."""
        self._ensure_job_exists(item.job_id)
        sql = """
        INSERT OR REPLACE INTO knowledge (
            item_id, job_id, item_type, title, statement, conditions_json, exceptions_json, reasoning,
            validation_status, confidence, start_time, end_time,
            evidence_json, parent_id, tags_json, serial_number, is_committed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    item.item_id,
                    item.job_id,
                    item.item_type.value,
                    item.title,
                    item.statement,
                    json.dumps(item.conditions),
                    json.dumps(item.exceptions),
                    item.reasoning,
                    item.validation_status.value,
                    item.confidence,
                    item.start_time,
                    item.end_time,
                    json.dumps([e.model_dump() for e in item.evidence]),
                    item.parent_id,
                    json.dumps(item.tags),
                    serial_number,
                    1 if is_committed else 0,
                ),
            )

    def save_knowledge(self, items: List[KnowledgeItem], is_committed: bool = True) -> None:
        """Persist a list of knowledge items."""
        for item in items:
            self.save_knowledge_item(item, is_committed=is_committed)

    def commit_knowledge_for_serial(self, job_id: str, serial_number: int) -> int:
        """Mark all knowledge items for a given serial as committed. Returns number of rows updated."""
        sql = "UPDATE knowledge SET is_committed = 1 WHERE job_id = ? AND serial_number = ?;"
        with self._db.connection() as conn:
            cursor = conn.execute(sql, (job_id, serial_number))
            return cursor.rowcount

    def get_knowledge(
        self,
        job_id: Optional[str] = None,
        only_committed: bool = True,
        item_type: Optional[KnowledgeType] = None,
    ) -> List[KnowledgeItem]:
        """Fetch knowledge items for a job (or all jobs if job_id is None), optionally filtered to committed-only or specific type."""
        conditions = []
        params: List[Any] = []

        if job_id is not None:
            conditions.append("job_id = ?")
            params.append(job_id)

        if only_committed:
            conditions.append("is_committed = 1")
        if item_type:
            conditions.append("item_type = ?")
            params.append(item_type.value)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM knowledge {where_clause} ORDER BY start_time ASC;"
        with self._db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            items = []
            for row in rows:
                ev_data = json.loads(row["evidence_json"])
                evidence = [EvidenceTrace(**e) for e in ev_data]
                keys = row.keys()
                conds = json.loads(row["conditions_json"]) if "conditions_json" in keys else []
                excs = json.loads(row["exceptions_json"]) if "exceptions_json" in keys else []
                items.append(
                    KnowledgeItem(
                        item_id=row["item_id"],
                        job_id=row["job_id"],
                        item_type=KnowledgeType(row["item_type"]),
                        title=row["title"],
                        statement=row["statement"],
                        conditions=conds,
                        exceptions=excs,
                        reasoning=row["reasoning"],
                        validation_status=KnowledgeValidationStatus(row["validation_status"]),
                        confidence=row["confidence"],
                        start_time=row["start_time"],
                        end_time=row["end_time"],
                        evidence=evidence,
                        parent_id=row["parent_id"],
                        tags=json.loads(row["tags_json"]),
                    )
                )
            return items

    def search_knowledge(
        self,
        job_id: str,
        query: str,
        item_types: Optional[List[KnowledgeType]] = None,
        limit: int = 10,
    ) -> List[KnowledgeItem]:
        """Basic text search across committed knowledge items (RAG retrieval baseline)."""
        conditions = ["job_id = ?", "is_committed = 1", "(title LIKE ? OR statement LIKE ? OR tags_json LIKE ?)"]
        search_pattern = f"%{query}%"
        params: List[Any] = [job_id, search_pattern, search_pattern, search_pattern]

        if item_types:
            placeholders = ",".join("?" for _ in item_types)
            conditions.append(f"item_type IN ({placeholders})")
            params.extend([t.value for t in item_types])

        sql = f"SELECT * FROM knowledge WHERE {' AND '.join(conditions)} LIMIT {limit};"
        with self._db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            results = []
            for row in rows:
                ev_data = json.loads(row["evidence_json"])
                evidence = [EvidenceTrace(**e) for e in ev_data]
                results.append(
                    KnowledgeItem(
                        item_id=row["item_id"],
                        job_id=row["job_id"],
                        item_type=KnowledgeType(row["item_type"]),
                        title=row["title"],
                        statement=row["statement"],
                        reasoning=row["reasoning"],
                        validation_status=KnowledgeValidationStatus(row["validation_status"]),
                        confidence=row["confidence"],
                        start_time=row["start_time"],
                        end_time=row["end_time"],
                        evidence=evidence,
                        parent_id=row["parent_id"],
                        tags=json.loads(row["tags_json"]),
                    )
                )
            return results

    # -------------------------------------------------------------
    # NOTES
    # -------------------------------------------------------------
    def save_notes(self, notes: LectureNotes) -> None:
        """Insert or replace structured lecture notes."""
        sql = """
        INSERT OR REPLACE INTO notes (
            job_id, title, overview, concepts_json, definitions_json,
            rules_json, examples_json, exceptions_json, visual_explanations_json,
            important_statements_json, relationships_json, uncertainty_json,
            evidence_timestamps_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    notes.job_id,
                    notes.title,
                    notes.overview,
                    json.dumps([e.model_dump() for e in notes.concepts]),
                    json.dumps([e.model_dump() for e in notes.definitions]),
                    json.dumps([e.model_dump() for e in notes.rules]),
                    json.dumps([e.model_dump() for e in notes.examples]),
                    json.dumps([e.model_dump() for e in notes.exceptions]),
                    json.dumps([e.model_dump() for e in notes.visual_explanations]),
                    json.dumps([e.model_dump() for e in notes.important_statements]),
                    json.dumps([e.model_dump() for e in notes.relationships]),
                    json.dumps([e.model_dump() for e in notes.uncertainty]),
                    json.dumps(notes.evidence_timestamps),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_notes(self, job_id: str) -> Optional[LectureNotes]:
        """Fetch structured lecture notes for a job."""
        sql = "SELECT * FROM notes WHERE job_id = ?;"
        with self._db.connection() as conn:
            row = conn.execute(sql, (job_id,)).fetchone()
            if not row:
                return None
            return LectureNotes(
                job_id=row["job_id"],
                title=row["title"],
                overview=row["overview"],
                concepts=[TimestampedNoteEntry(**e) for e in json.loads(row["concepts_json"])],
                definitions=[TimestampedNoteEntry(**e) for e in json.loads(row["definitions_json"])],
                rules=[TimestampedNoteEntry(**e) for e in json.loads(row["rules_json"])],
                examples=[TimestampedNoteEntry(**e) for e in json.loads(row["examples_json"])],
                exceptions=[TimestampedNoteEntry(**e) for e in json.loads(row["exceptions_json"])],
                visual_explanations=[TimestampedNoteEntry(**e) for e in json.loads(row["visual_explanations_json"])],
                important_statements=[TimestampedNoteEntry(**e) for e in json.loads(row["important_statements_json"])],
                relationships=[TimestampedNoteEntry(**e) for e in json.loads(row["relationships_json"])],
                uncertainty=[TimestampedNoteEntry(**e) for e in json.loads(row["uncertainty_json"])],
                evidence_timestamps=json.loads(row["evidence_timestamps_json"]),
            )

    # -------------------------------------------------------------
    # SUPERVISOR & TASK UNITS
    # -------------------------------------------------------------
    def save_task_unit(self, unit: TaskUnit) -> None:
        """Insert or update a task unit."""
        self._ensure_job_exists(unit.job_id)
        sql = """
        INSERT INTO task_units (
            unit_id, job_id, serial_number, stage, start_time, end_time,
            status, payload_json, result_json, retry_count, max_retries,
            last_error, error_severity, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(job_id, serial_number) DO UPDATE SET
            status = excluded.status,
            payload_json = excluded.payload_json,
            result_json = excluded.result_json,
            retry_count = excluded.retry_count,
            last_error = excluded.last_error,
            error_severity = excluded.error_severity,
            updated_at = excluded.updated_at;
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    unit.unit_id,
                    unit.job_id,
                    unit.serial_number,
                    unit.stage.value,
                    unit.start_time,
                    unit.end_time,
                    unit.status.value,
                    json.dumps(unit.payload),
                    json.dumps(unit.result) if unit.result else None,
                    unit.retry_count,
                    unit.max_retries,
                    unit.last_error,
                    unit.error_severity.value if unit.error_severity else None,
                    unit.created_at.isoformat(),
                    unit.updated_at.isoformat(),
                ),
            )

    def get_task_unit(self, job_id: str, serial_number: int) -> Optional[TaskUnit]:
        """Fetch a specific task unit by job ID and serial number."""
        sql = "SELECT * FROM task_units WHERE job_id = ? AND serial_number = ?;"
        with self._db.connection() as conn:
            row = conn.execute(sql, (job_id, serial_number)).fetchone()
            if not row:
                return None
            return TaskUnit(
                unit_id=row["unit_id"],
                job_id=row["job_id"],
                serial_number=row["serial_number"],
                stage=PipelineStage(row["stage"]),
                start_time=row["start_time"],
                end_time=row["end_time"],
                status=TaskUnitStatus(row["status"]),
                payload=json.loads(row["payload_json"]),
                result=json.loads(row["result_json"]) if row["result_json"] else None,
                retry_count=row["retry_count"],
                max_retries=row["max_retries"],
                last_error=row["last_error"],
                error_severity=ErrorSeverity(row["error_severity"]) if row["error_severity"] else None,
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_task_units(self, job_id: str) -> List[TaskUnit]:
        """List all task units for a job sorted strictly by serial number."""
        sql = "SELECT * FROM task_units WHERE job_id = ? ORDER BY serial_number ASC;"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (job_id,)).fetchall()
            return [
                TaskUnit(
                    unit_id=row["unit_id"],
                    job_id=row["job_id"],
                    serial_number=row["serial_number"],
                    stage=PipelineStage(row["stage"]),
                    start_time=row["start_time"],
                    end_time=row["end_time"],
                    status=TaskUnitStatus(row["status"]),
                    payload=json.loads(row["payload_json"]),
                    result=json.loads(row["result_json"]) if row["result_json"] else None,
                    retry_count=row["retry_count"],
                    max_retries=row["max_retries"],
                    last_error=row["last_error"],
                    error_severity=ErrorSeverity(row["error_severity"]) if row["error_severity"] else None,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in rows
            ]

    # -------------------------------------------------------------
    # COMMIT BARRIER STATE
    # -------------------------------------------------------------
    def save_commit_barrier(self, state: CommitBarrierState) -> None:
        """Persist the Knowledge Commit Barrier state."""
        sql = """
        INSERT OR REPLACE INTO commit_barriers (
            job_id, highest_committed_serial, buffered_units_json,
            blocked_on_serial, updated_at
        ) VALUES (?, ?, ?, ?, ?);
        """
        with self._db.get_connection() as conn:
            conn.execute(
                sql,
                (
                    state.job_id,
                    state.highest_committed_serial,
                    json.dumps(state.buffered_units),
                    state.blocked_on_serial,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_commit_barrier(self, job_id: str) -> CommitBarrierState:
        """Retrieve commit barrier state, initializing clean state if not yet tracked."""
        sql = "SELECT * FROM commit_barriers WHERE job_id = ?;"
        with self._db.get_connection() as conn:
            row = conn.execute(sql, (job_id,)).fetchone()
            if not row:
                return CommitBarrierState(
                    job_id=job_id,
                    highest_committed_serial=0,
                    buffered_units=[],
                    blocked_on_serial=None,
                )
            return CommitBarrierState(
                job_id=row["job_id"],
                highest_committed_serial=row["highest_committed_serial"],
                buffered_units=json.loads(row["buffered_units_json"]),
                blocked_on_serial=row["blocked_on_serial"],
            )

    # -------------------------------------------------------------
    # TOKEN USAGE & ERRORS
    # -------------------------------------------------------------
    def save_token_usage(self, record: TokenUsageRecord) -> None:
        """Log token consumption to SQLite for audit."""
        sql = """
        INSERT INTO token_usage (
            job_id, serial_number, stage, model, input_tokens,
            output_tokens, total_tokens, cached_tokens, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.get_connection() as conn:
            conn.execute(
                sql,
                (
                    record.job_id,
                    record.serial_number,
                    record.stage,
                    record.model,
                    record.input_tokens,
                    record.output_tokens,
                    record.total_tokens,
                    record.cached_tokens,
                    record.timestamp.isoformat(),
                ),
            )

    def log_error(
        self,
        job_id: str,
        serial_number: int,
        stage: str,
        error_message: str,
        severity: ErrorSeverity,
        retry_count: int = 0,
        is_recovered: bool = False,
    ) -> None:
        """Log a processing failure with recovery categorization."""
        sql = """
        INSERT INTO errors (
            job_id, serial_number, stage, error_message, error_severity,
            is_recovered, retry_count, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.get_connection() as conn:
            conn.execute(
                sql,
                (
                    job_id,
                    serial_number,
                    stage,
                    error_message,
                    severity.value,
                    1 if is_recovered else 0,
                    retry_count,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    # ------------------------------------------------------------------
    # CONTEXT SNAPSHOTS (Step 2)
    # ------------------------------------------------------------------
    def save_context_snapshot(self, snapshot: ContextSnapshot) -> None:
        """Persist an active context window checkpoint for restart-recovery."""
        sql = """
        INSERT OR REPLACE INTO context_snapshots (
            snapshot_id, job_id, serial_number, window_tokens,
            window_items_json, pruned_items_json, compression_summary, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    snapshot.snapshot_id,
                    snapshot.job_id,
                    snapshot.serial_number,
                    snapshot.window_tokens,
                    json.dumps(snapshot.window_items_json),
                    json.dumps(snapshot.pruned_items_json),
                    snapshot.compression_summary,
                    snapshot.created_at.isoformat(),
                ),
            )

    def get_latest_context_snapshot(
        self, job_id: str
    ) -> Optional[ContextSnapshot]:
        """Load the most recent context snapshot for a job (highest serial)."""
        sql = """
        SELECT * FROM context_snapshots
        WHERE job_id = ?
        ORDER BY serial_number DESC, created_at DESC
        LIMIT 1;
        """
        with self._db.connection() as conn:
            row = conn.execute(sql, (job_id,)).fetchone()
            if not row:
                return None
            return ContextSnapshot(
                snapshot_id=row["snapshot_id"],
                job_id=row["job_id"],
                serial_number=row["serial_number"],
                window_tokens=row["window_tokens"],
                window_items_json=json.loads(row["window_items_json"]),
                pruned_items_json=json.loads(row["pruned_items_json"]),
                compression_summary=row["compression_summary"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def list_context_snapshots(self, job_id: str) -> List[ContextSnapshot]:
        """Return all context snapshots for a job ordered by serial."""
        sql = """
        SELECT * FROM context_snapshots WHERE job_id = ?
        ORDER BY serial_number ASC;
        """
        with self._db.connection() as conn:
            rows = conn.execute(sql, (job_id,)).fetchall()
            return [
                ContextSnapshot(
                    snapshot_id=row["snapshot_id"],
                    job_id=row["job_id"],
                    serial_number=row["serial_number"],
                    window_tokens=row["window_tokens"],
                    window_items_json=json.loads(row["window_items_json"]),
                    pruned_items_json=json.loads(row["pruned_items_json"]),
                    compression_summary=row["compression_summary"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    # ------------------------------------------------------------------
    # MEMORY EMBEDDINGS (Step 2)
    # ------------------------------------------------------------------
    def save_memory_embedding(self, emb: MemoryEmbedding) -> None:
        """Upsert a memory embedding record."""
        self._ensure_job_exists(emb.job_id)
        sql = """
        INSERT OR REPLACE INTO memory_embeddings (
            embedding_id, job_id, source_type, source_id, text_content,
            embedding_json, model_name, importance_score, access_count,
            last_accessed_at, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    emb.embedding_id,
                    emb.job_id,
                    emb.source_type,
                    emb.source_id,
                    emb.text_content,
                    json.dumps(emb.embedding),
                    emb.model_name,
                    emb.importance_score,
                    emb.access_count,
                    emb.last_accessed_at.isoformat() if emb.last_accessed_at else None,
                    emb.created_at.isoformat(),
                ),
            )

    def get_memory_embeddings(
        self,
        job_id: str,
        source_type: Optional[str] = None,
        limit: int = 500,
    ) -> List[MemoryEmbedding]:
        """Fetch all (or source-type filtered) embedding records for a job."""
        if source_type:
            sql = """
            SELECT * FROM memory_embeddings
            WHERE job_id = ? AND source_type = ?
            ORDER BY importance_score DESC
            LIMIT ?;
            """
            params: tuple = (job_id, source_type, limit)
        else:
            sql = """
            SELECT * FROM memory_embeddings
            WHERE job_id = ?
            ORDER BY importance_score DESC
            LIMIT ?;
            """
            params = (job_id, limit)

        with self._db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                MemoryEmbedding(
                    embedding_id=row["embedding_id"],
                    job_id=row["job_id"],
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    text_content=row["text_content"],
                    embedding=json.loads(row["embedding_json"]),
                    model_name=row["model_name"],
                    importance_score=row["importance_score"],
                    access_count=row["access_count"],
                    last_accessed_at=(
                        datetime.fromisoformat(row["last_accessed_at"])
                        if row["last_accessed_at"]
                        else None
                    ),
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    def get_embedding_by_source(
        self, job_id: str, source_id: str
    ) -> Optional[MemoryEmbedding]:
        """Retrieve a single embedding record by its source artifact ID."""
        sql = "SELECT * FROM memory_embeddings WHERE job_id = ? AND source_id = ?;"
        with self._db.connection() as conn:
            row = conn.execute(sql, (job_id, source_id)).fetchone()
            if not row:
                return None
            return MemoryEmbedding(
                embedding_id=row["embedding_id"],
                job_id=row["job_id"],
                source_type=row["source_type"],
                source_id=row["source_id"],
                text_content=row["text_content"],
                embedding=json.loads(row["embedding_json"]),
                model_name=row["model_name"],
                importance_score=row["importance_score"],
                access_count=row["access_count"],
                last_accessed_at=(
                    datetime.fromisoformat(row["last_accessed_at"])
                    if row["last_accessed_at"]
                    else None
                ),
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def increment_embedding_access(
        self, embedding_id: str, accessed_at: Optional[datetime] = None
    ) -> None:
        """Increment access counter and update last_accessed_at timestamp."""
        ts = (accessed_at or datetime.now(timezone.utc)).isoformat()
        sql = """
        UPDATE memory_embeddings
        SET access_count = access_count + 1,
            last_accessed_at = ?
        WHERE embedding_id = ?;
        """
        with self._db.connection() as conn:
            conn.execute(sql, (ts, embedding_id))

    # -------------------------------------------------------------
    # STEP 3: SOURCES
    # -------------------------------------------------------------
    def save_source(self, source: SourceMetadata) -> None:
        """Insert or update a learning source metadata record."""
        sql = """
        INSERT OR REPLACE INTO sources (
            source_id, source_type, source_location, title, language,
            normalized_language, permissions_json, acquisition_json,
            processing_status, error_reason, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    source.source_id,
                    source.source_type.value,
                    source.source_location,
                    source.title,
                    source.language,
                    source.normalized_language,
                    json.dumps(source.permissions),
                    json.dumps(source.acquisition),
                    source.processing_status.value,
                    source.error_reason,
                    source.created_at.isoformat(),
                    source.updated_at.isoformat(),
                ),
            )

    def get_source(self, source_id: str) -> Optional[SourceMetadata]:
        """Retrieve source metadata by source ID."""
        sql = "SELECT * FROM sources WHERE source_id = ?;"
        with self._db.connection() as conn:
            row = conn.execute(sql, (source_id,)).fetchone()
            if not row:
                return None
            return SourceMetadata(
                source_id=row["source_id"],
                source_type=SourceType(row["source_type"]),
                source_location=row["source_location"],
                title=row["title"],
                language=row["language"],
                normalized_language=row["normalized_language"],
                permissions=json.loads(row["permissions_json"]),
                acquisition=json.loads(row["acquisition_json"]),
                processing_status=SourceStatus(row["processing_status"]),
                error_reason=row["error_reason"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_sources(
        self, source_type: Optional[SourceType] = None, limit: int = 50
    ) -> List[SourceMetadata]:
        """List all learning sources, optionally filtered by source type."""
        params: List[Any] = []
        sql = "SELECT * FROM sources"
        if source_type:
            sql += " WHERE source_type = ?"
            params.append(source_type.value)
        sql += " ORDER BY created_at DESC LIMIT ?;"
        params.append(limit)

        with self._db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                SourceMetadata(
                    source_id=row["source_id"],
                    source_type=SourceType(row["source_type"]),
                    source_location=row["source_location"],
                    title=row["title"],
                    language=row["language"],
                    normalized_language=row["normalized_language"],
                    permissions=json.loads(row["permissions_json"]),
                    acquisition=json.loads(row["acquisition_json"]),
                    processing_status=SourceStatus(row["processing_status"]),
                    error_reason=row["error_reason"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                for row in rows
            ]

    # -------------------------------------------------------------
    # STEP 3: KNOWLEDGE VERSIONS & AUDIT TRAIL
    # -------------------------------------------------------------
    def save_knowledge_version(self, version: KnowledgeVersion) -> None:
        """Persist a knowledge revision version entry."""
        sql = """
        INSERT OR REPLACE INTO knowledge_versions (
            version_id, item_id, version_number, job_id, title, statement,
            conditions_json, exceptions_json, reasoning, validation_status,
            confidence, evidence_json, correction_notes, corrected_by,
            corrected_at, superseded_by_version, is_active, provenance_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    version.version_id,
                    version.item_id,
                    version.version_number,
                    version.job_id,
                    version.title,
                    version.statement,
                    json.dumps(version.conditions),
                    json.dumps(version.exceptions),
                    version.reasoning,
                    version.validation_status.value,
                    version.confidence,
                    version.evidence_json,
                    version.correction_notes,
                    version.corrected_by,
                    version.corrected_at.isoformat(),
                    version.superseded_by_version,
                    1 if version.is_active else 0,
                    json.dumps(version.provenance),
                ),
            )

    def get_knowledge_versions(self, item_id: str) -> List[KnowledgeVersion]:
        """Fetch all versions of a knowledge item ordered by version number ascending."""
        sql = "SELECT * FROM knowledge_versions WHERE item_id = ? ORDER BY version_number ASC;"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (item_id,)).fetchall()
            return [
                KnowledgeVersion(
                    version_id=row["version_id"],
                    item_id=row["item_id"],
                    version_number=row["version_number"],
                    job_id=row["job_id"],
                    title=row["title"],
                    statement=row["statement"],
                    conditions=json.loads(row["conditions_json"]),
                    exceptions=json.loads(row["exceptions_json"]),
                    reasoning=row["reasoning"],
                    validation_status=KnowledgeValidationStatus(row["validation_status"]),
                    confidence=row["confidence"],
                    evidence_json=row["evidence_json"],
                    correction_notes=row["correction_notes"],
                    corrected_by=row["corrected_by"],
                    corrected_at=datetime.fromisoformat(row["corrected_at"]),
                    superseded_by_version=row["superseded_by_version"],
                    is_active=bool(row["is_active"]),
                    provenance=json.loads(row["provenance_json"]),
                )
                for row in rows
            ]

    def get_active_knowledge_version(self, item_id: str) -> Optional[KnowledgeVersion]:
        """Fetch the currently active revision for a knowledge item."""
        sql = "SELECT * FROM knowledge_versions WHERE item_id = ? AND is_active = 1 ORDER BY version_number DESC LIMIT 1;"
        with self._db.connection() as conn:
            row = conn.execute(sql, (item_id,)).fetchone()
            if not row:
                return None
            return KnowledgeVersion(
                version_id=row["version_id"],
                item_id=row["item_id"],
                version_number=row["version_number"],
                job_id=row["job_id"],
                title=row["title"],
                statement=row["statement"],
                conditions=json.loads(row["conditions_json"]),
                exceptions=json.loads(row["exceptions_json"]),
                reasoning=row["reasoning"],
                validation_status=KnowledgeValidationStatus(row["validation_status"]),
                confidence=row["confidence"],
                evidence_json=row["evidence_json"],
                correction_notes=row["correction_notes"],
                corrected_by=row["corrected_by"],
                corrected_at=datetime.fromisoformat(row["corrected_at"]),
                superseded_by_version=row["superseded_by_version"],
                is_active=bool(row["is_active"]),
                provenance=json.loads(row["provenance_json"]),
            )

    def archive_previous_versions(self, item_id: str, superseded_by_version: int) -> None:
        """Mark all prior versions as inactive and set superseded_by_version."""
        sql = """
        UPDATE knowledge_versions
        SET is_active = 0, superseded_by_version = ?
        WHERE item_id = ? AND is_active = 1;
        """
        with self._db.connection() as conn:
            conn.execute(sql, (superseded_by_version, item_id))

    # -------------------------------------------------------------
    # STEP 3: BACKGROUND JOBS & PROCESS TRACKING
    # -------------------------------------------------------------
    def save_background_job(self, job: BackgroundJobState) -> None:
        """Insert or update a background processing job state record."""
        sql = """
        INSERT OR REPLACE INTO background_jobs (
            job_id, source_id, status, progress, current_stage,
            current_serial, total_units, completed_units, failed_units,
            started_at, updated_at, error, retry_count, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    job.job_id,
                    job.source_id,
                    job.status.value,
                    job.progress,
                    job.current_stage.value,
                    job.current_serial,
                    job.total_units,
                    job.completed_units,
                    job.failed_units,
                    job.started_at.isoformat(),
                    job.updated_at.isoformat(),
                    job.error,
                    job.retry_count,
                    json.dumps(job.metadata),
                ),
            )

    def get_background_job(self, job_id: str) -> Optional[BackgroundJobState]:
        """Retrieve a background job state by job ID."""
        sql = "SELECT * FROM background_jobs WHERE job_id = ?;"
        with self._db.connection() as conn:
            row = conn.execute(sql, (job_id,)).fetchone()
            if not row:
                return None
            return BackgroundJobState(
                job_id=row["job_id"],
                source_id=row["source_id"],
                status=JobStatus(row["status"]),
                progress=row["progress"],
                current_stage=PipelineStage(row["current_stage"]),
                current_serial=row["current_serial"],
                total_units=row["total_units"],
                completed_units=row["completed_units"],
                failed_units=row["failed_units"],
                started_at=datetime.fromisoformat(row["started_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                error=row["error"],
                retry_count=row["retry_count"],
                metadata=json.loads(row["metadata_json"]),
            )

    def list_background_jobs(
        self, status: Optional[JobStatus] = None, limit: int = 50
    ) -> List[BackgroundJobState]:
        """List background jobs optionally filtered by status."""
        params: List[Any] = []
        sql = "SELECT * FROM background_jobs"
        if status:
            sql += " WHERE status = ?"
            params.append(status.value)
        sql += " ORDER BY updated_at DESC LIMIT ?;"
        params.append(limit)

        with self._db.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                BackgroundJobState(
                    job_id=row["job_id"],
                    source_id=row["source_id"],
                    status=JobStatus(row["status"]),
                    progress=row["progress"],
                    current_stage=PipelineStage(row["current_stage"]),
                    current_serial=row["current_serial"],
                    total_units=row["total_units"],
                    completed_units=row["completed_units"],
                    failed_units=row["failed_units"],
                    started_at=datetime.fromisoformat(row["started_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    error=row["error"],
                    retry_count=row["retry_count"],
                    metadata=json.loads(row["metadata_json"]),
                )
                for row in rows
            ]

    # -------------------------------------------------------------
    # STEP 3: USER COMMANDS
    # -------------------------------------------------------------
    def save_user_command(self, cmd: UserCommand) -> None:
        """Persist a parsed user command record."""
        sql = """
        INSERT OR REPLACE INTO user_commands (
            command_id, raw_text, intent, target_knowledge_id,
            parameters_json, source, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    cmd.command_id,
                    cmd.raw_text,
                    cmd.intent.value,
                    cmd.target_knowledge_id,
                    json.dumps(cmd.parameters),
                    cmd.source,
                    cmd.created_at.isoformat(),
                ),
            )

    def list_user_commands(self, limit: int = 50) -> List[UserCommand]:
        """List historical user commands ordered by timestamp descending."""
        sql = "SELECT * FROM user_commands ORDER BY created_at DESC LIMIT ?;"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (limit,)).fetchall()
            return [
                UserCommand(
                    command_id=row["command_id"],
                    raw_text=row["raw_text"],
                    intent=CommandIntent(row["intent"]),
                    target_knowledge_id=row["target_knowledge_id"],
                    parameters=json.loads(row["parameters_json"]),
                    source=row["source"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    # -------------------------------------------------------------
    # STEP 3: NOTES, CONCEPTS & DOUBTS RETRIEVAL
    # -------------------------------------------------------------
    def get_doubts(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve all identified doubts, caveats, contradictions, and uncertain items for a job."""
        doubts: List[Dict[str, Any]] = []

        # 1. Epistemic uncertainty in knowledge items
        knowledge_items = self.get_knowledge(job_id=job_id, only_committed=False)
        for k in knowledge_items:
            if k.validation_status in (
                KnowledgeValidationStatus.UNCERTAIN,
                KnowledgeValidationStatus.CONTRADICTED,
                KnowledgeValidationStatus.NEEDS_REVIEW,
            ):
                doubts.append({
                    "doubt_id": f"doubt_k_{k.item_id}",
                    "source": "knowledge",
                    "item_id": k.item_id,
                    "title": k.title,
                    "statement": k.statement,
                    "status": k.validation_status.value,
                    "reasoning": k.reasoning,
                    "confidence": k.confidence,
                    "timestamps": [k.start_time, k.end_time],
                })

        # 2. Lecture notes uncertainty entries
        notes = self.get_notes(job_id)
        if notes and notes.uncertainty:
            for idx, u in enumerate(notes.uncertainty):
                doubts.append({
                    "doubt_id": f"doubt_notes_{idx}",
                    "source": "notes_uncertainty",
                    "item_id": None,
                    "title": u.title,
                    "statement": u.content,
                    "status": KnowledgeValidationStatus.UNCERTAIN.value,
                    "reasoning": "Reported in lecture notes caveats",
                    "confidence": 0.5,
                    "timestamps": [u.start_time, u.end_time],
                })

        # 3. Transcripts uncertainty
        transcripts = self.get_transcripts(job_id)
        for t in transcripts:
            for u in t.uncertainty:
                doubts.append({
                    "doubt_id": f"doubt_tr_{t.segment_id}_{hash(u) % 10000}",
                    "source": "transcript_uncertainty",
                    "item_id": None,
                    "title": "Transcript Uncertainty",
                    "statement": u,
                    "status": KnowledgeValidationStatus.UNCERTAIN.value,
                    "reasoning": f"Unclear speech or doubt in segment {t.start_time:.1f}s-{t.end_time:.1f}s",
                    "confidence": t.confidence,
                    "timestamps": [t.start_time, t.end_time],
                })

        return doubts

    def get_concepts(self, job_id: str) -> List[Dict[str, Any]]:
        """Retrieve all core concepts, definitions, and terminology identified for a job."""
        concepts: List[Dict[str, Any]] = []

        # 1. Knowledge items of concept / definition / terminology
        knowledge_items = self.get_knowledge(job_id=job_id, only_committed=False)
        for k in knowledge_items:
            if k.item_type in (
                KnowledgeType.CONCEPT,
                KnowledgeType.DEFINITION,
                KnowledgeType.TERMINOLOGY,
            ):
                concepts.append({
                    "concept_id": k.item_id,
                    "item_type": k.item_type.value,
                    "title": k.title,
                    "statement": k.statement,
                    "status": k.validation_status.value,
                    "confidence": k.confidence,
                    "start_time": k.start_time,
                    "end_time": k.end_time,
                    "tags": k.tags,
                })

        # 2. Lecture notes concepts and definitions
        notes = self.get_notes(job_id)
        if notes:
            for idx, c in enumerate(notes.concepts):
                concepts.append({
                    "concept_id": f"notes_concept_{idx}",
                    "item_type": KnowledgeType.CONCEPT.value,
                    "title": c.title,
                    "statement": c.content,
                    "status": KnowledgeValidationStatus.OBSERVED.value,
                    "confidence": 1.0,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "tags": ["notes_concept"],
                })
            for idx, d in enumerate(notes.definitions):
                concepts.append({
                    "concept_id": f"notes_def_{idx}",
                    "item_type": KnowledgeType.DEFINITION.value,
                    "title": d.title,
                    "statement": d.content,
                    "status": KnowledgeValidationStatus.OBSERVED.value,
                    "confidence": 1.0,
                    "start_time": d.start_time,
                    "end_time": d.end_time,
                    "tags": ["notes_definition"],
                })

        return concepts

    # =================================================================
    # STEP 4: Market Intelligence, Multi-Timeframe, Markings & Privacy
    # =================================================================

    def save_market_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Persist or update a market price snapshot."""
        snapshot_id = f"{snapshot.symbol}_{snapshot.timeframe}"
        candles_data = [c.model_dump() for c in snapshot.candles]
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO market_snapshots
                    (snapshot_id, symbol, timeframe, latest_price, candles_json, status, timestamp, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    latest_price=excluded.latest_price,
                    candles_json=excluded.candles_json,
                    status=excluded.status,
                    timestamp=excluded.timestamp,
                    error=excluded.error
                """,
                (
                    snapshot_id,
                    snapshot.symbol,
                    snapshot.timeframe,
                    snapshot.latest_price,
                    json.dumps(candles_data),
                    snapshot.status.value,
                    snapshot.timestamp,
                    snapshot.error,
                ),
            )

    def get_market_snapshot(self, symbol: str, timeframe: str) -> Optional[MarketSnapshot]:
        """Fetch the latest market price snapshot for a symbol and timeframe."""
        snapshot_id = f"{symbol}_{timeframe}"
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM market_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
            if not row:
                return None
            candles_raw = json.loads(row["candles_json"])
            candles = [Candle(**c) for c in candles_raw]
            return MarketSnapshot(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                candles=candles,
                latest_price=row["latest_price"],
                timestamp=row["timestamp"],
                status=MarketDataStatus(row["status"]),
                error=row["error"],
            )

    def get_latest_market_snapshot(self, symbol: str, timeframe: str) -> Optional[MarketSnapshot]:
        """Fetch the latest market price snapshot for a symbol and timeframe."""
        return self.get_market_snapshot(symbol, timeframe)

    def save_market_level(self, level: MarketLevel) -> None:
        """Persist an identified critical price level."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO market_levels
                    (level_id, symbol, timeframe, price, level_type, rule_id, evidence_json, confidence, validity, invalidation_condition, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    level.level_id,
                    level.symbol,
                    level.timeframe,
                    level.price,
                    level.level_type.value,
                    level.rule_id,
                    json.dumps(level.evidence_json),
                    level.confidence,
                    level.validity.value,
                    level.invalidation_condition,
                    level.created_at.isoformat(),
                ),
            )

    def get_market_levels(
        self, symbol: str, timeframe: Optional[str] = None, only_valid: bool = True
    ) -> List[MarketLevel]:
        """Retrieve market levels for a symbol, optionally filtered by timeframe and validity."""
        query = "SELECT * FROM market_levels WHERE symbol = ?"
        params: List[Any] = [symbol]
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if only_valid:
            query += " AND validity = ?"
            params.append(MarkingValidity.VALID.value)
        query += " ORDER BY price DESC"

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            levels = []
            for r in rows:
                levels.append(
                    MarketLevel(
                        level_id=r["level_id"],
                        symbol=r["symbol"],
                        timeframe=r["timeframe"],
                        price=r["price"],
                        level_type=LevelType(r["level_type"]),
                        rule_id=r["rule_id"],
                        evidence_json=json.loads(r["evidence_json"]),
                        confidence=r["confidence"],
                        validity=MarkingValidity(r["validity"]),
                        invalidation_condition=r["invalidation_condition"],
                        created_at=datetime.fromisoformat(r["created_at"]),
                    )
                )
            return levels

    def save_chart_marking(self, marking: ChartMarking) -> None:
        """Persist or update a visual chart marking with canonical metadata."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO chart_markings
                    (marking_id, symbol, timeframe, marking_type, coordinates_json, rule_id, evidence_json,
                     validity, confidence, invalidation_condition, priority, concept, price, price_range_json,
                     direction, source, rule_reference, rule_version, status, visibility, created_by, broker_id,
                     market_context_json, backtest_run_id, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    marking.marking_id,
                    marking.symbol,
                    marking.timeframe,
                    marking.marking_type.value,
                    json.dumps(marking.coordinates_json),
                    marking.rule_id,
                    json.dumps(marking.evidence_json),
                    marking.validity.value,
                    marking.confidence,
                    marking.invalidation_condition,
                    marking.priority,
                    marking.concept,
                    marking.price,
                    json.dumps(marking.price_range or {}),
                    marking.direction,
                    marking.source,
                    marking.rule_reference,
                    marking.rule_version,
                    marking.status,
                    marking.visibility,
                    marking.created_by,
                    marking.broker_id,
                    json.dumps(marking.market_context or {}),
                    marking.backtest_run_id,
                    marking.timestamp,
                    marking.created_at.isoformat(),
                ),
            )

    def get_chart_markings(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        only_valid: bool = True,
        source: Optional[str] = None,
        broker_id: Optional[str] = None,
        backtest_run_id: Optional[str] = None,
    ) -> List[ChartMarking]:
        """Retrieve visual markings for a symbol, respecting clean-chart filtering and context."""
        query = "SELECT * FROM chart_markings WHERE symbol = ?"
        params: List[Any] = [symbol]
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if only_valid:
            query += " AND validity = ?"
            params.append(MarkingValidity.VALID.value)
        if source:
            query += " AND source = ?"
            params.append(source)
        if broker_id:
            query += " AND broker_id = ?"
            params.append(broker_id)
        if backtest_run_id:
            query += " AND backtest_run_id = ?"
            params.append(backtest_run_id)
        query += " ORDER BY priority DESC, created_at DESC"

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            markings = []
            for r in rows:
                keys = r.keys() if hasattr(r, "keys") else []
                price_range = json.loads(r["price_range_json"]) if "price_range_json" in keys and r["price_range_json"] else None
                market_context = json.loads(r["market_context_json"]) if "market_context_json" in keys and r["market_context_json"] else {}
                ts = r["timestamp"] if "timestamp" in keys and r["timestamp"] is not None else datetime.fromisoformat(r["created_at"]).timestamp()
                markings.append(
                    ChartMarking(
                        marking_id=r["marking_id"],
                        symbol=r["symbol"],
                        timeframe=r["timeframe"],
                        marking_type=MarkingType(r["marking_type"]),
                        coordinates_json=json.loads(r["coordinates_json"]),
                        rule_id=r["rule_id"],
                        evidence_json=json.loads(r["evidence_json"]),
                        validity=MarkingValidity(r["validity"]),
                        confidence=r["confidence"],
                        invalidation_condition=r["invalidation_condition"],
                        priority=r["priority"],
                        concept=r["concept"] if "concept" in keys else None,
                        price=r["price"] if "price" in keys else None,
                        price_range=price_range,
                        direction=r["direction"] if "direction" in keys else None,
                        source=r["source"] if "source" in keys and r["source"] else "SUPERVISOR",
                        rule_reference=r["rule_reference"] if "rule_reference" in keys else None,
                        rule_version=r["rule_version"] if "rule_version" in keys and r["rule_version"] else "1.0",
                        status=r["status"] if "status" in keys and r["status"] else "VALIDATED",
                        visibility=r["visibility"] if "visibility" in keys and r["visibility"] else "VISIBLE",
                        created_by=r["created_by"] if "created_by" in keys and r["created_by"] else "MARKET_SUPERVISOR",
                        broker_id=r["broker_id"] if "broker_id" in keys else None,
                        market_context=market_context,
                        backtest_run_id=r["backtest_run_id"] if "backtest_run_id" in keys else None,
                        timestamp=ts,
                        created_at=datetime.fromisoformat(r["created_at"]),
                    )
                )
            return markings

    def get_chart_markings_for_backtest(self, backtest_id: str) -> List[ChartMarking]:
        """Retrieve canonical visual markings for a specific historical backtest run."""
        query = "SELECT * FROM chart_markings WHERE backtest_run_id = ? ORDER BY timestamp ASC, created_at ASC"
        with self._db.connection() as conn:
            rows = conn.execute(query, (backtest_id,)).fetchall()
            markings = []
            for r in rows:
                keys = r.keys() if hasattr(r, "keys") else []
                price_range = json.loads(r["price_range_json"]) if "price_range_json" in keys and r["price_range_json"] else None
                market_context = json.loads(r["market_context_json"]) if "market_context_json" in keys and r["market_context_json"] else {}
                ts = r["timestamp"] if "timestamp" in keys and r["timestamp"] is not None else datetime.fromisoformat(r["created_at"]).timestamp()
                markings.append(
                    ChartMarking(
                        marking_id=r["marking_id"],
                        symbol=r["symbol"],
                        timeframe=r["timeframe"],
                        marking_type=MarkingType(r["marking_type"]),
                        coordinates_json=json.loads(r["coordinates_json"]),
                        rule_id=r["rule_id"],
                        evidence_json=json.loads(r["evidence_json"]),
                        validity=MarkingValidity(r["validity"]),
                        confidence=r["confidence"],
                        invalidation_condition=r["invalidation_condition"],
                        priority=r["priority"],
                        concept=r["concept"] if "concept" in keys else None,
                        price=r["price"] if "price" in keys else None,
                        price_range=price_range,
                        direction=r["direction"] if "direction" in keys else None,
                        source=r["source"] if "source" in keys and r["source"] else "BACKTEST",
                        rule_reference=r["rule_reference"] if "rule_reference" in keys else None,
                        rule_version=r["rule_version"] if "rule_version" in keys and r["rule_version"] else "1.0",
                        status=r["status"] if "status" in keys and r["status"] else "VALIDATED",
                        visibility=r["visibility"] if "visibility" in keys and r["visibility"] else "VISIBLE",
                        created_by=r["created_by"] if "created_by" in keys and r["created_by"] else "BACKTEST",
                        broker_id=r["broker_id"] if "broker_id" in keys else None,
                        market_context=market_context,
                        backtest_run_id=r["backtest_run_id"] if "backtest_run_id" in keys else None,
                        timestamp=ts,
                        created_at=datetime.fromisoformat(r["created_at"]),
                    )
                )
            return markings

    def update_chart_marking_validity(self, marking_id: str, validity: MarkingValidity) -> None:
        """Update validity status (e.g. INVALIDATED, EXPIRED, SUPPRESSED) for a chart marking."""
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE chart_markings SET validity = ? WHERE marking_id = ?",
                (validity.value, marking_id),
            )

    def save_market_setup(self, setup: MarketSetup) -> None:
        """Persist a lecture-validated market setup with lifecycle state."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO market_setups
                    (setup_id, symbol, timeframe, setup_type, direction, status, applicable_rule_id,
                     conditions_satisfied_json, exceptions_checked_json, invalidation_checked_json,
                     supporting_evidence_json, contradicting_evidence_json, confidence, created_at,
                     detection_timestamp, confirmation_timestamp, invalidation_timestamp,
                     invalidation_condition, confirmation_evidence_json, previous_status,
                     status_transition_reason, is_displayed, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    setup.setup_id,
                    setup.symbol,
                    setup.timeframe,
                    setup.setup_type,
                    setup.direction.value,
                    setup.status.value,
                    setup.applicable_rule_id,
                    json.dumps(setup.conditions_satisfied),
                    json.dumps(setup.exceptions_checked),
                    json.dumps(setup.invalidation_checked),
                    json.dumps(setup.supporting_evidence),
                    json.dumps(setup.contradicting_evidence),
                    setup.confidence,
                    setup.created_at.isoformat(),
                    setup.detection_timestamp,
                    setup.confirmation_timestamp,
                    setup.invalidation_timestamp,
                    setup.invalidation_condition,
                    json.dumps(setup.confirmation_evidence),
                    setup.previous_status.value if setup.previous_status else None,
                    setup.status_transition_reason,
                    1 if setup.is_displayed else 0,
                    setup.updated_at.isoformat(),
                ),
            )

    def get_market_setups(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        status: Optional[SetupStatus] = None,
    ) -> List[MarketSetup]:
        """Fetch market setups for a symbol, optionally filtered by timeframe or validation status."""
        query = "SELECT * FROM market_setups WHERE symbol = ?"
        params: List[Any] = [symbol]
        if timeframe:
            query += " AND timeframe = ?"
            params.append(timeframe)
        if status:
            query += " AND status = ?"
            params.append(status.value)
        query += " ORDER BY created_at DESC"

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            setups = []
            for r in rows:
                c_ev = []
                if "confirmation_evidence_json" in r.keys() and r["confirmation_evidence_json"]:
                    try:
                        c_ev = json.loads(r["confirmation_evidence_json"])
                    except Exception:
                        c_ev = []

                prev_stat = None
                if "previous_status" in r.keys() and r["previous_status"]:
                    try:
                        prev_stat = SetupStatus(r["previous_status"])
                    except ValueError:
                        prev_stat = None

                setups.append(
                    MarketSetup(
                        setup_id=r["setup_id"],
                        symbol=r["symbol"],
                        timeframe=r["timeframe"],
                        setup_type=r["setup_type"],
                        direction=MarketDirection(r["direction"]),
                        status=SetupStatus(r["status"]),
                        applicable_rule_id=r["applicable_rule_id"],
                        conditions_satisfied=json.loads(r["conditions_satisfied_json"]),
                        exceptions_checked=json.loads(r["exceptions_checked_json"]),
                        invalidation_checked=json.loads(r["invalidation_checked_json"]),
                        supporting_evidence=json.loads(r["supporting_evidence_json"]),
                        contradicting_evidence=json.loads(r["contradicting_evidence_json"]),
                        confidence=r["confidence"],
                        detection_timestamp=r["detection_timestamp"] if "detection_timestamp" in r.keys() and r["detection_timestamp"] is not None else 0.0,
                        confirmation_timestamp=r["confirmation_timestamp"] if "confirmation_timestamp" in r.keys() else None,
                        invalidation_timestamp=r["invalidation_timestamp"] if "invalidation_timestamp" in r.keys() else None,
                        invalidation_condition=r["invalidation_condition"] if "invalidation_condition" in r.keys() else None,
                        confirmation_evidence=c_ev,
                        previous_status=prev_stat,
                        status_transition_reason=r["status_transition_reason"] if "status_transition_reason" in r.keys() else None,
                        is_displayed=bool(r["is_displayed"]) if "is_displayed" in r.keys() and r["is_displayed"] is not None else False,
                        created_at=datetime.fromisoformat(r["created_at"]),
                        updated_at=datetime.fromisoformat(r["updated_at"]) if "updated_at" in r.keys() and r["updated_at"] else datetime.fromisoformat(r["created_at"]),
                    )
                )
            return setups

    def save_market_analysis(self, analysis: UnifiedMarketAnalysis) -> None:
        """Persist a unified multi-timeframe market analysis record."""
        htf_str = json.dumps(analysis.htf_analysis.model_dump(mode="json")) if analysis.htf_analysis else None
        mtf_str = json.dumps(analysis.mtf_analysis.model_dump(mode="json")) if analysis.mtf_analysis else None
        ltf_str = json.dumps(analysis.ltf_analysis.model_dump(mode="json")) if analysis.ltf_analysis else None
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO market_analysis
                    (analysis_id, symbol, htf_json, mtf_json, ltf_json, overall_bias, conflict_notes_json, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis.analysis_id,
                    analysis.symbol,
                    htf_str,
                    mtf_str,
                    ltf_str,
                    analysis.overall_bias.value,
                    json.dumps(analysis.conflict_notes),
                    analysis.timestamp,
                    analysis.created_at.isoformat(),
                ),
            )

    def get_latest_market_analysis(self, symbol: str) -> Optional[UnifiedMarketAnalysis]:
        """Fetch the most recent multi-timeframe analysis for a symbol."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM market_analysis WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if not row:
                return None
            htf = TimeframeAnalysis(**json.loads(row["htf_json"])) if row["htf_json"] else None
            mtf = TimeframeAnalysis(**json.loads(row["mtf_json"])) if row["mtf_json"] else None
            ltf = TimeframeAnalysis(**json.loads(row["ltf_json"])) if row["ltf_json"] else None

            # Load active setups and markings
            setups = self.get_market_setups(symbol=symbol)
            markings = self.get_chart_markings(symbol=symbol, only_valid=True)

            return UnifiedMarketAnalysis(
                analysis_id=row["analysis_id"],
                symbol=row["symbol"],
                htf_analysis=htf,
                mtf_analysis=mtf,
                ltf_analysis=ltf,
                overall_bias=MarketBias(row["overall_bias"]),
                active_setups=setups,
                active_markings=markings,
                conflict_notes=json.loads(row["conflict_notes_json"]),
                timestamp=row["timestamp"],
                created_at=datetime.fromisoformat(r_time) if (r_time := row["created_at"]) else datetime.now(timezone.utc),
            )

    def save_watchlist_monitor_state(self, state: WatchlistMonitorState) -> None:
        """Persist or update state for a symbol in the background watchlist queue."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO watchlist_monitor_state
                    (symbol, timeframe, last_processed_timestamp, latest_status, worker_state, retry_count, last_error, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    timeframe=excluded.timeframe,
                    last_processed_timestamp=excluded.last_processed_timestamp,
                    latest_status=excluded.latest_status,
                    worker_state=excluded.worker_state,
                    retry_count=excluded.retry_count,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at
                """,
                (
                    state.symbol,
                    state.timeframe,
                    state.last_processed_timestamp,
                    state.latest_status.value,
                    state.worker_state,
                    state.retry_count,
                    state.last_error,
                    state.updated_at.isoformat(),
                ),
            )

    def get_watchlist_monitor_state(self, symbol: str) -> Optional[WatchlistMonitorState]:
        """Fetch monitor state for a specific watchlist symbol."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM watchlist_monitor_state WHERE symbol = ?",
                (symbol,),
            ).fetchone()
            if not row:
                return None
            return WatchlistMonitorState(
                symbol=row["symbol"],
                timeframe=row["timeframe"],
                last_processed_timestamp=row["last_processed_timestamp"],
                latest_status=WatchlistMonitorStatus(row["latest_status"]),
                worker_state=row["worker_state"],
                retry_count=row["retry_count"],
                last_error=row["last_error"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_watchlist_monitor_states(self) -> List[WatchlistMonitorState]:
        """List all tracked symbols and their background monitor states."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM watchlist_monitor_state ORDER BY updated_at DESC"
            ).fetchall()
            states = []
            for r in rows:
                states.append(
                    WatchlistMonitorState(
                        symbol=r["symbol"],
                        timeframe=r["timeframe"],
                        last_processed_timestamp=r["last_processed_timestamp"],
                        latest_status=WatchlistMonitorStatus(r["latest_status"]),
                        worker_state=r["worker_state"],
                        retry_count=r["retry_count"],
                        last_error=r["last_error"],
                        updated_at=datetime.fromisoformat(r["updated_at"]),
                    )
                )
            return states

    def save_screen_permission(self, perm: ScreenPermission) -> None:
        """Persist or update an explicit screen access permission."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO screen_permissions
                    (permission_id, app_name, window_title, region_box_json, status, requested_at, granted_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    perm.permission_id,
                    perm.app_name,
                    perm.window_title,
                    json.dumps(perm.region_box) if perm.region_box else None,
                    perm.status.value,
                    perm.requested_at.isoformat(),
                    perm.granted_at.isoformat() if perm.granted_at else None,
                    perm.revoked_at.isoformat() if perm.revoked_at else None,
                ),
            )

    def get_screen_permission(self, permission_id: str) -> Optional[ScreenPermission]:
        """Fetch screen permission by ID."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM screen_permissions WHERE permission_id = ?",
                (permission_id,),
            ).fetchone()
            if not row:
                return None
            return ScreenPermission(
                permission_id=row["permission_id"],
                app_name=row["app_name"],
                window_title=row["window_title"],
                region_box=json.loads(row["region_box_json"]) if row["region_box_json"] else None,
                status=ScreenAccessStatus(row["status"]),
                requested_at=datetime.fromisoformat(row["requested_at"]),
                granted_at=datetime.fromisoformat(row["granted_at"]) if row["granted_at"] else None,
                revoked_at=datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None,
            )

    def get_active_screen_permission(self, app_name: Optional[str] = None) -> Optional[ScreenPermission]:
        """Fetch currently granted screen access permission, if any."""
        query = "SELECT * FROM screen_permissions WHERE status = ?"
        params: List[Any] = [ScreenAccessStatus.SCREEN_ACCESS_GRANTED.value]
        if app_name:
            query += " AND app_name = ?"
            params.append(app_name)
        query += " ORDER BY granted_at DESC LIMIT 1"

        with self._db.connection() as conn:
            row = conn.execute(query, tuple(params)).fetchone()
            if not row:
                return None
            return ScreenPermission(
                permission_id=row["permission_id"],
                app_name=row["app_name"],
                window_title=row["window_title"],
                region_box=json.loads(row["region_box_json"]) if row["region_box_json"] else None,
                status=ScreenAccessStatus(row["status"]),
                requested_at=datetime.fromisoformat(row["requested_at"]),
                granted_at=datetime.fromisoformat(row["granted_at"]) if row["granted_at"] else None,
                revoked_at=datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None,
            )

    def list_screen_permissions(self) -> List[ScreenPermission]:
        """Fetch all screen access permission records."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM screen_permissions ORDER BY requested_at DESC").fetchall()
            return [
                ScreenPermission(
                    permission_id=row["permission_id"],
                    app_name=row["app_name"],
                    window_title=row["window_title"],
                    region_box=json.loads(row["region_box_json"]) if row["region_box_json"] else None,
                    status=ScreenAccessStatus(row["status"]),
                    requested_at=datetime.fromisoformat(row["requested_at"]),
                    granted_at=datetime.fromisoformat(row["granted_at"]) if row["granted_at"] else None,
                    revoked_at=datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None,
                )
                for row in rows
            ]

    # -------------------------------------------------------------
    # STEP 5: CURRENT SITUATION SNAPSHOTS
    # -------------------------------------------------------------
    def save_current_situation(self, snapshot: CurrentSituationSnapshot) -> None:
        """Persist a current decision-context situation snapshot including fundamental context."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO current_situation_snapshots
                    (snapshot_id, symbol, current_price, overall_bias, bias_summary,
                     nearest_major_liquidity_json, current_poi_json, active_confirmed_setup_json,
                     setup_confirmation_state, invalidation_level, confirmation_level,
                     timeframe_conflicts_json, recent_changes_json, visible_markings_json,
                     background_tracking_count, timestamp, created_at,
                     fundamental_context, relevant_news_json, next_high_impact_event_json,
                     news_time_distance_minutes, news_risk, technical_fundamental_relationship,
                     technical_fundamental_conflicts_json, recommended_action_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.symbol,
                    snapshot.current_price,
                    snapshot.overall_bias.value,
                    snapshot.bias_summary,
                    json.dumps(snapshot.nearest_major_liquidity.model_dump(mode="json")) if snapshot.nearest_major_liquidity else None,
                    json.dumps(snapshot.current_poi.model_dump(mode="json")) if snapshot.current_poi else None,
                    json.dumps(snapshot.active_confirmed_setup.model_dump(mode="json")) if snapshot.active_confirmed_setup else None,
                    snapshot.setup_confirmation_state,
                    snapshot.invalidation_level,
                    snapshot.confirmation_level,
                    json.dumps(snapshot.timeframe_conflicts),
                    json.dumps(snapshot.recent_changes),
                    json.dumps([m.model_dump(mode="json") for m in snapshot.visible_markings]),
                    snapshot.background_tracking_count,
                    snapshot.timestamp,
                    snapshot.created_at.isoformat(),
                    snapshot.fundamental_context,
                    json.dumps([e.model_dump(mode="json") for e in snapshot.relevant_news]),
                    json.dumps(snapshot.next_high_impact_event.model_dump(mode="json")) if snapshot.next_high_impact_event else None,
                    snapshot.news_time_distance_minutes,
                    snapshot.news_risk,
                    snapshot.technical_fundamental_relationship,
                    json.dumps(snapshot.technical_fundamental_conflicts),
                    snapshot.recommended_action_context,
                ),
            )

    def get_latest_current_situation(self, symbol: str) -> Optional[CurrentSituationSnapshot]:
        """Retrieve the most recent current situation snapshot for a symbol."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM current_situation_snapshots WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if not row:
                return None

            liq = MarketLevel(**json.loads(row["nearest_major_liquidity_json"])) if row["nearest_major_liquidity_json"] else None
            poi = MarketLevel(**json.loads(row["current_poi_json"])) if row["current_poi_json"] else None
            setup = MarketSetup(**json.loads(row["active_confirmed_setup_json"])) if row["active_confirmed_setup_json"] else None
            markings = [ChartMarking(**m) for m in json.loads(row["visible_markings_json"])] if row["visible_markings_json"] else []

            # Step 6 fundamental fields with backwards compatibility
            fund_ctx = row["fundamental_context"] if "fundamental_context" in row.keys() else None
            rel_news = (
                [FundamentalEvent(**e) for e in json.loads(row["relevant_news_json"])]
                if "relevant_news_json" in row.keys() and row["relevant_news_json"]
                else []
            )
            next_event = (
                FundamentalEvent(**json.loads(row["next_high_impact_event_json"]))
                if "next_high_impact_event_json" in row.keys() and row["next_high_impact_event_json"]
                else None
            )
            time_dist = row["news_time_distance_minutes"] if "news_time_distance_minutes" in row.keys() else None
            news_risk = row["news_risk"] if "news_risk" in row.keys() else "NO_RISK"
            rel = row["technical_fundamental_relationship"] if "technical_fundamental_relationship" in row.keys() else "TECHNICALLY_VALID"
            conflicts = (
                json.loads(row["technical_fundamental_conflicts_json"])
                if "technical_fundamental_conflicts_json" in row.keys() and row["technical_fundamental_conflicts_json"]
                else []
            )
            action_ctx = row["recommended_action_context"] if "recommended_action_context" in row.keys() else "PROCEED_WITH_STANDARD_CONFIRMATION"

            return CurrentSituationSnapshot(
                snapshot_id=row["snapshot_id"],
                symbol=row["symbol"],
                current_price=row["current_price"],
                overall_bias=MarketBias(row["overall_bias"]),
                bias_summary=row["bias_summary"],
                nearest_major_liquidity=liq,
                current_poi=poi,
                active_confirmed_setup=setup,
                setup_confirmation_state=row["setup_confirmation_state"],
                invalidation_level=row["invalidation_level"],
                confirmation_level=row["confirmation_level"],
                timeframe_conflicts=json.loads(row["timeframe_conflicts_json"]),
                recent_changes=json.loads(row["recent_changes_json"]),
                visible_markings=markings,
                background_tracking_count=row["background_tracking_count"],
                fundamental_context=fund_ctx,
                relevant_news=rel_news,
                next_high_impact_event=next_event,
                news_time_distance_minutes=time_dist,
                news_risk=news_risk,
                technical_fundamental_relationship=rel,
                technical_fundamental_conflicts=conflicts,
                recommended_action_context=action_ctx,
                timestamp=row["timestamp"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    # -------------------------------------------------------------
    # STEP 6 FUNDAMENTAL DATA & CORRELATION PERSISTENCE
    # -------------------------------------------------------------
    def save_fundamental_event(self, event: FundamentalEvent) -> None:
        """Persist a single normalized fundamental news event."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fundamental_events
                    (event_id, source, title, currency, event_time, impact,
                     previous, forecast, actual, status, retrieved_at, source_url,
                     confidence, evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.source,
                    event.title,
                    event.currency.upper(),
                    event.event_time.isoformat(),
                    event.impact.value,
                    event.previous,
                    event.forecast,
                    event.actual,
                    event.status.value,
                    event.retrieved_at.isoformat(),
                    event.source_url,
                    event.confidence,
                    json.dumps(event.evidence),
                ),
            )

    def save_fundamental_events(self, events: List[FundamentalEvent]) -> None:
        """Batch persist multiple normalized economic events."""
        with self._db.connection() as conn:
            for event in events:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO fundamental_events
                        (event_id, source, title, currency, event_time, impact,
                         previous, forecast, actual, status, retrieved_at, source_url,
                         confidence, evidence_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.source,
                        event.title,
                        event.currency.upper(),
                        event.event_time.isoformat(),
                        event.impact.value,
                        event.previous,
                        event.forecast,
                        event.actual,
                        event.status.value,
                        event.retrieved_at.isoformat(),
                        event.source_url,
                        event.confidence,
                        json.dumps(event.evidence),
                    ),
                )

    def get_fundamental_events(
        self,
        currency: Optional[str] = None,
        impact: Optional[EventImpact] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[FundamentalEvent]:
        """Query stored fundamental events with optional filters."""
        query = "SELECT * FROM fundamental_events WHERE 1=1"
        params: List[Any] = []

        if currency:
            query += " AND currency = ?"
            params.append(currency.upper())

        if impact:
            query += " AND impact = ?"
            params.append(impact.value)

        if start_time:
            query += " AND event_time >= ?"
            params.append(start_time.isoformat())

        if end_time:
            query += " AND event_time <= ?"
            params.append(end_time.isoformat())

        query += " ORDER BY event_time ASC LIMIT ?"
        params.append(limit)

        with self._db.connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                FundamentalEvent(
                    event_id=r["event_id"],
                    source=r["source"],
                    title=r["title"],
                    currency=r["currency"],
                    event_time=datetime.fromisoformat(r["event_time"]),
                    impact=EventImpact(r["impact"]),
                    previous=r["previous"],
                    forecast=r["forecast"],
                    actual=r["actual"],
                    status=EventStatus(r["status"]),
                    retrieved_at=datetime.fromisoformat(r["retrieved_at"]),
                    source_url=r["source_url"],
                    confidence=r["confidence"],
                    evidence=json.loads(r["evidence_json"]),
                )
                for r in rows
            ]

    def save_provider_state(self, state: FundamentalProviderState) -> None:
        """Persist fundamental provider health, errors, and staleness."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fundamental_provider_state
                    (provider_name, status, last_successful_fetch, stale_after_seconds,
                     is_stale, error_message, unavailable_reason, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.provider_name,
                    state.status.value,
                    state.last_successful_fetch.isoformat() if state.last_successful_fetch else None,
                    state.stale_after_seconds,
                    1 if state.is_stale else 0,
                    state.error_message,
                    state.unavailable_reason,
                    state.updated_at.isoformat(),
                ),
            )

    def get_provider_state(self, provider_name: str = "forex_factory") -> Optional[FundamentalProviderState]:
        """Retrieve stored status and health for a provider."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM fundamental_provider_state WHERE provider_name = ?",
                (provider_name,),
            ).fetchone()
            if not row:
                return None
            return FundamentalProviderState(
                provider_name=row["provider_name"],
                status=ProviderStatus(row["status"]),
                last_successful_fetch=datetime.fromisoformat(row["last_successful_fetch"]) if row["last_successful_fetch"] else None,
                stale_after_seconds=row["stale_after_seconds"],
                is_stale=bool(row["is_stale"]),
                error_message=row["error_message"],
                unavailable_reason=row["unavailable_reason"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def save_tech_fund_analysis(self, analysis: TechnicalFundamentalAnalysis) -> None:
        """Persist synthesized technical-fundamental analysis."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tech_fund_correlations
                    (analysis_id, symbol, relationship, technical_bias, news_risk,
                     timing_context_json, conflict_json, action_context, explanation,
                     evidence_json, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    analysis.analysis_id,
                    analysis.symbol,
                    analysis.relationship.value,
                    analysis.technical_bias,
                    analysis.news_risk.value,
                    json.dumps(analysis.timing_context.model_dump(mode="json")) if analysis.timing_context else "{}",
                    json.dumps(analysis.conflict.model_dump(mode="json")),
                    analysis.action_context,
                    analysis.explanation,
                    json.dumps(analysis.evidence),
                    analysis.timestamp,
                    analysis.created_at.isoformat(),
                ),
            )

    def get_latest_tech_fund_analysis(self, symbol: str) -> Optional[TechnicalFundamentalAnalysis]:
        """Retrieve the most recent synthesized correlation analysis for a symbol."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM tech_fund_correlations WHERE symbol = ? ORDER BY timestamp DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if not row:
                return None

            timing_raw = json.loads(row["timing_context_json"])
            timing = NewsTimingContext(**timing_raw) if timing_raw else None
            conflict = TechFundConflict(**json.loads(row["conflict_json"]))

            return TechnicalFundamentalAnalysis(
                analysis_id=row["analysis_id"],
                symbol=row["symbol"],
                relationship=TechFundRelationship(row["relationship"]),
                technical_bias=row["technical_bias"],
                news_risk=NewsRiskLevel(row["news_risk"]),
                timing_context=timing,
                conflict=conflict,
                action_context=row["action_context"],
                explanation=row["explanation"],
                evidence=json.loads(row["evidence_json"]),
                timestamp=row["timestamp"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    # =========================================================================
    # Step 7 — AI Trading Companion Persistence Methods
    # =========================================================================

    def save_alert_event(self, alert: AlertEvent) -> None:
        """Persist or update a proactive alert event."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alert_events
                    (alert_id, symbol, alert_type, priority, lifecycle_state, decision,
                     quality, title, message, evidence_json, is_acknowledged,
                     acknowledged_at, dedup_key, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.alert_id,
                    alert.symbol,
                    alert.alert_type.value,
                    alert.priority.value,
                    alert.lifecycle_state.value,
                    alert.decision.value,
                    alert.quality.value,
                    alert.title,
                    alert.message,
                    json.dumps(alert.evidence),
                    1 if alert.is_acknowledged else 0,
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    alert.dedup_key,
                    alert.created_at.isoformat(),
                    alert.updated_at.isoformat(),
                ),
            )

    def get_alert_event(self, alert_id: str) -> Optional[AlertEvent]:
        """Fetch alert event by its unique ID."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM alert_events WHERE alert_id = ?",
                (alert_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_alert_event(row)

    def get_alerts(
        self,
        symbol: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> List[AlertEvent]:
        """Retrieve recent alerts, optionally filtered by symbol and acknowledgment status."""
        with self._db.connection() as conn:
            query = "SELECT * FROM alert_events WHERE 1=1"
            params: List[Any] = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if unacknowledged_only:
                query += " AND is_acknowledged = 0"
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_alert_event(r) for r in rows]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged by the user."""
        with self._db.connection() as conn:
            now_iso = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                UPDATE alert_events
                SET is_acknowledged = 1, acknowledged_at = ?, updated_at = ?
                WHERE alert_id = ?
                """,
                (now_iso, now_iso, alert_id),
            )
            return cursor.rowcount > 0

    def _row_to_alert_event(self, row: Any) -> AlertEvent:
        return AlertEvent(
            alert_id=row["alert_id"],
            symbol=row["symbol"],
            alert_type=AlertType(row["alert_type"]),
            priority=AlertPriority(row["priority"]),
            lifecycle_state=AlertLifecycleState(row["lifecycle_state"]),
            decision=AlertDecision(row["decision"]),
            quality=AlertQuality(row["quality"]),
            title=row["title"],
            message=row["message"],
            evidence=json.loads(row["evidence_json"]),
            is_acknowledged=bool(row["is_acknowledged"]),
            acknowledged_at=datetime.fromisoformat(row["acknowledged_at"]) if row["acknowledged_at"] else None,
            dedup_key=row["dedup_key"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def save_alert_state(self, state: AlertStateRecord) -> None:
        """Persist deduplication state for an alert condition."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO alert_state
                    (dedup_key, symbol, last_alerted_at, last_lifecycle_state,
                     alert_count, suppression_reason, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.dedup_key,
                    state.symbol,
                    state.last_alerted_at,
                    state.last_lifecycle_state.value,
                    state.alert_count,
                    state.suppression_reason,
                    state.updated_at.isoformat(),
                ),
            )

    def get_alert_state(self, dedup_key: str) -> Optional[AlertStateRecord]:
        """Fetch alert deduplication record by dedup key."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM alert_state WHERE dedup_key = ?",
                (dedup_key,),
            ).fetchone()
            if not row:
                return None
            return AlertStateRecord(
                dedup_key=row["dedup_key"],
                symbol=row["symbol"],
                last_alerted_at=row["last_alerted_at"],
                last_lifecycle_state=AlertLifecycleState(row["last_lifecycle_state"]),
                alert_count=row["alert_count"],
                suppression_reason=row["suppression_reason"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def save_discipline_state(self, state: DisciplineState) -> None:
        """Persist user trading discipline parameters, statistics, and violations."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO discipline_state
                    (user_id, config_json, trades_today_count, daily_realized_loss,
                     last_trade_timestamp, cooldown_active_until,
                     recent_trades_timestamps_json, active_violations_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.user_id,
                    json.dumps(state.config.model_dump(mode="json")),
                    state.trades_today_count,
                    state.daily_realized_loss,
                    state.last_trade_timestamp,
                    state.cooldown_active_until.isoformat() if state.cooldown_active_until else None,
                    json.dumps(state.recent_trades_timestamps),
                    json.dumps(state.active_violations),
                    state.updated_at.isoformat(),
                ),
            )

    def get_discipline_state(self, user_id: str = "default_user") -> Optional[DisciplineState]:
        """Fetch discipline metrics and configuration for a user."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM discipline_state WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return None
            config_dict = json.loads(row["config_json"])
            config = DisciplineConfig(**config_dict) if config_dict else DisciplineConfig(user_id=user_id)
            return DisciplineState(
                user_id=row["user_id"],
                config=config,
                trades_today_count=row["trades_today_count"],
                daily_realized_loss=row["daily_realized_loss"],
                last_trade_timestamp=row["last_trade_timestamp"],
                cooldown_active_until=datetime.fromisoformat(row["cooldown_active_until"]) if row["cooldown_active_until"] else None,
                recent_trades_timestamps=json.loads(row["recent_trades_timestamps_json"]),
                active_violations=json.loads(row["active_violations_json"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def save_scenario_analysis(self, scenario: ScenarioCondition) -> None:
        """Persist conditional IF-THEN-UNLESS scenario analysis."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scenario_analyses
                    (scenario_id, symbol, level, direction, if_condition,
                     then_outcome, unless_condition, invalidated_when,
                     next_poi_or_liquidity, confidence_quality, evidence_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario.scenario_id,
                    scenario.symbol,
                    scenario.level,
                    scenario.direction,
                    scenario.if_condition,
                    scenario.then_outcome,
                    scenario.unless_condition,
                    scenario.invalidated_when,
                    scenario.next_poi_or_liquidity,
                    scenario.confidence_quality.value,
                    json.dumps(scenario.evidence),
                    scenario.created_at.isoformat(),
                ),
            )

    def get_scenarios(self, symbol: str, limit: int = 10) -> List[ScenarioCondition]:
        """Fetch recent scenario analyses for a symbol."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scenario_analyses WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
            return [
                ScenarioCondition(
                    scenario_id=r["scenario_id"],
                    symbol=r["symbol"],
                    level=r["level"],
                    direction=r["direction"],
                    if_condition=r["if_condition"],
                    then_outcome=r["then_outcome"],
                    unless_condition=r["unless_condition"],
                    invalidated_when=r["invalidated_when"],
                    next_poi_or_liquidity=r["next_poi_or_liquidity"],
                    confidence_quality=AlertQuality(r["confidence_quality"]),
                    evidence=json.loads(r["evidence_json"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    def save_companion_interaction(self, record: CompanionInteractionRecord) -> None:
        """Persist conversational interaction and companion query response."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO companion_interactions
                    (interaction_id, symbol, user_query, intent, answer,
                     context_summary, evidence_json, scenarios_json, timestamp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.interaction_id,
                    record.symbol,
                    record.user_query,
                    record.intent.value,
                    record.answer,
                    record.context_summary,
                    json.dumps(record.evidence),
                    json.dumps([s.model_dump(mode="json") for s in record.scenarios]),
                    record.timestamp,
                    record.created_at.isoformat(),
                ),
            )

    def get_companion_interactions(
        self, symbol: Optional[str] = None, limit: int = 20
    ) -> List[CompanionInteractionRecord]:
        """Retrieve recent companion interaction logs."""
        with self._db.connection() as conn:
            query = "SELECT * FROM companion_interactions"
            params: List[Any] = []
            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, tuple(params)).fetchall()
            results = []
            for r in rows:
                scenarios_raw = json.loads(r["scenarios_json"])
                scenarios = [ScenarioCondition(**s) for s in scenarios_raw]
                results.append(
                    CompanionInteractionRecord(
                        interaction_id=r["interaction_id"],
                        symbol=r["symbol"],
                        user_query=r["user_query"],
                        intent=VoiceIntent(r["intent"]),
                        answer=r["answer"],
                        context_summary=r["context_summary"],
                        evidence=json.loads(r["evidence_json"]),
                        scenarios=scenarios,
                        timestamp=r["timestamp"],
                        created_at=datetime.fromisoformat(r["created_at"]),
                    )
                )
            return results

    # =============================================================
    # STEP 8: HEDGE FUND EXECUTION & RISK MANAGEMENT PERSISTENCE
    # =============================================================

    def save_trade_proposal(self, proposal: TradeProposal) -> None:
        """Insert or replace a trade proposal."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_proposals (
                    proposal_id, symbol, direction, market_regime, htf_context, mtf_context,
                    overall_bias, liquidity_context, poi_context, lecture_rule_evidence_json,
                    confluences_json, entry_trigger, entry_price, hard_stop_loss, tp1, tp2,
                    final_target, invalidation_condition, risk_assessment_json, proposed_quantity,
                    engagement_mode, execution_environment, news_event_context, trade_thesis_json, quality,
                    lifecycle_state, user_approval_required, is_approved, rejection_reason,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(proposal_id) DO UPDATE SET
                    symbol = excluded.symbol,
                    direction = excluded.direction,
                    market_regime = excluded.market_regime,
                    htf_context = excluded.htf_context,
                    mtf_context = excluded.mtf_context,
                    overall_bias = excluded.overall_bias,
                    liquidity_context = excluded.liquidity_context,
                    poi_context = excluded.poi_context,
                    lecture_rule_evidence_json = excluded.lecture_rule_evidence_json,
                    confluences_json = excluded.confluences_json,
                    entry_trigger = excluded.entry_trigger,
                    entry_price = excluded.entry_price,
                    hard_stop_loss = excluded.hard_stop_loss,
                    tp1 = excluded.tp1,
                    tp2 = excluded.tp2,
                    final_target = excluded.final_target,
                    invalidation_condition = excluded.invalidation_condition,
                    risk_assessment_json = excluded.risk_assessment_json,
                    proposed_quantity = excluded.proposed_quantity,
                    engagement_mode = excluded.engagement_mode,
                    execution_environment = excluded.execution_environment,
                    news_event_context = excluded.news_event_context,
                    trade_thesis_json = excluded.trade_thesis_json,
                    quality = excluded.quality,
                    lifecycle_state = excluded.lifecycle_state,
                    user_approval_required = excluded.user_approval_required,
                    is_approved = excluded.is_approved,
                    rejection_reason = excluded.rejection_reason,
                    updated_at = excluded.updated_at
                """,
                (
                    proposal.proposal_id,
                    proposal.symbol,
                    proposal.direction,
                    proposal.market_regime,
                    proposal.htf_context,
                    proposal.mtf_context,
                    proposal.overall_bias.value if hasattr(proposal.overall_bias, "value") else str(proposal.overall_bias),
                    proposal.liquidity_context,
                    proposal.poi_context,
                    json.dumps(proposal.lecture_rule_evidence),
                    json.dumps([c.model_dump(mode="json") for c in proposal.confluences]),
                    proposal.entry_trigger,
                    proposal.entry_price,
                    proposal.hard_stop_loss,
                    proposal.tp1,
                    proposal.tp2,
                    proposal.final_target,
                    proposal.invalidation_condition,
                    json.dumps(proposal.risk_assessment.model_dump(mode="json")) if proposal.risk_assessment else None,
                    proposal.proposed_quantity,
                    proposal.engagement_mode.value if hasattr(proposal.engagement_mode, "value") else str(proposal.engagement_mode),
                    proposal.execution_environment.value if hasattr(proposal.execution_environment, "value") else str(proposal.execution_environment),
                    proposal.news_event_context,
                    json.dumps(proposal.trade_thesis.model_dump(mode="json")) if proposal.trade_thesis else None,
                    proposal.quality.value if hasattr(proposal.quality, "value") else str(proposal.quality),
                    proposal.lifecycle_state.value if hasattr(proposal.lifecycle_state, "value") else str(proposal.lifecycle_state),
                    1 if proposal.user_approval_required else 0,
                    1 if proposal.is_approved else 0,
                    proposal.rejection_reason,
                    proposal.created_at.isoformat(),
                    proposal.updated_at.isoformat(),
                ),
            )

    def get_trade_proposal(self, proposal_id: str) -> Optional[TradeProposal]:
        """Retrieve a specific trade proposal by ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM trade_proposals WHERE proposal_id = ?", (proposal_id,)).fetchone()
            if not row:
                return None
            return self._row_to_trade_proposal(row)

    def get_trade_proposals(
        self,
        symbol: Optional[str] = None,
        environment: Optional[ExecutionEnvironment] = None,
        limit: int = 50,
    ) -> List[TradeProposal]:
        """Retrieve recent trade proposals, optionally filtered by symbol and environment."""
        with self._db.connection() as conn:
            query = "SELECT * FROM trade_proposals WHERE 1=1"
            params: List[Any] = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if environment:
                query += " AND execution_environment = ?"
                params.append(environment.value if hasattr(environment, "value") else str(environment))
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_trade_proposal(r) for r in rows]

    def _row_to_trade_proposal(self, r: Any) -> TradeProposal:
        confluences_raw = json.loads(r["confluences_json"])
        confluences = [TradeConfluence(**c) for c in confluences_raw]
        risk_raw = json.loads(r["risk_assessment_json"]) if r["risk_assessment_json"] else None
        risk = RiskAssessment(**risk_raw) if risk_raw else None
        thesis_raw = json.loads(r["trade_thesis_json"]) if r["trade_thesis_json"] else None
        thesis = TradeThesis(**thesis_raw) if thesis_raw else None

        row_keys = r.keys() if hasattr(r, "keys") else []
        env_val = r["execution_environment"] if "execution_environment" in row_keys and r["execution_environment"] else "PAPER"

        return TradeProposal(
            proposal_id=r["proposal_id"],
            symbol=r["symbol"],
            direction=r["direction"],
            market_regime=r["market_regime"],
            htf_context=r["htf_context"],
            mtf_context=r["mtf_context"],
            overall_bias=MarketBias(r["overall_bias"]),
            liquidity_context=r["liquidity_context"],
            poi_context=r["poi_context"],
            lecture_rule_evidence=json.loads(r["lecture_rule_evidence_json"]),
            confluences=confluences,
            entry_trigger=r["entry_trigger"],
            entry_price=r["entry_price"],
            hard_stop_loss=r["hard_stop_loss"],
            tp1=r["tp1"],
            tp2=r["tp2"],
            final_target=r["final_target"],
            invalidation_condition=r["invalidation_condition"],
            risk_assessment=risk,
            proposed_quantity=r["proposed_quantity"],
            engagement_mode=EngagementMode(r["engagement_mode"]),
            execution_environment=ExecutionEnvironment(env_val),
            news_event_context=r["news_event_context"],
            trade_thesis=thesis,
            quality=AlertQuality(r["quality"]),
            lifecycle_state=ProposalLifecycleState(r["lifecycle_state"]),
            user_approval_required=bool(r["user_approval_required"]),
            is_approved=bool(r["is_approved"]),
            rejection_reason=r["rejection_reason"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )

    def save_execution_policy(self, policy: ExecutionPolicy) -> None:
        """Insert or replace an execution policy."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO execution_policies (
                    user_id, execution_mode, execution_environment, live_execution_enabled,
                    auto_execution_enabled, max_auto_risk_pct, min_rr_ratio, preferred_rr_ratio,
                    default_tp1_r_multiple, default_tp1_close_pct, move_sl_to_entry_at_tp1,
                    min_confluence_count, allowed_symbols_json, defensive_size_pct, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    policy.user_id,
                    policy.execution_mode.value if hasattr(policy.execution_mode, "value") else str(policy.execution_mode),
                    policy.execution_environment.value if hasattr(policy.execution_environment, "value") else str(policy.execution_environment),
                    1 if policy.live_execution_enabled else 0,
                    1 if policy.auto_execution_enabled else 0,
                    policy.max_auto_risk_pct,
                    policy.min_rr_ratio,
                    policy.preferred_rr_ratio,
                    policy.default_tp1_r_multiple,
                    policy.default_tp1_close_pct,
                    1 if policy.move_sl_to_entry_at_tp1 else 0,
                    policy.min_confluence_count,
                    json.dumps(policy.allowed_symbols),
                    policy.defensive_size_pct,
                    policy.updated_at.isoformat(),
                ),
            )

    def get_execution_policy(self, user_id: str = "default_user") -> Optional[ExecutionPolicy]:
        """Retrieve user execution policy."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM execution_policies WHERE user_id = ?", (user_id,)).fetchone()
            if not row:
                return None
            row_keys = row.keys() if hasattr(row, "keys") else []
            env_val = row["execution_environment"] if "execution_environment" in row_keys and row["execution_environment"] else "PAPER"
            live_en = bool(row["live_execution_enabled"]) if "live_execution_enabled" in row_keys else False

            return ExecutionPolicy(
                user_id=row["user_id"],
                execution_mode=ExecutionMode(row["execution_mode"]),
                execution_environment=ExecutionEnvironment(env_val),
                live_execution_enabled=live_en,
                auto_execution_enabled=bool(row["auto_execution_enabled"]),
                max_auto_risk_pct=row["max_auto_risk_pct"],
                min_rr_ratio=row["min_rr_ratio"],
                preferred_rr_ratio=row["preferred_rr_ratio"],
                default_tp1_r_multiple=row["default_tp1_r_multiple"],
                default_tp1_close_pct=row["default_tp1_close_pct"],
                move_sl_to_entry_at_tp1=bool(row["move_sl_to_entry_at_tp1"]),
                min_confluence_count=row["min_confluence_count"],
                allowed_symbols=json.loads(row["allowed_symbols_json"]),
                defensive_size_pct=row["defensive_size_pct"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def save_execution_intent(self, intent: ExecutionIntent) -> None:
        """Insert or replace an execution intent."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO execution_intents (
                    intent_id, proposal_id, symbol, execution_mode, execution_environment,
                    order_type, direction, quantity, entry_price, stop_loss, take_profit,
                    tp1, status, anti_duplicate_token, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent.intent_id,
                    intent.proposal_id,
                    intent.symbol,
                    intent.execution_mode.value if hasattr(intent.execution_mode, "value") else str(intent.execution_mode),
                    intent.execution_environment.value if hasattr(intent.execution_environment, "value") else str(intent.execution_environment),
                    intent.order_type.value if hasattr(intent.order_type, "value") else str(intent.order_type),
                    intent.direction,
                    intent.quantity,
                    intent.entry_price,
                    intent.stop_loss,
                    intent.take_profit,
                    intent.tp1,
                    intent.status,
                    intent.anti_duplicate_token,
                    intent.created_at.isoformat(),
                ),
            )

    def get_execution_intent_by_token(self, token: str) -> Optional[ExecutionIntent]:
        """Check for existing duplicate execution intent."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM execution_intents WHERE anti_duplicate_token = ?", (token,)).fetchone()
            if not row:
                return None
            row_keys = row.keys() if hasattr(row, "keys") else []
            env_val = row["execution_environment"] if "execution_environment" in row_keys and row["execution_environment"] else "PAPER"
            return ExecutionIntent(
                intent_id=row["intent_id"],
                proposal_id=row["proposal_id"],
                symbol=row["symbol"],
                execution_mode=ExecutionMode(row["execution_mode"]),
                execution_environment=ExecutionEnvironment(env_val),
                order_type=OrderType(row["order_type"]),
                direction=row["direction"],
                quantity=row["quantity"],
                entry_price=row["entry_price"],
                stop_loss=row["stop_loss"],
                take_profit=row["take_profit"],
                tp1=row["tp1"],
                status=row["status"],
                anti_duplicate_token=row["anti_duplicate_token"],
                created_at=datetime.fromisoformat(row["created_at"]),
            )

    def save_order(self, order: Order) -> None:
        """Insert or replace an order record."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO orders (
                    order_id, broker_order_id, intent_id, proposal_id, symbol,
                    execution_environment, is_simulated, order_type, direction,
                    quantity, limit_price, stop_price, status, filled_quantity,
                    average_fill_price, broker_response_json, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order.order_id,
                    order.broker_order_id,
                    order.intent_id,
                    order.proposal_id,
                    order.symbol,
                    order.execution_environment.value if hasattr(order.execution_environment, "value") else str(order.execution_environment),
                    1 if order.is_simulated else 0,
                    order.order_type.value if hasattr(order.order_type, "value") else str(order.order_type),
                    order.direction,
                    order.quantity,
                    order.limit_price,
                    order.stop_price,
                    order.status.value if hasattr(order.status, "value") else str(order.status),
                    order.filled_quantity,
                    order.average_fill_price,
                    json.dumps(order.broker_response),
                    order.error_message,
                    order.created_at.isoformat(),
                    order.updated_at.isoformat(),
                ),
            )

    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
            if not row:
                return None
            return self._row_to_order(row)

    def get_orders(
        self,
        symbol: Optional[str] = None,
        environment: Optional[ExecutionEnvironment] = None,
        limit: int = 50,
    ) -> List[Order]:
        """Retrieve recent orders, optionally filtered by symbol and environment."""
        with self._db.connection() as conn:
            query = "SELECT * FROM orders WHERE 1=1"
            params: List[Any] = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if environment:
                query += " AND execution_environment = ?"
                params.append(environment.value if hasattr(environment, "value") else str(environment))
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_order(r) for r in rows]

    def _row_to_order(self, r: Any) -> Order:
        row_keys = r.keys() if hasattr(r, "keys") else []
        env_val = r["execution_environment"] if "execution_environment" in row_keys and r["execution_environment"] else "PAPER"
        is_sim = bool(r["is_simulated"]) if "is_simulated" in row_keys else True
        return Order(
            order_id=r["order_id"],
            broker_order_id=r["broker_order_id"],
            intent_id=r["intent_id"],
            proposal_id=r["proposal_id"],
            symbol=r["symbol"],
            execution_environment=ExecutionEnvironment(env_val),
            is_simulated=is_sim,
            order_type=OrderType(r["order_type"]),
            direction=r["direction"],
            quantity=r["quantity"],
            limit_price=r["limit_price"],
            stop_price=r["stop_price"],
            status=OrderStatus(r["status"]),
            filled_quantity=r["filled_quantity"],
            average_fill_price=r["average_fill_price"],
            broker_response=json.loads(r["broker_response_json"]),
            error_message=r["error_message"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )

    def save_position(self, position: Position) -> None:
        """Insert or replace an actively managed position."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO positions (
                    position_id, proposal_id, symbol, direction, execution_environment,
                    entry_price, current_quantity, initial_quantity, initial_sl, current_sl,
                    tp1, tp2, final_target, status, unrealized_pnl, realized_pnl,
                    tp1_reached, sl_moved_to_entry, engagement_mode, opened_at,
                    updated_at, closed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.position_id,
                    position.proposal_id,
                    position.symbol,
                    position.direction,
                    position.execution_environment.value if hasattr(position.execution_environment, "value") else str(position.execution_environment),
                    position.entry_price,
                    position.current_quantity,
                    position.initial_quantity,
                    position.initial_sl,
                    position.current_sl,
                    position.tp1,
                    position.tp2,
                    position.final_target,
                    position.status.value if hasattr(position.status, "value") else str(position.status),
                    position.unrealized_pnl,
                    position.realized_pnl,
                    1 if position.tp1_reached else 0,
                    1 if position.sl_moved_to_entry else 0,
                    position.engagement_mode.value if hasattr(position.engagement_mode, "value") else str(position.engagement_mode),
                    position.opened_at.isoformat(),
                    position.updated_at.isoformat(),
                    position.closed_at.isoformat() if position.closed_at else None,
                ),
            )

    def get_position(self, position_id: str) -> Optional[Position]:
        """Retrieve a specific position by ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM positions WHERE position_id = ?", (position_id,)).fetchone()
            if not row:
                return None
            return self._row_to_position(row)

    def get_open_positions(
        self,
        symbol: Optional[str] = None,
        environment: Optional[ExecutionEnvironment] = None,
    ) -> List[Position]:
        """Retrieve currently open or partially closed positions, optionally isolated by environment."""
        with self._db.connection() as conn:
            query = "SELECT * FROM positions WHERE status IN ('OPEN', 'PARTIALLY_CLOSED')"
            params: List[Any] = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if environment:
                query += " AND execution_environment = ?"
                params.append(environment.value if hasattr(environment, "value") else str(environment))
            query += " ORDER BY opened_at DESC"
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_position(r) for r in rows]

    def get_positions(
        self,
        symbol: Optional[str] = None,
        environment: Optional[ExecutionEnvironment] = None,
        limit: int = 50,
    ) -> List[Position]:
        """Retrieve recent positions (both open and closed), optionally isolated by environment."""
        with self._db.connection() as conn:
            query = "SELECT * FROM positions WHERE 1=1"
            params: List[Any] = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if environment:
                query += " AND execution_environment = ?"
                params.append(environment.value if hasattr(environment, "value") else str(environment))
            query += " ORDER BY opened_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_position(r) for r in rows]

    def _row_to_position(self, r: Any) -> Position:
        row_keys = r.keys() if hasattr(r, "keys") else []
        env_val = r["execution_environment"] if "execution_environment" in row_keys and r["execution_environment"] else "PAPER"
        return Position(
            position_id=r["position_id"],
            proposal_id=r["proposal_id"],
            symbol=r["symbol"],
            direction=r["direction"],
            execution_environment=ExecutionEnvironment(env_val),
            entry_price=r["entry_price"],
            current_quantity=r["current_quantity"],
            initial_quantity=r["initial_quantity"],
            initial_sl=r["initial_sl"],
            current_sl=r["current_sl"],
            tp1=r["tp1"],
            tp2=r["tp2"],
            final_target=r["final_target"],
            status=PositionStatus(r["status"]),
            unrealized_pnl=r["unrealized_pnl"],
            realized_pnl=r["realized_pnl"],
            tp1_reached=bool(r["tp1_reached"]),
            sl_moved_to_entry=bool(r["sl_moved_to_entry"]),
            engagement_mode=EngagementMode(r["engagement_mode"]),
            opened_at=datetime.fromisoformat(r["opened_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
            closed_at=datetime.fromisoformat(r["closed_at"]) if r["closed_at"] else None,
        )

    def save_position_event(self, event: PositionEvent) -> None:
        """Insert a position management audit event."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO position_events (
                    event_id, position_id, action, price, quantity_affected,
                    realized_pnl_delta, notes, timestamp, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.position_id,
                    event.action.value if hasattr(event.action, "value") else str(event.action),
                    event.price,
                    event.quantity_affected,
                    event.realized_pnl_delta,
                    event.notes,
                    event.timestamp,
                    event.created_at.isoformat(),
                ),
            )

    def get_position_events(self, position_id: str) -> List[PositionEvent]:
        """Retrieve audit history for a specific position."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM position_events WHERE position_id = ? ORDER BY timestamp ASC",
                (position_id,),
            ).fetchall()
            return [
                PositionEvent(
                    event_id=r["event_id"],
                    position_id=r["position_id"],
                    action=PositionManagementAction(r["action"]),
                    price=r["price"],
                    quantity_affected=r["quantity_affected"],
                    realized_pnl_delta=r["realized_pnl_delta"],
                    notes=r["notes"],
                    timestamp=r["timestamp"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                )
                for r in rows
            ]

    # =============================================================
    # STEP 9: BACKTESTING, TRADING-BRAIN & PROMOTION PERSISTENCE
    # =============================================================

    def save_rule_version(self, rule: RuleVersion) -> None:
        """Insert or update an immutable versioned rule."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO rule_versions (
                    rule_id, version, title, rule_text, conditions_json,
                    exceptions_json, source_lecture_id, source_timestamp,
                    provenance, state, tags_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rule_id, version) DO UPDATE SET
                    title = excluded.title,
                    rule_text = excluded.rule_text,
                    conditions_json = excluded.conditions_json,
                    exceptions_json = excluded.exceptions_json,
                    source_lecture_id = excluded.source_lecture_id,
                    source_timestamp = excluded.source_timestamp,
                    provenance = excluded.provenance,
                    state = excluded.state,
                    tags_json = excluded.tags_json,
                    updated_at = excluded.updated_at
                """,
                (
                    rule.rule_id,
                    rule.version,
                    rule.title,
                    rule.rule_text,
                    json.dumps(rule.conditions),
                    json.dumps(rule.exceptions),
                    rule.source_lecture_id,
                    rule.source_timestamp,
                    rule.provenance,
                    rule.state.value if hasattr(rule.state, "value") else str(rule.state),
                    json.dumps(rule.tags),
                    rule.created_at.isoformat(),
                    rule.updated_at.isoformat(),
                ),
            )

    def get_rule_version(self, rule_id: str, version: Optional[int] = None) -> Optional[RuleVersion]:
        """Retrieve a specific rule version or the latest version."""
        with self._db.connection() as conn:
            if version is not None:
                row = conn.execute(
                    "SELECT * FROM rule_versions WHERE rule_id = ? AND version = ?",
                    (rule_id, version),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM rule_versions WHERE rule_id = ? ORDER BY version DESC LIMIT 1",
                    (rule_id,),
                ).fetchone()
            if not row:
                return None
            return self._row_to_rule_version(row)

    def get_rule_versions(
        self,
        rule_id: Optional[str] = None,
        state: Optional[RuleLifecycleState] = None,
    ) -> List[RuleVersion]:
        """Retrieve rule versions, optionally filtered by ID and state."""
        with self._db.connection() as conn:
            query = "SELECT * FROM rule_versions WHERE 1=1"
            params: List[Any] = []
            if rule_id:
                query += " AND rule_id = ?"
                params.append(rule_id)
            if state:
                query += " AND state = ?"
                params.append(state.value if hasattr(state, "value") else str(state))
            query += " ORDER BY created_at DESC"
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_rule_version(r) for r in rows]

    def _row_to_rule_version(self, r: Any) -> RuleVersion:
        return RuleVersion(
            rule_id=r["rule_id"],
            version=r["version"],
            title=r["title"],
            rule_text=r["rule_text"],
            conditions=json.loads(r["conditions_json"]),
            exceptions=json.loads(r["exceptions_json"]),
            source_lecture_id=r["source_lecture_id"],
            source_timestamp=r["source_timestamp"],
            provenance=r["provenance"],
            state=RuleLifecycleState(r["state"]),
            tags=json.loads(r["tags_json"]),
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )

    def save_strategy_version(self, strategy: StrategyVersion) -> None:
        """Insert or update a composite strategy version."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO strategy_versions (
                    strategy_id, version, name, description, rule_version_ids_json,
                    asset_classes_json, symbols_json, timeframes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(strategy_id, version) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    rule_version_ids_json = excluded.rule_version_ids_json,
                    asset_classes_json = excluded.asset_classes_json,
                    symbols_json = excluded.symbols_json,
                    timeframes_json = excluded.timeframes_json
                """,
                (
                    strategy.strategy_id,
                    strategy.version,
                    strategy.name,
                    strategy.description,
                    json.dumps(strategy.rule_version_ids),
                    json.dumps([ac.value if hasattr(ac, "value") else str(ac) for ac in strategy.asset_classes]),
                    json.dumps(strategy.symbols),
                    json.dumps(strategy.timeframes),
                    strategy.created_at.isoformat(),
                ),
            )

    def get_strategy_version(self, strategy_id: str, version: Optional[int] = None) -> Optional[StrategyVersion]:
        """Retrieve a specific strategy version or latest."""
        with self._db.connection() as conn:
            if version is not None:
                row = conn.execute(
                    "SELECT * FROM strategy_versions WHERE strategy_id = ? AND version = ?",
                    (strategy_id, version),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM strategy_versions WHERE strategy_id = ? ORDER BY version DESC LIMIT 1",
                    (strategy_id,),
                ).fetchone()
            if not row:
                return None
            return self._row_to_strategy_version(row)

    def get_strategy_versions(self) -> List[StrategyVersion]:
        """Retrieve all strategy versions."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM strategy_versions ORDER BY created_at DESC").fetchall()
            return [self._row_to_strategy_version(r) for r in rows]

    def _row_to_strategy_version(self, r: Any) -> StrategyVersion:
        raw_acs = json.loads(r["asset_classes_json"])
        return StrategyVersion(
            strategy_id=r["strategy_id"],
            version=r["version"],
            name=r["name"],
            description=r["description"],
            rule_version_ids=json.loads(r["rule_version_ids_json"]),
            asset_classes=[AssetClass(ac) for ac in raw_acs],
            symbols=json.loads(r["symbols_json"]),
            timeframes=json.loads(r["timeframes_json"]),
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    def save_backtest_run(self, run: BacktestRun) -> None:
        """Insert or update a backtest execution job."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO backtest_runs (
                    backtest_id, mode, asset_class, symbol, timeframe,
                    strategy_version_id, rule_version_id, config_json, status,
                    progress_pct, current_candle_index, total_candles, metrics_json,
                    is_reproducible, dataset_identifier, error_message, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(backtest_id) DO UPDATE SET
                    mode = excluded.mode,
                    asset_class = excluded.asset_class,
                    symbol = excluded.symbol,
                    timeframe = excluded.timeframe,
                    strategy_version_id = excluded.strategy_version_id,
                    rule_version_id = excluded.rule_version_id,
                    config_json = excluded.config_json,
                    status = excluded.status,
                    progress_pct = excluded.progress_pct,
                    current_candle_index = excluded.current_candle_index,
                    total_candles = excluded.total_candles,
                    metrics_json = excluded.metrics_json,
                    is_reproducible = excluded.is_reproducible,
                    dataset_identifier = excluded.dataset_identifier,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at
                """,
                (
                    run.backtest_id,
                    run.mode.value if hasattr(run.mode, "value") else str(run.mode),
                    run.asset_class.value if hasattr(run.asset_class, "value") else str(run.asset_class),
                    run.symbol,
                    run.timeframe,
                    run.strategy_version_id,
                    run.rule_version_id,
                    json.dumps(run.config.model_dump(mode="json")),
                    run.status.value if hasattr(run.status, "value") else str(run.status),
                    run.progress_pct,
                    run.current_candle_index,
                    run.total_candles,
                    json.dumps(run.metrics.model_dump(mode="json")) if run.metrics else None,
                    1 if run.is_reproducible else 0,
                    run.dataset_identifier,
                    run.error_message,
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                ),
            )

    def get_backtest_run(self, backtest_id: str) -> Optional[BacktestRun]:
        """Retrieve a specific backtest run by ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM backtest_runs WHERE backtest_id = ?", (backtest_id,)).fetchone()
            if not row:
                return None
            return self._row_to_backtest_run(row)

    def get_backtest_runs(
        self,
        asset_class: Optional[AssetClass] = None,
        symbol: Optional[str] = None,
        limit: int = 50,
    ) -> List[BacktestRun]:
        """Retrieve recent backtest runs with asset class and symbol filtering."""
        with self._db.connection() as conn:
            query = "SELECT * FROM backtest_runs WHERE 1=1"
            params: List[Any] = []
            if asset_class:
                query += " AND asset_class = ?"
                params.append(asset_class.value if hasattr(asset_class, "value") else str(asset_class))
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_backtest_run(r) for r in rows]

    def _row_to_backtest_run(self, r: Any) -> BacktestRun:
        config_dict = json.loads(r["config_json"]) if r["config_json"] else {}
        metrics_dict = json.loads(r["metrics_json"]) if r["metrics_json"] else None
        return BacktestRun(
            backtest_id=r["backtest_id"],
            mode=BacktestMode(r["mode"]),
            asset_class=AssetClass(r["asset_class"]),
            symbol=r["symbol"],
            timeframe=r["timeframe"],
            strategy_version_id=r["strategy_version_id"],
            rule_version_id=r["rule_version_id"],
            config=BacktestConfig(**config_dict) if config_dict else BacktestConfig(),
            status=BacktestStatus(r["status"]),
            progress_pct=r["progress_pct"],
            current_candle_index=r["current_candle_index"],
            total_candles=r["total_candles"],
            metrics=BacktestMetrics(**metrics_dict) if metrics_dict else None,
            is_reproducible=bool(r["is_reproducible"]),
            dataset_identifier=r["dataset_identifier"],
            error_message=r["error_message"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )

    def save_backtest_trade(self, trade: BacktestTrade) -> None:
        """Insert or replace a backtest trade record."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO backtest_trades (
                    trade_id, backtest_id, symbol, asset_class, direction,
                    entry_time, entry_price, exit_time, exit_price, stop_loss,
                    target_price, quantity, pnl, r_multiple, result,
                    rule_version_id, source, confluence_summary, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    exit_time = excluded.exit_time,
                    exit_price = excluded.exit_price,
                    pnl = excluded.pnl,
                    r_multiple = excluded.r_multiple,
                    result = excluded.result,
                    notes = excluded.notes
                """,
                (
                    trade.trade_id,
                    trade.backtest_id,
                    trade.symbol,
                    trade.asset_class.value if hasattr(trade.asset_class, "value") else str(trade.asset_class),
                    trade.direction,
                    trade.entry_time,
                    trade.entry_price,
                    trade.exit_time,
                    trade.exit_price,
                    trade.stop_loss,
                    trade.target_price,
                    trade.quantity,
                    trade.pnl,
                    trade.r_multiple,
                    trade.result,
                    trade.rule_version_id,
                    trade.source.value if hasattr(trade.source, "value") else str(trade.source),
                    trade.confluence_summary,
                    trade.notes,
                    trade.created_at.isoformat(),
                ),
            )

    def get_backtest_trades(self, backtest_id: str) -> List[BacktestTrade]:
        """Retrieve all trades generated within a backtest run."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_trades WHERE backtest_id = ? ORDER BY entry_time ASC",
                (backtest_id,),
            ).fetchall()
            return [self._row_to_backtest_trade(r) for r in rows]

    def _row_to_backtest_trade(self, r: Any) -> BacktestTrade:
        return BacktestTrade(
            trade_id=r["trade_id"],
            backtest_id=r["backtest_id"],
            symbol=r["symbol"],
            asset_class=AssetClass(r["asset_class"]),
            direction=r["direction"],
            entry_time=r["entry_time"],
            entry_price=r["entry_price"],
            exit_time=r["exit_time"],
            exit_price=r["exit_price"],
            stop_loss=r["stop_loss"],
            target_price=r["target_price"],
            quantity=r["quantity"],
            pnl=r["pnl"],
            r_multiple=r["r_multiple"],
            result=r["result"],
            rule_version_id=r["rule_version_id"],
            source=TradeSource(r["source"]),
            confluence_summary=r["confluence_summary"],
            notes=r["notes"],
            created_at=datetime.fromisoformat(r["created_at"]),
        )

    def save_rule_promotion(self, promo: RulePromotionRecord) -> None:
        """Insert or replace a rule promotion audit record."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO rule_promotions (
                    promotion_id, rule_version_id, current_state, backtest_id,
                    user_backtest_approved, paper_validation_approved,
                    user_live_approved, paper_trade_count, paper_pnl,
                    notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(promotion_id) DO UPDATE SET
                    current_state = excluded.current_state,
                    backtest_id = excluded.backtest_id,
                    user_backtest_approved = excluded.user_backtest_approved,
                    paper_validation_approved = excluded.paper_validation_approved,
                    user_live_approved = excluded.user_live_approved,
                    paper_trade_count = excluded.paper_trade_count,
                    paper_pnl = excluded.paper_pnl,
                    notes = excluded.notes,
                    updated_at = excluded.updated_at
                """,
                (
                    promo.promotion_id,
                    promo.rule_version_id,
                    promo.current_state.value if hasattr(promo.current_state, "value") else str(promo.current_state),
                    promo.backtest_id,
                    1 if promo.user_backtest_approved else 0,
                    1 if promo.paper_validation_approved else 0,
                    1 if promo.user_live_approved else 0,
                    promo.paper_trade_count,
                    promo.paper_pnl,
                    promo.notes,
                    promo.created_at.isoformat(),
                    promo.updated_at.isoformat(),
                ),
            )

    def get_rule_promotion(self, rule_version_id: str) -> Optional[RulePromotionRecord]:
        """Retrieve promotion status for a specific rule version."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM rule_promotions WHERE rule_version_id = ? ORDER BY updated_at DESC LIMIT 1",
                (rule_version_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_rule_promotion(row)

    def get_rule_promotions(self) -> List[RulePromotionRecord]:
        """Retrieve all active rule promotion records."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM rule_promotions ORDER BY updated_at DESC").fetchall()
            return [self._row_to_rule_promotion(r) for r in rows]

    def _row_to_rule_promotion(self, r: Any) -> RulePromotionRecord:
        return RulePromotionRecord(
            promotion_id=r["promotion_id"],
            rule_version_id=r["rule_version_id"],
            current_state=RuleLifecycleState(r["current_state"]),
            backtest_id=r["backtest_id"],
            user_backtest_approved=bool(r["user_backtest_approved"]),
            paper_validation_approved=bool(r["paper_validation_approved"]),
            user_live_approved=bool(r["user_live_approved"]),
            paper_trade_count=r["paper_trade_count"],
            paper_pnl=r["paper_pnl"],
            notes=r["notes"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )

    # -------------------------------------------------------------
    # STEP 10: TRADING JOURNAL
    # -------------------------------------------------------------
    def save_journal_entry(self, entry: TradeJournalEntry) -> None:
        """Insert or replace a trade journal record."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO trade_journal_entries (
                    journal_id, trade_id, environment, backtest_type, asset_class,
                    symbol, timeframe, direction, rule_version_id, strategy_version_id,
                    lecture_source_id, setup_name, market_regime, htf_context,
                    mtf_context, ltf_context, news_context, confluences_json,
                    engagement_mode, trade_thesis, planned_json, actual_json,
                    rule_adherence_score, rule_violations_json, behaviors_json,
                    notes, evidence_references_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(journal_id) DO UPDATE SET
                    trade_id = excluded.trade_id,
                    environment = excluded.environment,
                    backtest_type = excluded.backtest_type,
                    asset_class = excluded.asset_class,
                    symbol = excluded.symbol,
                    timeframe = excluded.timeframe,
                    direction = excluded.direction,
                    rule_version_id = excluded.rule_version_id,
                    strategy_version_id = excluded.strategy_version_id,
                    lecture_source_id = excluded.lecture_source_id,
                    setup_name = excluded.setup_name,
                    market_regime = excluded.market_regime,
                    htf_context = excluded.htf_context,
                    mtf_context = excluded.mtf_context,
                    ltf_context = excluded.ltf_context,
                    news_context = excluded.news_context,
                    confluences_json = excluded.confluences_json,
                    engagement_mode = excluded.engagement_mode,
                    trade_thesis = excluded.trade_thesis,
                    planned_json = excluded.planned_json,
                    actual_json = excluded.actual_json,
                    rule_adherence_score = excluded.rule_adherence_score,
                    rule_violations_json = excluded.rule_violations_json,
                    behaviors_json = excluded.behaviors_json,
                    notes = excluded.notes,
                    evidence_references_json = excluded.evidence_references_json,
                    updated_at = excluded.updated_at
                """,
                (
                    entry.journal_id,
                    entry.trade_id,
                    entry.environment.value if hasattr(entry.environment, "value") else str(entry.environment),
                    entry.backtest_type.value if hasattr(entry.backtest_type, "value") else str(entry.backtest_type),
                    entry.asset_class.value if hasattr(entry.asset_class, "value") else str(entry.asset_class),
                    entry.symbol,
                    entry.timeframe,
                    entry.direction,
                    entry.rule_version_id,
                    entry.strategy_version_id,
                    entry.lecture_source_id,
                    entry.setup_name,
                    entry.market_regime,
                    entry.htf_context,
                    entry.mtf_context,
                    entry.ltf_context,
                    entry.news_context,
                    json.dumps(entry.confluences),
                    entry.engagement_mode.value if hasattr(entry.engagement_mode, "value") else str(entry.engagement_mode),
                    entry.trade_thesis,
                    json.dumps(entry.planned.model_dump(mode="json")),
                    json.dumps(entry.actual.model_dump(mode="json")),
                    entry.rule_adherence_score,
                    json.dumps(entry.rule_violations),
                    json.dumps([b.model_dump(mode="json") for b in entry.behaviors]),
                    entry.notes,
                    json.dumps(entry.evidence_references),
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                ),
            )

    def get_journal_entry(self, journal_id: str) -> Optional[TradeJournalEntry]:
        """Fetch a journal entry by its primary key."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM trade_journal_entries WHERE journal_id = ?", (journal_id,)).fetchone()
            if not row:
                return None
            return self._row_to_journal_entry(row)

    def get_journal_entry_by_trade_id(self, trade_id: str) -> Optional[TradeJournalEntry]:
        """Fetch a journal entry by linked trade/position ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM trade_journal_entries WHERE trade_id = ? ORDER BY created_at DESC LIMIT 1", (trade_id,)).fetchone()
            if not row:
                return None
            return self._row_to_journal_entry(row)

    def list_journal_entries(
        self,
        environment: Optional[JournalEnvironment] = None,
        asset_class: Optional[AssetClass] = None,
        symbol: Optional[str] = None,
        backtest_type: Optional[JournalBacktestType] = None,
        limit: int = 100,
    ) -> List[TradeJournalEntry]:
        """Retrieve journal entries with optional environment, asset class, and symbol filtering."""
        query = "SELECT * FROM trade_journal_entries WHERE 1=1"
        params: List[Any] = []
        if environment:
            query += " AND environment = ?"
            params.append(environment.value if hasattr(environment, "value") else str(environment))
        if asset_class:
            query += " AND asset_class = ?"
            params.append(asset_class.value if hasattr(asset_class, "value") else str(asset_class))
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if backtest_type:
            query += " AND backtest_type = ?"
            params.append(backtest_type.value if hasattr(backtest_type, "value") else str(backtest_type))
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_journal_entry(r) for r in rows]

    def _row_to_journal_entry(self, r: Any) -> TradeJournalEntry:
        planned_dict = json.loads(r["planned_json"])
        actual_dict = json.loads(r["actual_json"])
        behaviors_list = json.loads(r["behaviors_json"])

        return TradeJournalEntry(
            journal_id=r["journal_id"],
            trade_id=r["trade_id"],
            environment=JournalEnvironment(r["environment"]),
            backtest_type=JournalBacktestType(r["backtest_type"]) if r["backtest_type"] else JournalBacktestType.NOT_APPLICABLE,
            asset_class=AssetClass(r["asset_class"]),
            symbol=r["symbol"],
            timeframe=r["timeframe"],
            direction=r["direction"],
            rule_version_id=r["rule_version_id"],
            strategy_version_id=r["strategy_version_id"],
            lecture_source_id=r["lecture_source_id"],
            setup_name=r["setup_name"],
            market_regime=r["market_regime"],
            htf_context=r["htf_context"],
            mtf_context=r["mtf_context"],
            ltf_context=r["ltf_context"],
            news_context=r["news_context"],
            confluences=json.loads(r["confluences_json"]),
            engagement_mode=EngagementMode(r["engagement_mode"]),
            trade_thesis=r["trade_thesis"],
            planned=PlannedTradeValues(**planned_dict),
            actual=ActualTradeValues(**actual_dict),
            rule_adherence_score=r["rule_adherence_score"],
            rule_violations=json.loads(r["rule_violations_json"]),
            behaviors=[ObservedBehavior(**b) for b in behaviors_list],
            notes=r["notes"],
            evidence_references=json.loads(r["evidence_references_json"]),
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )

    # -------------------------------------------------------------
    # STEP 10: LESSONS
    # -------------------------------------------------------------
    def save_journal_lesson(self, lesson: JournalLesson) -> None:
        """Insert or update a journal-derived lesson."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO journal_lessons (
                    lesson_id, source_journal_ids_json, symbol, asset_class,
                    affected_rule_version_id, title, observation, explanation,
                    proposed_action, confidence, state, reviewer_notes,
                    candidate_rule_version_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(lesson_id) DO UPDATE SET
                    source_journal_ids_json = excluded.source_journal_ids_json,
                    symbol = excluded.symbol,
                    asset_class = excluded.asset_class,
                    affected_rule_version_id = excluded.affected_rule_version_id,
                    title = excluded.title,
                    observation = excluded.observation,
                    explanation = excluded.explanation,
                    proposed_action = excluded.proposed_action,
                    confidence = excluded.confidence,
                    state = excluded.state,
                    reviewer_notes = excluded.reviewer_notes,
                    candidate_rule_version_id = excluded.candidate_rule_version_id,
                    updated_at = excluded.updated_at
                """,
                (
                    lesson.lesson_id,
                    json.dumps(lesson.source_journal_ids),
                    lesson.symbol,
                    lesson.asset_class.value if lesson.asset_class and hasattr(lesson.asset_class, "value") else (str(lesson.asset_class) if lesson.asset_class else None),
                    lesson.affected_rule_version_id,
                    lesson.title,
                    lesson.observation,
                    lesson.explanation,
                    lesson.proposed_action,
                    lesson.confidence,
                    lesson.state.value if hasattr(lesson.state, "value") else str(lesson.state),
                    lesson.reviewer_notes,
                    lesson.candidate_rule_version_id,
                    lesson.created_at.isoformat(),
                    lesson.updated_at.isoformat(),
                ),
            )

    def get_journal_lesson(self, lesson_id: str) -> Optional[JournalLesson]:
        """Fetch a specific lesson by ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM journal_lessons WHERE lesson_id = ?", (lesson_id,)).fetchone()
            if not row:
                return None
            return self._row_to_journal_lesson(row)

    def list_journal_lessons(
        self,
        state: Optional[LessonLifecycleState] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[JournalLesson]:
        """List journal lessons with optional state and symbol filter."""
        query = "SELECT * FROM journal_lessons WHERE 1=1"
        params: List[Any] = []
        if state:
            query += " AND state = ?"
            params.append(state.value if hasattr(state, "value") else str(state))
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_journal_lesson(r) for r in rows]

    def _row_to_journal_lesson(self, r: Any) -> JournalLesson:
        return JournalLesson(
            lesson_id=r["lesson_id"],
            source_journal_ids=json.loads(r["source_journal_ids_json"]),
            symbol=r["symbol"],
            asset_class=AssetClass(r["asset_class"]) if r["asset_class"] else None,
            affected_rule_version_id=r["affected_rule_version_id"],
            title=r["title"],
            observation=r["observation"],
            explanation=r["explanation"],
            proposed_action=r["proposed_action"],
            confidence=r["confidence"],
            state=LessonLifecycleState(r["state"]),
            reviewer_notes=r["reviewer_notes"],
            candidate_rule_version_id=r["candidate_rule_version_id"],
            created_at=datetime.fromisoformat(r["created_at"]),
            updated_at=datetime.fromisoformat(r["updated_at"]),
        )

    # -------------------------------------------------------------
    # STEP 10: PRE-TRADE REVIEWS
    # -------------------------------------------------------------
    def save_pretrade_review(self, review: PreTradeReviewResult) -> None:
        """Insert or replace a pre-trade decision support review result."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO journal_reviews (
                    review_id, proposal_id, symbol, status, plan_is_complete,
                    missing_plan_elements_json, historical_similar_trades_count,
                    historical_win_rate, relevant_lessons_json, relevant_mistakes_json,
                    discipline_warnings_json, risk_reward_ratio, recommendation, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review.review_id,
                    getattr(review, "proposal_id", None),
                    getattr(review, "symbol", "EUR/USD"),
                    review.status.value if hasattr(review.status, "value") else str(review.status),
                    1 if review.plan_is_complete else 0,
                    json.dumps(review.missing_plan_elements),
                    review.historical_similar_trades_count,
                    review.historical_win_rate,
                    json.dumps([l.model_dump(mode="json") for l in review.relevant_lessons]),
                    json.dumps(review.relevant_mistakes),
                    json.dumps(review.discipline_warnings),
                    review.risk_reward_ratio,
                    review.recommendation,
                    review.reviewed_at.isoformat(),
                ),
            )

    def get_pretrade_review(self, review_id: str) -> Optional[PreTradeReviewResult]:
        """Fetch pre-trade review by ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM journal_reviews WHERE review_id = ?", (review_id,)).fetchone()
            if not row:
                return None
            lessons_raw = json.loads(row["relevant_lessons_json"])
            return PreTradeReviewResult(
                review_id=row["review_id"],
                status=PreTradeReviewStatus(row["status"]),
                plan_is_complete=bool(row["plan_is_complete"]),
                missing_plan_elements=json.loads(row["missing_plan_elements_json"]),
                historical_similar_trades_count=row["historical_similar_trades_count"],
                historical_win_rate=row["historical_win_rate"],
                relevant_lessons=[JournalLesson(**l) for l in lessons_raw],
                relevant_mistakes=json.loads(row["relevant_mistakes_json"]),
                discipline_warnings=json.loads(row["discipline_warnings_json"]),
                risk_reward_ratio=row["risk_reward_ratio"],
                recommendation=row["recommendation"],
                reviewed_at=datetime.fromisoformat(row["reviewed_at"]),
            )

    # -------------------------------------------------------------
    # STEP 10: RESILIENCE CHECKPOINTS
    # -------------------------------------------------------------
    def save_journal_checkpoint(self, chk: JournalProcessingCheckpoint) -> None:
        """Save progress checkpoint for crash-resilient journal processing."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO journal_checkpoints (
                    job_id, last_processed_trade_id, processed_count,
                    total_count, context_token_estimate, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chk.job_id,
                    chk.last_processed_trade_id,
                    chk.processed_count,
                    chk.total_count,
                    chk.context_token_estimate,
                    chk.status,
                    chk.updated_at.isoformat(),
                ),
            )

    def get_journal_checkpoint(self, job_id: str) -> Optional[JournalProcessingCheckpoint]:
        """Fetch checkpoint by job ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM journal_checkpoints WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return JournalProcessingCheckpoint(
                job_id=row["job_id"],
                last_processed_trade_id=row["last_processed_trade_id"],
                processed_count=row["processed_count"],
                total_count=row["total_count"],
                context_token_estimate=row["context_token_estimate"],
                status=row["status"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    # -------------------------------------------------------------
    # STEP 11: HIGH-PERFORMANCE ADAPTIVE AI ARCHITECTURE
    # -------------------------------------------------------------
    def save_component_upgrade(self, record: ComponentUpgradeRecord) -> None:
        """Persist or update a component/model upgrade audit record."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO component_upgrades (
                    upgrade_id, component_name, current_version, candidate_version,
                    status, benchmark_accuracy, benchmark_latency_ms, regression_pass_rate,
                    user_approved, approved_by, notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.upgrade_id,
                    record.component_name,
                    record.current_version,
                    record.candidate_version,
                    record.status.value if hasattr(record.status, "value") else str(record.status),
                    record.benchmark_accuracy,
                    record.benchmark_latency_ms,
                    record.regression_pass_rate,
                    1 if record.user_approved else 0,
                    record.approved_by,
                    record.notes,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )

    def get_component_upgrade(self, upgrade_id: str) -> Optional[ComponentUpgradeRecord]:
        """Fetch component upgrade record by ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM component_upgrades WHERE upgrade_id = ?", (upgrade_id,)).fetchone()
            if not row:
                return None
            return ComponentUpgradeRecord(
                upgrade_id=row["upgrade_id"],
                component_name=row["component_name"],
                current_version=row["current_version"],
                candidate_version=row["candidate_version"],
                status=UpgradeStatus(row["status"]),
                benchmark_accuracy=row["benchmark_accuracy"],
                benchmark_latency_ms=row["benchmark_latency_ms"],
                regression_pass_rate=row["regression_pass_rate"],
                user_approved=bool(row["user_approved"]),
                approved_by=row["approved_by"],
                notes=row["notes"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_component_upgrades(self, limit: int = 50) -> List[ComponentUpgradeRecord]:
        """List all component upgrade audit candidates."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM component_upgrades ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
            return [
                ComponentUpgradeRecord(
                    upgrade_id=r["upgrade_id"],
                    component_name=r["component_name"],
                    current_version=r["current_version"],
                    candidate_version=r["candidate_version"],
                    status=UpgradeStatus(r["status"]),
                    benchmark_accuracy=r["benchmark_accuracy"],
                    benchmark_latency_ms=r["benchmark_latency_ms"],
                    regression_pass_rate=r["regression_pass_rate"],
                    user_approved=bool(r["user_approved"]),
                    approved_by=r["approved_by"],
                    notes=r["notes"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                    updated_at=datetime.fromisoformat(r["updated_at"]),
                )
                for r in rows
            ]

    def save_latency_telemetry(self, telemetry: LatencyTelemetry, symbol: Optional[str] = None) -> None:
        """Save a performance and latency telemetry measurement."""
        telem_id = f"tel_{uuid.uuid4().hex[:10]}"
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO performance_telemetry (
                    telemetry_id, symbol, ingestion_latency_ms, normalization_latency_ms,
                    technical_analysis_ms, mtf_synthesis_ms, gemini_reasoning_ms,
                    end_to_end_decision_ms, queue_wait_ms, measured_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    telem_id,
                    symbol,
                    telemetry.ingestion_latency_ms,
                    telemetry.normalization_latency_ms,
                    telemetry.technical_analysis_ms,
                    telemetry.mtf_synthesis_ms,
                    telemetry.gemini_reasoning_ms,
                    telemetry.end_to_end_decision_ms,
                    telemetry.queue_wait_ms,
                    telemetry.measured_at.isoformat(),
                ),
            )

    def get_recent_latency_telemetry(self, limit: int = 20) -> List[LatencyTelemetry]:
        """Fetch the most recent latency telemetry records."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM performance_telemetry ORDER BY measured_at DESC LIMIT ?", (limit,)).fetchall()
            return [
                LatencyTelemetry(
                    ingestion_latency_ms=r["ingestion_latency_ms"],
                    normalization_latency_ms=r["normalization_latency_ms"],
                    technical_analysis_ms=r["technical_analysis_ms"],
                    mtf_synthesis_ms=r["mtf_synthesis_ms"],
                    gemini_reasoning_ms=r["gemini_reasoning_ms"],
                    end_to_end_decision_ms=r["end_to_end_decision_ms"],
                    queue_wait_ms=r["queue_wait_ms"],
                    measured_at=datetime.fromisoformat(r["measured_at"]),
                )
                for r in rows
            ]

    # -------------------------------------------------------------
    # STEP 12: SECURITY, DEVICES, CAPABILITIES, NOTIFICATIONS & AUDIT
    # -------------------------------------------------------------
    def save_trusted_device(self, device: DeviceRecord) -> None:
        """Insert or update a trusted client device record."""
        sql = """
        INSERT OR REPLACE INTO trusted_devices (
            device_id, device_name, device_type, status,
            public_key_or_token_hash, paired_at, last_seen,
            session_token_hash, revoked_at, revocation_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    device.device_id,
                    device.device_name,
                    device.device_type,
                    device.status.value,
                    device.public_key_or_token_hash,
                    device.paired_at.isoformat() if device.paired_at else None,
                    device.last_seen.isoformat() if device.last_seen else None,
                    device.session_token_hash,
                    device.revoked_at.isoformat() if device.revoked_at else None,
                    device.revocation_reason,
                ),
            )

    def get_trusted_device(self, device_id: str) -> Optional[DeviceRecord]:
        """Fetch a trusted device by its unique ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM trusted_devices WHERE device_id = ?", (device_id,)).fetchone()
            if not row:
                return None
            return DeviceRecord(
                device_id=row["device_id"],
                device_name=row["device_name"],
                device_type=row["device_type"],
                status=DeviceStatus(row["status"]),
                public_key_or_token_hash=row["public_key_or_token_hash"],
                paired_at=datetime.fromisoformat(row["paired_at"]) if row["paired_at"] else None,
                last_seen=datetime.fromisoformat(row["last_seen"]) if row["last_seen"] else None,
                session_token_hash=row["session_token_hash"],
                revoked_at=datetime.fromisoformat(row["revoked_at"]) if row["revoked_at"] else None,
                revocation_reason=row["revocation_reason"],
            )

    def list_trusted_devices(self) -> List[DeviceRecord]:
        """Return all known devices."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM trusted_devices ORDER BY last_seen DESC").fetchall()
            return [
                DeviceRecord(
                    device_id=r["device_id"],
                    device_name=r["device_name"],
                    device_type=r["device_type"],
                    status=DeviceStatus(r["status"]),
                    public_key_or_token_hash=r["public_key_or_token_hash"],
                    paired_at=datetime.fromisoformat(r["paired_at"]) if r["paired_at"] else None,
                    last_seen=datetime.fromisoformat(r["last_seen"]) if r["last_seen"] else None,
                    session_token_hash=r["session_token_hash"],
                    revoked_at=datetime.fromisoformat(r["revoked_at"]) if r["revoked_at"] else None,
                    revocation_reason=r["revocation_reason"],
                )
                for r in rows
            ]

    def save_capability_permission(self, rule: PCCapabilityRule) -> None:
        """Insert or update a capability permission rule."""
        sql = """
        INSERT OR REPLACE INTO capability_permissions (
            capability, state, requires_user_approval, is_dangerous,
            last_updated, approved_by, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    rule.capability.value,
                    rule.state.value,
                    1 if rule.requires_user_approval else 0,
                    1 if rule.is_dangerous else 0,
                    rule.last_updated.isoformat(),
                    rule.approved_by,
                    rule.reason,
                ),
            )

    def get_capability_permission(self, capability: PCCapability) -> Optional[PCCapabilityRule]:
        """Fetch a capability permission rule."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM capability_permissions WHERE capability = ?", (capability.value,)).fetchone()
            if not row:
                return None
            return PCCapabilityRule(
                capability=PCCapability(row["capability"]),
                state=CapabilityState(row["state"]),
                requires_user_approval=bool(row["requires_user_approval"]),
                is_dangerous=bool(row["is_dangerous"]),
                last_updated=datetime.fromisoformat(row["last_updated"]),
                approved_by=row["approved_by"],
                reason=row["reason"],
            )

    def list_capability_permissions(self) -> List[PCCapabilityRule]:
        """List all capability permission rules."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM capability_permissions").fetchall()
            return [
                PCCapabilityRule(
                    capability=PCCapability(r["capability"]),
                    state=CapabilityState(r["state"]),
                    requires_user_approval=bool(r["requires_user_approval"]),
                    is_dangerous=bool(r["is_dangerous"]),
                    last_updated=datetime.fromisoformat(r["last_updated"]),
                    approved_by=r["approved_by"],
                    reason=r["reason"],
                )
                for r in rows
            ]

    def save_security_audit_entry(self, entry: SecurityAuditEntry) -> None:
        """Persist an auditable security entry."""
        sql = """
        INSERT OR REPLACE INTO security_audit_logs (
            audit_id, timestamp, capability, action, requesting_component,
            decision, is_user_authorized, authorized_by, device_id, details_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    entry.audit_id,
                    entry.timestamp.isoformat(),
                    entry.capability,
                    entry.action,
                    entry.requesting_component,
                    entry.decision,
                    1 if entry.is_user_authorized else 0,
                    entry.authorized_by,
                    entry.device_id,
                    json.dumps(entry.details),
                ),
            )

    def get_recent_security_audit_entries(self, limit: int = 50) -> List[SecurityAuditEntry]:
        """Retrieve recent security audit entries."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM security_audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [
                SecurityAuditEntry(
                    audit_id=r["audit_id"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    capability=r["capability"],
                    action=r["action"],
                    requesting_component=r["requesting_component"],
                    decision=r["decision"],
                    is_user_authorized=bool(r["is_user_authorized"]),
                    authorized_by=r["authorized_by"],
                    device_id=r["device_id"],
                    details=json.loads(r["details_json"]),
                )
                for r in rows
            ]

    def save_cross_device_notification(self, notif: NotificationItem) -> None:
        """Persist a cross-device notification."""
        sql = """
        INSERT OR REPLACE INTO cross_device_notifications (
            notification_id, title, message, priority, category,
            timestamp, read, delivered_devices_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    notif.notification_id,
                    notif.title,
                    notif.message,
                    notif.priority.value,
                    notif.category,
                    notif.timestamp.isoformat(),
                    1 if notif.read else 0,
                    json.dumps(notif.delivered_devices),
                ),
            )

    def get_cross_device_notifications(self, limit: int = 50) -> List[NotificationItem]:
        """Fetch recent cross-device notifications."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM cross_device_notifications ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [
                NotificationItem(
                    notification_id=r["notification_id"],
                    title=r["title"],
                    message=r["message"],
                    priority=NotificationPriority(r["priority"]),
                    category=r["category"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    read=bool(r["read"]),
                    delivered_devices=json.loads(r["delivered_devices_json"]),
                )
                for r in rows
            ]

    # =========================================================================
    # STEP 13: RESILIENCE, INTEGRITY MANIFEST, BACKUPS, CHECKPOINTS & THREATS
    # =========================================================================

    def save_knowledge_integrity_record(self, record: KnowledgeIntegrityRecord) -> None:
        """Persist or update cryptographic knowledge integrity manifest entry."""
        sql = """
        INSERT OR REPLACE INTO knowledge_integrity_manifest (
            item_id, version_id, sha256_hash, content_length,
            epistemic_state, provenance_hash, last_verified_at, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    record.item_id,
                    record.version_id,
                    record.sha256_hash,
                    record.content_length,
                    record.epistemic_state,
                    record.provenance_hash,
                    record.last_verified_at.isoformat(),
                    record.status.value,
                ),
            )

    def get_knowledge_integrity_record(self, item_id: str) -> Optional[KnowledgeIntegrityRecord]:
        """Fetch cryptographic integrity record for a knowledge item."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM knowledge_integrity_manifest WHERE item_id = ?", (item_id,)).fetchone()
            if not row:
                return None
            return KnowledgeIntegrityRecord(
                item_id=row["item_id"],
                version_id=row["version_id"],
                sha256_hash=row["sha256_hash"],
                content_length=row["content_length"],
                epistemic_state=row["epistemic_state"],
                provenance_hash=row["provenance_hash"],
                last_verified_at=datetime.fromisoformat(row["last_verified_at"]),
                status=KnowledgeProtectionState(row["status"]),
            )

    def save_recovery_backup(self, backup: BackupMetadata) -> None:
        """Persist an isolated backup manifest entry."""
        sql = """
        INSERT OR REPLACE INTO recovery_backups (
            backup_id, created_at, backup_path, sha256_checksum,
            total_items, file_size_bytes, is_verified, retention_tag, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    backup.backup_id,
                    backup.created_at.isoformat(),
                    backup.backup_path,
                    backup.sha256_checksum,
                    backup.total_items,
                    backup.file_size_bytes,
                    1 if backup.is_verified else 0,
                    backup.retention_tag,
                    backup.notes,
                ),
            )

    def get_recovery_backup(self, backup_id: str) -> Optional[BackupMetadata]:
        """Fetch a specific backup manifest record."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM recovery_backups WHERE backup_id = ?", (backup_id,)).fetchone()
            if not row:
                return None
            return BackupMetadata(
                backup_id=row["backup_id"],
                created_at=datetime.fromisoformat(row["created_at"]),
                backup_path=row["backup_path"],
                sha256_checksum=row["sha256_checksum"],
                total_items=row["total_items"],
                file_size_bytes=row["file_size_bytes"],
                is_verified=bool(row["is_verified"]),
                retention_tag=row["retention_tag"],
                notes=row["notes"],
            )

    def list_recovery_backups(self) -> List[BackupMetadata]:
        """List all registered recovery backups."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM recovery_backups ORDER BY created_at DESC").fetchall()
            return [
                BackupMetadata(
                    backup_id=r["backup_id"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                    backup_path=r["backup_path"],
                    sha256_checksum=r["sha256_checksum"],
                    total_items=r["total_items"],
                    file_size_bytes=r["file_size_bytes"],
                    is_verified=bool(r["is_verified"]),
                    retention_tag=r["retention_tag"],
                    notes=r["notes"],
                )
                for r in rows
            ]

    def delete_recovery_backup(self, backup_id: str) -> None:
        """Remove a pruned backup manifest record."""
        with self._db.connection() as conn:
            conn.execute("DELETE FROM recovery_backups WHERE backup_id = ?", (backup_id,))

    def save_recovery_checkpoint(self, checkpoint: RecoveryCheckpoint) -> None:
        """Persist a resilient task recovery checkpoint."""
        sql = """
        INSERT OR REPLACE INTO recovery_checkpoints (
            checkpoint_id, job_type, job_id, serial_number,
            state_payload_json, sha256_hash, created_at, classification, requires_user_approval
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    checkpoint.checkpoint_id,
                    checkpoint.job_type,
                    checkpoint.job_id,
                    checkpoint.serial_number,
                    json.dumps(checkpoint.state_payload),
                    checkpoint.sha256_hash,
                    checkpoint.created_at.isoformat(),
                    checkpoint.classification.value,
                    1 if checkpoint.requires_user_approval else 0,
                ),
            )

    def get_recovery_checkpoint(self, checkpoint_id: str) -> Optional[RecoveryCheckpoint]:
        """Fetch a specific recovery checkpoint."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM recovery_checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)).fetchone()
            if not row:
                return None
            return RecoveryCheckpoint(
                checkpoint_id=row["checkpoint_id"],
                job_type=row["job_type"],
                job_id=row["job_id"],
                serial_number=row["serial_number"],
                state_payload=json.loads(row["state_payload_json"]),
                sha256_hash=row["sha256_hash"],
                created_at=datetime.fromisoformat(row["created_at"]),
                classification=RecoveryClassification(row["classification"]),
                requires_user_approval=bool(row["requires_user_approval"]),
            )

    def save_threat_incident(self, incident: ThreatIncident) -> None:
        """Persist a cyber wall threat or suspicious activity record."""
        sql = """
        INSERT OR REPLACE INTO threat_incidents (
            incident_id, timestamp, threat_level, source,
            target_path_or_resource, description, details_json, action_taken, is_mitigated
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    incident.incident_id,
                    incident.timestamp.isoformat(),
                    incident.threat_level.value,
                    incident.source,
                    incident.target_path_or_resource,
                    incident.description,
                    json.dumps(incident.details),
                    incident.action_taken,
                    1 if incident.is_mitigated else 0,
                ),
            )

    def list_threat_incidents(self, limit: int = 50) -> List[ThreatIncident]:
        """Fetch recent threat incidents."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM threat_incidents ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [
                ThreatIncident(
                    incident_id=r["incident_id"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    threat_level=ThreatLevel(r["threat_level"]),
                    source=r["source"],
                    target_path_or_resource=r["target_path_or_resource"],
                    description=r["description"],
                    details=json.loads(r["details_json"]),
                    action_taken=r["action_taken"],
                    is_mitigated=bool(r["is_mitigated"]),
                )
                for r in rows
            ]

    # -------------------------------------------------------------
    # STEP 14: SEPARATE FILE ARCHITECTURE & MULTI-SOURCE ANALYSIS
    # -------------------------------------------------------------
    def save_normalized_knowledge(self, item: Any) -> None:
        """Persist or update a normalized knowledge item with permanent provenance."""
        sql = """
        INSERT OR REPLACE INTO normalized_knowledge (
            item_id, domain, source_type, source_id, source_document_id, source_session_id,
            source_location, concept_type, title, statement, conditions_json, exceptions_json,
            confidence, authority, authority_priority, version, evidence_references_json,
            tags_json, status, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    item.item_id,
                    item.domain.value if hasattr(item.domain, "value") else str(item.domain),
                    item.source_type.value if hasattr(item.source_type, "value") else str(item.source_type),
                    item.source_id,
                    item.source_document_id,
                    item.source_session_id,
                    item.source_location,
                    item.concept_type,
                    item.title,
                    item.statement,
                    json.dumps(item.conditions),
                    json.dumps(item.exceptions),
                    item.confidence,
                    item.authority.value if hasattr(item.authority, "value") else str(item.authority),
                    item.authority_priority,
                    item.version,
                    json.dumps(item.evidence_references),
                    json.dumps(item.tags),
                    item.status,
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )

    def get_normalized_knowledge(self, item_id: str) -> Optional[Any]:
        """Retrieve a specific normalized knowledge unit by ID."""
        from trading_ai.schemas.trade_logic import NormalizedKnowledgeItem
        from trading_ai.schemas.enums import SourceDomain, SourceType, SourceAuthority
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM normalized_knowledge WHERE item_id = ?", (item_id,)).fetchone()
            if not row:
                return None
            return NormalizedKnowledgeItem(
                item_id=row["item_id"],
                domain=SourceDomain(row["domain"]),
                source_type=SourceType(row["source_type"]),
                source_id=row["source_id"],
                source_document_id=row["source_document_id"],
                source_session_id=row["source_session_id"],
                source_location=row["source_location"],
                concept_type=row["concept_type"],
                title=row["title"],
                statement=row["statement"],
                conditions=json.loads(row["conditions_json"]),
                exceptions=json.loads(row["exceptions_json"]),
                confidence=row["confidence"],
                authority=SourceAuthority(row["authority"]),
                authority_priority=row["authority_priority"],
                version=row["version"],
                evidence_references=json.loads(row["evidence_references_json"]),
                tags=json.loads(row["tags_json"]),
                status=row["status"],
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

    def list_normalized_knowledge(
        self,
        domain: Optional[Any] = None,
        concept_type: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
    ) -> List[Any]:
        """Query normalized knowledge units with optional domain and concept filters."""
        from trading_ai.schemas.trade_logic import NormalizedKnowledgeItem
        from trading_ai.schemas.enums import SourceDomain, SourceType, SourceAuthority
        query = "SELECT * FROM normalized_knowledge WHERE 1=1"
        params: List[Any] = []
        if domain:
            dom_val = domain.value if hasattr(domain, "value") else str(domain)
            query += " AND domain = ?"
            params.append(dom_val)
        if concept_type:
            query += " AND concept_type = ?"
            params.append(concept_type)
        query += " ORDER BY authority_priority ASC, confidence DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            results = []
            for row in rows:
                item = NormalizedKnowledgeItem(
                    item_id=row["item_id"],
                    domain=SourceDomain(row["domain"]),
                    source_type=SourceType(row["source_type"]),
                    source_id=row["source_id"],
                    source_document_id=row["source_document_id"],
                    source_session_id=row["source_session_id"],
                    source_location=row["source_location"],
                    concept_type=row["concept_type"],
                    title=row["title"],
                    statement=row["statement"],
                    conditions=json.loads(row["conditions_json"]),
                    exceptions=json.loads(row["exceptions_json"]),
                    confidence=row["confidence"],
                    authority=SourceAuthority(row["authority"]),
                    authority_priority=row["authority_priority"],
                    version=row["version"],
                    evidence_references=json.loads(row["evidence_references_json"]),
                    tags=json.loads(row["tags_json"]),
                    status=row["status"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )
                if tag and tag not in item.tags:
                    continue
                results.append(item)
            return results

    get_normalized_knowledge = list_normalized_knowledge

    def save_knowledge_relationship(self, rel: Any) -> None:
        """Persist a cross-source knowledge relationship link."""
        sql = """
        INSERT OR REPLACE INTO knowledge_relationships (
            relationship_id, source_item_id, target_item_id, relationship_type, description, created_at
        ) VALUES (?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    rel.relationship_id,
                    rel.source_item_id,
                    rel.target_item_id,
                    rel.relationship_type.value if hasattr(rel.relationship_type, "value") else str(rel.relationship_type),
                    rel.description,
                    rel.created_at.isoformat(),
                ),
            )

    def list_knowledge_relationships(
        self,
        item_id: Optional[str] = None,
        rel_type: Optional[Any] = None,
    ) -> List[Any]:
        """Retrieve cross-source relationship links."""
        from trading_ai.schemas.trade_logic import KnowledgeRelationship
        from trading_ai.schemas.enums import RelationshipType
        query = "SELECT * FROM knowledge_relationships WHERE 1=1"
        params: List[Any] = []
        if item_id:
            query += " AND (source_item_id = ? OR target_item_id = ?)"
            params.extend([item_id, item_id])
        if rel_type:
            rel_val = rel_type.value if hasattr(rel_type, "value") else str(rel_type)
            query += " AND relationship_type = ?"
            params.append(rel_val)
        query += " ORDER BY created_at DESC"

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [
                KnowledgeRelationship(
                    relationship_id=row["relationship_id"],
                    source_item_id=row["source_item_id"],
                    target_item_id=row["target_item_id"],
                    relationship_type=RelationshipType(row["relationship_type"]),
                    description=row["description"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                )
                for row in rows
            ]

    def save_cross_source_confluence(self, conf: Any, proposal_id: Optional[str] = None) -> None:
        """Persist a multi-source confluence evaluation audit record."""
        sql = """
        INSERT OR REPLACE INTO cross_source_confluences (
            confluence_id, symbol, timeframe, decision_state, confluence_score,
            matched_rule_ids_json, conflicting_rule_ids_json, supporting_lesson_ids_json,
            reasons_to_trade_json, reasons_not_to_trade_json, proposal_id, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        matched_ids = [m.item_id if hasattr(m, "item_id") else str(m) for m in conf.matched_rules]
        conflicting_ids = [c.get("item_id", "") if isinstance(c, dict) else str(c) for c in conf.conflicting_rules]
        supporting_ids = [s.get("lesson_id", "") if isinstance(s, dict) else str(s) for s in conf.supporting_lessons]

        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    conf.confluence_id,
                    conf.symbol,
                    conf.timeframe,
                    conf.decision_state.value if hasattr(conf.decision_state, "value") else str(conf.decision_state),
                    conf.confluence_score,
                    json.dumps(matched_ids),
                    json.dumps(conflicting_ids),
                    json.dumps(supporting_ids),
                    json.dumps(conf.reasons_to_trade),
                    json.dumps(conf.reasons_not_to_trade),
                    proposal_id,
                    conf.evaluated_at.isoformat(),
                ),
            )

    # ------------------------------------------------------------------ #
    # Step 15: Plugin Architecture & AI Core Persistence Methods         #
    # ------------------------------------------------------------------ #

    def save_plugin_record(self, plugin_data: Dict[str, Any]) -> None:
        """Persist or update plugin registration in plugin_registry."""
        sql = """
        INSERT OR REPLACE INTO plugin_registry (
            plugin_id, name, version, category, description, author,
            capabilities_json, status, is_enabled, permissions_json,
            config_json, consecutive_failures, health_status, last_error,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    plugin_data["plugin_id"],
                    plugin_data.get("name", ""),
                    plugin_data.get("version", "1.0.0"),
                    plugin_data.get("category", "EXTERNAL_DATA"),
                    plugin_data.get("description", ""),
                    plugin_data.get("author", ""),
                    json.dumps(plugin_data.get("capabilities", [])),
                    plugin_data.get("status", "LOADED"),
                    1 if plugin_data.get("is_enabled", True) else 0,
                    json.dumps(plugin_data.get("permissions", [])),
                    json.dumps(plugin_data.get("config", {})),
                    int(plugin_data.get("consecutive_failures", 0)),
                    plugin_data.get("health_status", "HEALTHY"),
                    plugin_data.get("last_error"),
                    plugin_data.get("created_at", now_str),
                    now_str,
                ),
            )

    def get_plugin_record(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single plugin registration by ID."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM plugin_registry WHERE plugin_id = ?", (plugin_id,)).fetchone()
            if not row:
                return None
            return {
                "plugin_id": row["plugin_id"],
                "name": row["name"],
                "version": row["version"],
                "category": row["category"],
                "description": row["description"],
                "author": row["author"],
                "capabilities": json.loads(row["capabilities_json"]),
                "status": row["status"],
                "is_enabled": bool(row["is_enabled"]),
                "permissions": json.loads(row["permissions_json"]),
                "config": json.loads(row["config_json"]),
                "consecutive_failures": row["consecutive_failures"],
                "health_status": row["health_status"],
                "last_error": row["last_error"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def list_plugin_records(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM plugin_registry ORDER BY name ASC").fetchall()
            return [
                {
                    "plugin_id": row["plugin_id"],
                    "name": row["name"],
                    "version": row["version"],
                    "category": row["category"],
                    "description": row["description"],
                    "author": row["author"],
                    "capabilities": json.loads(row["capabilities_json"]),
                    "status": row["status"],
                    "is_enabled": bool(row["is_enabled"]),
                    "permissions": json.loads(row["permissions_json"]),
                    "config": json.loads(row["config_json"]),
                    "consecutive_failures": row["consecutive_failures"],
                    "health_status": row["health_status"],
                    "last_error": row["last_error"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def save_plugin_audit_log(self, plugin_id: str, action: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Record an immutable plugin security/execution audit event."""
        log_id = f"plog_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO plugin_audit_logs (log_id, plugin_id, action, status, details_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(sql, (log_id, plugin_id, action, status, json.dumps(details or {}), now_str))

    def list_plugin_audit_logs(self, plugin_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent plugin audit log records."""
        query = "SELECT * FROM plugin_audit_logs"
        params: List[Any] = []
        if plugin_id:
            query += " WHERE plugin_id = ?"
            params.append(plugin_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [
                {
                    "log_id": row["log_id"],
                    "plugin_id": row["plugin_id"],
                    "action": row["action"],
                    "status": row["status"],
                    "details": json.loads(row["details_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def save_video_processing_job(self, job_data: Dict[str, Any]) -> None:
        """Persist or update video streaming processing job state."""
        sql = """
        INSERT OR REPLACE INTO video_processing_jobs (
            job_id, file_path, file_size_bytes, chunk_duration,
            total_chunks, completed_chunks, current_chunk, status,
            checkpoint_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    job_data["job_id"],
                    job_data.get("file_path", ""),
                    int(job_data.get("file_size_bytes", 0)),
                    float(job_data.get("chunk_duration", 30.0)),
                    int(job_data.get("total_chunks", 1)),
                    int(job_data.get("completed_chunks", 0)),
                    int(job_data.get("current_chunk", 1)),
                    job_data.get("status", "PENDING"),
                    json.dumps(job_data.get("checkpoint", {})),
                    job_data.get("created_at", now_str),
                    now_str,
                ),
            )

    def get_video_processing_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch video processing job state by job_id."""
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM video_processing_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                return None
            return {
                "job_id": row["job_id"],
                "file_path": row["file_path"],
                "file_size_bytes": row["file_size_bytes"],
                "chunk_duration": row["chunk_duration"],
                "total_chunks": row["total_chunks"],
                "completed_chunks": row["completed_chunks"],
                "current_chunk": row["current_chunk"],
                "status": row["status"],
                "checkpoint": json.loads(row["checkpoint_json"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def list_video_processing_jobs(self) -> List[Dict[str, Any]]:
        """List all video processing jobs."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM video_processing_jobs ORDER BY created_at DESC").fetchall()
            return [
                {
                    "job_id": row["job_id"],
                    "file_path": row["file_path"],
                    "file_size_bytes": row["file_size_bytes"],
                    "chunk_duration": row["chunk_duration"],
                    "total_chunks": row["total_chunks"],
                    "completed_chunks": row["completed_chunks"],
                    "current_chunk": row["current_chunk"],
                    "status": row["status"],
                    "checkpoint": json.loads(row["checkpoint_json"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def save_comprehension_review(self, review_data: Dict[str, Any]) -> None:
        """Save an iterative comprehension review attempt."""
        review_id = review_data.get("review_id") or f"rev_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT OR REPLACE INTO comprehension_reviews (
            review_id, job_id, concept_id, iteration, confidence,
            epistemic_state, review_notes, is_resolved, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    review_id,
                    review_data.get("job_id", ""),
                    review_data.get("concept_id", ""),
                    int(review_data.get("iteration", 1)),
                    float(review_data.get("confidence", 0.0)),
                    review_data.get("epistemic_state", "NEEDS_REVIEW"),
                    review_data.get("review_notes", ""),
                    1 if review_data.get("is_resolved", False) else 0,
                    now_str,
                ),
            )

    def list_comprehension_reviews(self, job_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List comprehension reviews, optionally filtered by job_id."""
        query = "SELECT * FROM comprehension_reviews"
        params: List[Any] = []
        if job_id:
            query += " WHERE job_id = ?"
            params.append(job_id)
        query += " ORDER BY created_at DESC"
        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [
                {
                    "review_id": row["review_id"],
                    "job_id": row["job_id"],
                    "concept_id": row["concept_id"],
                    "iteration": row["iteration"],
                    "confidence": row["confidence"],
                    "epistemic_state": row["epistemic_state"],
                    "review_notes": row["review_notes"],
                    "is_resolved": bool(row["is_resolved"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def save_live_decision_scenario(self, scenario_data: Dict[str, Any]) -> None:
        """Persist a live decision forecast scenario record."""
        scenario_id = scenario_data.get("scenario_id") or f"scen_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT OR REPLACE INTO live_decision_scenarios (
            scenario_id, symbol, timeframe, active_concept, learned_rule_id,
            scenarios_json, confirmation_conditions_json, invalidation_conditions_json,
            decision_state, hindi_explanation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        with self._db.connection() as conn:
            conn.execute(
                sql,
                (
                    scenario_id,
                    scenario_data.get("symbol", ""),
                    scenario_data.get("timeframe", "15m"),
                    scenario_data.get("active_concept", ""),
                    scenario_data.get("learned_rule_id"),
                    json.dumps(scenario_data.get("scenarios", [])),
                    json.dumps(scenario_data.get("confirmation_conditions", [])),
                    json.dumps(scenario_data.get("invalidation_conditions", [])),
                    scenario_data.get("decision_state", "WAIT"),
                    scenario_data.get("hindi_explanation", ""),
                    now_str,
                ),
            )

    def list_live_decision_scenarios(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent live decision scenarios."""
        query = "SELECT * FROM live_decision_scenarios"
        params: List[Any] = []
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [
                {
                    "scenario_id": row["scenario_id"],
                    "symbol": row["symbol"],
                    "timeframe": row["timeframe"],
                    "active_concept": row["active_concept"],
                    "learned_rule_id": row["learned_rule_id"],
                    "scenarios": json.loads(row["scenarios_json"]),
                    "confirmation_conditions": json.loads(row["confirmation_conditions_json"]),
                    "invalidation_conditions": json.loads(row["invalidation_conditions_json"]),
                    "decision_state": row["decision_state"],
                    "hindi_explanation": row["hindi_explanation"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    # =========================================================================
    # Step 16: Cloud Runtime, Event Stream, Notifications & Task Queue
    # =========================================================================

    def save_cloud_runtime_state(self, state: Dict[str, Any]) -> None:
        """Persist current 24×7 cloud runtime state and heartbeat."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO cloud_runtime_state (
                    runtime_id, state, active_workers, uptime_seconds, last_heartbeat, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(runtime_id) DO UPDATE SET
                    state = excluded.state,
                    active_workers = excluded.active_workers,
                    uptime_seconds = excluded.uptime_seconds,
                    last_heartbeat = excluded.last_heartbeat,
                    metadata_json = excluded.metadata_json
                """,
                (
                    state.get("runtime_id", "primary_cloud"),
                    state.get("state", "ONLINE"),
                    int(state.get("active_workers", 4)),
                    float(state.get("uptime_seconds", 0.0)),
                    state.get("last_heartbeat", now_str),
                    json.dumps(state.get("metadata", {})),
                ),
            )

    def get_cloud_runtime_state(self, runtime_id: str = "primary_cloud") -> Optional[Dict[str, Any]]:
        """Retrieve latest persisted cloud runtime state."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM cloud_runtime_state WHERE runtime_id = ?",
                (runtime_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "runtime_id": row["runtime_id"],
                "state": row["state"],
                "active_workers": row["active_workers"],
                "uptime_seconds": row["uptime_seconds"],
                "last_heartbeat": row["last_heartbeat"],
                "metadata": json.loads(row["metadata_json"]),
            }

    def save_stream_event(self, event: Any) -> None:
        """Persist an authoritative stream event."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO event_stream_log (
                    event_id, event_type, priority, symbol, source, confidence,
                    evidence_json, payload_json, deduplication_key, correlation_id,
                    delivery_status, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    delivery_status = excluded.delivery_status
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.priority.value if hasattr(event.priority, "value") else str(event.priority),
                    event.symbol,
                    event.source,
                    float(event.confidence),
                    json.dumps(event.evidence_reference),
                    json.dumps(event.payload),
                    event.deduplication_key,
                    event.correlation_id,
                    event.delivery_status.value if hasattr(event.delivery_status, "value") else str(event.delivery_status),
                    event.timestamp.isoformat() if hasattr(event.timestamp, "isoformat") else str(event.timestamp),
                ),
            )

    def get_stream_events(
        self,
        limit: int = 50,
        min_priority: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query persisted stream events for client replay or auditing."""
        query = "SELECT * FROM event_stream_log"
        clauses = []
        params: List[Any] = []
        if min_priority:
            clauses.append("priority = ?")
            params.append(min_priority)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [
                {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "priority": row["priority"],
                    "symbol": row["symbol"],
                    "source": row["source"],
                    "confidence": row["confidence"],
                    "evidence_reference": json.loads(row["evidence_json"]),
                    "payload": json.loads(row["payload_json"]),
                    "deduplication_key": row["deduplication_key"],
                    "correlation_id": row["correlation_id"],
                    "delivery_status": row["delivery_status"],
                    "timestamp": row["timestamp"],
                }
                for row in rows
            ]

    def save_notification_record(self, record: Any) -> None:
        """Persist a notification delivery record."""
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO notification_records (
                    notification_id, event_id, priority, title, message_hindi,
                    message_english, symbol, channel, delivery_status, privacy_level,
                    deduplication_key, target_device_id, retry_count, last_error,
                    created_at, delivered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(notification_id) DO UPDATE SET
                    delivery_status = excluded.delivery_status,
                    retry_count = excluded.retry_count,
                    last_error = excluded.last_error,
                    delivered_at = excluded.delivered_at
                """,
                (
                    record.notification_id,
                    record.event_id,
                    record.priority.value if hasattr(record.priority, "value") else str(record.priority),
                    record.title,
                    record.message_hindi,
                    record.message_english,
                    record.symbol,
                    record.channel.value if hasattr(record.channel, "value") else str(record.channel),
                    record.delivery_status.value if hasattr(record.delivery_status, "value") else str(record.delivery_status),
                    record.privacy_level.value if hasattr(record.privacy_level, "value") else str(record.privacy_level),
                    record.deduplication_key,
                    record.target_device_id,
                    int(record.retry_count),
                    record.last_error,
                    record.created_at.isoformat() if hasattr(record.created_at, "isoformat") else str(record.created_at),
                    record.delivered_at.isoformat() if record.delivered_at and hasattr(record.delivered_at, "isoformat") else None,
                ),
            )

    def is_notification_duplicate(self, dedup_key: str, within_seconds: int = 3600) -> bool:
        """Check if identical notification was dispatched within cooldown window."""
        if not dedup_key:
            return False
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=within_seconds)).isoformat()
        with self._db.connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM notification_records
                WHERE deduplication_key = ? AND created_at >= ? AND delivery_status IN ('SUCCESS', 'QUEUED')
                LIMIT 1
                """,
                (dedup_key, cutoff),
            ).fetchone()
            return row is not None

    def get_notification_records(
        self,
        limit: int = 50,
        channel: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recent notification records."""
        query = "SELECT * FROM notification_records"
        params: List[Any] = []
        if channel:
            query += " WHERE channel = ?"
            params.append(channel)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._db.connection() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            return [
                {
                    "notification_id": row["notification_id"],
                    "event_id": row["event_id"],
                    "priority": row["priority"],
                    "title": row["title"],
                    "message_hindi": row["message_hindi"],
                    "message_english": row["message_english"],
                    "symbol": row["symbol"],
                    "channel": row["channel"],
                    "delivery_status": row["delivery_status"],
                    "privacy_level": row["privacy_level"],
                    "deduplication_key": row["deduplication_key"],
                    "target_device_id": row["target_device_id"],
                    "retry_count": row["retry_count"],
                    "last_error": row["last_error"],
                    "created_at": row["created_at"],
                    "delivered_at": row["delivered_at"],
                }
                for row in rows
            ]

    def save_user_notification_preferences(self, user_id: str, prefs_dict: Dict[str, Any]) -> None:
        """Save user notification configuration."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO user_notification_preferences (user_id, preferences_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    preferences_json = excluded.preferences_json,
                    updated_at = excluded.updated_at
                """,
                (user_id, json.dumps(prefs_dict), now_str),
            )

    def get_user_notification_preferences(self, user_id: str = "default_user") -> Optional[Dict[str, Any]]:
        """Fetch user notification configuration."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT preferences_json FROM user_notification_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return None
            return json.loads(row["preferences_json"])

    def save_device_push_token(self, device_id: str, push_token: str, platform: str = "web") -> None:
        """Register or update mobile/web push token for a paired device."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO device_push_tokens (device_id, push_token, platform, last_updated)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    push_token = excluded.push_token,
                    platform = excluded.platform,
                    last_updated = excluded.last_updated
                """,
                (device_id, push_token, platform, now_str),
            )

    def get_device_push_token(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Fetch push token for specific device."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM device_push_tokens WHERE device_id = ?",
                (device_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "device_id": row["device_id"],
                "push_token": row["push_token"],
                "platform": row["platform"],
                "last_updated": row["last_updated"],
            }

    def list_device_push_tokens(self) -> List[Dict[str, Any]]:
        """List all active device push tokens."""
        with self._db.connection() as conn:
            rows = conn.execute("SELECT * FROM device_push_tokens").fetchall()
            return [
                {
                    "device_id": row["device_id"],
                    "push_token": row["push_token"],
                    "platform": row["platform"],
                    "last_updated": row["last_updated"],
                }
                for row in rows
            ]

    def save_task_queue_job(self, job_dict: Dict[str, Any]) -> None:
        """Persist a job handled by the C3 persistent task queue."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO persistent_task_queue (
                    job_id, task_type, priority, payload_json, status,
                    retry_count, max_retries, serial_number, created_at, updated_at, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    retry_count = excluded.retry_count,
                    serial_number = excluded.serial_number,
                    updated_at = excluded.updated_at,
                    error_message = excluded.error_message
                """,
                (
                    job_dict["job_id"],
                    job_dict["task_type"],
                    int(job_dict.get("priority", 2)),
                    json.dumps(job_dict.get("payload", {})),
                    job_dict.get("status", "PENDING"),
                    int(job_dict.get("retry_count", 0)),
                    int(job_dict.get("max_retries", 3)),
                    job_dict.get("serial_number"),
                    job_dict.get("created_at", now_str),
                    now_str,
                    job_dict.get("error_message"),
                ),
            )

    def get_task_queue_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Fetch task queue job by ID."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM persistent_task_queue WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if not row:
                return None
            return {
                "job_id": row["job_id"],
                "task_type": row["task_type"],
                "priority": row["priority"],
                "payload": json.loads(row["payload_json"]),
                "status": row["status"],
                "retry_count": row["retry_count"],
                "max_retries": row["max_retries"],
                "serial_number": row["serial_number"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "error_message": row["error_message"],
            }

    def get_pending_task_queue_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch pending jobs ordered by priority (lower number = higher priority)."""
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM persistent_task_queue
                WHERE status IN ('PENDING', 'RETRYING')
                ORDER BY priority ASC, created_at ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "job_id": row["job_id"],
                    "task_type": row["task_type"],
                    "priority": row["priority"],
                    "payload": json.loads(row["payload_json"]),
                    "status": row["status"],
                    "retry_count": row["retry_count"],
                    "max_retries": row["max_retries"],
                    "serial_number": row["serial_number"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "error_message": row["error_message"],
                }
                for row in rows
            ]

    # ========================================================================
    # Step 17: Master Decision Engine & Concept Modules Persistence
    # ========================================================================

    def save_decision_audit(self, audit_dict: Dict[str, Any]) -> str:
        """Persist an immutable decision audit record into SQLite."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO decision_audit_log (
                    decision_id, symbol, timeframe, timestamp, final_state, direction,
                    confluence_score, threshold, concept_results_json, evidence_references_json,
                    rule_ids_json, source_provenance_json, epistemic_state_json, conflicts_json,
                    missing_evidence_json, risk_result_json, entry_zone, invalidation_level,
                    target_level, primary_reason, user_summary_en, user_summary_hi,
                    notification_event_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_id) DO UPDATE SET
                    final_state = excluded.final_state,
                    confluence_score = excluded.confluence_score,
                    concept_results_json = excluded.concept_results_json,
                    conflicts_json = excluded.conflicts_json,
                    user_summary_en = excluded.user_summary_en,
                    user_summary_hi = excluded.user_summary_hi,
                    notification_event_id = excluded.notification_event_id
                """,
                (
                    audit_dict["decision_id"],
                    audit_dict["symbol"],
                    audit_dict.get("timeframe", "15m"),
                    float(audit_dict.get("timestamp", 0.0)),
                    audit_dict["final_state"],
                    audit_dict.get("direction", "NEUTRAL"),
                    float(audit_dict.get("confluence_score", 0.0)),
                    float(audit_dict.get("threshold", 8.0)),
                    audit_dict.get("concept_results_json", "[]"),
                    audit_dict.get("evidence_references_json", "[]"),
                    audit_dict.get("rule_ids_json", "[]"),
                    audit_dict.get("source_provenance_json", "[]"),
                    audit_dict.get("epistemic_state_json", "{}"),
                    audit_dict.get("conflicts_json", "[]"),
                    audit_dict.get("missing_evidence_json", "[]"),
                    audit_dict.get("risk_result_json", "{}"),
                    audit_dict.get("entry_zone"),
                    str(audit_dict.get("invalidation_level", "")) if audit_dict.get("invalidation_level") is not None else None,
                    str(audit_dict.get("target_level", "")) if audit_dict.get("target_level") is not None else None,
                    audit_dict.get("primary_reason"),
                    audit_dict.get("user_summary_en"),
                    audit_dict.get("user_summary_hi"),
                    audit_dict.get("notification_event_id"),
                    audit_dict.get("created_at", now_str),
                ),
            )
        return audit_dict["decision_id"]

    def get_decision_audit(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific decision audit trace by ID."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM decision_audit_log WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
            if not row:
                return None
            return dict(row)

    def list_decision_audits(
        self, symbol: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List decision audit records, optionally filtered by symbol."""
        with self._db.connection() as conn:
            if symbol:
                rows = conn.execute(
                    """
                    SELECT * FROM decision_audit_log
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (symbol.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM decision_audit_log
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def register_concept_module(self, module_dict: Dict[str, Any]) -> None:
        """Register or update a trading concept module definition."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO concept_modules_registry (
                    module_id, concept_name, concept_type, authority_priority,
                    is_enabled, is_approved, data_requirements_json, config_json,
                    version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(module_id) DO UPDATE SET
                    concept_name = excluded.concept_name,
                    concept_type = excluded.concept_type,
                    authority_priority = excluded.authority_priority,
                    is_enabled = excluded.is_enabled,
                    is_approved = excluded.is_approved,
                    data_requirements_json = excluded.data_requirements_json,
                    config_json = excluded.config_json,
                    version = excluded.version,
                    updated_at = excluded.updated_at
                """,
                (
                    module_dict["module_id"],
                    module_dict["concept_name"],
                    module_dict.get("concept_type", "CONCEPT"),
                    int(module_dict.get("authority_priority", 4)),
                    1 if module_dict.get("is_enabled", True) else 0,
                    1 if module_dict.get("is_approved", True) else 0,
                    json.dumps(module_dict.get("data_requirements", [])),
                    json.dumps(module_dict.get("config", {})),
                    module_dict.get("version", "1.0.0"),
                    module_dict.get("created_at", now_str),
                    now_str,
                ),
            )

    def get_concept_modules(self) -> List[Dict[str, Any]]:
        """Retrieve all registered concept modules."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM concept_modules_registry ORDER BY authority_priority ASC, module_id ASC"
            ).fetchall()
            modules = []
            for r in rows:
                m = dict(r)
                m["is_enabled"] = bool(m["is_enabled"])
                m["is_approved"] = bool(m["is_approved"])
                m["data_requirements"] = json.loads(m["data_requirements_json"])
                m["config"] = json.loads(m["config_json"])
                modules.append(m)
            return modules

    def toggle_concept_module(self, module_id: str, is_enabled: bool) -> None:
        """Enable or disable a concept module."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE concept_modules_registry SET is_enabled = ?, updated_at = ? WHERE module_id = ?",
                (1 if is_enabled else 0, now_str, module_id),
            )

    def set_concept_module_approval(self, module_id: str, is_approved: bool) -> None:
        """Update approval status for a newly taught concept module."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE concept_modules_registry SET is_approved = ?, updated_at = ? WHERE module_id = ?",
                (1 if is_approved else 0, now_str, module_id),
            )

    # =========================================================================
    # Step 18: Universal Providers & Connection Configurations
    # =========================================================================

    def save_universal_provider(self, provider: Any) -> None:
        """Persist or update universal provider metadata."""
        now_str = datetime.now(timezone.utc).isoformat()
        p_dict = provider.model_dump(mode="json") if hasattr(provider, "model_dump") else dict(provider)
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO universal_providers
                    (provider_id, provider_name, provider_type, version, status,
                     trading_state, capabilities_json, supported_symbols_json,
                     supported_markets_json, supported_timeframes_json,
                     authentication_type, metadata_json, enabled, approved,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider_id) DO UPDATE SET
                    provider_name = excluded.provider_name,
                    provider_type = excluded.provider_type,
                    version = excluded.version,
                    status = excluded.status,
                    trading_state = excluded.trading_state,
                    capabilities_json = excluded.capabilities_json,
                    supported_symbols_json = excluded.supported_symbols_json,
                    supported_markets_json = excluded.supported_markets_json,
                    supported_timeframes_json = excluded.supported_timeframes_json,
                    authentication_type = excluded.authentication_type,
                    metadata_json = excluded.metadata_json,
                    enabled = excluded.enabled,
                    approved = excluded.approved,
                    updated_at = excluded.updated_at
                """,
                (
                    p_dict["provider_id"],
                    p_dict["provider_name"],
                    p_dict["provider_type"] if isinstance(p_dict["provider_type"], str) else p_dict["provider_type"].value,
                    p_dict.get("version", "1.0.0"),
                    p_dict["status"] if isinstance(p_dict["status"], str) else p_dict["status"].value,
                    p_dict["trading_state"] if isinstance(p_dict["trading_state"], str) else p_dict["trading_state"].value,
                    json.dumps(p_dict.get("capabilities", []), default=str),
                    json.dumps(p_dict.get("supported_symbols", []), default=str),
                    json.dumps(p_dict.get("supported_markets", []), default=str),
                    json.dumps(p_dict.get("supported_timeframes", []), default=str),
                    p_dict.get("authentication_type", "NONE"),
                    json.dumps(p_dict, default=str),
                    1 if p_dict.get("enabled", True) else 0,
                    1 if p_dict.get("approved", False) else 0,
                    p_dict.get("created_at", now_str) if isinstance(p_dict.get("created_at"), str) else (p_dict.get("created_at").isoformat() if p_dict.get("created_at") else now_str),
                    now_str,
                ),
            )

    def get_universal_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve universal provider by ID."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM universal_providers WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()
            if not row:
                return None
            p = dict(row)
            p["capabilities"] = json.loads(p["capabilities_json"])
            p["supported_symbols"] = json.loads(p["supported_symbols_json"])
            p["supported_markets"] = json.loads(p["supported_markets_json"])
            p["supported_timeframes"] = json.loads(p["supported_timeframes_json"])
            p["metadata"] = json.loads(p["metadata_json"])
            p["enabled"] = bool(p["enabled"])
            p["approved"] = bool(p["approved"])
            return p

    def list_universal_providers(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List universal providers, optionally filtered by category."""
        with self._db.connection() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM universal_providers WHERE provider_type = ? ORDER BY provider_id ASC",
                    (category.upper(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM universal_providers ORDER BY provider_type ASC, provider_id ASC"
                ).fetchall()
            out = []
            for r in rows:
                p = dict(r)
                p["capabilities"] = json.loads(p["capabilities_json"])
                p["supported_symbols"] = json.loads(p["supported_symbols_json"])
                p["supported_markets"] = json.loads(p["supported_markets_json"])
                p["supported_timeframes"] = json.loads(p["supported_timeframes_json"])
                p["metadata"] = json.loads(p["metadata_json"])
                p["enabled"] = bool(p["enabled"])
                p["approved"] = bool(p["approved"])
                out.append(p)
            return out

    def toggle_universal_provider(self, provider_id: str, enabled: bool) -> None:
        """Toggle enabled state of universal provider."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE universal_providers SET enabled = ?, updated_at = ? WHERE provider_id = ?",
                (1 if enabled else 0, now_str, provider_id),
            )

    def approve_universal_provider(self, provider_id: str, approved: bool) -> None:
        """Set approval status of universal provider."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE universal_providers SET approved = ?, updated_at = ? WHERE provider_id = ?",
                (1 if approved else 0, now_str, provider_id),
            )

    def update_universal_provider_status(
        self,
        provider_id: str,
        status: str,
        trading_state: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> None:
        """Update provider lifecycle and trading state."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            if trading_state:
                conn.execute(
                    """
                    UPDATE universal_providers
                    SET status = ?, trading_state = ?, updated_at = ?
                    WHERE provider_id = ?
                    """,
                    (status, trading_state, now_str, provider_id),
                )
            else:
                conn.execute(
                    "UPDATE universal_providers SET status = ?, updated_at = ? WHERE provider_id = ?",
                    (status, now_str, provider_id),
                )

    def save_provider_connection_config(
        self,
        provider_id: str,
        config: Dict[str, Any],
        is_encrypted: bool = False,
    ) -> None:
        """Persist non-secret connection config for provider."""
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO provider_connection_configs (provider_id, config_json, is_encrypted, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(provider_id) DO UPDATE SET
                    config_json = excluded.config_json,
                    is_encrypted = excluded.is_encrypted,
                    updated_at = excluded.updated_at
                """,
                (provider_id, json.dumps(config), 1 if is_encrypted else 0, now_str),
            )

    def get_provider_connection_config(self, provider_id: str) -> Dict[str, Any]:
        """Fetch connection config for provider."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM provider_connection_configs WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()
            if not row:
                return {}
            return json.loads(row["config_json"])

    # =========================================================================
    # Step 18: Real Lecture Performance Benchmarking Persistence
    # =========================================================================

    def save_lecture_benchmark(self, record: Any) -> None:
        """Persist real lecture performance benchmark data."""
        now_str = datetime.now(timezone.utc).isoformat()
        r_dict = record.model_dump(mode="json") if hasattr(record, "model_dump") else dict(record)
        with self._db.connection() as conn:
            conn.execute(
                """
                INSERT INTO lecture_benchmarks
                    (lecture_id, file_name, file_hash, file_size_bytes, duration_seconds,
                     current_stage, is_complete, benchmark_data_json, timeline_json,
                     bottlenecks_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(lecture_id) DO UPDATE SET
                    file_name = excluded.file_name,
                    file_hash = excluded.file_hash,
                    file_size_bytes = excluded.file_size_bytes,
                    duration_seconds = excluded.duration_seconds,
                    current_stage = excluded.current_stage,
                    is_complete = excluded.is_complete,
                    benchmark_data_json = excluded.benchmark_data_json,
                    timeline_json = excluded.timeline_json,
                    bottlenecks_json = excluded.bottlenecks_json,
                    updated_at = excluded.updated_at
                """,
                (
                    r_dict["lecture_id"],
                    r_dict.get("file_name", ""),
                    r_dict.get("file_hash", ""),
                    int(r_dict.get("file_size_bytes", 0)),
                    float(r_dict.get("duration_seconds", 0.0)),
                    r_dict.get("current_stage", "QUEUED"),
                    1 if r_dict.get("is_complete", False) else 0,
                    json.dumps(r_dict, default=str),
                    json.dumps(r_dict.get("timeline", []), default=str),
                    json.dumps(r_dict.get("bottlenecks", {}), default=str),
                    r_dict.get("created_at", now_str) if isinstance(r_dict.get("created_at"), str) else (r_dict.get("created_at").isoformat() if r_dict.get("created_at") else now_str),
                    now_str,
                ),
            )

    def get_lecture_benchmark(self, lecture_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve lecture benchmark by lecture_id."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT * FROM lecture_benchmarks WHERE lecture_id = ?",
                (lecture_id,),
            ).fetchone()
            if not row:
                return None
            b = dict(row)
            data = json.loads(b["benchmark_data_json"])
            data["timeline"] = json.loads(b["timeline_json"])
            data["bottlenecks"] = json.loads(b["bottlenecks_json"])
            data["is_complete"] = bool(b["is_complete"])
            return data

    def list_lecture_benchmarks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List lecture benchmarks ordered by recency."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM lecture_benchmarks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            out = []
            for r in rows:
                b = dict(r)
                data = json.loads(b["benchmark_data_json"])
                data["timeline"] = json.loads(b["timeline_json"])
                data["bottlenecks"] = json.loads(b["bottlenecks_json"])
                data["is_complete"] = bool(b["is_complete"])
                out.append(data)
            return out

StorageRepository = Repository
