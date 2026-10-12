"""Pipeline orchestrator coordinating multimodal lecture analysis from ingestion to knowledge commit."""

import logging
import uuid
from typing import Dict, List, Optional
from trading_ai.ai import (
    AlignmentEngine,
    AudioProcessor,
    GeminiClient,
    NotesGenerator,
    ReasoningEngine,
    VisualProcessor,
)
from trading_ai.core import ModelRouter, Settings, TokenManager, get_settings
from trading_ai.core.context_manager import ContextManager
from trading_ai.core.memory_retriever import _text_to_mock_embedding
from trading_ai.core.model_router import ModelRole
from trading_ai.media import MediaIngestionService, VideoProcessor
from trading_ai.schemas import (
    BackgroundJobState,
    ContextItem,
    ContextItemSource,
    ErrorSeverity,
    JobMetadata,
    JobStatus,
    LectureNotes,
    MemoryEmbedding,
    PipelineStage,
    TaskUnitStatus,
)
from trading_ai.storage import DatabaseManager, Repository
from trading_ai.supervisor import CommitBarrier

logger = logging.getLogger(__name__)


class LecturePipelineOrchestrator:
    """End-to-end multimodal pipeline orchestrator.

    Pipeline:
    INGESTION -> MEDIA ANALYSIS -> AUDIO -> VISUAL -> TIMELINE ->
    ALIGNMENT -> EVIDENCE -> REASONING -> VALIDATION ->
    KNOWLEDGE COMMIT (via Supervisor CommitBarrier) -> STORAGE -> NOTES
    """

    def __init__(
        self,
        repository: Optional[Repository] = None,
        settings: Optional[Settings] = None,
        gemini_client: Optional[GeminiClient] = None,
    ):
        self._settings = settings or get_settings()
        self._repo = repository or Repository(DatabaseManager(self._settings.db_path))
        self._token_manager = TokenManager(self._settings)
        self._router = ModelRouter(self._settings)
        self._ai = gemini_client or GeminiClient(
            settings=self._settings,
            model_router=self._router,
            token_manager=self._token_manager,
            repository=self._repo,
        )

        # Stage services
        self._ingestion = MediaIngestionService(self._repo, self._settings)
        self._video_proc = VideoProcessor(self._settings)
        self._audio_proc = AudioProcessor(self._ai, self._repo)
        self._visual_proc = VisualProcessor(self._ai, self._repo)
        self._alignment_engine = AlignmentEngine(self._ai, self._repo)
        self._reasoning_engine = ReasoningEngine(self._ai, self._repo)
        self._notes_generator = NotesGenerator(self._ai, self._repo)
        self._barrier = CommitBarrier(self._repo)

    @property
    def repository(self) -> Repository:
        return self._repo

    def run_pipeline(
        self,
        source_media_path: str,
        filename: Optional[str] = None,
        job_id: Optional[str] = None,
        chunk_duration: float = 30.0,
    ) -> JobMetadata:
        """Execute the complete multimodal pipeline synchronously or resume from failure."""
        # 1. Ingestion
        job = self._ingestion.ingest_file(
            source_path=source_media_path,
            filename=filename,
            job_id=job_id,
        )
        job_id = job.job_id

        # Initialize or retrieve BackgroundJobState
        bg_job = self._repo.get_background_job(job_id)
        if not bg_job:
            bg_job = BackgroundJobState(
                job_id=job_id,
                source_id=job.file_hash or job_id,
                status=JobStatus.INGESTING,
                progress=5.0,
                current_stage=PipelineStage.INGESTION,
                total_units=1,
                completed_units=0,
                failed_units=0,
            )
            self._repo.save_background_job(bg_job)

        try:
            job.status = JobStatus.PROCESSING
            job.current_stage = PipelineStage.MEDIA_ANALYSIS
            self._repo.save_job(job)

            bg_job.status = JobStatus.PROCESSING
            bg_job.current_stage = PipelineStage.MEDIA_ANALYSIS
            self._repo.save_background_job(bg_job)

            # 2. Keyframes & Audio extraction (Adaptive sampling)
            keyframes = self._video_proc.extract_keyframes(
                video_path=job.file_path,
                job_id=job_id,
                sample_interval_seconds=min(10.0, chunk_duration / 2),
            )
            audio_path = self._video_proc.extract_audio_track(job.file_path, job_id)

            duration = max(30.0, job.duration)

            # 3. Timeline Chunking & TaskUnits (Long Video Strategy & Supervisor)
            num_chunks = max(1, int(duration // chunk_duration))
            bg_job.total_units = num_chunks
            self._repo.save_background_job(bg_job)

            all_alignments = self._repo.get_alignments(job_id)
            all_evidence = self._repo.get_evidence(job_id)
            all_knowledge = self._repo.get_knowledge(job_id, only_committed=False)

            barrier_state = self._barrier.get_barrier_state(job_id)

            for i in range(num_chunks):
                chunk_serial = i + 1
                chunk_start = i * chunk_duration
                chunk_end = min(duration, (i + 1) * chunk_duration)

                # Check if job was paused or cancelled by user
                current_bg = self._repo.get_background_job(job_id)
                if current_bg and current_bg.status in (JobStatus.PAUSED, JobStatus.CANCELLED):
                    logger.info(f"Job {job_id} is {current_bg.status.value}. Halting chunk loop.")
                    job.status = current_bg.status
                    self._repo.save_job(job)
                    return job

                # Check if this chunk was already committed in a prior run (Resumability)
                if barrier_state.highest_committed_serial >= chunk_serial:
                    logger.info(
                        f"Chunk {chunk_serial}/{num_chunks} already committed. Skipping reprocessing (Resuming)."
                    )
                    continue

                # Register unit with strictly increasing serial number
                task_unit = self._barrier.register_task_unit(
                    job_id=job_id,
                    stage=PipelineStage.TIMELINE,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    payload={"chunk_index": chunk_serial, "duration": chunk_duration},
                )
                serial = task_unit.serial_number
                job.serial_number = serial

                try:
                    # Audio Processing for this chunk
                    job.current_stage = PipelineStage.AUDIO_PROCESSING
                    self._repo.save_job(job)

                    bg_job.current_stage = PipelineStage.AUDIO
                    bg_job.current_serial = serial
                    self._repo.save_background_job(bg_job)

                    chunk_transcripts = self._audio_proc.process_audio(
                        job_id=job_id,
                        audio_file_path=audio_path,
                        duration=chunk_duration,
                        serial_number=serial,
                    )

                    # Visual Processing for this chunk
                    job.current_stage = PipelineStage.VISUAL_PROCESSING
                    self._repo.save_job(job)

                    bg_job.current_stage = PipelineStage.VISUAL
                    self._repo.save_background_job(bg_job)

                    chunk_keyframes = [
                        kf for kf in keyframes
                        if chunk_start <= kf.get("timestamp", 0.0) <= chunk_end
                    ] or keyframes[:2]

                    chunk_visuals = self._visual_proc.process_keyframes(
                        job_id=job_id,
                        keyframes=chunk_keyframes,
                        serial_number=serial,
                    )

                    # Audio/Visual Alignment
                    job.current_stage = PipelineStage.AUDIO_VISUAL_ALIGNMENT
                    self._repo.save_job(job)

                    bg_job.current_stage = PipelineStage.ALIGNMENT
                    self._repo.save_background_job(bg_job)

                    chunk_alignments = self._alignment_engine.align_segments(
                        job_id=job_id,
                        transcripts=chunk_transcripts,
                        visual_events=chunk_visuals,
                        serial_number=serial,
                    )
                    all_alignments.extend(chunk_alignments)

                    # Gemini Reasoning & Epistemic Validation
                    job.current_stage = PipelineStage.GEMINI_REASONING
                    self._repo.save_job(job)

                    bg_job.current_stage = PipelineStage.REASONING
                    self._repo.save_background_job(bg_job)

                    chunk_ev, chunk_k = self._reasoning_engine.reason_and_validate(
                        job_id=job_id,
                        alignments=chunk_alignments,
                        serial_number=serial,
                    )
                    all_evidence.extend(chunk_ev)
                    all_knowledge.extend(chunk_k)

                    # Commit Barrier: ordered commit or buffer if predecessor failed
                    committed_serials = self._barrier.complete_task_unit(
                        job_id=job_id,
                        serial_number=serial,
                        result={"status": "success", "alignments": len(chunk_alignments)},
                        knowledge_items=chunk_k,
                    )
                    logger.info(f"Unit {serial} processed. Newly committed serials: {committed_serials}")

                    # Step 2: Update ContextManager and save memory embeddings for committed items
                    if committed_serials:
                        ctx_mgr = ContextManager(
                            job_id=job_id,
                            repository=self._repo,
                            token_manager=self._token_manager,
                            settings=self._settings,
                        )
                        ctx_items = [
                            ContextItem(
                                item_id=f"ctx_{k.item_id}",
                                source_type=ContextItemSource.KNOWLEDGE,
                                text_content=f"[{k.item_type.value.upper()}] {k.title}: {k.statement}",
                                estimated_tokens=self._token_manager.estimate_text_tokens(k.statement),
                                serial_number=serial,
                                start_time=k.start_time,
                                end_time=k.end_time,
                                importance_score=k.confidence,
                                metadata={"source_id": k.item_id},
                            )
                            for k in chunk_k
                        ]
                        ctx_mgr.add_items(ctx_items)
                        ctx_mgr.advance_serial(new_serial=serial)

                        for k in chunk_k:
                            emb_text = f"{k.title}: {k.statement}"
                            emb_record = MemoryEmbedding(
                                embedding_id=f"emb_{uuid.uuid4().hex[:12]}",
                                job_id=job_id,
                                source_type="knowledge",
                                source_id=k.item_id,
                                text_content=emb_text,
                                embedding=_text_to_mock_embedding(emb_text),
                                model_name=self._router.get_model(ModelRole.EMBEDDING),
                                importance_score=k.confidence,
                            )
                            self._repo.save_memory_embedding(emb_record)

                        # Update background job progress
                        barrier_state = self._barrier.get_barrier_state(job_id)
                        bg_job.completed_units = barrier_state.highest_committed_serial
                        bg_job.progress = round(
                            min(95.0, (bg_job.completed_units / max(1, num_chunks)) * 90.0 + 5.0),
                            2,
                        )
                        self._repo.save_background_job(bg_job)

                except Exception as chunk_err:
                    logger.error(f"Error processing unit {serial} of job {job_id}: {chunk_err}")
                    self._barrier.fail_task_unit(
                        job_id=job_id,
                        serial_number=serial,
                        error_message=str(chunk_err),
                        severity=ErrorSeverity.RECOVERABLE,
                    )
                    job.retry_count += 1
                    job.last_error = str(chunk_err)
                    self._repo.save_job(job)

                    bg_job.failed_units += 1
                    bg_job.status = JobStatus.FAILED
                    bg_job.error = str(chunk_err)
                    self._repo.save_background_job(bg_job)
                    # Stop processing subsequent chunks until error is resolved
                    break

            # 4. Synthesize 11-Section Lecture Notes
            job.current_stage = PipelineStage.NOTES_GENERATION
            self._repo.save_job(job)

            bg_job.current_stage = PipelineStage.NOTES
            self._repo.save_background_job(bg_job)

            notes = self._notes_generator.generate_notes(
                job_id=job_id,
                alignments=all_alignments,
                knowledge_items=all_knowledge,
                evidence_traces=all_evidence,
                serial_number=job.serial_number,
            )

            # Finalize Job
            barrier_state = self._barrier.get_barrier_state(job_id)
            if barrier_state.blocked_on_serial is None:
                job.status = JobStatus.COMPLETED
                job.current_stage = PipelineStage.COMPLETED

                bg_job.status = JobStatus.COMPLETED
                bg_job.current_stage = PipelineStage.COMPLETED
                bg_job.progress = 100.0
                bg_job.completed_units = num_chunks
            else:
                job.status = JobStatus.FAILED
                job.current_stage = PipelineStage.GEMINI_REASONING
                job.last_error = f"Blocked on serial {barrier_state.blocked_on_serial}"

                bg_job.status = JobStatus.FAILED
                bg_job.error = job.last_error

            self._repo.save_job(job)
            self._repo.save_background_job(bg_job)
            return job

        except Exception as e:
            logger.exception(f"Pipeline failure for job {job_id}: {e}")
            job.status = JobStatus.FAILED
            job.last_error = str(e)
            job.retry_count += 1
            self._repo.save_job(job)

            if bg_job:
                bg_job.status = JobStatus.FAILED
                bg_job.error = str(e)
                self._repo.save_background_job(bg_job)
            return job

    def pause_job(self, job_id: str) -> Optional[BackgroundJobState]:
        """Request pause for an active background job."""
        bg_job = self._repo.get_background_job(job_id)
        if not bg_job:
            return None
        bg_job.status = JobStatus.PAUSED
        self._repo.save_background_job(bg_job)
        job = self._repo.get_job(job_id)
        if job:
            job.status = JobStatus.PAUSED
            self._repo.save_job(job)
        return bg_job

    def resume_job(self, job_id: str) -> Optional[BackgroundJobState]:
        """Resume a paused or interrupted background job from its last committed chunk."""
        bg_job = self._repo.get_background_job(job_id)
        if not bg_job:
            return None
        bg_job.status = JobStatus.PROCESSING
        self._repo.save_background_job(bg_job)
        job = self._repo.get_job(job_id)
        if job:
            job.status = JobStatus.PROCESSING
            self._repo.save_job(job)
        return bg_job

    def cancel_job(self, job_id: str) -> Optional[BackgroundJobState]:
        """Cancel an in-progress or queued background job."""
        bg_job = self._repo.get_background_job(job_id)
        if not bg_job:
            return None
        bg_job.status = JobStatus.CANCELLED
        self._repo.save_background_job(bg_job)
        job = self._repo.get_job(job_id)
        if job:
            job.status = JobStatus.CANCELLED
            self._repo.save_job(job)
        return bg_job

    def get_background_job_state(self, job_id: str) -> Optional[BackgroundJobState]:
        """Retrieve background job tracking state."""
        return self._repo.get_background_job(job_id)
