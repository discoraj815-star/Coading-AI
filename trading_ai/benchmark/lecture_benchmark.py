"""Step 18: Real Lecture Performance Benchmarking Engine.

Provides deep runtime instrumentation around the lecture-processing pipeline:
- Stage timing and granular metric aggregation
- Processing counts and token cost estimation
- Knowledge extraction and quality tracking
- Persistent milestone timeline
- Automatic bottleneck percentage calculation
- Full SQLite WAL persistence
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import logging
import time
from typing import Any, Dict, List, Optional

from trading_ai.schemas.benchmark import (
    BenchmarkStage,
    LectureBenchmarkRecord,
    TimelineCheckpoint,
)
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class LectureBenchmarkEngine:
    """Central instrumentation engine for real user lecture processing benchmarks."""

    def __init__(self, repository: Optional[Repository] = None):
        self._repo = repository
        self._active_benchmarks: Dict[str, LectureBenchmarkRecord] = {}

    def start_benchmark(
        self,
        lecture_id: str,
        file_name: str = "",
        file_hash: str = "",
        file_size_bytes: int = 0,
        duration_seconds: float = 0.0,
        resolution: str = "1280x720",
        fps: float = 30.0,
    ) -> LectureBenchmarkRecord:
        """Initialize and persist a benchmark session for a lecture."""
        record = LectureBenchmarkRecord(
            lecture_id=lecture_id,
            file_name=file_name,
            file_hash=file_hash,
            file_size_bytes=file_size_bytes,
            duration_seconds=duration_seconds,
            resolution=resolution,
            fps=fps,
            current_stage=BenchmarkStage.QUEUED,
        )
        # Add initial timeline milestone
        record.timeline.append(
            TimelineCheckpoint(
                stage="QUEUED",
                description="Lecture ingested and benchmark session initialized.",
                elapsed_seconds=0.0,
            )
        )
        self._active_benchmarks[lecture_id] = record
        if self._repo:
            self._repo.save_lecture_benchmark(record)
        return record

    def get_benchmark(self, lecture_id: str) -> Optional[LectureBenchmarkRecord]:
        """Fetch active benchmark in memory or from database."""
        if lecture_id in self._active_benchmarks:
            return self._active_benchmarks[lecture_id]
        if self._repo:
            data = self._repo.get_lecture_benchmark(lecture_id)
            if data:
                return LectureBenchmarkRecord(**data)
        return None

    def record_checkpoint(self, lecture_id: str, stage: str, description: str) -> None:
        """Append an immutable persistent milestone to the lecture timeline."""
        record = self.get_benchmark(lecture_id)
        if not record:
            return
        elapsed = record.metrics.get("total_processing_time", 0.0)
        checkpoint = TimelineCheckpoint(
            stage=stage,
            description=description,
            elapsed_seconds=round(elapsed, 2),
        )
        record.timeline.append(checkpoint)
        self._active_benchmarks[lecture_id] = record
        if self._repo:
            self._repo.save_lecture_benchmark(record)

    @contextmanager
    def benchmark_stage(self, lecture_id: str, stage: BenchmarkStage):
        """Context manager to measure exact duration of a processing stage."""
        record = self.get_benchmark(lecture_id)
        t_start = time.time()
        stage_name = stage.value.lower()
        if record:
            record.current_stage = stage
            record.timings[f"{stage_name}_start"] = t_start
        try:
            yield
        finally:
            t_end = time.time()
            elapsed = t_end - t_start
            if record:
                record.timings[f"{stage_name}_end"] = t_end
                metric_key = f"{stage_name}_processing_time" if f"{stage_name}_processing_time" in record.metrics else f"{stage_name}_time"
                record.metrics[metric_key] = record.metrics.get(metric_key, 0.0) + elapsed
                record.metrics["total_processing_time"] = record.metrics.get("total_processing_time", 0.0) + elapsed
                
                # Checkpoint
                record.timeline.append(
                    TimelineCheckpoint(
                        stage=stage.value,
                        description=f"{stage.value} processing finished in {elapsed:.2f}s.",
                        elapsed_seconds=round(record.metrics["total_processing_time"], 2),
                    )
                )
                self._compute_bottlenecks(record)
                self._active_benchmarks[lecture_id] = record
                if self._repo:
                    self._repo.save_lecture_benchmark(record)

    def update_counts(
        self,
        lecture_id: str,
        frames_sampled: int = 0,
        frames_processed: int = 0,
        audio_segments: int = 0,
        visual_segments: int = 0,
        aligned_segments: int = 0,
        gemini_calls: int = 0,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> None:
        """Update granular resource consumption counts."""
        record = self.get_benchmark(lecture_id)
        if not record:
            return
        record.counts["sampled_frames"] += frames_sampled
        record.counts["processed_frames"] += frames_processed
        record.counts["audio_segments"] += audio_segments
        record.counts["visual_segments"] += visual_segments
        record.counts["aligned_segments"] += aligned_segments
        record.counts["gemini_calls"] += gemini_calls
        record.counts["tokens_input"] += tokens_input
        record.counts["tokens_output"] += tokens_output
        
        # Approximate cost calculation for Flash 3.8 ($0.15/1M input, $0.60/1M output)
        cost_in = (record.counts["tokens_input"] / 1_000_000.0) * 0.15
        cost_out = (record.counts["tokens_output"] / 1_000_000.0) * 0.60
        record.estimated_cost_usd = round(cost_in + cost_out, 6)

        self._active_benchmarks[lecture_id] = record
        if self._repo:
            self._repo.save_lecture_benchmark(record)

    def update_knowledge_metrics(
        self,
        lecture_id: str,
        concepts: int = 0,
        rules: int = 0,
        evidence: int = 0,
        exceptions: int = 0,
        examples: int = 0,
        uncertainties: int = 0,
        needs_review: int = 0,
    ) -> None:
        """Update extracted knowledge statistics."""
        record = self.get_benchmark(lecture_id)
        if not record:
            return
        record.knowledge_metrics["concepts_extracted"] += concepts
        record.knowledge_metrics["rules_extracted"] += rules
        record.knowledge_metrics["evidence_items"] += evidence
        record.knowledge_metrics["exceptions"] += exceptions
        record.knowledge_metrics["examples"] += examples
        record.knowledge_metrics["uncertainties"] += uncertainties
        record.knowledge_metrics["needs_review_items"] += needs_review
        self._active_benchmarks[lecture_id] = record
        if self._repo:
            self._repo.save_lecture_benchmark(record)

    def finalize_benchmark(self, lecture_id: str, is_complete: bool = True) -> LectureBenchmarkRecord:
        """Finalize benchmark, calculate bottlenecks, and record completion."""
        record = self.get_benchmark(lecture_id)
        if not record:
            raise ValueError(f"Benchmark for lecture '{lecture_id}' not found.")
        record.is_complete = is_complete
        record.current_stage = BenchmarkStage.COMPLETE if is_complete else BenchmarkStage.FAILED
        self._compute_bottlenecks(record)
        record.timeline.append(
            TimelineCheckpoint(
                stage="COMPLETE" if is_complete else "FAILED",
                description="Lecture processing benchmark finalized.",
                elapsed_seconds=round(record.metrics.get("total_processing_time", 0.0), 2),
            )
        )
        self._active_benchmarks[lecture_id] = record
        if self._repo:
            self._repo.save_lecture_benchmark(record)
        return record

    def _compute_bottlenecks(self, record: LectureBenchmarkRecord) -> None:
        """Automatically identify and calculate percentage breakdown of stage bottlenecks."""
        total = record.metrics.get("total_processing_time", 0.0)
        if total <= 0.001:
            record.bottlenecks = {}
            return

        stages = {
            "VISUAL_ANALYSIS": record.metrics.get("visual_processing_time", 0.0),
            "AUDIO_PROCESSING": record.metrics.get("audio_processing_time", 0.0),
            "ALIGNMENT": record.metrics.get("alignment_time", 0.0),
            "KNOWLEDGE_EXTRACTION": record.metrics.get("knowledge_time", 0.0),
            "DATABASE_PERSISTENCE": record.metrics.get("database_time", 0.0),
            "GEMINI_REASONING": record.metrics.get("gemini_time", 0.0),
        }
        breakdown = {}
        for name, dur in stages.items():
            pct = (dur / total) * 100.0
            if pct > 0.0:
                breakdown[name] = round(pct, 1)
        record.bottlenecks = breakdown
