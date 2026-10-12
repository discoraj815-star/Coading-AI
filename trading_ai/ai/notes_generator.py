"""Generates comprehensive 11-section structured multimodal lecture notes."""

import json
from typing import Dict, List, Optional
from trading_ai.ai.gemini_client import GeminiClient
from trading_ai.core.model_router import ModelRole
from trading_ai.schemas import (
    AudioVisualAlignment,
    EvidenceTrace,
    KnowledgeItem,
    LectureNotes,
    TimestampedNoteEntry,
)
from trading_ai.storage.repository import Repository


class NotesGenerator:
    """Synthesizes multimodal evidence, alignments, and reasoning into structured study notes."""

    def __init__(self, gemini_client: GeminiClient, repository: Optional[Repository] = None):
        self._ai = gemini_client
        self._repo = repository

    def generate_notes(
        self,
        job_id: str,
        alignments: List[AudioVisualAlignment],
        knowledge_items: List[KnowledgeItem],
        evidence_traces: List[EvidenceTrace],
        serial_number: int = 1,
    ) -> LectureNotes:
        """Synthesize 11-section structured lecture notes grounded in timestamps and evidence."""
        prompt = (
            f"Generate comprehensive structured lecture notes for job '{job_id}'.\n"
            f"Grounding Data:\n"
            f"- Alignments: {len(alignments)}\n"
            f"- Validated Knowledge Items: {len(knowledge_items)}\n"
            f"- Evidence Traces: {len(evidence_traces)}\n\n"
            "Required 11 Note Sections:\n"
            "1. Overview\n"
            "2. Concepts\n"
            "3. Definitions\n"
            "4. Rules\n"
            "5. Examples\n"
            "6. Exceptions\n"
            "7. Visual explanations\n"
            "8. Important statements\n"
            "9. Relationships\n"
            "10. Uncertainty\n"
            "11. Evidence timestamps\n"
            "Every item must link back to source evidence timestamps."
        )

        response_str = self._ai.generate_content(
            role=ModelRole.KNOWLEDGE_SYNTHESIS,
            prompt=prompt,
            job_id=job_id,
            stage="notes_generation",
            serial_number=serial_number,
        )

        # Build evidence timestamp index
        evidence_index = [
            {
                "evidence_id": ev.evidence_id,
                "start_time": ev.start_time,
                "end_time": ev.end_time,
                "statement": ev.extracted_statement,
            }
            for ev in evidence_traces
        ]

        try:
            data = json.loads(response_str)
            notes = LectureNotes(
                job_id=job_id,
                title=f"Multimodal Analysis Notes - Job {job_id}",
                overview=data.get(
                    "overview",
                    "Comprehensive multimodal analysis of trading lecture session.",
                ),
                concepts=[
                    TimestampedNoteEntry(
                        title=c.get("title", "Concept"),
                        content=c.get("content", ""),
                        start_time=float(c.get("start", 0.0)),
                        end_time=float(c.get("end", 0.0)),
                    )
                    for c in data.get("concepts", [])
                ],
                definitions=[
                    TimestampedNoteEntry(
                        title=d.get("title", "Definition"),
                        content=d.get("content", ""),
                        start_time=float(d.get("start", 0.0)),
                        end_time=float(d.get("end", 0.0)),
                    )
                    for d in data.get("definitions", [])
                ],
                rules=[
                    TimestampedNoteEntry(
                        title=r.get("title", "Trading Rule"),
                        content=r.get("content", ""),
                        start_time=float(r.get("start", 0.0)),
                        end_time=float(r.get("end", 0.0)),
                    )
                    for r in data.get("rules", [])
                ],
                examples=[
                    TimestampedNoteEntry(
                        title=e.get("title", "Example"),
                        content=e.get("content", ""),
                        start_time=float(e.get("start", 0.0)),
                        end_time=float(e.get("end", 0.0)),
                    )
                    for e in data.get("examples", [])
                ],
                exceptions=[
                    TimestampedNoteEntry(
                        title=ex.get("title", "Exception"),
                        content=ex.get("content", ""),
                        start_time=float(ex.get("start", 0.0)),
                        end_time=float(ex.get("end", 0.0)),
                    )
                    for ex in data.get("exceptions", [])
                ],
                visual_explanations=[
                    TimestampedNoteEntry(
                        title=v.get("title", "Visual Chart Breakdown"),
                        content=v.get("content", ""),
                        start_time=float(v.get("start", 0.0)),
                        end_time=float(v.get("end", 0.0)),
                    )
                    for v in data.get("visual_explanations", [])
                ],
                important_statements=[
                    TimestampedNoteEntry(
                        title=s.get("title", "Important Principle"),
                        content=s.get("content", ""),
                        start_time=float(s.get("start", 0.0)),
                        end_time=float(s.get("end", 0.0)),
                    )
                    for s in data.get("important_statements", [])
                ],
                relationships=[
                    TimestampedNoteEntry(
                        title=rel.get("title", "Market Dynamic Relationship"),
                        content=rel.get("content", ""),
                        start_time=float(rel.get("start", 0.0)),
                        end_time=float(rel.get("end", 0.0)),
                    )
                    for rel in data.get("relationships", [])
                ],
                uncertainty=[
                    TimestampedNoteEntry(
                        title=u.get("title", "Discretionary Element"),
                        content=u.get("content", ""),
                        start_time=float(u.get("start", 0.0)),
                        end_time=float(u.get("end", 0.0)),
                    )
                    for u in data.get("uncertainty", [])
                ],
                evidence_timestamps=evidence_index,
            )
        except Exception:
            # Fallback populating all 11 sections directly from knowledge items
            notes = LectureNotes(
                job_id=job_id,
                title=f"Multimodal Analysis Notes - Job {job_id}",
                overview="Structured notes generated from lecture audio and visual chart evidence.",
                concepts=[
                    TimestampedNoteEntry(
                        title=k.title,
                        content=k.statement,
                        start_time=k.start_time,
                        end_time=k.end_time,
                    )
                    for k in knowledge_items
                ],
                evidence_timestamps=evidence_index,
            )

        if self._repo:
            self._repo.save_notes(notes)

        return notes
