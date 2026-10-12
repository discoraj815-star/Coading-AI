"""Reasoning Brain separating Perception != Reasoning != Memory and validating evidence."""

import json
import uuid
from typing import List, Optional, Tuple
from trading_ai.ai.gemini_client import GeminiClient
from trading_ai.core.model_router import ModelRole
from trading_ai.schemas import (
    AlignmentRelationship,
    AudioVisualAlignment,
    EvidenceTrace,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    TranscriptSegment,
    VisualEvent,
)
from trading_ai.storage.repository import Repository


class ReasoningEngine:
    """Validates epistemic consistency between audio and visual evidence before knowledge commit."""

    def __init__(self, gemini_client: GeminiClient, repository: Optional[Repository] = None):
        self._ai = gemini_client
        self._repo = repository

    def reason_and_validate(
        self,
        job_id: str,
        alignments: List[AudioVisualAlignment],
        serial_number: int = 1,
    ) -> Tuple[List[EvidenceTrace], List[KnowledgeItem]]:
        """Subject aligned evidence to Gemini reasoning to extract validated knowledge items."""
        if not alignments:
            return [], []

        evidence_traces: List[EvidenceTrace] = []
        knowledge_items: List[KnowledgeItem] = []

        # Convert alignments into structured reasoning input
        summary_payload = []
        for al in alignments:
            audio_text = al.audio.transcript if al.audio else ""
            visual_desc = "; ".join([v.visual_description for v in al.visual])
            summary_payload.append(
                {
                    "start_time": al.start_time,
                    "end_time": al.end_time,
                    "spoken": audio_text,
                    "visual": visual_desc,
                    "relationship": al.relationship.value,
                    "temporal_sequence": al.temporal_sequence,
                    "conflicts": [c.model_dump() for c in al.conflicts],
                    "deictic_resolutions": [d.model_dump() for d in al.deictic_resolutions],
                }
            )

        prompt = (
            f"Perform deep trading reasoning on aligned evidence for job '{job_id}':\n"
            f"{json.dumps(summary_payload, indent=2)}\n\n"
            "Rigorous Reasoning Checklist:\n"
            "1. Validate whether audio and visual evidence match or contradict.\n"
            "2. Verify whether chart examples demonstrate the spoken trading rule.\n"
            "3. Identify conditional statements ('if price closes...', 'only when...').\n"
            "4. Detect exceptions and invalidations.\n"
            "5. Assign Epistemic Status: OBSERVED, INFERRED, UNCERTAIN, CONTRADICTED, VALIDATED.\n"
            "6. Classify items into KnowledgeTypes: concept, definition, rule, sub_rule, example, counterexample, exception, condition, relationship, warning, terminology.\n"
            "Never invent facts without evidence. Distinguish examples from universal rules."
        )

        response_str = self._ai.generate_content(
            role=ModelRole.REASONING,
            prompt=prompt,
            job_id=job_id,
            stage="gemini_reasoning",
            serial_number=serial_number,
        )

        try:
            data = json.loads(response_str)
            raw_items = data.get("reasoned_items", [])
            for idx, r in enumerate(raw_items):
                item_type_str = r.get("type", "rule").lower()
                status_str = r.get("epistemic_status", "VALIDATED").upper()

                # Select matching alignment window
                target_al = alignments[min(idx, len(alignments) - 1)]

                # Check if target alignment has cross-modal conflict
                has_conflict = len(target_al.conflicts) > 0 or target_al.relationship == AlignmentRelationship.CONTRADICTING

                # Epistemic guard: If spoken audio classified this as an EXAMPLE, do NOT allow it to become a universal RULE
                stmt_text = r.get("statement", "")
                if target_al.audio and target_al.audio.statement_types:
                    classified_type = target_al.audio.statement_types.get(stmt_text, "")
                    if classified_type == "EXAMPLE" or "for example" in stmt_text.lower():
                        item_type_str = "example"

                # Map to schema enums safely
                try:
                    ktype = KnowledgeType(item_type_str)
                except ValueError:
                    ktype = KnowledgeType.RULE

                if has_conflict:
                    kstatus = KnowledgeValidationStatus.CONTRADICTED
                else:
                    try:
                        kstatus = KnowledgeValidationStatus(status_str)
                    except ValueError:
                        kstatus = KnowledgeValidationStatus.OBSERVED

                ev = EvidenceTrace(
                    evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
                    job_id=job_id,
                    start_time=target_al.start_time,
                    end_time=target_al.end_time,
                    audio_quote=target_al.audio.transcript if target_al.audio else None,
                    visual_reference=target_al.visual[0].event_id if target_al.visual else None,
                    extracted_statement=r.get("statement", ""),
                    confidence=float(r.get("confidence", 0.92)) if not has_conflict else 0.50,
                    notes=f"Validated via reasoning engine under relationship {target_al.relationship.value}. Conflicts: {len(target_al.conflicts)}",
                )
                evidence_traces.append(ev)
                if self._repo:
                    self._repo.save_evidence(ev)

                reasoning_note = f"Reasoned from cross-modal alignment [{target_al.start_time}s - {target_al.end_time}s]. Evidence relationship: {target_al.relationship.value}."
                if has_conflict:
                    reasoning_note += f" [CROSS_MODAL_CONFLICT]: Spoken claim contradicts visual chart reality."

                k_item = KnowledgeItem(
                    item_id=f"k_{uuid.uuid4().hex[:8]}",
                    job_id=job_id,
                    item_type=ktype,
                    title=r.get("title", f"{ktype.value.capitalize()} Rule"),
                    statement=stmt_text,
                    reasoning=reasoning_note,
                    validation_status=kstatus,
                    confidence=float(r.get("confidence", 0.92)) if not has_conflict else 0.45,
                    start_time=target_al.start_time,
                    end_time=target_al.end_time,
                    evidence=[ev],
                    tags=["ICT", "SmartMoney", ktype.value],
                )
                knowledge_items.append(k_item)

        except Exception:
            # Deterministic fallback based directly on alignments
            for al in alignments:
                if al.relationship in [AlignmentRelationship.SUPPORTING, AlignmentRelationship.PARTIALLY_SUPPORTING]:
                    ev = EvidenceTrace(
                        evidence_id=f"ev_{uuid.uuid4().hex[:8]}",
                        job_id=job_id,
                        start_time=al.start_time,
                        end_time=al.end_time,
                        audio_quote=al.audio.transcript if al.audio else None,
                        visual_reference=al.visual[0].event_id if al.visual else None,
                        extracted_statement=al.audio.transcript if al.audio else "Visual chart price action observed.",
                        confidence=al.confidence,
                    )
                    evidence_traces.append(ev)
                    if self._repo:
                        self._repo.save_evidence(ev)

                    k_item = KnowledgeItem(
                        item_id=f"k_{uuid.uuid4().hex[:8]}",
                        job_id=job_id,
                        item_type=KnowledgeType.RULE,
                        title="Trading Rule Observation",
                        statement=al.audio.transcript if al.audio else "Price action confirmation rule.",
                        reasoning=al.reasoning,
                        validation_status=KnowledgeValidationStatus.VALIDATED,
                        confidence=al.confidence,
                        start_time=al.start_time,
                        end_time=al.end_time,
                        evidence=[ev],
                        tags=["trading", "rule"],
                    )
                    knowledge_items.append(k_item)

        return evidence_traces, knowledge_items
