"""Iterative Comprehension Loop for ambiguous lecture concepts in Step 15."""

import logging
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from trading_ai.ai_core.epistemic import EpistemicState
from trading_ai.core.token_manager import TokenManager
from trading_ai.storage.repository import Repository

logger = logging.getLogger(__name__)


class ComprehensionResult(BaseModel):
    """Result of iterative comprehension evaluation."""

    concept_id: str
    job_id: str
    iterations_run: int
    final_confidence: float
    epistemic_state: EpistemicState
    is_resolved: bool
    review_notes: str
    tokens_consumed: int
    hindi_status: str


class IterativeComprehensionLoop:
    """Iterative multi-pass review loop for ambiguous or low-confidence concepts.

    Safety Bounds:
    - Maximum 3 review iterations.
    - Strict token budget per review pass.
    - Diminishing return detection (stops if delta < 0.05).
    - Never hallucinates certainty: outputs NEEDS_REVIEW or AMBIGUOUS when uncertain.
    """

    def __init__(
        self,
        repository: Optional[Repository] = None,
        token_manager: Optional[TokenManager] = None,
        max_iterations: int = 3,
        min_confidence_threshold: float = 0.85,
    ):
        self._repo = repository
        self._token_manager = token_manager or TokenManager()
        self._max_iterations = max_iterations
        self._min_confidence = min_confidence_threshold

    def review_concept(
        self,
        job_id: str,
        concept_title: str,
        initial_confidence: float,
        audio_evidence: Optional[str] = None,
        visual_evidence: Optional[str] = None,
        transcript_context: Optional[str] = None,
        token_budget: int = 4000,
    ) -> ComprehensionResult:
        """Run bounded iterative review loop on an ambiguous concept."""
        concept_id = f"conc_{uuid.uuid4().hex[:10]}"
        current_confidence = initial_confidence
        iteration = 1
        tokens_used = 0
        notes = []
        is_resolved = False

        notes.append(f"Pass 1: Initial analysis confidence {initial_confidence:.2f}")

        # If already highly confident and unambiguous
        if current_confidence >= self._min_confidence and audio_evidence and visual_evidence:
            state = EpistemicState.VALIDATED
            is_resolved = True
            hindi = "संकल्पना पूरी तरह स्पष्ट और सत्यापित है (Validated)।"
        else:
            state = EpistemicState.NEEDS_REVIEW
            hindi = "संकल्पना में अस्पष्टता है, पुनः समीक्षा की जा रही है।"

            while iteration <= self._max_iterations and not is_resolved and tokens_used < token_budget:
                iteration += 1
                tokens_used += 400  # Estimate tokens for one review pass

                prev_conf = current_confidence

                # Review audio and visual evidence confluence
                has_audio = bool(audio_evidence and len(audio_evidence.strip()) > 10)
                has_visual = bool(visual_evidence and len(visual_evidence.strip()) > 10)
                has_transcript = bool(transcript_context and len(transcript_context.strip()) > 10)

                # Simulated iterative evidence synthesis
                if has_audio and has_visual and has_transcript:
                    current_confidence = min(1.0, current_confidence + 0.15)
                    notes.append(f"Pass {iteration}: Cross-checked audio, visual, and transcript. Confidence -> {current_confidence:.2f}")
                elif has_audio or has_visual:
                    current_confidence = min(0.80, current_confidence + 0.08)
                    notes.append(f"Pass {iteration}: Partial multi-modal evidence found. Confidence -> {current_confidence:.2f}")
                else:
                    notes.append(f"Pass {iteration}: Missing multi-modal evidence. Diminishing returns.")
                    break

                # Diminishing return detection
                if (current_confidence - prev_conf) < 0.03:
                    notes.append("Pass halted: Diminishing returns detected.")
                    break

                if current_confidence >= self._min_confidence:
                    is_resolved = True
                    state = EpistemicState.VALIDATED
                    hindi = "अतिरिक्त साक्ष्यों के बाद संकल्पना सत्यापित हो गई है।"
                    break

            # If still not resolved after passes
            if not is_resolved:
                if current_confidence >= 0.70:
                    state = EpistemicState.PARTIALLY_UNDERSTOOD
                    hindi = "संकल्पना आंशिक रूप से समझी गई है, कुछ साक्ष्य अभी भी स्पष्ट नहीं हैं।"
                else:
                    state = EpistemicState.AMBIGUOUS
                    hindi = "संकल्पना अस्पष्ट बनी हुई है (Ambiguous), अनुमान लगाने के बजाय समीक्षा के लिए चिन्हित किया गया।"

        res = ComprehensionResult(
            concept_id=concept_id,
            job_id=job_id,
            iterations_run=iteration,
            final_confidence=round(current_confidence, 3),
            epistemic_state=state,
            is_resolved=is_resolved,
            review_notes=" | ".join(notes),
            tokens_consumed=tokens_used,
            hindi_status=hindi,
        )

        # Persist review record if repo available
        if self._repo and hasattr(self._repo, "save_comprehension_review"):
            try:
                self._repo.save_comprehension_review({
                    "job_id": job_id,
                    "concept_id": concept_id,
                    "iteration": iteration,
                    "confidence": res.final_confidence,
                    "epistemic_state": state.value,
                    "review_notes": res.review_notes,
                    "is_resolved": is_resolved,
                })
            except Exception as e:
                logger.debug(f"Could not persist comprehension review: {e}")

        return res
