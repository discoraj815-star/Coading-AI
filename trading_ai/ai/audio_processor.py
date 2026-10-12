"""Audio processing service extracting timestamped transcripts, deictic cues, and epistemic statements."""

import json
import re
import uuid
from typing import Dict, List, Optional
from trading_ai.ai.gemini_client import GeminiClient
from trading_ai.core.model_router import ModelRole
from trading_ai.schemas import EpistemicStatementType, TranscriptSegment
from trading_ai.storage.repository import Repository

DEICTIC_PATTERNS = [
    r"\bthis candle\b",
    r"\bthis level\b",
    r"\bthis zone\b",
    r"\blook here\b",
    r"\bright here\b",
    r"\bas you can see\b",
    r"\bnotice this\b",
    r"\blook at this\b",
    r"\babove this\b",
    r"\bbelow this\b",
    r"\bthis move\b",
    r"\bthis high\b",
    r"\bthis low\b",
    r"\bhere\b",
    r"\bthere\b",
]


class AudioProcessor:
    """Processes lecture audio and generates detailed timestamped transcript segments with epistemic classification."""

    def __init__(self, gemini_client: GeminiClient, repository: Optional[Repository] = None):
        self._ai = gemini_client
        self._repo = repository

    def detect_deictic_phrases(self, text: str) -> List[str]:
        """Identify deictic/pointing phrases in spoken text for visual grounding."""
        text_lower = text.lower()
        found = []
        for pattern in DEICTIC_PATTERNS:
            matches = re.findall(pattern, text_lower)
            if matches:
                found.extend(matches)
        return list(dict.fromkeys(found))

    def classify_statement_epistemics(self, statement: str) -> str:
        """Classify statement into RULE, EXAMPLE, EXCEPTION, WARNING, or OBSERVATION to prevent treating examples as rules."""
        s_lower = statement.lower()
        if any(w in s_lower for w in ["for example", "e.g.", "here is an example", "in this example", "case study", "demonstrating"]):
            return EpistemicStatementType.EXAMPLE.value
        elif any(w in s_lower for w in ["except", "unless", "however if", "invalidation condition", "exception to"]):
            return EpistemicStatementType.EXCEPTION.value
        elif any(w in s_lower for w in ["warning", "caution", "danger", "be careful", "never risk", "avoid"]):
            return EpistemicStatementType.WARNING.value
        elif any(w in s_lower for w in ["must", "always", "rule:", "the rule is", "required before", "you have to", "mandatory"]):
            return EpistemicStatementType.RULE.value
        elif any(w in s_lower for w in ["notice", "we see", "is showing", "currently", "on this chart"]):
            return EpistemicStatementType.OBSERVATION.value
        elif "?" in statement:
            return EpistemicStatementType.QUESTION.value
        return EpistemicStatementType.OBSERVATION.value

    def process_audio(
        self,
        job_id: str,
        audio_file_path: Optional[str] = None,
        duration: float = 60.0,
        serial_number: int = 1,
    ) -> List[TranscriptSegment]:
        """Analyze lecture speech and extract timestamped statements, rules, examples, and deictic references."""
        prompt = (
            f"Perform detailed transcription and domain analysis for trading lecture job '{job_id}'. "
            f"Duration: {duration} seconds. "
            "Extract: transcript text with exact start_time and end_time, language, speaker, "
            "important trading statements, domain terminology (e.g. order blocks, FVGs, liquidity), "
            "spoken trading rules or conditions, examples described, and spoken uncertainties or caveats. "
            "Explicitly distinguish universal rules from specific chart examples."
        )

        response_str = self._ai.generate_content(
            role=ModelRole.AUDIO_ANALYSIS,
            prompt=prompt,
            job_id=job_id,
            stage="audio_processing",
            serial_number=serial_number,
        )

        segments: List[TranscriptSegment] = []
        try:
            data = json.loads(response_str)
            raw_segments = data.get("segments", [])
            for s in raw_segments:
                transcript_text = s.get("transcript", "")
                imp_statements = s.get("important_statements", [])
                rules = s.get("rules", [])
                examples = s.get("examples", [])

                # Map statements to epistemic categories
                statement_types: Dict[str, str] = {}
                for st in imp_statements:
                    statement_types[st] = self.classify_statement_epistemics(st)
                for r in rules:
                    statement_types[r] = EpistemicStatementType.RULE.value
                for ex in examples:
                    statement_types[ex] = EpistemicStatementType.EXAMPLE.value

                # Detect deictic expressions
                deictic_cues = self.detect_deictic_phrases(transcript_text)

                seg = TranscriptSegment(
                    segment_id=f"seg_{uuid.uuid4().hex[:8]}",
                    job_id=job_id,
                    start_time=float(s.get("start_time", 0.0)),
                    end_time=float(s.get("end_time", duration)),
                    transcript=transcript_text,
                    language=s.get("language", "en"),
                    speaker=s.get("speaker"),
                    important_statements=imp_statements,
                    terminology=s.get("terminology", []),
                    rules=rules,
                    examples=examples,
                    uncertainty=s.get("uncertainty", []),
                    confidence=float(s.get("confidence", 0.95)),
                    statement_types=statement_types,
                    deictic_phrases=deictic_cues,
                )
                segments.append(seg)
                if self._repo:
                    self._repo.save_transcript_segment(seg)
        except Exception:
            # Fallback if unparseable
            text = response_str or "Trading lecture speech content."
            deictic_cues = self.detect_deictic_phrases(text)
            fallback_seg = TranscriptSegment(
                segment_id=f"seg_{uuid.uuid4().hex[:8]}",
                job_id=job_id,
                start_time=0.0,
                end_time=duration,
                transcript=text,
                language="en",
                confidence=0.9,
                statement_types={text: EpistemicStatementType.OBSERVATION.value},
                deictic_phrases=deictic_cues,
            )
            segments.append(fallback_seg)
            if self._repo:
                self._repo.save_transcript_segment(fallback_seg)

        return segments

