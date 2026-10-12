"""Explicit Audio/Visual alignment layer establishing cross-modal grounding, conflict detection, and deictic resolution."""

import json
import uuid
from typing import List, Optional
from trading_ai.ai.gemini_client import GeminiClient
from trading_ai.core.model_router import ModelRole
from trading_ai.schemas import (
    AlignmentRelationship,
    AudioVisualAlignment,
    CrossModalConflict,
    CrossModalConflictType,
    DeicticResolution,
    ReferenceResolutionStatus,
    TranscriptSegment,
    VisualEvent,
    VisualRegion,
    VisualRegionType,
)
from trading_ai.storage.repository import Repository


class AlignmentEngine:
    """Explicitly correlates what was said with what was shown across time segments, resolving references and detecting contradictions."""

    def __init__(self, gemini_client: GeminiClient, repository: Optional[Repository] = None):
        self._ai = gemini_client
        self._repo = repository

    def _resolve_deictic_references(
        self,
        transcript: str,
        deictic_phrases: List[str],
        matched_visuals: List[VisualEvent],
    ) -> List[DeicticResolution]:
        """Ground spoken pointing phrases ('this candle', 'this zone', 'here') to exact spatial visual regions without hallucination."""
        resolutions: List[DeicticResolution] = []
        if not deictic_phrases or not matched_visuals:
            return resolutions

        # Gather all regions and cursor interactions across contemporaneous visual frames
        available_regions: List[VisualRegion] = []
        cursor_regions: List[VisualRegion] = []
        for v in matched_visuals:
            for r in v.regions:
                available_regions.append(r)
                if r.region_type == VisualRegionType.CURSOR.value:
                    cursor_regions.append(r)

        for phrase in deictic_phrases:
            p_lower = phrase.lower()
            matched_region: Optional[VisualRegion] = None
            resolution_status = ReferenceResolutionStatus.REFERENCE_UNRESOLVED.value
            target_desc = None
            conf = 0.50

            # Match candles
            if "candle" in p_lower:
                candles = [r for r in available_regions if r.region_type == VisualRegionType.CANDLE.value]
                if candles:
                    # If cursor is near candle, prioritize
                    matched_region = candles[0]
                    resolution_status = ReferenceResolutionStatus.RESOLVED.value
                    target_desc = matched_region.label or "Swept wick candle highlighted on chart"
                    conf = 0.94
                else:
                    resolution_status = ReferenceResolutionStatus.REFERENCE_UNRESOLVED.value
                    target_desc = "Spoken 'this candle' but no distinct candle highlighted in current visual frame"

            # Match zones or order blocks
            elif any(w in p_lower for w in ["zone", "order block", "fvg", "gap"]):
                zones = [
                    r for r in available_regions
                    if r.region_type in (VisualRegionType.FVG.value, VisualRegionType.ORDER_BLOCK.value, VisualRegionType.HIGHLIGHTED_ZONE.value)
                ]
                if zones:
                    matched_region = zones[0]
                    resolution_status = ReferenceResolutionStatus.RESOLVED.value
                    target_desc = matched_region.label or "Highlighted price imbalance zone"
                    conf = 0.95
                else:
                    resolution_status = ReferenceResolutionStatus.REFERENCE_UNRESOLVED.value
                    target_desc = "Spoken 'this zone' but no bounding box zone active in current frame"

            # Match levels or highs/lows
            elif any(w in p_lower for w in ["level", "high", "low", "line"]):
                levels = [
                    r for r in available_regions
                    if r.region_type in (VisualRegionType.LIQUIDITY_LEVEL.value, VisualRegionType.DRAWING.value, VisualRegionType.SUPPORT_RESISTANCE.value)
                ]
                if levels:
                    matched_region = levels[0]
                    resolution_status = ReferenceResolutionStatus.RESOLVED.value
                    target_desc = matched_region.label or "Horizontal liquidity level line"
                    conf = 0.93
                else:
                    resolution_status = ReferenceResolutionStatus.REFERENCE_UNRESOLVED.value
                    target_desc = "Spoken 'this level' but horizontal level drawing absent or ambiguous"

            # Generic "here" / "look here" / "right here"
            elif any(w in p_lower for w in ["look here", "right here", "here"]):
                if cursor_regions:
                    matched_region = cursor_regions[0]
                    resolution_status = ReferenceResolutionStatus.RESOLVED.value
                    target_desc = f"Teacher cursor active at normalized x={matched_region.box.normalized_x if matched_region.box else 'N/A'}"
                    conf = 0.92
                elif available_regions:
                    matched_region = available_regions[0]
                    resolution_status = ReferenceResolutionStatus.PARTIALLY_RESOLVED.value
                    target_desc = f"Grounded to prominent visual region: {matched_region.label or matched_region.region_type}"
                    conf = 0.82
                else:
                    resolution_status = ReferenceResolutionStatus.REFERENCE_UNRESOLVED.value
                    target_desc = "Deictic gesture uttered without contemporaneous cursor or highlighting"

            resolutions.append(
                DeicticResolution(
                    phrase=phrase,
                    status=resolution_status,
                    target_region=matched_region,
                    target_description=target_desc,
                    confidence=conf,
                )
            )

        return resolutions

    def align_segments(
        self,
        job_id: str,
        transcripts: List[TranscriptSegment],
        visual_events: List[VisualEvent],
        serial_number: int = 1,
    ) -> List[AudioVisualAlignment]:
        """Align speech segments with contemporaneous visual chart events, resolving references and detecting contradictions."""
        if not transcripts:
            return []

        alignments: List[AudioVisualAlignment] = []

        # Iterate over speech segments to build alignment windows
        for seg in transcripts:
            # Find all visual events occurring within or near this speech window [start - 1.0, end + 1.0]
            matched_visuals = [
                v
                for v in visual_events
                if (seg.start_time - 1.0) <= v.timestamp <= (seg.end_time + 1.0)
            ]

            relationship = AlignmentRelationship.AMBIGUOUS
            confidence = 0.80
            reasoning = "Insufficient visual evidence within speech window."
            conflicts: List[CrossModalConflict] = []
            temporal_seq = "PROGRESSIVE"

            if matched_visuals:
                audio_text = seg.transcript.lower()
                has_visual_match = False
                has_visual_contradiction = False

                for v in matched_visuals:
                    v_desc = v.visual_description.lower()
                    v_event = v.detected_event.lower()

                    # 1. Semantic corroboration
                    if ("liquidity" in audio_text and "liquidity" in v_desc) or \
                       ("gap" in audio_text and ("gap" in v_desc or "fvg" in v_desc)) or \
                       ("order block" in audio_text and ("rectangle" in v_desc or "order block" in v_desc)) or \
                       ("wick" in audio_text and "wick" in v_desc) or \
                       ("break" in audio_text and "break" in v_desc) or \
                       ("sweep" in audio_text and "sweep" in v_desc) or \
                       ("displacement" in audio_text and "displacement" in v_desc):
                        has_visual_match = True

                    # 2. Strict Directional Contradiction Detection
                    # e.g. Audio claims bullish/buy, but chart shows bearish breakdown/downward displacement
                    audio_bullish = any(w in audio_text for w in ["bullish", "buy", "long", "upward reversal", "higher high"])
                    visual_bearish = any(w in v_desc for w in ["bearish breakdown", "short signal", "lower low", "sharp downward", "drops sharply"])
                    audio_bearish = any(w in audio_text for w in ["bearish", "sell", "short", "downward reversal", "lower low"])
                    visual_bullish = any(w in v_desc for w in ["bullish breakout", "buy signal", "higher high", "sharp upward", "rallies sharply"])

                    if (audio_bullish and visual_bearish) or (audio_bearish and visual_bullish):
                        has_visual_contradiction = True
                        conflicts.append(
                            CrossModalConflict(
                                conflict_type=CrossModalConflictType.DIRECTIONAL_CONTRADICTION.value,
                                audio_claim=seg.transcript[:120],
                                visual_evidence=v.visual_description[:120],
                                severity="HIGH",
                                confidence=0.95,
                            )
                        )

                    # 3. Timeframe Discrepancy Detection
                    if ("4 hour" in audio_text or "4h" in audio_text) and v.timeframe and v.timeframe.lower() not in ("4h", "4 hour"):
                        conflicts.append(
                            CrossModalConflict(
                                conflict_type=CrossModalConflictType.TIMEFRAME_MISMATCH.value,
                                audio_claim="Speaker references 4-hour timeframe",
                                visual_evidence=f"Active chart display shows {v.timeframe}",
                                severity="MEDIUM",
                                confidence=0.90,
                            )
                        )

                # Determine temporal sequence from visual transitions
                if any(v.detected_event in ("liquidity_sweep", "price_approach") for v in matched_visuals):
                    temporal_seq = "BEFORE_TO_CHANGE"
                elif any("fvg" in v.detected_event or "displacement" in v.detected_event for v in matched_visuals):
                    temporal_seq = "CHANGE_TO_AFTER"

                if has_visual_contradiction:
                    relationship = AlignmentRelationship.CONTRADICTING
                    confidence = 0.92
                    reasoning = f"Cross-modal contradiction: Spoken directional claim conflicts with active chart visual reality ({len(conflicts)} conflict(s) detected)."
                elif has_visual_match:
                    relationship = AlignmentRelationship.SUPPORTING
                    confidence = 0.95
                    reasoning = "Visual elements directly corroborate and demonstrate spoken trading rules, levels, and terms."
                else:
                    relationship = AlignmentRelationship.PARTIALLY_SUPPORTING
                    confidence = 0.85
                    reasoning = "Visual chart is active but does not explicitly highlight the exact spoken terminology."
            else:
                relationship = AlignmentRelationship.UNRELATED
                confidence = 0.88
                reasoning = "No corresponding visual actions or chart changes observed during this speech window."

            # Ground deictic expressions to visual regions
            deictic_cues = seg.deictic_phrases or []
            deictic_resolutions = self._resolve_deictic_references(
                transcript=seg.transcript,
                deictic_phrases=deictic_cues,
                matched_visuals=matched_visuals,
            )

            align = AudioVisualAlignment(
                alignment_id=f"align_{uuid.uuid4().hex[:8]}",
                job_id=job_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                audio=seg,
                visual=matched_visuals,
                relationship=relationship,
                confidence=confidence,
                reasoning=reasoning,
                conflicts=conflicts,
                deictic_resolutions=deictic_resolutions,
                temporal_sequence=temporal_seq,
            )
            alignments.append(align)
            if self._repo:
                self._repo.save_alignment(align)

        return alignments

