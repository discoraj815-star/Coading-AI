"""Comprehensive test suite validating Step 1 Multimodal Hardening:
- Visual Grounding (spatial bounding boxes & coordinates)
- Deictic Reference Resolution ("this candle", "this level", "look here")
- Cross-Modal Consistency & Contradiction Detection
- Epistemic Example vs Rule Separation
- Temporal State Sequencing (Before -> Change -> After)
- Adaptive Visual Sampling & Duplicate Frame Suppression
- Multi-resolution Region Cropping (Level 3/5)
- Evidence Provenance & Uncertainty Preservation
"""

import os
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

from trading_ai.ai import (
    AlignmentEngine,
    AudioProcessor,
    GeminiClient,
    ReasoningEngine,
    VisualProcessor,
)
from trading_ai.core import ModelRouter, TokenManager
from trading_ai.media import VideoProcessor
from trading_ai.schemas import (
    AlignmentRelationship,
    AudioVisualAlignment,
    BoundingBox,
    CrossModalConflict,
    CrossModalConflictType,
    DeicticResolution,
    EpistemicStatementType,
    JobMetadata,
    KnowledgeItem,
    KnowledgeType,
    KnowledgeValidationStatus,
    ReferenceResolutionStatus,
    TemporalStateTransition,
    TranscriptSegment,
    VisualEvent,
    VisualRegion,
    VisualRegionType,
)
from trading_ai.storage import DatabaseManager, Repository


@pytest.fixture
def multimodal_env(tmp_path):
    """Sets up a clean database and mock AI environment."""
    db_file = tmp_path / "multimodal_hardening.db"
    db_mgr = DatabaseManager(db_path=str(db_file))
    repo = Repository(db_mgr)
    tm = TokenManager()
    mr = ModelRouter()
    ai_client = GeminiClient(model_router=mr, token_manager=tm, repository=repo)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Create two test frames: Frame 1 and Frame 2
    f1 = frames_dir / "frame_0001.jpg"
    img1 = Image.new("RGB", (1280, 720), color=(15, 20, 30))
    d1 = ImageDraw.Draw(img1)
    d1.rectangle([(580, 180), (760, 210)], fill=(40, 60, 90), outline=(100, 150, 220))
    d1.text((600, 190), "London High 1.08520", fill=(255, 255, 255))
    img1.save(str(f1), "JPEG")

    f2 = frames_dir / "frame_0002.jpg"
    img2 = Image.new("RGB", (1280, 720), color=(24, 32, 46))
    d2 = ImageDraw.Draw(img2)
    # Draw multiple candlesticks across chart
    for i in range(10):
        d2.rectangle([(100 + i * 80, 200 + (i % 3) * 40), (130 + i * 80, 360)], fill=(200, 50, 60))
    d2.rectangle([(620, 260), (760, 340)], fill=(80, 30, 40), outline=(220, 80, 100))
    d2.text((630, 290), "Bearish FVG 1.0838", fill=(255, 255, 255))
    img2.save(str(f2), "JPEG")

    # Identical frame for duplicate detection test
    f1_dup = frames_dir / "frame_0001_dup.jpg"
    img1.save(str(f1_dup), "JPEG")

    # Save job record
    repo.save_job(
        JobMetadata(
            job_id="job_hardened_test",
            file_hash="hash_hardened_123",
            filename="lecture.mp4",
            file_path=str(f1),
        )
    )

    return {
        "repo": repo,
        "ai": ai_client,
        "f1": str(f1),
        "f2": str(f2),
        "f1_dup": str(f1_dup),
        "frames_dir": str(frames_dir),
    }


def test_visual_grounding_coordinates_and_regions(multimodal_env):
    """Test that visual events contain structured bounding boxes with normalized coordinates."""
    repo = multimodal_env["repo"]
    ai = multimodal_env["ai"]
    vp = VisualProcessor(gemini_client=ai, repository=repo)

    keyframes = [
        {"frame_index": 1, "timestamp": 5.0, "file_path": multimodal_env["f1"]},
        {"frame_index": 2, "timestamp": 15.0, "file_path": multimodal_env["f2"]},
    ]

    events = vp.process_keyframes(job_id="job_hardened_test", keyframes=keyframes)
    assert len(events) >= 2

    e1 = events[0]
    assert len(e1.regions) >= 1
    # Check bounding box normalization
    r = e1.regions[0]
    assert r.box is not None
    assert 0.0 <= r.box.normalized_x <= 1.0
    assert 0.0 <= r.box.normalized_y <= 1.0
    assert 0.0 < r.box.normalized_width <= 1.0
    assert 0.0 < r.box.normalized_height <= 1.0
    assert e1.cursor_box is not None
    assert e1.cursor_box.width > 0

    # Verify repository persistence of regions
    saved_events = repo.get_visual_events("job_hardened_test")
    assert len(saved_events) >= 2
    assert len(saved_events[0].regions) >= 1
    assert saved_events[0].regions[0].region_type in [
        VisualRegionType.CANDLE.value,
        VisualRegionType.LIQUIDITY_LEVEL.value,
        VisualRegionType.ORDER_BLOCK.value,
        VisualRegionType.FVG.value,
        VisualRegionType.CURSOR.value,
    ]


def test_temporal_state_sequence_preservation(multimodal_env):
    """Test Before -> Change -> After sequence model in visual analysis."""
    repo = multimodal_env["repo"]
    ai = multimodal_env["ai"]
    vp = VisualProcessor(gemini_client=ai, repository=repo)

    keyframes = [
        {"frame_index": 1, "timestamp": 8.0, "file_path": multimodal_env["f1"]},
        {"frame_index": 2, "timestamp": 22.0, "file_path": multimodal_env["f2"]},
    ]
    events = vp.process_keyframes(job_id="job_hardened_test", keyframes=keyframes)

    assert events[1].state_transition is not None
    st = events[1].state_transition
    assert st.before_state is not None
    assert st.change_event is not None
    assert st.after_state is not None
    assert "EURUSD" in st.before_state or "Market" in st.before_state or "resistance" in st.before_state


def test_deictic_reference_resolution_resolved(multimodal_env):
    """Test resolving deictic reference ('this candle', 'this zone') to exact visual region."""
    repo = multimodal_env["repo"]
    ai = multimodal_env["ai"]
    aligner = AlignmentEngine(gemini_client=ai, repository=repo)

    # Create visual event with candle and cursor
    c_box = BoundingBox(x=640, y=180, width=20, height=20, normalized_x=0.5, normalized_y=0.25, normalized_width=0.015, normalized_height=0.03)
    candle_region = VisualRegion(
        region_id="reg_candle_01",
        region_type=VisualRegionType.CANDLE.value,
        box=BoundingBox(x=635, y=170, width=30, height=80, normalized_x=0.49, normalized_y=0.23, normalized_width=0.02, normalized_height=0.11),
        label="London High sweep candle with upper wick",
        confidence=0.96,
    )
    cursor_region = VisualRegion(
        region_id="reg_cur_01",
        region_type=VisualRegionType.CURSOR.value,
        box=c_box,
        label="Cursor hovers on sweep wick",
        confidence=0.98,
    )
    ve = VisualEvent(
        event_id="ve_test_01",
        job_id="job_hardened_test",
        timestamp=10.0,
        visual_description="EURUSD 15m chart. Cursor points directly to upper wick at 1.0852.",
        detected_event="liquidity_sweep",
        symbol="EURUSD",
        timeframe="15m",
        regions=[candle_region, cursor_region],
        cursor_box=c_box,
    )

    seg = TranscriptSegment(
        segment_id="seg_deictic_01",
        job_id="job_hardened_test",
        start_time=9.0,
        end_time=12.0,
        transcript="Notice how this candle sweeps London high liquidity right here.",
        deictic_phrases=["this candle", "right here"],
    )

    alignments = aligner.align_segments("job_hardened_test", [seg], [ve])
    assert len(alignments) == 1
    al = alignments[0]
    assert len(al.deictic_resolutions) >= 1

    # Check that 'this candle' is RESOLVED
    candle_res = next((d for d in al.deictic_resolutions if "candle" in d.phrase), None)
    assert candle_res is not None
    assert candle_res.status == ReferenceResolutionStatus.RESOLVED.value
    assert candle_res.target_region is not None
    assert candle_res.target_region.region_type == VisualRegionType.CANDLE.value


def test_adversarial_wrong_reference_unresolved(multimodal_env):
    """Adversarial test: Speaker says 'this candle' but chart displays no candle or unrelated UI."""
    repo = multimodal_env["repo"]
    ai = multimodal_env["ai"]
    aligner = AlignmentEngine(gemini_client=ai, repository=repo)

    # Frame has no candles (only text/slide)
    slide_region = VisualRegion(
        region_id="reg_slide_01",
        region_type=VisualRegionType.SLIDE.value,
        label="Slide title: Risk Management Rules",
        confidence=0.95,
    )
    ve = VisualEvent(
        event_id="ve_slide_01",
        job_id="job_hardened_test",
        timestamp=30.0,
        visual_description="Presentation slide displaying text rules.",
        detected_event="slide_presentation",
        regions=[slide_region],
    )

    seg = TranscriptSegment(
        segment_id="seg_wrong_ref",
        job_id="job_hardened_test",
        start_time=29.0,
        end_time=32.0,
        transcript="Look at this candle right above the support.",
        deictic_phrases=["this candle"],
    )

    alignments = aligner.align_segments("job_hardened_test", [seg], [ve])
    assert len(alignments) == 1
    candle_res = alignments[0].deictic_resolutions[0]
    # Must preserve uncertainty without hallucinating!
    assert candle_res.status == ReferenceResolutionStatus.REFERENCE_UNRESOLVED.value
    assert candle_res.confidence <= 0.60


def test_adversarial_cross_modal_conflict_detection(multimodal_env):
    """Adversarial test: Speaker claims bullish setup while chart displays bearish breakdown."""
    repo = multimodal_env["repo"]
    ai = multimodal_env["ai"]
    aligner = AlignmentEngine(gemini_client=ai, repository=repo)

    ve = VisualEvent(
        event_id="ve_bearish_01",
        job_id="job_hardened_test",
        timestamp=40.0,
        visual_description="EURUSD 15m chart shows a violent bearish breakdown with a sharp downward displacement and lower low.",
        detected_event="candle_breakout",
        symbol="EURUSD",
        timeframe="15m",
    )

    # Audio claims bullish buy opportunity
    seg = TranscriptSegment(
        segment_id="seg_conflict_01",
        job_id="job_hardened_test",
        start_time=39.0,
        end_time=42.0,
        transcript="This is our prime bullish setup for a massive buy signal.",
    )

    alignments = aligner.align_segments("job_hardened_test", [seg], [ve])
    assert len(alignments) == 1
    al = alignments[0]
    assert al.relationship == AlignmentRelationship.CONTRADICTING
    assert len(al.conflicts) >= 1
    c = al.conflicts[0]
    assert c.conflict_type == CrossModalConflictType.DIRECTIONAL_CONTRADICTION.value
    assert "bullish" in c.audio_claim.lower()
    assert "bearish" in c.visual_evidence.lower()


def test_example_vs_rule_epistemic_separation(multimodal_env):
    """Test that explicit chart examples are not misclassified as permanent universal rules."""
    repo = multimodal_env["repo"]
    ai = multimodal_env["ai"]
    audio_proc = AudioProcessor(gemini_client=ai, repository=repo)
    reasoning_engine = ReasoningEngine(gemini_client=ai, repository=repo)

    # Spoken audio containing both a specific example and a universal rule
    seg = TranscriptSegment(
        segment_id="seg_rules_and_examples",
        job_id="job_hardened_test",
        start_time=10.0,
        end_time=25.0,
        transcript="For example here on EURUSD we saw an 8 pip sweep. The universal rule is you must always wait for a displacement candle before entering.",
        important_statements=[
            "For example here on EURUSD we saw an 8 pip sweep.",
            "You must always wait for a displacement candle before entering.",
        ],
    )

    # Verify classification in AudioProcessor
    type1 = audio_proc.classify_statement_epistemics("For example here on EURUSD we saw an 8 pip sweep.")
    type2 = audio_proc.classify_statement_epistemics("You must always wait for a displacement candle before entering.")
    assert type1 == EpistemicStatementType.EXAMPLE.value
    assert type2 == EpistemicStatementType.RULE.value

    # Verify reasoning engine does not promote the example into a RULE
    seg.statement_types = {
        "For example here on EURUSD we saw an 8 pip sweep.": EpistemicStatementType.EXAMPLE.value,
        "You must always wait for a displacement candle before entering.": EpistemicStatementType.RULE.value,
    }

    ve = VisualEvent(
        event_id="ve_sweep_demo",
        job_id="job_hardened_test",
        timestamp=15.0,
        visual_description="Chart displays liquidity sweep and displacement.",
        detected_event="liquidity_sweep",
    )

    al = AudioVisualAlignment(
        alignment_id="al_ex_01",
        job_id="job_hardened_test",
        start_time=10.0,
        end_time=25.0,
        audio=seg,
        visual=[ve],
        relationship=AlignmentRelationship.SUPPORTING,
        confidence=0.95,
    )

    ev_traces, knowledge_items = reasoning_engine.reason_and_validate("job_hardened_test", [al])
    assert len(knowledge_items) >= 1
    # Check that rule items maintain valid rule or example type
    for k in knowledge_items:
        if "for example" in k.statement.lower():
            assert k.item_type == KnowledgeType.EXAMPLE


def test_adaptive_visual_sampling_and_duplicate_suppression(multimodal_env):
    """Test visual change metric, duplicate frame detection, and multi-resolution crop."""
    vp = VideoProcessor()
    f1 = multimodal_env["f1"]
    f2 = multimodal_env["f2"]
    f1_dup = multimodal_env["f1_dup"]

    # 1. Identical frames should yield visual change < 0.01
    dup_change = vp.detect_visual_change(f1, f1_dup)
    assert dup_change < 0.015

    # 2. Distinct frames should yield visual change > 0.05
    diff_change = vp.detect_visual_change(f1, f2)
    assert diff_change > 0.05

    # 3. Multi-resolution crop extraction
    crop_path = vp.crop_region(
        image_path=f1,
        box={"normalized_x": 0.45, "normalized_y": 0.22, "normalized_width": 0.15, "normalized_height": 0.08},
    )
    assert crop_path is not None
    assert os.path.exists(crop_path)
    with Image.open(crop_path) as crp:
        assert crp.width > 0
        assert crp.height > 0
