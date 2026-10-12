"""Schemas for visual processing and on-screen event detection."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates for grounded visual perception."""

    x: float = Field(default=0.0, description="Top-left X coordinate in pixels")
    y: float = Field(default=0.0, description="Top-left Y coordinate in pixels")
    width: float = Field(default=0.0, description="Box width in pixels")
    height: float = Field(default=0.0, description="Box height in pixels")
    normalized_x: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized X (0.0 to 1.0)")
    normalized_y: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized Y (0.0 to 1.0)")
    normalized_width: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized width (0.0 to 1.0)")
    normalized_height: float = Field(default=0.0, ge=0.0, le=1.0, description="Normalized height (0.0 to 1.0)")


class VisualRegion(BaseModel):
    """Specific grounded visual region representing an on-screen chart object or annotation."""

    region_id: str = Field(description="Unique region identifier")
    region_type: str = Field(
        default="UNKNOWN",
        description="Visual region category: CHART, CANDLE, ORDER_BLOCK, FVG, LIQUIDITY_LEVEL, CURSOR, ANNOTATION, etc.",
    )
    box: Optional[BoundingBox] = Field(default=None, description="Spatial coordinates of the region")
    label: Optional[str] = Field(default=None, description="Human/model-assigned label or text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Grounding confidence")
    observed_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Raw objective visual features (colors, wick length, price level) separate from interpretation",
    )


class TemporalStateTransition(BaseModel):
    """Preserves Before -> Change -> After sequence across chart frames."""

    before_state: Optional[str] = Field(
        default=None, description="State of chart/market prior to transition (e.g. price approaching level)"
    )
    change_event: Optional[str] = Field(
        default=None, description="Active change action (e.g. teacher draws FVG, candle pierces liquidity)"
    )
    after_state: Optional[str] = Field(
        default=None, description="State of chart/market after transition (e.g. sharp displacement down)"
    )


class VisualEvent(BaseModel):
    """Detailed visual observation preserving micro-details on charts/screens."""

    event_id: str = Field(description="Unique event identifier")
    job_id: str = Field(description="Associated lecture job ID")
    timestamp: float = Field(description="Timestamp in seconds")
    frame_ref: Optional[str] = Field(
        default=None, description="Path or hash reference to the analyzed keyframe image"
    )
    visual_description: str = Field(
        description="Comprehensive objective description of what is shown on screen"
    )
    detected_event: str = Field(
        description="Event category: e.g. timeframe_change, candle_breakout, zone_draw, teacher_pointing"
    )
    symbol: Optional[str] = Field(
        default=None, description="Active instrument or symbol (e.g. EURUSD, NQ, BTC)"
    )
    timeframe: Optional[str] = Field(
        default=None, description="Active chart timeframe (e.g. 1m, 5m, 15m, 1h, 4h, Daily)"
    )
    candles: List[str] = Field(
        default_factory=list, description="Notable candle patterns (wicks, engulfing, rejections)"
    )
    drawings: List[str] = Field(
        default_factory=list, description="Trendlines, rays, horizontal levels, channels"
    )
    highlighted_zones: List[str] = Field(
        default_factory=list, description="Boxes, rectangles, fair value gaps, order blocks"
    )
    arrows: List[str] = Field(
        default_factory=list, description="Arrows or directional markers on chart"
    )
    labels: List[str] = Field(
        default_factory=list, description="Text labels or notes drawn on screen"
    )
    screen_text: List[str] = Field(
        default_factory=list, description="OCR/detected readable text in interface"
    )
    cursor_interaction: Optional[str] = Field(
        default=None, description="Detected cursor hover, drawing action, or teacher pointing"
    )
    chart_changes: List[str] = Field(
        default_factory=list, description="Changes in chart layout, indicators, or scale"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Visual detection confidence"
    )
    # Step 1 Hardening Extensions
    regions: List[VisualRegion] = Field(
        default_factory=list, description="Spatially grounded bounding-box regions detected in this frame"
    )
    cursor_box: Optional[BoundingBox] = Field(
        default=None, description="Spatial coordinates of the mouse cursor if detected"
    )
    state_transition: Optional[TemporalStateTransition] = Field(
        default=None, description="Before -> Change -> After sequence model"
    )
    is_static_duplicate: bool = Field(
        default=False, description="Flag indicating this frame is identical to the prior frame"
    )
    scene_type: Optional[str] = Field(
        default="chart", description="Visual scene class: chart, slide, browser, face, terminal"
    )

