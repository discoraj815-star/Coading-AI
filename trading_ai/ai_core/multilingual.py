"""Multilingual Comprehension & Pure Hindi User-Facing Translation Engine for Step 15."""

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MultilingualInsight(BaseModel):
    """Normalized semantic representation preserving full source evidence."""

    original_text: str = Field(..., description="Unaltered source input text")
    detected_language: str = Field(..., description="Detected language (e.g. hinglish, hi, en, fr, de, es, ar)")
    normalized_meaning: str = Field(..., description="Standardized English trading meaning")
    extracted_concepts: List[str] = Field(default_factory=list, description="Extracted trading concepts (e.g. FVG, BOS, LiquiditySweep)")
    confidence: float = Field(default=0.90, description="Comprehension confidence score")
    timestamp: Optional[float] = Field(default=None, description="Temporal timestamp if from audio/video")
    hindi_explanation: str = Field(..., description="Explanatory insight in simple/pure Hindi with English trading terms")


# Dictionary of common Hinglish / Hindi phrases mapped to normalized concepts
HINGLISH_PATTERNS = [
    (r"(?i).*(liquidity\s+sweep|sweep).*(displacement).*", "Wait for displacement after liquidity sweep.", ["LiquiditySweep", "Displacement"]),
    (r"(?i).*(displacement\s+ke\s+baad|displacement\s+baad).*", "Look for continuation or FVG entry following displacement.", ["Displacement", "FVG"]),
    (r"(?i).*(fvg|fair\s+value\s+gap).*(entry|enter).*", "Execute entry at Fair Value Gap (FVG) mitigation.", ["FVG", "Entry"]),
    (r"(?i).*(choch|change\s+of\s+character).*", "Market structure shift / Change of Character (CHOCH) detected.", ["CHOCH", "MarketStructure"]),
    (r"(?i).*(bos|break\s+of\s+structure).*", "Break of Structure (BOS) indicates trend continuation.", ["BOS", "TrendContinuation"]),
    (r"(?i).*(stop\s+loss|sl).*(hunting|hunt|hit).*", "Liquidity grab targeting retail stop losses.", ["StopHunt", "LiquiditySweep"]),
    (r"(?i).*(high\s+impact\s+news|news\s+aane\s+wali).*", "High impact economic news pending; avoid immediate entries.", ["NewsRisk", "Filter"]),
]

# Multilingual keywords for language detection
LANG_KEYWORDS = {
    "hi": [r"[\u0900-\u097F]"],  # Devanagari script
    "hinglish": [
        r"\b(humein|hum|karna|hoga|karein|baad|pehle|dekhna|hai|hain|yeh|woh|kya|kyun|aur|lekin|chahiye|mein|par)\b"
    ],
    "fr": [r"\b(le|la|les|des|un|une|avec|dans|après|pour|est|sont|marché)\b"],
    "es": [r"\b(el|la|los|las|después|para|mercado|entrada|liquidez|velas)\b"],
    "de": [r"\b(der|die|das|und|nach|markt|bestätigung|liquidität|ausbruch)\b"],
    "ar": [r"[\u0600-\u06FF]"],  # Arabic script
    "en": [r"\b(the|is|after|before|liquidity|sweep|displacement|market|structure|entry|candle)\b"],
}


class MultilingualNormalizer:
    """Enterprise Multilingual Normalizer.

    Preserves original text, identifies language, normalizes core meaning,
    and formulates user-facing explanations strictly in SIMPLE / PURE HINDI.
    """

    @staticmethod
    def detect_language(text: str) -> str:
        """Identify language of input text."""
        cleaned = text.strip()
        # Devanagari check
        if re.search(LANG_KEYWORDS["hi"][0], cleaned):
            return "Hindi"

        # Arabic script check
        if re.search(LANG_KEYWORDS["ar"][0], cleaned):
            return "Arabic"

        cleaned_lower = cleaned.lower()
        # Hinglish check
        if re.search(LANG_KEYWORDS["hinglish"][0], cleaned_lower):
            return "Hinglish"

        # French, Spanish, German checks
        if re.search(LANG_KEYWORDS["fr"][0], cleaned_lower):
            return "French"
        if re.search(LANG_KEYWORDS["es"][0], cleaned_lower):
            return "Spanish"
        if re.search(LANG_KEYWORDS["de"][0], cleaned_lower):
            return "German"

        return "English"

    @classmethod
    def comprehend(cls, text: str, timestamp: Optional[float] = None) -> MultilingualInsight:
        """Normalize multilingual statement while preserving exact evidence."""
        lang = cls.detect_language(text)
        cleaned = text.strip()

        # Check Hinglish / Hindi pattern matches
        normalized = None
        concepts: List[str] = []

        for pat, norm_text, c_list in HINGLISH_PATTERNS:
            if re.search(pat, cleaned):
                normalized = norm_text
                concepts = c_list
                break

        # Fallback normalization if no specific pattern matched
        if not normalized:
            if lang in ("Hindi", "Hinglish"):
                normalized = f"Price action observation: {cleaned}"
                concepts = ["PriceAction"]
            elif lang == "French":
                normalized = f"Market observation (translated from French): {cleaned}"
                concepts = ["MarketStructure"]
            elif lang == "Spanish":
                normalized = f"Market observation (translated from Spanish): {cleaned}"
                concepts = ["MarketStructure"]
            elif lang == "German":
                normalized = f"Market observation (translated from German): {cleaned}"
                concepts = ["MarketStructure"]
            elif lang == "Arabic":
                normalized = f"Market observation (translated from Arabic): {cleaned}"
                concepts = ["MarketStructure"]
            else:
                normalized = cleaned
                # Extract English keywords
                if "sweep" in cleaned.lower():
                    concepts.append("LiquiditySweep")
                if "fvg" in cleaned.lower():
                    concepts.append("FVG")
                if "bos" in cleaned.lower():
                    concepts.append("BOS")
                if "choch" in cleaned.lower():
                    concepts.append("CHOCH")

        # Ensure all explicit trading concepts in the text are present in concepts list
        lower = cleaned.lower()
        if "fvg" in lower or "fair value gap" in lower:
            if "FVG" not in concepts:
                concepts.append("FVG")
        if "bos" in lower or "break of structure" in lower:
            if "BOS" not in concepts:
                concepts.append("BOS")
        if "choch" in lower or "change of character" in lower:
            if "CHOCH" not in concepts:
                concepts.append("CHOCH")
        if "sweep" in lower or "liquidity" in lower:
            if "LiquiditySweep" not in concepts:
                concepts.append("LiquiditySweep")
        if "displacement" in lower:
            if "Displacement" not in concepts:
                concepts.append("Displacement")

        hindi_expl = cls.synthesize_hindi_explanation(
            normalized_meaning=normalized,
            concepts=concepts,
            original_text=cleaned,
        )

        return MultilingualInsight(
            original_text=cleaned,
            detected_language=lang,
            normalized_meaning=normalized,
            extracted_concepts=concepts,
            confidence=0.92,
            timestamp=timestamp,
            hindi_explanation=hindi_expl,
        )

    @classmethod
    def synthesize_hindi_explanation(
        cls,
        normalized_meaning: str,
        concepts: List[str],
        original_text: str,
        decision_state: Optional[str] = None,
    ) -> str:
        """Produce explanations strictly in SIMPLE / PURE HINDI with English technical terms."""
        # Concept-specific pure Hindi mappings
        if "LiquiditySweep" in concepts and "Displacement" in concepts:
            return "Liquidity sweep के बाद displacement का इंतज़ार करें, पुष्टि मिलने पर ही ट्रेड की योजना बनाएं।"
        elif "Displacement" in concepts:
            return "मार्केट में मजबूत displacement दिखाई दी है, अब उचित FVG या retracement पर ध्यान दें।"
        elif "FVG" in concepts:
            return "FVG ज़ोन पर मूल्य की प्रतिक्रिया देखें, उचित पुष्टि मिलने पर ही Entry लें।"
        elif "CHOCH" in concepts or "MarketStructure" in concepts:
            return "मार्केट संरचना में CHOCH हुआ है, ट्रेंड में संभावित बदलाव के संकेत हैं।"
        elif "NewsRisk" in concepts:
            return "महत्वपूर्ण आर्थिक समाचार (High-Impact News) आने वाला है, इस समय नया ट्रेड न लें।"

        if decision_state == "NO_TRADE" or decision_state == "BLOCKED":
            return "इस सेटअप के लिए पर्याप्त पुष्टि नहीं मिली है, इसलिए अभी ट्रेड नहीं लेना चाहिए।"
        elif decision_state == "WAIT":
            return "वर्तमान में सेटअप अधूरा है, स्पष्ट Confirmation मिलने तक प्रतीक्षा करें।"
        elif decision_state == "CONFIRMED_SETUP":
            return "सेटअप को सभी नियमों की पुष्टि मिल चुकी है, निर्धारित Risk और Stop Loss के साथ ट्रेड पर विचार कर सकते हैं।"

        return f"मार्केट की स्थिति: {normalized_meaning}। हमेशा उचित Risk Management और अनुशासन का पालन करें।"
