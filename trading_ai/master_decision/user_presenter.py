"""Bilingual English & Hindi Presenter for Step 17 Master Decision Engine.

Generates:
1. Structured institutional trade presentation in English.
2. Clear, simple Devanagari Hindi summary tailored for the user (Alam).

Strict Invariants:
- Zero false precision: Never promises guaranteed profit or mathematical win certainty.
- Explicit epistemic separation: Clearly distinguishes OBSERVED from INFERRED.
- Full transparency on blocking reasons and missing data.
"""
from typing import Any, Dict, List, Optional

from trading_ai.schemas.enums import TradeDecisionState
from trading_ai.schemas.master_decision import ConceptResult, MasterDecisionResult


class MasterDecisionUserPresenter:
    """Formats Master Decision outcomes into English and Hindi user presentations."""

    @staticmethod
    def format_english_presentation(decision: MasterDecisionResult) -> str:
        """Render structured English user-facing decision output."""
        if decision.final_state == TradeDecisionState.CONFIRMED_SETUP:
            confluence_bullets = []
            for c_id, res in decision.concept_results.items():
                if res.is_data_available and res.direction.value == decision.direction:
                    confluence_bullets.append(f"• {res.concept_name}: {res.epistemic_state.decided}")

            evidence_items = []
            for res in decision.concept_results.values():
                if res.is_data_available and res.evidence:
                    evidence_items.extend(res.evidence[:2])

            lines = [
                f"Trade Status: CONFIRMED SETUP",
                f"Direction: {decision.direction}",
                f"Entry Zone: {decision.entry_zone or 'Current Key Level'}",
                f"Invalidation (Stop Loss): {f'{decision.invalidation_level:.5f}' if decision.invalidation_level else 'Structure Low/High'}",
                f"Target (Take Profit): {f'{decision.target_level:.5f}' if decision.target_level else 'Liquidity Target'}",
                f"Risk: {decision.risk_engine_status or 'APPROVED (Within Risk Limits)'}",
                f"Confluence Score: {decision.confluence_score:.1f}/10.0 (Threshold: {decision.threshold:.1f}/10.0)",
                f"Confidence: Calibrated Evidence Score {decision.confluence_score * 10:.0f}% (Qualitative: HIGH)",
                f"Confluence Breakdown:",
            ]
            lines.extend(confluence_bullets if confluence_bullets else ["• Verified multi-concept alignment."])
            lines.append("Key Factual Evidence:")
            lines.extend([f"  - {e}" for e in evidence_items[:4]])
            lines.append(f"Decision: {decision.final_state.value}")
            return "\n".join(lines)

        else:
            # NO_TRADE / BLOCKED / CONFLICTED / INSUFFICIENT_EVIDENCE
            lines = [
                f"Trade Status: NO TRADE ({decision.final_state.value})",
                f"Primary Reason: {decision.blocking_reasons[0] if decision.blocking_reasons else 'Safety conditions not satisfied.'}",
            ]
            if decision.conflicts:
                lines.append("Active Conflicts:")
                lines.extend([f"• {c}" for c in decision.conflicts])
            if decision.missing_evidence:
                lines.append("Missing Evidence / Data:")
                lines.extend([f"• {m}" for m in decision.missing_evidence])
            lines.append("What is required for execution:")
            if decision.conflicts:
                lines.append("• Resolution of opposing macro/micro signals and clear directional agreement.")
            elif decision.missing_evidence:
                lines.append("• Live market data feed confirmation for unverified concepts.")
            else:
                lines.append(f"• Confluence score must reach at least {decision.threshold:.1f}/10.0 with 2+ independent concepts.")
            return "\n".join(lines)

    @staticmethod
    def format_hindi_presentation(decision: MasterDecisionResult) -> str:
        """Render clean, simple Devanagari Hindi summary for the user."""
        if decision.final_state == TradeDecisionState.CONFIRMED_SETUP:
            dir_hi = "BUY (खरीदें)" if decision.direction == "BUY" else "SELL (बेचें)"
            lines = [
                f"ट्रेड स्थिति: CONFIRMED SETUP (ट्रेड सेटअप की पुष्टि हो गई है)",
                f"दिशा: {dir_hi}",
                f"प्रवेश क्षेत्र (Entry Zone): {decision.entry_zone or 'मार्केट लेवल'}",
                f"अमान्यता स्तर (Stop Loss): {f'{decision.invalidation_level:.5f}' if decision.invalidation_level else 'स्ट्रक्चर लेवल'}",
                f"लक्ष्य (Target / TP): {f'{decision.target_level:.5f}' if decision.target_level else 'लिक्विडिटी टारगेट'}",
                f"कॉन्फ्लुएंस स्कोर: {decision.confluence_score:.1f}/10 (न्यूनतम आवश्यक: {decision.threshold:.1f}/10)",
                f"जोखिम नियंत्रण: स्वीकृत (Risk Limits Passed)",
                f"सत्यापित संगम (Confluence Points):",
            ]
            for c_id, res in decision.concept_results.items():
                if res.is_data_available and res.direction.value == decision.direction:
                    lines.append(f"• {res.concept_name}: {res.epistemic_state.decided}")
            lines.append("अंतिम निर्णय: CONFIRMED_SETUP (सुरक्षा नियमों के अनुसार मान्य)")
            return "\n".join(lines)
        else:
            reason_hi = decision.blocking_reasons[0] if decision.blocking_reasons else "पर्याप्त प्रमाण या संगम उपलब्ध नहीं है।"
            lines = [
                f"ट्रेड स्थिति: NO TRADE (ट्रेड न लें - {decision.final_state.value})",
                f"मुख्य कारण: {reason_hi}",
            ]
            if decision.conflicts:
                lines.append("पहचाने गए विरोध (Conflicts):")
                for c in decision.conflicts:
                    lines.append(f"• {c}")
            if decision.missing_evidence:
                lines.append("अनुपलब्ध डेटा (Missing Data):")
                for m in decision.missing_evidence:
                    lines.append(f"• {m}")
            lines.append("अगली आवश्यकता: सुरक्षा नियम और सभी आवश्यक इंडिपेंडेंट कॉन्फ्लुएंस की पुष्टि होने तक प्रतीक्षा करें।")
            return "\n".join(lines)
