"""User command parser and intent extractor for Trading AI Assistant."""

import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.enums import CommandIntent
from trading_ai.schemas.commands import CommandParseResult, UserCommand

logger = logging.getLogger(__name__)


class UserCommandParser:
    """
    Parses and categorizes explicit user instructions into structured system actions.

    Optimized for token conservation: uses high-precision pattern matching
    first (0 tokens) and falls back to model only when ambiguous.
    """

    INTENT_PATTERNS = [
        # Save rule
        (
            CommandIntent.SAVE_RULE,
            [
                r"^\s*save\s+(this\s+)?rule\b",
                r"^\s*save\s+rule[:\s]",
                r"^\s*add\s+rule[:\s]",
                r"^\s*record\s+(this\s+)?rule\b",
            ],
        ),
        # Correction
        (
            CommandIntent.CORRECT,
            [
                r"^\s*correct\s+(this\s+)?(mistake|rule|interpretation|one)?\b",
                r"^\s*no[,.]?\s+correct\s+it\b",
                r"^\s*you\s+are\s+wrong\b",
                r"^\s*that\s+interpretation\s+was\s+wrong\b",
                r"^\s*that('?s|\s+is)\s+incorrect\b",
                r"^\s*fix\s+(this\s+)?rule\b",
                r"^\s*update\s+(this\s+)?rule\b",
            ],
        ),
        # Remember / Add Knowledge
        (
            CommandIntent.REMEMBER,
            [
                r"^\s*remember\s+(this|that)?\b",
                r"^\s*keep\s+in\s+mind\b",
            ],
        ),
        # Add to trading knowledge
        (
            CommandIntent.ADD_KNOWLEDGE,
            [
                r"^\s*add\s+(this\s+)?to\s+(my\s+)?trading\s+knowledge\b",
                r"^\s*store\s+(this\s+)?knowledge\b",
            ],
        ),
        # Forget
        (
            CommandIntent.FORGET,
            [
                r"^\s*forget\s+(this\s+)?(incorrect\s+)?(interpretation|rule|mistake)?\b",
                r"^\s*delete\s+(this\s+)?rule\b",
                r"^\s*remove\s+(this\s+)?knowledge\b",
            ],
        ),
        # Review
        (
            CommandIntent.REVIEW,
            [
                r"^\s*review\s+(this\s+)?again\b",
                r"^\s*re-?evaluate\b",
                r"^\s*check\s+(this\s+)?again\b",
            ],
        ),
        # Explain
        (
            CommandIntent.EXPLAIN,
            [
                r"^\s*explain\s+why\s+(this\s+)?rule\s+applies\b",
                r"^\s*explain\s+(this\s+)?rule\b",
                r"^\s*why\s+does\s+this\s+rule\s+apply\b",
            ],
        ),
        # Mark exception
        (
            CommandIntent.MARK_EXCEPTION,
            [
                r"^\s*mark\s+(this\s+)?as\s+(an\s+)?exception\b",
                r"^\s*this\s+is\s+an\s+exception\b",
                r"^\s*add\s+exception[:\s]",
            ],
        ),
        # Use rule in future analysis
        (
            CommandIntent.USE_RULE,
            [
                r"^\s*use\s+this\s+rule\s+in\s+future\s+analysis\b",
                r"^\s*apply\s+this\s+rule\s+in\s+future\b",
                r"^\s*always\s+use\s+this\s+rule\b",
            ],
        ),
        # Step 4: Market Analysis & Monitoring Commands
        (
            CommandIntent.ANALYZE_WATCHLIST,
            [
                r"^\s*analyze\s+(?:my\s+)?(?:whole\s+)?watchlist\b",
                r"^\s*scan\s+(?:all\s+)?watchlist\b",
            ],
        ),
        (
            CommandIntent.ANALYZE_CHART,
            [
                r"^\s*analyze\s+(?:this|the|current)\s+chart\b",
            ],
        ),
        (
            CommandIntent.ANALYZE_SYMBOL,
            [
                r"^\s*analyze\s+(?:symbol\s+)?([A-Z0-9/_-]{2,15})\b",
            ],
        ),
        (
            CommandIntent.CHECK_HTF,
            [
                r"^\s*check\s+(?:the\s+)?(?:higher\s+timeframe|htf)\b",
            ],
        ),
        (
            CommandIntent.FILTER_VALID_SETUPS,
            [
                r"^\s*(?:show|filter|display)\s+valid\s+setups(?:\s+only)?\b",
            ],
        ),
        (
            CommandIntent.USE_LECTURE_RULES_ONLY,
            [
                r"^\s*use\s+(?:my\s+)?lecture\s+rules\s+only\b",
                r"^\s*lecture\s+rules\s+only\b",
            ],
        ),
        (
            CommandIntent.EXPLAIN_SETUP,
            [
                r"^\s*why\s+is\s+this\s+setup\s+(valid|invalid)\b",
                r"^\s*explain\s+setup\b",
            ],
        ),
        (
            CommandIntent.MARK_PATTERN,
            [
                r"^\s*mark\s+(?:valid\s+)?([A-Za-z0-9_\s]{2,20})\b",
            ],
        ),
        (
            CommandIntent.STOP_MONITORING,
            [
                r"^\s*stop\s+(?:background\s+)?monitoring\b",
                r"^\s*pause\s+monitoring\b",
            ],
        ),
        (
            CommandIntent.RESUME_MONITORING,
            [
                r"^\s*(?:resume|start)\s+(?:background\s+)?monitoring\b",
            ],
        ),
        # Step 5 Command Intents
        (
            CommandIntent.SHOW_ALL_LIQUIDITY,
            [
                r"^\s*(?:show|display)\s+(?:all\s+)?liquidity\b",
                r"^\s*reveal\s+liquidity\b",
            ],
        ),
        (
            CommandIntent.SHOW_HIDDEN_SETUPS,
            [
                r"^\s*(?:show|display|reveal)\s+(?:all\s+)?hidden\s+setups\b",
                r"^\s*show\s+background\s+setups\b",
            ],
        ),
        (
            CommandIntent.SHOW_FVGS,
            [
                r"^\s*(?:show|display)\s+(?:all\s+)?fvgs?\b",
                r"^\s*(?:show|display)\s+fair\s+value\s+gaps?\b",
            ],
        ),
        (
            CommandIntent.SHOW_ORDER_BLOCKS,
            [
                r"^\s*(?:show|display)\s+(?:all\s+)?(?:order\s+blocks?|obs?)\b",
            ],
        ),
        (
            CommandIntent.SHOW_UNCONFIRMED_SETUPS,
            [
                r"^\s*(?:show|display)\s+unconfirmed\s+setups?\b",
            ],
        ),
        (
            CommandIntent.SHOW_CONFIRMATION_EVIDENCE,
            [
                r"^\s*(?:show|explain)\s+(?:why\s+this\s+setup\s+is\s+confirmed|confirmation\s+evidence)\b",
            ],
        ),
        (
            CommandIntent.SHOW_INVALIDATION_CONDITION,
            [
                r"^\s*(?:show|explain)\s+invalidation\s+conditions?\b",
                r"^\s*what\s+invalidates\s+this\s+setup\b",
            ],
        ),
        (
            CommandIntent.HIDE_BACKGROUND_MARKINGS,
            [
                r"^\s*hide\s+(?:background\s+)?markings\b",
                r"^\s*clean\s+chart\b",
                r"^\s*reset\s+chart\s+view\b",
            ],
        ),
        (
            CommandIntent.FOCUS_CURRENT_SETUP,
            [
                r"^\s*focus\s+(?:on\s+)?(?:the\s+)?current\s+setup\b",
            ],
        ),
        (
            CommandIntent.SHOW_HTF_CONTEXT,
            [
                r"^\s*show\s+(?:higher\s+timeframe|htf)\s+context\b",
            ],
        ),
    ]

    def parse(self, raw_text: str, target_knowledge_id: Optional[str] = None) -> CommandParseResult:
        """Parse raw user input into a validated UserCommand."""
        clean_text = raw_text.strip()
        command_id = f"cmd_{uuid.uuid4().hex[:12]}"

        if not clean_text:
            cmd = UserCommand(
                command_id=command_id,
                raw_text=raw_text,
                intent=CommandIntent.UNKNOWN,
                parameters={"error": "Empty command text"},
            )
            return CommandParseResult(
                command=cmd,
                parsed_success=False,
                suggested_action="none",
                explanation="Received empty command text.",
            )

        # 1. Match intent
        detected_intent = self._detect_intent(clean_text)

        # 2. Extract structured parameters based on intent
        params = self._extract_parameters(clean_text, detected_intent, target_knowledge_id)

        # Extract explicit target_id from parameters if found
        extracted_target_id = target_knowledge_id or params.get("target_knowledge_id")

        cmd = UserCommand(
            command_id=command_id,
            raw_text=raw_text,
            intent=detected_intent,
            target_knowledge_id=extracted_target_id,
            parameters=params,
            source="user",
        )

        action, explanation = self._describe_action(cmd)
        return CommandParseResult(
            command=cmd,
            parsed_success=detected_intent != CommandIntent.UNKNOWN,
            suggested_action=action,
            explanation=explanation,
        )

    def _detect_intent(self, text: str) -> CommandIntent:
        """Test regex patterns to classify user intent."""
        for intent, patterns in self.INTENT_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return intent

        # Secondary keywords check if no anchor match
        lower = text.lower()
        if "correct" in lower or "wrong" in lower or "mistake" in lower:
            return CommandIntent.CORRECT
        if "save rule" in lower or "save this rule" in lower:
            return CommandIntent.SAVE_RULE
        if "remember" in lower:
            return CommandIntent.REMEMBER
        if "trading knowledge" in lower:
            return CommandIntent.ADD_KNOWLEDGE
        if "exception" in lower:
            return CommandIntent.MARK_EXCEPTION
        if "explain" in lower:
            return CommandIntent.EXPLAIN

        return CommandIntent.UNKNOWN

    def _extract_parameters(
        self, text: str, intent: CommandIntent, target_knowledge_id: Optional[str]
    ) -> Dict[str, Any]:
        """Extract rule, conditions, exceptions, and corrections from user prompt."""
        params: Dict[str, Any] = {}

        # Look for explicit item ID in text (e.g. item_abc123 or [item_123])
        id_match = re.search(r"\b(item_[a-zA-Z0-9_-]+)\b", text)
        if id_match:
            params["target_knowledge_id"] = id_match.group(1)
        elif target_knowledge_id:
            params["target_knowledge_id"] = target_knowledge_id

        # Rule extraction
        # Look for content after colons, quotes, or keywords
        content = text
        for intent_type, patterns in self.INTENT_PATTERNS:
            if intent_type == intent:
                for pat in patterns:
                    m = re.search(pat, text, re.IGNORECASE)
                    if m:
                        content = text[m.end():].strip(" :-\n\t\"'")
                        break
                break

        # Check for Conditions: "when ...", "if ...", "condition: ...", "only when ..."
        conditions: List[str] = []
        cond_matches = re.findall(
            r"(?:only\s+when|when|if|condition:)\s+([^.;,\n]+(?:\s+exists)?)",
            text,
            re.IGNORECASE,
        )
        if cond_matches:
            for c in cond_matches:
                cleaned = c.strip()
                if cleaned and cleaned not in conditions:
                    conditions.append(cleaned)
        params["conditions"] = conditions if conditions else None

        # Check for Exceptions: "unless ...", "except ...", "exception: ..."
        exceptions: List[str] = []
        exc_matches = re.findall(
            r"(?:unless|except\s+when|except|exception:)\s+([^.;,\n]+)",
            text,
            re.IGNORECASE,
        )
        if exc_matches:
            for e in exc_matches:
                cleaned = e.strip()
                if cleaned and cleaned not in exceptions:
                    exceptions.append(cleaned)
        params["exceptions"] = exceptions if exceptions else None

        # Store primary statement or payload
        params["statement"] = content if content else text
        if intent == CommandIntent.SAVE_RULE:
            params["rule"] = content if content else text
        elif intent == CommandIntent.CORRECT:
            params["correction_statement"] = content if content else text
            # Extract possible reason if phrased as "No, correct it. Rule X applies only when..."
            params["reason"] = f"User direct correction: {text}"
        elif intent == CommandIntent.MARK_EXCEPTION:
            params["exception_details"] = content if content else text
        elif intent == CommandIntent.ANALYZE_SYMBOL:
            sym_match = re.search(r"^\s*analyze\s+(?:symbol\s+)?([A-Z0-9/_-]+)", text, re.IGNORECASE)
            if sym_match:
                params["symbol"] = sym_match.group(1).upper()
        elif intent == CommandIntent.MARK_PATTERN:
            pat_match = re.search(r"^\s*mark\s+(?:valid\s+)?([A-Za-z0-9_\s]+)", text, re.IGNORECASE)
            if pat_match:
                params["pattern"] = pat_match.group(1).strip()
        elif intent == CommandIntent.EXPLAIN_SETUP:
            if "invalid" in text.lower():
                params["query_type"] = "why_invalid"
            else:
                params["query_type"] = "why_valid"

        return params

    def _describe_action(self, cmd: UserCommand) -> Tuple[str, str]:
        """Generate human-readable action name and explanation."""
        if cmd.intent == CommandIntent.SAVE_RULE:
            return "save_rule", "Save persistent trading rule with conditions and exceptions."
        if cmd.intent == CommandIntent.CORRECT:
            return "apply_correction", "Create versioned correction preserving previous knowledge provenance."
        if cmd.intent == CommandIntent.REMEMBER:
            return "remember", "Store information in long-term memory."
        if cmd.intent == CommandIntent.ADD_KNOWLEDGE:
            return "add_knowledge", "Incorporate new rule/concept into active trading knowledge."
        if cmd.intent == CommandIntent.FORGET:
            return "archive_knowledge", "Archive and deactivate superseded or erroneous interpretation."
        if cmd.intent == CommandIntent.REVIEW:
            return "review", "Re-examine knowledge item and supporting multimodal evidence."
        if cmd.intent == CommandIntent.EXPLAIN:
            return "explain", "Retrieve evidence and reasoning behind rule application."
        if cmd.intent == CommandIntent.MARK_EXCEPTION:
            return "mark_exception", "Attach an exception case to trading rule."
        if cmd.intent == CommandIntent.USE_RULE:
            return "activate_rule", "Mark rule as active priority for future market analysis."
        if cmd.intent == CommandIntent.ANALYZE_SYMBOL:
            sym = cmd.parameters.get("symbol", "selected asset")
            return "analyze_symbol", f"Perform multi-timeframe lecture-constrained analysis for {sym}."
        if cmd.intent == CommandIntent.ANALYZE_CHART:
            return "analyze_chart", "Analyze visible chart structure, levels, and setups."
        if cmd.intent == CommandIntent.ANALYZE_WATCHLIST:
            return "analyze_watchlist", "Independently analyze all symbols in the background watchlist."
        if cmd.intent == CommandIntent.CHECK_HTF:
            return "check_htf", "Examine macro higher-timeframe context and trend."
        if cmd.intent == CommandIntent.FILTER_VALID_SETUPS:
            return "filter_valid_setups", "Filter chart and analysis to display valid setups only."
        if cmd.intent == CommandIntent.USE_LECTURE_RULES_ONLY:
            return "use_lecture_rules_only", "Constrain market intelligence strictly to persistent lecture rules."
        if cmd.intent == CommandIntent.EXPLAIN_SETUP:
            return "explain_setup", "Retrieve conditions, exceptions, and evidence for setup validation."
        if cmd.intent == CommandIntent.MARK_PATTERN:
            pat = cmd.parameters.get("pattern", "pattern")
            return "mark_pattern", f"Plot validated {pat} on chart."
        if cmd.intent == CommandIntent.STOP_MONITORING:
            return "stop_monitoring", "Halt background watchlist monitoring."
        if cmd.intent == CommandIntent.RESUME_MONITORING:
            return "resume_monitoring", "Resume background watchlist monitoring."
        return "general_action", "Process general instruction."
