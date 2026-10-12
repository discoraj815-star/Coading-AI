"""Base Concept Module Contract & Interfaces for Step 17.

Every trading concept (SMC/ICT, Orderflow, Footprint, Volume Profile, Wyckoff, Plugins)
must inherit from BaseConceptModule and produce a strictly normalized ConceptResult.

Hard Epistemic Invariants:
1. Data Availability Rule: A module may ONLY produce a directional bias when its required data actually exists.
2. Four-Layer Epistemic Model: Must populate OBSERVED, LEARNED, INFERRED, DECIDED separately.
3. No False Precision: Calibrated confidence; never claims guaranteed win rate.
4. Authority Hierarchy: Preserves P1 User Override -> P2 Mentorship -> P3 Journal -> P4 Lecture -> P5 General.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from trading_ai.schemas.enums import ConceptDirection
from trading_ai.schemas.master_decision import ConceptResult, EpistemicConceptState


class BaseConceptModule(ABC):
    """Abstract base class for all modular trading theory/concept evaluators."""

    def __init__(
        self,
        concept_id: str,
        concept_name: str,
        authority_priority: int = 4,
        required_data_fields: Optional[List[str]] = None,
        is_approved: bool = True,
        is_macro: bool = False,
    ):
        self.concept_id = concept_id
        self.concept_name = concept_name
        self.authority_priority = authority_priority  # 1=User, 2=Mentor, 3=Journal, 4=Lecture, 5=General
        self.required_data_fields = required_data_fields or []
        self.is_approved = is_approved
        self.is_macro = is_macro  # True = Macro/Context layer; False = Micro/Execution layer

    def validate_data_availability(
        self, market_data: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Validate whether all mandatory data fields for this concept exist in the input.

        Returns:
            (is_available, available_fields, missing_fields)
        """
        available = []
        missing = []
        for field in self.required_data_fields:
            val = market_data.get(field)
            if val is not None and val != "" and val != [] and val != {}:
                available.append(field)
            else:
                missing.append(field)

        is_available = len(missing) == 0
        return is_available, available, missing

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        timeframe: str,
        market_data: Dict[str, Any],
        candidate_rules: Optional[List[Any]] = None,
        as_of_timestamp: Optional[float] = None,
    ) -> ConceptResult:
        """
        Execute concept analysis and return normalized ConceptResult.
        Must respect as_of_timestamp to prevent lookahead in backtests.
        """
        pass
