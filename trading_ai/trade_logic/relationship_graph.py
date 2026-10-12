"""Cross-Source Knowledge Relationship Graph for Step 14.

Manages semantic and epistemic links between knowledge items across the 4 isolated domains:
SUPPORTS, CONTRADICTS, REFINES, OVERRIDES, EXCEPTION_TO, DERIVED_FROM, VALIDATED_BY, FAILED_BY, RELATED_TO.
"""
from typing import Dict, List, Optional, Tuple

from trading_ai.schemas.enums import RelationshipType
from trading_ai.schemas.trade_logic import KnowledgeRelationship
from trading_ai.storage.repository import Repository


class CrossSourceRelationshipGraph:
    """In-memory and persisted graph tracking cross-source knowledge relationships."""

    def __init__(self, repository: Optional[Repository] = None, repo: Optional[Repository] = None):
        self.repository = repository or repo
        self._memory_relationships: List[KnowledgeRelationship] = []

    def add_relationship(
        self,
        source_item_id: str,
        target_item_id: str,
        relationship_type: RelationshipType,
        description: str = "",
    ) -> KnowledgeRelationship:
        """Create and store a directed relationship between two knowledge units."""
        rel = KnowledgeRelationship(
            source_item_id=source_item_id,
            target_item_id=target_item_id,
            relationship_type=relationship_type,
            description=description,
        )
        self._memory_relationships.append(rel)
        if self.repository and hasattr(self.repository, "save_knowledge_relationship"):
            self.repository.save_knowledge_relationship(rel)
        return rel

    def link_items(
        self,
        source_item_id: str,
        target_item_id: str,
        relationship_type: RelationshipType,
        description: str = "",
    ) -> KnowledgeRelationship:
        """Convenience alias for add_relationship."""
        return self.add_relationship(source_item_id, target_item_id, relationship_type, description)


    def get_relationships_for_item(self, item_id: str) -> List[KnowledgeRelationship]:
        """Fetch all relationships where item is either source or target."""
        if self.repository and hasattr(self.repository, "list_knowledge_relationships"):
            repo_rels = self.repository.list_knowledge_relationships(item_id=item_id)
            if repo_rels:
                return repo_rels
        return [
            r for r in self._memory_relationships
            if r.source_item_id == item_id or r.target_item_id == item_id
        ]

    def find_contradictions(self, item_id: str) -> List[KnowledgeRelationship]:
        """Return all contradiction links associated with an item."""
        all_rels = self.get_relationships_for_item(item_id)
        return [r for r in all_rels if r.relationship_type == RelationshipType.CONTRADICTS]

    def find_refinements_or_overrides(self, item_id: str) -> List[KnowledgeRelationship]:
        """Return all refinement, override, or exception links targeting this item."""
        all_rels = self.get_relationships_for_item(item_id)
        return [
            r for r in all_rels
            if r.target_item_id == item_id
            and r.relationship_type in (
                RelationshipType.REFINES,
                RelationshipType.OVERRIDES,
                RelationshipType.EXCEPTION_TO,
            )
        ]
