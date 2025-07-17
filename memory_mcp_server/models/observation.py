"""Observation model for storing contextual information about entities."""

from typing import Any, Dict, Optional

from pydantic import Field, field_validator

from .base import BaseMemory


class Observation(BaseMemory):
    """Model for storing observations related to other entities."""

    content: str = Field(..., description="Content of the observation", min_length=1)
    entity_type: str = Field(
        ..., description="Type of the related entity", min_length=1
    )
    entity_id: str = Field(
        ..., description="Identifier of the related entity", min_length=1
    )
    context: Optional[Dict[str, Any]] = Field(
        None, description="Additional context information"
    )

    @field_validator("content", "entity_type", "entity_id")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate text fields are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Content, entity_type, and entity_id cannot be empty")
        return v.strip()

    @field_validator("context")
    @classmethod
    def validate_context(cls, v):
        """Validate context is a valid dictionary if provided."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("Context must be a dictionary")
        return v

    def get_entity_reference(self) -> str:
        """Get a string reference to the related entity.

        Returns:
            String in format "entity_type:entity_id"
        """
        return f"{self.entity_type}:{self.entity_id}"

    def has_context(self) -> bool:
        """Check if the observation has context information."""
        return bool(self.context)

    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get a value from the context dictionary.

        Args:
            key: The context key to retrieve
            default: Default value if key is not found

        Returns:
            The context value or default
        """
        return self.context.get(key, default) if self.context else default

    def set_context_value(self, key: str, value: Any) -> None:
        """Set a value in the context dictionary.

        Args:
            key: The context key to set
            value: The value to set
        """
        if self.context is None:
            self.context = {}
        self.context[key] = value
        self.update_timestamp()

    def remove_context_value(self, key: str) -> None:
        """Remove a value from the context dictionary.

        Args:
            key: The context key to remove
        """
        if self.context and key in self.context:
            del self.context[key]
            self.update_timestamp()

    def matches_entity(self, entity_type: str = None, entity_id: str = None) -> bool:
        """Check if the observation matches entity criteria.

        Args:
            entity_type: Entity type to match (optional)
            entity_id: Entity ID to match (optional)

        Returns:
            True if the observation matches the criteria
        """
        if entity_type and self.entity_type.lower() != entity_type.lower():
            return False
        if entity_id and self.entity_id.lower() != entity_id.lower():
            return False
        return True

    def matches_search(self, query: str) -> bool:
        """Check if the observation matches a search query.

        Args:
            query: Search query to match against

        Returns:
            True if the observation matches the query
        """
        query_lower = query.lower()
        return (
            query_lower in self.content.lower()
            or query_lower in self.entity_type.lower()
            or query_lower in self.entity_id.lower()
            or (
                self.context
                and any(
                    query_lower in str(v).lower()
                    for v in self.context.values()
                    if v is not None
                )
            )
        )

    def __str__(self) -> str:
        """String representation of the observation."""
        return f"[{self.entity_type}:{self.entity_id}] {self.content[:50]}{'...' if len(self.content) > 50 else ''}"
