"""Alias model for bidirectional term mappings."""

from typing import Optional
from pydantic import Field, field_validator
from .base import BaseMemory


class Alias(BaseMemory):
    """Model for storing bidirectional aliases between terms or phrases."""

    source: str = Field(..., description="Source term or phrase", min_length=1)
    target: str = Field(..., description="Target term or phrase", min_length=1)
    bidirectional: bool = Field(
        True, description="Whether the alias works in both directions"
    )

    @field_validator("source", "target")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate source and target are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Source and target cannot be empty")
        return v.strip()

    @field_validator("target")
    @classmethod
    def validate_different_source_target(cls, v, info):
        """Ensure source and target are different."""
        if (
            info.data
            and "source" in info.data
            and v.strip().lower() == info.data["source"].strip().lower()
        ):
            raise ValueError("Source and target cannot be the same")
        return v

    def get_mapping(self, query: str) -> Optional[str]:
        """Get the mapping for a given query term.

        Args:
            query: The term to look up

        Returns:
            The mapped term if found, None otherwise
        """
        query = query.strip().lower()
        source_lower = self.source.strip().lower()
        target_lower = self.target.strip().lower()

        if query == source_lower:
            return self.target
        elif self.bidirectional and query == target_lower:
            return self.source
        return None

    def matches_query(self, query: str) -> bool:
        """Check if this alias matches a query term.

        Args:
            query: The term to check

        Returns:
            True if the alias matches the query
        """
        return self.get_mapping(query) is not None

    def is_word_alias(self) -> bool:
        """Check if this is a word-to-word alias (no spaces)."""
        return " " not in self.source and " " not in self.target

    def is_phrase_alias(self) -> bool:
        """Check if this is a phrase alias (contains spaces)."""
        return " " in self.source or " " in self.target

    def __str__(self) -> str:
        """String representation of the alias."""
        arrow = "↔" if self.bidirectional else "→"
        return f"{self.source} {arrow} {self.target}"
