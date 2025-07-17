"""Note model for storing arbitrary user notes."""

from typing import Optional
from pydantic import Field, field_validator
from .base import BaseMemory


class Note(BaseMemory):
    """Model for storing user notes and personal information."""

    title: str = Field(
        ..., description="Title of the note", min_length=1, max_length=200
    )
    content: str = Field(..., description="Content of the note", min_length=1)
    category: Optional[str] = Field(None, description="Category for organizing notes")

    @field_validator("title", "content")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate title and content are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Title and content cannot be empty")
        return v.strip()

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        """Validate category is a non-empty string if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v

    def get_preview(self, max_length: int = 100) -> str:
        """Get a preview of the note content.

        Args:
            max_length: Maximum length of the preview

        Returns:
            Truncated content with ellipsis if needed
        """
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length].rstrip() + "..."

    def word_count(self) -> int:
        """Get the word count of the note content."""
        return len(self.content.split())

    def char_count(self) -> int:
        """Get the character count of the note content."""
        return len(self.content)

    def has_category(self) -> bool:
        """Check if the note has a category assigned."""
        return self.category is not None and self.category.strip() != ""

    def matches_search(self, query: str) -> bool:
        """Check if the note matches a search query.

        Args:
            query: Search query to match against

        Returns:
            True if the note matches the query
        """
        query_lower = query.lower()
        return (
            query_lower in self.title.lower()
            or query_lower in self.content.lower()
            or (self.category and query_lower in self.category.lower())
        )

    def __str__(self) -> str:
        """String representation of the note."""
        category_str = f" [{self.category}]" if self.category else ""
        return f"{self.title}{category_str}: {self.get_preview(50)}"
