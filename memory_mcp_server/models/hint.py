"""Hint model for storing LLM interaction guidance."""

from typing import Optional
from pydantic import Field, field_validator
from .base import BaseMemory


class Hint(BaseMemory):
    """Model for storing hints for LLM interactions and workflows."""

    content: str = Field(..., description="Content of the hint", min_length=1)
    category: str = Field(..., description="Category of the hint", min_length=1)
    priority: int = Field(1, description="Priority level (1=low, 5=high)", ge=1, le=5)
    workflow_context: Optional[str] = Field(
        None, description="Specific workflow context"
    )

    @field_validator("content", "category")
    @classmethod
    def validate_text_fields(cls, v):
        """Validate content and category are non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Content and category cannot be empty")
        return v.strip()

    @field_validator("workflow_context")
    @classmethod
    def validate_workflow_context(cls, v):
        """Validate workflow_context is a non-empty string if provided."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v):
        """Validate priority is within valid range."""
        if not 1 <= v <= 5:
            raise ValueError("Priority must be between 1 and 5")
        return v

    def is_high_priority(self) -> bool:
        """Check if this is a high priority hint (4 or 5)."""
        return self.priority >= 4

    def is_low_priority(self) -> bool:
        """Check if this is a low priority hint (1 or 2)."""
        return self.priority <= 2

    def get_priority_label(self) -> str:
        """Get a human-readable priority label."""
        priority_labels = {
            1: "Very Low",
            2: "Low",
            3: "Medium",
            4: "High",
            5: "Very High",
        }
        return priority_labels.get(self.priority, "Unknown")

    def has_workflow_context(self) -> bool:
        """Check if the hint has workflow context specified."""
        return self.workflow_context is not None and self.workflow_context.strip() != ""

    def matches_category(self, category: str) -> bool:
        """Check if the hint matches a specific category.

        Args:
            category: Category to match against

        Returns:
            True if the hint matches the category
        """
        return self.category.lower() == category.lower()

    def matches_workflow(self, workflow: str) -> bool:
        """Check if the hint matches a specific workflow context.

        Args:
            workflow: Workflow context to match against

        Returns:
            True if the hint matches the workflow context
        """
        if not self.workflow_context:
            return False
        return self.workflow_context.lower() == workflow.lower()

    def matches_search(self, query: str) -> bool:
        """Check if the hint matches a search query.

        Args:
            query: Search query to match against

        Returns:
            True if the hint matches the query
        """
        query_lower = query.lower()
        return (
            query_lower in self.content.lower()
            or query_lower in self.category.lower()
            or (self.workflow_context and query_lower in self.workflow_context.lower())
        )

    def __str__(self) -> str:
        """String representation of the hint."""
        workflow_str = f" ({self.workflow_context})" if self.workflow_context else ""
        priority_str = "!" * self.priority if self.priority > 3 else ""
        return f"[{self.category}]{workflow_str}{priority_str}: {self.content[:50]}{'...' if len(self.content) > 50 else ''}"
