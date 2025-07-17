"""Base Pydantic models for the Memory MCP Server."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MemoryType(str, Enum):
    """Enumeration of memory types."""

    ALIAS = "alias"
    NOTE = "note"
    OBSERVATION = "observation"
    HINT = "hint"


class BaseMemory(BaseModel):
    """Base model for all memory types."""

    id: Optional[int] = Field(None, description="Unique identifier for the memory")
    user_id: Optional[str] = Field(
        None, description="User identifier for data separation"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate tags are non-empty strings."""
        if v is None:
            return []
        return [tag.strip() for tag in v if tag and tag.strip()]

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id is a non-empty string if provided."""
        if v is not None and isinstance(v, str):
            v = v.strip()
            if not v:  # Empty string after stripping
                return None
        return v

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.utcnow()

    def add_tag(self, tag: str) -> None:
        """Add a tag if it doesn't already exist."""
        tag = tag.strip()
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag if it exists."""
        tag = tag.strip()
        if tag in self.tags:
            self.tags.remove(tag)

    def has_tag(self, tag: str) -> bool:
        """Check if the memory has a specific tag."""
        return tag.strip() in self.tags
