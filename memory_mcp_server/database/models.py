"""SQLAlchemy database models for the Memory MCP Server."""

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import TypeDecorator

Base = declarative_base()


class JSONType(TypeDecorator):
    """Custom SQLAlchemy type for storing JSON data."""

    impl = Text
    cache_ok = True

    def process_bind_param(self, value, dialect):
        """Convert Python object to JSON string for storage."""
        if value is None:
            return None
        return json.dumps(value)

    def process_result_value(self, value, dialect):
        """Convert JSON string back to Python object."""
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None


class User(Base):
    """User table for data separation."""

    __tablename__ = "users"

    id = Column(String(255), primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    aliases = relationship(
        "AliasDB", back_populates="user", cascade="all, delete-orphan"
    )
    notes = relationship("NoteDB", back_populates="user", cascade="all, delete-orphan")
    observations = relationship(
        "ObservationDB", back_populates="user", cascade="all, delete-orphan"
    )
    hints = relationship("HintDB", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User(id='{self.id}', created_at='{self.created_at}')>"


class AliasDB(Base):
    """Alias table for bidirectional term mappings."""

    __tablename__ = "aliases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=True)
    source = Column(String(500), nullable=False)
    target = Column(String(500), nullable=False)
    bidirectional = Column(Boolean, default=True, nullable=False)
    tags = Column(JSONType, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="aliases")

    def __repr__(self) -> str:
        arrow = "↔" if self.bidirectional else "→"
        return (
            f"<AliasDB(id={self.id}, source='{self.source}' {arrow} '{self.target}')>"
        )


class NoteDB(Base):
    """Note table for storing user notes."""

    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=True)
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    tags = Column(JSONType, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="notes")

    def __repr__(self) -> str:
        return (
            f"<NoteDB(id={self.id}, title='{self.title}', category='{self.category}')>"
        )


class ObservationDB(Base):
    """Observation table for storing contextual information about entities."""

    __tablename__ = "observations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=True)
    content = Column(Text, nullable=False)
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(String(255), nullable=False)
    context = Column(JSONType, default=dict)
    tags = Column(JSONType, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="observations")

    def __repr__(self) -> str:
        return f"<ObservationDB(id={self.id}, entity='{self.entity_type}:{self.entity_id}')>"


class HintDB(Base):
    """Hint table for storing LLM interaction guidance."""

    __tablename__ = "hints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey("users.id"), nullable=True)
    content = Column(Text, nullable=False)
    category = Column(String(100), nullable=False)
    priority = Column(Integer, default=1, nullable=False)
    workflow_context = Column(String(200), nullable=True)
    tags = Column(JSONType, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    # Relationships
    user = relationship("User", back_populates="hints")

    def __repr__(self) -> str:
        return f"<HintDB(id={self.id}, category='{self.category}', priority={self.priority})>"


class EmbeddingDB(Base):
    """Embedding table for storing vector embeddings for RAG functionality."""

    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    memory_type = Column(String(50), nullable=False)  # alias, note, observation, hint
    memory_id = Column(Integer, nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Serialized vector data
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<EmbeddingDB(id={self.id}, memory_type='{self.memory_type}', memory_id={self.memory_id})>"
