"""Pytest configuration and fixtures for memory_mcp_server tests."""

import asyncio
import os
import tempfile
from datetime import datetime, UTC
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from memory_mcp_server.database.connection import DatabaseManager
from memory_mcp_server.database.models import Base
from memory_mcp_server.models.alias import Alias
from memory_mcp_server.models.hint import Hint
from memory_mcp_server.models.note import Note
from memory_mcp_server.models.observation import Observation


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path():
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sync_engine(temp_db_path):
    """Create a synchronous SQLAlchemy engine for testing."""
    engine = create_engine(f"sqlite:///{temp_db_path}", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def sync_session(sync_engine) -> Generator[Session, None, None]:
    """Create a synchronous database session for testing."""
    SessionLocal = sessionmaker(bind=sync_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest_asyncio.fixture
async def async_engine(temp_db_path):
    """Create an asynchronous SQLAlchemy engine for testing."""
    engine = create_async_engine(f"sqlite+aiosqlite:///{temp_db_path}", echo=False)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an asynchronous database session for testing."""
    async with AsyncSession(async_engine) as session:
        try:
            yield session
        finally:
            await session.rollback()


@pytest.fixture
def sample_alias():
    """Create a sample Alias for testing."""
    return Alias(
        source="ML",
        target="Machine Learning",
        user_id="test_user",
        bidirectional=True,
        tags=["tech", "ai"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )


@pytest.fixture
def sample_note():
    """Create a sample Note for testing."""
    return Note(
        title="Test Note",
        content="This is a test note about machine learning concepts.",
        category="education",
        user_id="test_user",
        tags=["learning", "notes"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )


@pytest.fixture
def sample_observation():
    """Create a sample Observation for testing."""
    return Observation(
        content="User shows strong interest in AI topics",
        entity_type="user",
        entity_id="test_user",
        context={"interest_level": "high", "topics": ["AI", "ML"]},
        user_id="test_user",
        tags=["behavior", "interests"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )


@pytest.fixture
def sample_hint():
    """Create a sample Hint for testing."""
    return Hint(
        content="When discussing ML, always provide practical examples",
        category="teaching",
        priority=4,
        workflow_context="education",
        user_id="test_user",
        tags=["teaching", "ml"],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC)
    )


@pytest.fixture
def db_manager(temp_db_path):
    """Create a DatabaseManager instance for testing."""
    return DatabaseManager(f"sqlite:///{temp_db_path}")


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    class MockEmbeddingConfig:
        model_name = "all-MiniLM-L6-v2"
        similarity_threshold = 0.3
        max_results = 10
        cache_embeddings = True
    
    class MockConfig:
        embedding = MockEmbeddingConfig()
    
    return MockConfig()