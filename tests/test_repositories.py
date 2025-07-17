"""Unit tests for repository layer."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from memory_mcp_server.database.repositories import (
    BaseRepository,
    UserRepository,
    AliasRepository,
    NoteRepository,
    ObservationRepository,
    HintRepository,
    EmbeddingRepository
)
from memory_mcp_server.database.models import (
    User,
    AliasDB,
    NoteDB,
    ObservationDB,
    HintDB,
    EmbeddingDB
)


class TestBaseRepository:
    """Test cases for BaseRepository."""

    def test_base_repository_initialization(self):
        """Test BaseRepository initialization."""
        repo = BaseRepository(User)
        assert repo.model_class == User

    def test_create(self, sync_session):
        """Test creating a record."""
        repo = UserRepository()
        user = repo.create(sync_session, id="test_user")
        
        assert user.id == "test_user"
        assert isinstance(user.created_at, datetime)

    def test_get_by_id(self, sync_session):
        """Test getting a record by ID."""
        repo = UserRepository()
        
        # Create a user first
        created_user = repo.create(sync_session, id="test_user")
        sync_session.commit()
        
        # Get the user by ID
        retrieved_user = repo.get_by_id(sync_session, "test_user")
        
        assert retrieved_user is not None
        assert retrieved_user.id == "test_user"

    def test_get_by_id_not_found(self, sync_session):
        """Test getting a non-existent record."""
        repo = UserRepository()
        user = repo.get_by_id(sync_session, "nonexistent")
        
        assert user is None

    def test_get_all_no_filters(self, sync_session):
        """Test getting all records without filters."""
        repo = UserRepository()
        
        # Create multiple users
        repo.create(sync_session, id="user1")
        repo.create(sync_session, id="user2")
        sync_session.commit()
        
        users = repo.get_all(sync_session)
        
        assert len(users) == 2
        user_ids = [user.id for user in users]
        assert "user1" in user_ids
        assert "user2" in user_ids

    def test_get_all_with_pagination(self, sync_session):
        """Test getting records with pagination."""
        repo = UserRepository()
        
        # Create multiple users
        for i in range(5):
            repo.create(sync_session, id=f"user{i}")
        sync_session.commit()
        
        # Test pagination
        page1 = repo.get_all(sync_session, skip=0, limit=2)
        page2 = repo.get_all(sync_session, skip=2, limit=2)
        
        assert len(page1) == 2
        assert len(page2) == 2
        
        # Ensure different records
        page1_ids = [user.id for user in page1]
        page2_ids = [user.id for user in page2]
        assert not set(page1_ids).intersection(set(page2_ids))

    def test_update(self, sync_session):
        """Test updating a record."""
        repo = AliasRepository()
        
        # Create an alias
        alias = repo.create(
            sync_session,
            user_id="test_user",
            source="ML",
            target="Machine Learning",
            bidirectional=True
        )
        sync_session.commit()
        
        # Update the alias
        updated_alias = repo.update(
            sync_session,
            alias.id,
            target="Artificial Intelligence"
        )
        
        assert updated_alias.target == "Artificial Intelligence"
        assert updated_alias.source == "ML"  # Should remain unchanged

    def test_update_not_found(self, sync_session):
        """Test updating a non-existent record."""
        repo = AliasRepository()
        
        result = repo.update(sync_session, 999, target="New Target")
        assert result is None

    def test_delete(self, sync_session):
        """Test deleting a record."""
        repo = AliasRepository()
        
        # Create an alias
        alias = repo.create(
            sync_session,
            user_id="test_user",
            source="ML",
            target="Machine Learning"
        )
        sync_session.commit()
        
        # Delete the alias
        success = repo.delete(sync_session, alias.id)
        assert success is True
        
        # Verify it's deleted
        retrieved = repo.get_by_id(sync_session, alias.id)
        assert retrieved is None

    def test_delete_not_found(self, sync_session):
        """Test deleting a non-existent record."""
        repo = AliasRepository()
        
        success = repo.delete(sync_session, 999)
        assert success is False


class TestUserRepository:
    """Test cases for UserRepository."""

    def test_get_or_create_new_user(self, sync_session):
        """Test creating a new user."""
        repo = UserRepository()
        
        user = repo.get_or_create(sync_session, "new_user")
        
        assert user.id == "new_user"
        assert isinstance(user.created_at, datetime)

    def test_get_or_create_existing_user(self, sync_session):
        """Test getting an existing user."""
        repo = UserRepository()
        
        # Create user first
        original_user = repo.create(sync_session, id="existing_user")
        sync_session.commit()
        
        # Get or create should return existing user
        retrieved_user = repo.get_or_create(sync_session, "existing_user")
        
        assert retrieved_user.id == original_user.id
        assert retrieved_user.created_at == original_user.created_at

    @pytest.mark.asyncio
    async def test_get_or_create_async(self, async_session):
        """Test async get_or_create."""
        repo = UserRepository()
        
        user = await repo.get_or_create_async(async_session, "async_user")
        
        assert user.id == "async_user"
        assert isinstance(user.created_at, datetime)


class TestAliasRepository:
    """Test cases for AliasRepository."""

    def test_create_alias(self, sync_session):
        """Test creating an alias."""
        repo = AliasRepository()
        
        alias = repo.create(
            sync_session,
            user_id="test_user",
            source="ML",
            target="Machine Learning",
            bidirectional=True,
            tags=["tech", "ai"]
        )
        
        assert alias.source == "ML"
        assert alias.target == "Machine Learning"
        assert alias.bidirectional is True
        assert alias.tags == ["tech", "ai"]

    def test_get_by_source(self, sync_session):
        """Test getting aliases by source."""
        repo = AliasRepository()
        
        # Create aliases
        repo.create(sync_session, source="ML", target="Machine Learning", user_id="user1")
        repo.create(sync_session, source="AI", target="Artificial Intelligence", user_id="user1")
        repo.create(sync_session, source="ML", target="Meta Learning", user_id="user2")
        sync_session.commit()
        
        # Test getting by source
        ml_aliases = repo.get_by_source(sync_session, "ML")
        assert len(ml_aliases) == 2
        
        # Test with user filter
        user1_ml_aliases = repo.get_by_source(sync_session, "ML", "user1")
        assert len(user1_ml_aliases) == 1
        assert user1_ml_aliases[0].target == "Machine Learning"

    def test_search_aliases(self, sync_session):
        """Test searching aliases."""
        repo = AliasRepository()
        
        # Create test aliases
        repo.create(sync_session, source="ML", target="Machine Learning", user_id="user1")
        repo.create(sync_session, source="AI", target="Artificial Intelligence", user_id="user1")
        repo.create(sync_session, source="DL", target="Deep Learning", user_id="user1")
        sync_session.commit()
        
        # Test search
        results = repo.search_aliases(sync_session, "Learning")
        assert len(results) == 2  # Machine Learning and Deep Learning
        
        # Test with user filter
        user_results = repo.search_aliases(sync_session, "Learning", "user1")
        assert len(user_results) == 2

    @pytest.mark.asyncio
    async def test_create_alias_async(self, async_session):
        """Test creating an alias asynchronously."""
        repo = AliasRepository()
        
        alias = await repo.create_async(
            async_session,
            user_id="test_user",
            source="ML",
            target="Machine Learning",
            bidirectional=True
        )
        
        assert alias.source == "ML"
        assert alias.target == "Machine Learning"


class TestNoteRepository:
    """Test cases for NoteRepository."""

    def test_create_note(self, sync_session):
        """Test creating a note."""
        repo = NoteRepository()
        
        note = repo.create(
            sync_session,
            user_id="test_user",
            title="Test Note",
            content="This is test content",
            category="test",
            tags=["note", "test"]
        )
        
        assert note.title == "Test Note"
        assert note.content == "This is test content"
        assert note.category == "test"
        assert note.tags == ["note", "test"]

    def test_search_notes(self, sync_session):
        """Test searching notes."""
        repo = NoteRepository()
        
        # Create test notes
        repo.create(
            sync_session,
            title="Machine Learning Basics",
            content="Introduction to ML concepts",
            user_id="user1"
        )
        repo.create(
            sync_session,
            title="Deep Learning",
            content="Advanced neural networks",
            user_id="user1"
        )
        repo.create(
            sync_session,
            title="Statistics",
            content="Basic statistics for data science",
            user_id="user2"
        )
        sync_session.commit()
        
        # Test search
        results = repo.search_notes(sync_session, "Learning")
        assert len(results) == 2
        
        # Test with user filter
        user_results = repo.search_notes(sync_session, "Learning", "user1")
        assert len(user_results) == 2
        
        # Test with category filter
        category_results = repo.search_notes(sync_session, "Learning", category="nonexistent")
        assert len(category_results) == 0  # Should find no results with nonexistent category


class TestObservationRepository:
    """Test cases for ObservationRepository."""

    def test_create_observation(self, sync_session):
        """Test creating an observation."""
        repo = ObservationRepository()
        
        observation = repo.create(
            sync_session,
            user_id="test_user",
            content="User shows interest in ML",
            entity_type="user",
            entity_id="user123",
            context={"level": "high"}
        )
        
        assert observation.content == "User shows interest in ML"
        assert observation.entity_type == "user"
        assert observation.entity_id == "user123"
        assert observation.context == {"level": "high"}

    def test_get_by_entity(self, sync_session):
        """Test getting observations by entity."""
        repo = ObservationRepository()
        
        # Create observations
        repo.create(
            sync_session,
            content="Observation 1",
            entity_type="user",
            entity_id="user123",
            user_id="observer1"
        )
        repo.create(
            sync_session,
            content="Observation 2",
            entity_type="user",
            entity_id="user123",
            user_id="observer2"
        )
        repo.create(
            sync_session,
            content="Observation 3",
            entity_type="project",
            entity_id="proj456",
            user_id="observer1"
        )
        sync_session.commit()
        
        # Test getting by entity
        user_observations = repo.get_by_entity(sync_session, "user", "user123")
        assert len(user_observations) == 2
        
        # Test with user filter
        filtered_observations = repo.get_by_entity(
            sync_session, "user", "user123", "observer1"
        )
        assert len(filtered_observations) == 1


class TestHintRepository:
    """Test cases for HintRepository."""

    def test_create_hint(self, sync_session):
        """Test creating a hint."""
        repo = HintRepository()
        
        hint = repo.create(
            sync_session,
            user_id="test_user",
            content="Always provide examples",
            category="teaching",
            priority=4,
            workflow_context="education"
        )
        
        assert hint.content == "Always provide examples"
        assert hint.category == "teaching"
        assert hint.priority == 4
        assert hint.workflow_context == "education"

    def test_get_by_category(self, sync_session):
        """Test getting hints by category."""
        repo = HintRepository()
        
        # Create hints
        repo.create(
            sync_session,
            content="Teaching hint 1",
            category="teaching",
            user_id="user1"
        )
        repo.create(
            sync_session,
            content="Teaching hint 2",
            category="teaching",
            user_id="user2"
        )
        repo.create(
            sync_session,
            content="Coding hint",
            category="coding",
            user_id="user1"
        )
        sync_session.commit()
        
        # Test getting by category
        teaching_hints = repo.get_by_category(sync_session, "teaching")
        assert len(teaching_hints) == 2
        
        # Test with user filter
        user_teaching_hints = repo.get_by_category(sync_session, "teaching", "user1")
        assert len(user_teaching_hints) == 1

    def test_search_hints(self, sync_session):
        """Test searching hints."""
        repo = HintRepository()
        
        # Create hints with different priorities
        repo.create(sync_session, content="Teaching hint", category="teaching", priority=1)
        repo.create(sync_session, content="Coding hint", category="coding", priority=3)
        repo.create(sync_session, content="Learning hint", category="teaching", priority=5)
        sync_session.commit()
        
        # Test search
        teaching_hints = repo.search_hints(sync_session, "Teaching")
        assert len(teaching_hints) == 1
        
        # Test search with category filter
        category_hints = repo.search_hints(sync_session, "hint", category="teaching")
        assert len(category_hints) == 2


class TestEmbeddingRepository:
    """Test cases for EmbeddingRepository."""

    def test_create_embedding(self, sync_session):
        """Test creating an embedding."""
        repo = EmbeddingRepository()
        
        embedding_data = b"fake_embedding_data"
        embedding = repo.create(
            sync_session,
            memory_type="alias",
            memory_id=1,
            embedding=embedding_data
        )
        
        assert embedding.memory_type == "alias"
        assert embedding.memory_id == 1
        assert embedding.embedding == embedding_data

    def test_get_by_memory(self, sync_session):
        """Test getting embedding by memory."""
        repo = EmbeddingRepository()
        
        # Create embedding
        embedding_data = b"fake_embedding_data"
        created_embedding = repo.create(
            sync_session,
            memory_type="alias",
            memory_id=1,
            embedding=embedding_data
        )
        sync_session.commit()
        
        # Get by memory
        retrieved_embedding = repo.get_by_memory(sync_session, "alias", 1)
        
        assert retrieved_embedding is not None
        assert retrieved_embedding.id == created_embedding.id
        assert retrieved_embedding.embedding == embedding_data

    def test_get_by_type(self, sync_session):
        """Test getting embeddings by type."""
        repo = EmbeddingRepository()
        
        # Create embeddings of different types
        repo.create(sync_session, memory_type="alias", memory_id=1, embedding=b"data1")
        repo.create(sync_session, memory_type="alias", memory_id=2, embedding=b"data2")
        repo.create(sync_session, memory_type="note", memory_id=1, embedding=b"data3")
        sync_session.commit()
        
        # Get by type
        alias_embeddings = repo.get_by_type(sync_session, "alias")
        assert len(alias_embeddings) == 2
        
        note_embeddings = repo.get_by_type(sync_session, "note")
        assert len(note_embeddings) == 1

    def test_delete_all(self, sync_session):
        """Test deleting all embeddings."""
        repo = EmbeddingRepository()
        
        # Create embeddings
        repo.create(sync_session, memory_type="alias", memory_id=1, embedding=b"data1")
        repo.create(sync_session, memory_type="note", memory_id=1, embedding=b"data2")
        sync_session.commit()
        
        # Delete all
        count = repo.delete_all(sync_session)
        assert count == 2
        
        # Verify deletion
        all_embeddings = repo.get_all(sync_session)
        assert len(all_embeddings) == 0

    @pytest.mark.asyncio
    async def test_create_embedding_async(self, async_session):
        """Test creating an embedding asynchronously."""
        repo = EmbeddingRepository()
        
        embedding_data = b"fake_embedding_data"
        embedding = await repo.create_async(
            async_session,
            memory_type="alias",
            memory_id=1,
            embedding=embedding_data
        )
        
        assert embedding.memory_type == "alias"
        assert embedding.memory_id == 1
        assert embedding.embedding == embedding_data