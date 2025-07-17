"""Unit tests for SearchService."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from memory_mcp_server.services.search_service import (
    SearchService,
    SearchServiceError,
    EmbeddingError
)
from memory_mcp_server.models.alias import Alias
from memory_mcp_server.models.note import Note
from memory_mcp_server.models.observation import Observation
from memory_mcp_server.models.hint import Hint


class TestSearchService:
    """Test cases for SearchService."""

    @pytest.fixture
    def search_service(self, mock_config):
        """Create a SearchService instance for testing."""
        with patch('memory_mcp_server.services.search_service.get_database_manager'):
            service = SearchService(mock_config)
            service.embedding_repo = Mock()
            return service

    def test_search_service_initialization(self, mock_config):
        """Test SearchService initialization."""
        with patch('memory_mcp_server.services.search_service.get_database_manager'):
            service = SearchService(mock_config)
            
            assert service.model_name == "all-MiniLM-L6-v2"
            assert service.similarity_threshold == 0.3
            assert service.max_results == 10
            assert service.cache_embeddings is True

    def test_serialize_deserialize_embedding(self, search_service):
        """Test embedding serialization and deserialization."""
        embedding = np.array([0.1, 0.2, 0.3])
        
        # Test serialization
        serialized = search_service._serialize_embedding(embedding)
        assert isinstance(serialized, bytes)
        
        # Test deserialization
        deserialized = search_service._deserialize_embedding(serialized)
        assert isinstance(deserialized, np.ndarray)
        np.testing.assert_array_equal(embedding, deserialized)

    def test_extract_text_from_memory_alias(self, search_service):
        """Test extracting text from Alias memory."""
        alias = Alias(source="ML", target="Machine Learning")
        
        result = search_service._extract_text_from_memory(alias)
        
        assert result == "ML Machine Learning"

    def test_extract_text_from_memory_note(self, search_service):
        """Test extracting text from Note memory."""
        note = Note(title="Test Note", content="This is test content")
        
        result = search_service._extract_text_from_memory(note)
        
        assert result == "Test Note This is test content"

    def test_extract_text_from_memory_observation(self, search_service):
        """Test extracting text from Observation memory."""
        observation = Observation(
            content="User shows interest",
            entity_type="user",
            entity_id="123"
        )
        
        result = search_service._extract_text_from_memory(observation)
        
        assert result == "User shows interest"

    def test_extract_text_from_memory_hint(self, search_service):
        """Test extracting text from Hint memory."""
        hint = Hint(content="Always provide examples", category="teaching")
        
        result = search_service._extract_text_from_memory(hint)
        
        assert result == "Always provide examples"

    def test_index_memory_no_id(self, search_service, sample_alias):
        """Test indexing memory without ID."""
        mock_session = Mock()
        sample_alias.id = None
        
        result = search_service.index_memory(mock_session, sample_alias)
        
        assert result is False

    def test_generate_embeddings_empty_text(self, search_service):
        """Test generating embeddings for empty text."""
        result = search_service.generate_embeddings("")
        
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_semantic_search_empty_query(self, search_service):
        """Test semantic search with empty query."""
        mock_session = Mock()
        result = search_service.semantic_search(mock_session, "")
        
        assert result == []

    def test_exact_search_empty_query(self, search_service):
        """Test exact search with empty query."""
        mock_session = Mock()
        result = search_service.exact_search(mock_session, "")
        
        assert result == []

    def test_semantic_search_no_embeddings(self, search_service):
        """Test semantic search when no embeddings exist."""
        mock_session = Mock()
        search_service.generate_embeddings = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        search_service.embedding_repo.get_by_type.return_value = []
        
        result = search_service.semantic_search(mock_session, "test query", memory_types=["alias"])
        
        assert result == []

    def test_embedding_error_handling(self, search_service):
        """Test embedding error handling."""
        embedding = np.array([0.1, 0.2, 0.3])
        
        # Test serialization error handling
        with patch('pickle.dumps', side_effect=Exception("Serialization failed")):
            with pytest.raises(EmbeddingError):
                search_service._serialize_embedding(embedding)

    def test_search_service_has_required_attributes(self, search_service):
        """Test that SearchService has all required attributes."""
        assert hasattr(search_service, 'model_name')
        assert hasattr(search_service, 'similarity_threshold')
        assert hasattr(search_service, 'max_results')
        assert hasattr(search_service, 'cache_embeddings')
        assert hasattr(search_service, 'embedding_repo')

    def test_search_service_has_required_methods(self, search_service):
        """Test that SearchService has all required methods."""
        assert hasattr(search_service, 'generate_embeddings')
        assert hasattr(search_service, 'index_memory')
        assert hasattr(search_service, 'semantic_search')
        assert hasattr(search_service, 'exact_search')
        assert hasattr(search_service, '_extract_text_from_memory')
        assert hasattr(search_service, '_serialize_embedding')
        assert hasattr(search_service, '_deserialize_embedding')

    def test_memory_type_models_mapping(self, search_service):
        """Test that memory type models mapping is correct."""
        expected_types = ["alias", "note", "observation", "hint"]
        for memory_type in expected_types:
            assert memory_type in search_service.memory_type_models

    def test_extract_text_from_unknown_memory_type(self, search_service):
        """Test extracting text from unknown memory type."""
        unknown_memory = Mock()
        unknown_memory.__str__ = Mock(return_value="unknown memory")
        
        result = search_service._extract_text_from_memory(unknown_memory)
        
        assert result == "unknown memory"


class TestSearchServiceAsyncMethods:
    """Test cases for async methods in SearchService."""

    @pytest.fixture
    def search_service(self, mock_config):
        """Create a SearchService instance for testing."""
        with patch('memory_mcp_server.services.search_service.get_database_manager'):
            service = SearchService(mock_config)
            service.embedding_repo = Mock()
            return service

    @pytest.mark.asyncio
    async def test_index_memory_async_no_id(self, search_service, sample_alias):
        """Test indexing memory asynchronously without ID."""
        mock_async_session = Mock()
        sample_alias.id = None
        
        result = await search_service.index_memory_async(mock_async_session, sample_alias)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_semantic_search_async_empty_query(self, search_service):
        """Test semantic search asynchronously with empty query."""
        mock_async_session = Mock()
        result = await search_service.semantic_search_async(mock_async_session, "")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_exact_search_async_empty_query(self, search_service):
        """Test exact search asynchronously with empty query."""
        mock_async_session = Mock()
        result = await search_service.exact_search_async(mock_async_session, "")
        
        assert result == []

    @pytest.mark.asyncio
    async def test_search_service_async_methods_exist(self, search_service):
        """Test that async methods exist."""
        assert hasattr(search_service, 'index_memory_async')
        assert hasattr(search_service, 'semantic_search_async')
        assert hasattr(search_service, 'exact_search_async')