"""Unit tests for Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from memory_mcp_server.models.base import BaseMemory, MemoryType
from memory_mcp_server.models.alias import Alias
from memory_mcp_server.models.note import Note
from memory_mcp_server.models.observation import Observation
from memory_mcp_server.models.hint import Hint


class TestBaseMemory:
    """Test cases for BaseMemory model."""

    def test_base_memory_creation(self):
        """Test creating a BaseMemory instance."""
        memory = BaseMemory(user_id="test_user", tags=["tag1", "tag2"])
        
        assert memory.user_id == "test_user"
        assert memory.tags == ["tag1", "tag2"]
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)

    def test_base_memory_defaults(self):
        """Test BaseMemory with default values."""
        memory = BaseMemory()
        
        assert memory.id is None
        assert memory.user_id is None
        assert memory.tags == []
        assert isinstance(memory.created_at, datetime)
        assert isinstance(memory.updated_at, datetime)

    def test_validate_tags_empty_strings(self):
        """Test that empty tags are filtered out."""
        memory = BaseMemory(tags=["valid", "", "  ", "another"])
        assert memory.tags == ["valid", "another"]

    def test_validate_user_id_empty_string(self):
        """Test that empty user_id becomes None."""
        memory = BaseMemory(user_id="  ")
        assert memory.user_id is None

    def test_update_timestamp(self):
        """Test updating timestamp."""
        memory = BaseMemory()
        original_time = memory.updated_at
        
        # Wait a tiny bit to ensure different timestamp
        import time
        time.sleep(0.001)
        
        memory.update_timestamp()
        assert memory.updated_at > original_time

    def test_add_tag(self):
        """Test adding tags."""
        memory = BaseMemory()
        memory.add_tag("new_tag")
        
        assert "new_tag" in memory.tags
        
        # Adding same tag shouldn't duplicate
        memory.add_tag("new_tag")
        assert memory.tags.count("new_tag") == 1

    def test_remove_tag(self):
        """Test removing tags."""
        memory = BaseMemory(tags=["tag1", "tag2"])
        memory.remove_tag("tag1")
        
        assert "tag1" not in memory.tags
        assert "tag2" in memory.tags

    def test_has_tag(self):
        """Test checking for tag existence."""
        memory = BaseMemory(tags=["tag1", "tag2"])
        
        assert memory.has_tag("tag1")
        assert memory.has_tag("tag2")
        assert not memory.has_tag("tag3")


class TestAlias:
    """Test cases for Alias model."""

    def test_alias_creation(self):
        """Test creating an Alias instance."""
        alias = Alias(
            source="ML",
            target="Machine Learning",
            user_id="test_user",
            bidirectional=True
        )
        
        assert alias.source == "ML"
        assert alias.target == "Machine Learning"
        assert alias.user_id == "test_user"
        assert alias.bidirectional is True

    def test_alias_validation_empty_source(self):
        """Test validation fails for empty source."""
        with pytest.raises(ValidationError) as exc_info:
            Alias(source="", target="Machine Learning")
        
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_alias_validation_empty_target(self):
        """Test validation fails for empty target."""
        with pytest.raises(ValidationError) as exc_info:
            Alias(source="ML", target="  ")
        
        assert "Source and target cannot be empty" in str(exc_info.value)

    def test_alias_validation_same_source_target(self):
        """Test validation fails when source equals target."""
        with pytest.raises(ValidationError) as exc_info:
            Alias(source="ML", target="ml")
        
        assert "Source and target cannot be the same" in str(exc_info.value)

    def test_alias_text_field_trimming(self):
        """Test that source and target are trimmed."""
        alias = Alias(source="  ML  ", target="  Machine Learning  ")
        
        assert alias.source == "ML"
        assert alias.target == "Machine Learning"

    def test_get_mapping_forward(self):
        """Test getting mapping in forward direction."""
        alias = Alias(source="ML", target="Machine Learning", bidirectional=True)
        
        result = alias.get_mapping("ML")
        assert result == "Machine Learning"

    def test_get_mapping_backward(self):
        """Test getting mapping in backward direction."""
        alias = Alias(source="ML", target="Machine Learning", bidirectional=True)
        
        result = alias.get_mapping("Machine Learning")
        assert result == "ML"  # Should return source

    def test_get_mapping_unidirectional(self):
        """Test mapping only works forward when not bidirectional."""
        alias = Alias(source="ML", target="Machine Learning", bidirectional=False)
        
        assert alias.get_mapping("ML") == "Machine Learning"
        assert alias.get_mapping("Machine Learning") is None

    def test_get_mapping_case_insensitive(self):
        """Test mapping is case insensitive."""
        alias = Alias(source="ML", target="Machine Learning", bidirectional=True)
        
        assert alias.get_mapping("ml") == "Machine Learning"
        assert alias.get_mapping("machine learning") == "ML"

    def test_matches_query(self):
        """Test query matching."""
        alias = Alias(source="ML", target="Machine Learning", bidirectional=True)
        
        assert alias.matches_query("ML")
        assert alias.matches_query("Machine Learning")
        assert not alias.matches_query("AI")

    def test_is_word_alias(self):
        """Test word alias detection."""
        word_alias = Alias(source="ML", target="AI")
        phrase_alias = Alias(source="Machine Learning", target="AI")
        
        assert word_alias.is_word_alias()
        assert not phrase_alias.is_word_alias()

    def test_is_phrase_alias(self):
        """Test phrase alias detection."""
        word_alias = Alias(source="ML", target="AI")
        phrase_alias = Alias(source="Machine Learning", target="AI")
        
        assert not word_alias.is_phrase_alias()
        assert phrase_alias.is_phrase_alias()

    def test_string_representation(self):
        """Test string representation."""
        bidirectional = Alias(source="ML", target="Machine Learning", bidirectional=True)
        unidirectional = Alias(source="ML", target="Machine Learning", bidirectional=False)
        
        assert "↔" in str(bidirectional)
        assert "→" in str(unidirectional)


class TestNote:
    """Test cases for Note model."""

    def test_note_creation(self):
        """Test creating a Note instance."""
        note = Note(
            title="Test Note",
            content="This is test content",
            category="test",
            user_id="test_user"
        )
        
        assert note.title == "Test Note"
        assert note.content == "This is test content"
        assert note.category == "test"
        assert note.user_id == "test_user"

    def test_note_validation_empty_title(self):
        """Test validation fails for empty title."""
        with pytest.raises(ValidationError) as exc_info:
            Note(title="", content="Content")
        
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_note_validation_empty_content(self):
        """Test validation fails for empty content."""
        with pytest.raises(ValidationError) as exc_info:
            Note(title="Title", content="  ")
        
        assert "Title and content cannot be empty" in str(exc_info.value)

    def test_note_text_field_trimming(self):
        """Test that title and content are trimmed."""
        note = Note(title="  Test Title  ", content="  Test Content  ")
        
        assert note.title == "Test Title"
        assert note.content == "Test Content"

    def test_note_category_validation(self):
        """Test category validation."""
        note = Note(title="Title", content="Content", category="  ")
        assert note.category is None
        
        note = Note(title="Title", content="Content", category="  test  ")
        assert note.category == "test"

    def test_get_preview(self):
        """Test content preview generation."""
        short_content = "Short content"
        long_content = "This is a very long content that should be truncated when generating a preview"
        
        short_note = Note(title="Title", content=short_content)
        long_note = Note(title="Title", content=long_content)
        
        assert short_note.get_preview() == short_content
        assert long_note.get_preview(50).endswith("...")
        assert len(long_note.get_preview(50)) <= 53  # 50 + "..."

    def test_word_count(self):
        """Test word counting."""
        note = Note(title="Title", content="This is a test content")
        assert note.word_count() == 5

    def test_char_count(self):
        """Test character counting."""
        note = Note(title="Title", content="Hello")
        assert note.char_count() == 5

    def test_has_category(self):
        """Test category checking."""
        note_with_category = Note(title="Title", content="Content", category="test")
        note_without_category = Note(title="Title", content="Content")
        
        assert note_with_category.has_category()
        assert not note_without_category.has_category()

    def test_matches_search(self):
        """Test search matching."""
        note = Note(
            title="Machine Learning",
            content="This is about AI and ML",
            category="tech"
        )
        
        assert note.matches_search("machine")
        assert note.matches_search("AI")
        assert note.matches_search("tech")
        assert not note.matches_search("biology")

    def test_string_representation(self):
        """Test string representation."""
        note = Note(
            title="Test Title",
            content="This is a long content that should be truncated in string representation",
            category="test"
        )
        
        str_repr = str(note)
        assert "Test Title" in str_repr
        assert "[test]" in str_repr
        assert "..." in str_repr  # Should be truncated


class TestObservation:
    """Test cases for Observation model."""

    def test_observation_creation(self):
        """Test creating an Observation instance."""
        observation = Observation(
            content="User shows interest in ML",
            entity_type="user",
            entity_id="user123",
            context={"level": "high"},
            user_id="test_user"
        )
        
        assert observation.content == "User shows interest in ML"
        assert observation.entity_type == "user"
        assert observation.entity_id == "user123"
        assert observation.context == {"level": "high"}
        assert observation.user_id == "test_user"

    def test_observation_validation_empty_fields(self):
        """Test validation fails for empty required fields."""
        with pytest.raises(ValidationError) as exc_info:
            Observation(content="", entity_type="user", entity_id="123")
        
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_observation_text_field_trimming(self):
        """Test that text fields are trimmed."""
        observation = Observation(
            content="  Content  ",
            entity_type="  user  ",
            entity_id="  123  "
        )
        
        assert observation.content == "Content"
        assert observation.entity_type == "user"
        assert observation.entity_id == "123"

    def test_observation_context_validation(self):
        """Test context validation."""
        # None context should become empty dict
        observation = Observation(
            content="Content",
            entity_type="user",
            entity_id="123",
            context=None
        )
        assert observation.context == {}

    def test_get_entity_reference(self):
        """Test entity reference generation."""
        observation = Observation(
            content="Content",
            entity_type="user",
            entity_id="123"
        )
        
        assert observation.get_entity_reference() == "user:123"

    def test_has_context(self):
        """Test context checking."""
        with_context = Observation(
            content="Content",
            entity_type="user",
            entity_id="123",
            context={"key": "value"}
        )
        without_context = Observation(
            content="Content",
            entity_type="user",
            entity_id="123"
        )
        
        assert with_context.has_context()
        assert not without_context.has_context()

    def test_context_operations(self):
        """Test context get/set/remove operations."""
        observation = Observation(
            content="Content",
            entity_type="user",
            entity_id="123",
            context={"existing": "value"}
        )
        
        # Test get
        assert observation.get_context_value("existing") == "value"
        assert observation.get_context_value("missing", "default") == "default"
        
        # Test set
        observation.set_context_value("new_key", "new_value")
        assert observation.context["new_key"] == "new_value"
        
        # Test remove
        observation.remove_context_value("existing")
        assert "existing" not in observation.context

    def test_matches_entity(self):
        """Test entity matching."""
        observation = Observation(
            content="Content",
            entity_type="user",
            entity_id="123"
        )
        
        assert observation.matches_entity("user", "123")
        assert observation.matches_entity("user", None)
        assert observation.matches_entity(None, "123")
        assert not observation.matches_entity("admin", "123")

    def test_matches_search(self):
        """Test search matching."""
        observation = Observation(
            content="User shows interest in ML",
            entity_type="user",
            entity_id="john_doe",
            context={"topic": "machine learning"}
        )
        
        assert observation.matches_search("interest")
        assert observation.matches_search("user")
        assert observation.matches_search("john")
        assert observation.matches_search("machine")
        assert not observation.matches_search("biology")


class TestHint:
    """Test cases for Hint model."""

    def test_hint_creation(self):
        """Test creating a Hint instance."""
        hint = Hint(
            content="Always provide examples",
            category="teaching",
            priority=4,
            workflow_context="education",
            user_id="test_user"
        )
        
        assert hint.content == "Always provide examples"
        assert hint.category == "teaching"
        assert hint.priority == 4
        assert hint.workflow_context == "education"
        assert hint.user_id == "test_user"

    def test_hint_validation_empty_fields(self):
        """Test validation fails for empty required fields."""
        with pytest.raises(ValidationError) as exc_info:
            Hint(content="", category="test")
        
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_hint_priority_validation(self):
        """Test priority validation."""
        with pytest.raises(ValidationError) as exc_info:
            Hint(content="Content", category="test", priority=0)
        
        assert "Input should be greater than or equal to 1" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            Hint(content="Content", category="test", priority=6)
        
        assert "Input should be less than or equal to 5" in str(exc_info.value)

    def test_hint_text_field_trimming(self):
        """Test that text fields are trimmed."""
        hint = Hint(
            content="  Content  ",
            category="  test  ",
            workflow_context="  context  "
        )
        
        assert hint.content == "Content"
        assert hint.category == "test"
        assert hint.workflow_context == "context"

    def test_priority_methods(self):
        """Test priority checking methods."""
        low_hint = Hint(content="Content", category="test", priority=2)
        medium_hint = Hint(content="Content", category="test", priority=3)
        high_hint = Hint(content="Content", category="test", priority=4)
        
        assert low_hint.is_low_priority()
        assert not low_hint.is_high_priority()
        
        assert not medium_hint.is_low_priority()
        assert not medium_hint.is_high_priority()
        
        assert not high_hint.is_low_priority()
        assert high_hint.is_high_priority()

    def test_get_priority_label(self):
        """Test priority label generation."""
        hint = Hint(content="Content", category="test", priority=3)
        assert hint.get_priority_label() == "Medium"

    def test_has_workflow_context(self):
        """Test workflow context checking."""
        with_context = Hint(
            content="Content",
            category="test",
            workflow_context="education"
        )
        without_context = Hint(content="Content", category="test")
        
        assert with_context.has_workflow_context()
        assert not without_context.has_workflow_context()

    def test_matches_category(self):
        """Test category matching."""
        hint = Hint(content="Content", category="Teaching")
        
        assert hint.matches_category("teaching")
        assert hint.matches_category("Teaching")
        assert not hint.matches_category("coding")

    def test_matches_workflow(self):
        """Test workflow matching."""
        hint = Hint(
            content="Content",
            category="test",
            workflow_context="Education"
        )
        
        assert hint.matches_workflow("education")
        assert hint.matches_workflow("Education")
        assert not hint.matches_workflow("coding")

    def test_matches_search(self):
        """Test search matching."""
        hint = Hint(
            content="Always provide ML examples",
            category="teaching",
            workflow_context="education"
        )
        
        assert hint.matches_search("ML")
        assert hint.matches_search("teaching")
        assert hint.matches_search("education")
        assert not hint.matches_search("biology")

    def test_string_representation(self):
        """Test string representation."""
        hint = Hint(
            content="This is a very long hint content that should be truncated",
            category="test",
            priority=5,
            workflow_context="context"
        )
        
        str_repr = str(hint)
        assert "[test]" in str_repr
        assert "(context)" in str_repr
        assert "!!!!!" in str_repr  # 5 exclamation marks for priority 5
        assert "..." in str_repr  # Should be truncated


class TestMemoryType:
    """Test cases for MemoryType enum."""

    def test_memory_type_values(self):
        """Test MemoryType enum values."""
        assert MemoryType.ALIAS == "alias"
        assert MemoryType.NOTE == "note"
        assert MemoryType.OBSERVATION == "observation"
        assert MemoryType.HINT == "hint"