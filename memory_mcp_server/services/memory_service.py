"""Core memory service for managing all memory types."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ..database.connection import get_database_manager
from ..database.models import AliasDB, HintDB, NoteDB, ObservationDB, User
from ..database.repositories import (
    AliasRepository,
    HintRepository,
    NoteRepository,
    ObservationRepository,
    UserRepository,
)
from ..models.alias import Alias
from ..models.base import BaseMemory
from ..models.hint import Hint
from ..models.note import Note
from ..models.observation import Observation
from .search_service import SearchService

logger = logging.getLogger(__name__)


class MemoryServiceError(Exception):
    """Base exception for memory service errors."""

    pass


class ValidationError(MemoryServiceError):
    """Exception raised for validation errors."""

    pass


class NotFoundError(MemoryServiceError):
    """Exception raised when a resource is not found."""

    pass


class MemoryService:
    """Core service for managing all memory types with user separation and validation."""

    def __init__(self):
        """Initialize the memory service with repositories."""
        self.user_repo = UserRepository()
        self.alias_repo = AliasRepository()
        self.note_repo = NoteRepository()
        self.observation_repo = ObservationRepository()
        self.hint_repo = HintRepository()
        self.db_manager = get_database_manager()
        self.search_service = SearchService()

    # User management methods
    def ensure_user(self, session: Session, user_id: Optional[str]) -> Optional[User]:
        """Ensure user exists if user_id is provided."""
        if not user_id:
            return None

        try:
            return self.user_repo.get_or_create(session, user_id)
        except Exception as e:
            logger.error(f"Failed to ensure user {user_id}: {e}")
            raise MemoryServiceError(f"Failed to ensure user: {e}") from None

    async def ensure_user_async(
        self, session: AsyncSession, user_id: Optional[str]
    ) -> Optional[User]:
        """Ensure user exists if user_id is provided (async)."""
        if not user_id:
            return None

        try:
            return await self.user_repo.get_or_create_async(session, user_id)
        except Exception as e:
            logger.error(f"Failed to ensure user {user_id}: {e}")
            raise MemoryServiceError(f"Failed to ensure user: {e}") from None

    def get_users(self, session: Session) -> List[str]:
        """Get all user IDs."""
        try:
            users = self.user_repo.get_all(session)
            return [user.id for user in users]
        except Exception as e:
            logger.error(f"Failed to get users: {e}")
            raise MemoryServiceError(f"Failed to get users: {e}") from None

    async def get_users_async(self, session: AsyncSession) -> List[str]:
        """Get all user IDs (async)."""
        try:
            users = await self.user_repo.get_all_async(session)
            return [user.id for user in users]
        except Exception as e:
            logger.error(f"Failed to get users: {e}")
            raise MemoryServiceError(f"Failed to get users: {e}") from None

    # Alias methods
    def create_alias(self, session: Session, alias: Alias) -> Alias:
        """Create a new alias."""
        try:
            # Validate the alias
            alias.model_validate(alias.model_dump())

            # Ensure user exists if specified
            self.ensure_user(session, alias.user_id)

            # Create database record
            alias_db = self.alias_repo.create(
                session,
                user_id=alias.user_id,
                source=alias.source,
                target=alias.target,
                bidirectional=alias.bidirectional,
                tags=alias.tags,
            )

            # Convert back to Pydantic model
            result = self._db_to_pydantic_alias(alias_db)

            # Index the alias for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    search_service.index_memory(session, result)
                session.commit()
            except Exception as e:
                logger.warning(f"Failed to index alias {result.id} for search: {e}")

            logger.info(f"Created alias: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to create alias: {e}")
            raise MemoryServiceError(f"Failed to create alias: {e}") from None

    async def create_alias_async(self, session: AsyncSession, alias: Alias) -> Alias:
        """Create a new alias (async)."""
        try:
            # Validate the alias
            alias.model_validate(alias.model_dump())

            # Ensure user exists if specified
            await self.ensure_user_async(session, alias.user_id)

            # Create database record
            alias_db = await self.alias_repo.create_async(
                session,
                user_id=alias.user_id,
                source=alias.source,
                target=alias.target,
                bidirectional=alias.bidirectional,
                tags=alias.tags,
            )

            # Convert back to Pydantic model
            result = self._db_to_pydantic_alias(alias_db)

            # Index the alias for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    await search_service.index_memory_async(session, result)
                await session.commit()
            except Exception as e:
                logger.warning(f"Failed to index alias {result.id} for search: {e}")

            logger.info(f"Created alias: {result}")
            return result

        except Exception as e:
            logger.error(f"Failed to create alias: {e}")
            raise MemoryServiceError(f"Failed to create alias: {e}") from None

    def get_aliases(
        self,
        session: Session,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Alias]:
        """Get aliases with optional filtering."""
        try:
            filters = {}
            if user_id:
                filters["user_id"] = user_id

            if query:
                # Use repository search method for query
                aliases_db = self.alias_repo.search_aliases(session, query, user_id)
            else:
                # Use regular get_all with filters
                aliases_db = self.alias_repo.get_all(session, skip, limit, filters)

            return [self._db_to_pydantic_alias(alias_db) for alias_db in aliases_db]

        except Exception as e:
            logger.error(f"Failed to get aliases: {e}")
            raise MemoryServiceError(f"Failed to get aliases: {e}") from None

    async def get_aliases_async(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Alias]:
        """Get aliases with optional filtering (async)."""
        try:
            filters = {}
            if user_id:
                filters["user_id"] = user_id

            if query:
                # Use repository search method for query
                aliases_db = await self.alias_repo.search_aliases_async(
                    session, query, user_id
                )
            else:
                # Use regular get_all with filters
                aliases_db = await self.alias_repo.get_all_async(
                    session, skip, limit, filters
                )

            return [self._db_to_pydantic_alias(alias_db) for alias_db in aliases_db]

        except Exception as e:
            logger.error(f"Failed to get aliases: {e}")
            raise MemoryServiceError(f"Failed to get aliases: {e}") from None

    def update_alias(
        self, session: Session, alias_id: int, **kwargs
    ) -> Optional[Alias]:
        """Update an alias."""
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.utcnow()

            alias_db = self.alias_repo.update(session, alias_id, **kwargs)
            if not alias_db:
                raise NotFoundError(f"Alias with ID {alias_id} not found")

            result = self._db_to_pydantic_alias(alias_db)

            # Re-index the updated alias for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    search_service.index_memory(session, result)
                session.commit()
            except Exception as e:
                logger.warning(
                    f"Failed to re-index updated alias {result.id} for search: {e}"
                )

            logger.info(f"Updated alias {alias_id}")
            return result

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update alias {alias_id}: {e}")
            raise MemoryServiceError(f"Failed to update alias: {e}") from None

    async def update_alias_async(
        self, session: AsyncSession, alias_id: int, **kwargs
    ) -> Optional[Alias]:
        """Update an alias (async)."""
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.utcnow()

            alias_db = await self.alias_repo.update_async(session, alias_id, **kwargs)
            if not alias_db:
                raise NotFoundError(f"Alias with ID {alias_id} not found")

            result = self._db_to_pydantic_alias(alias_db)
            logger.info(f"Updated alias {alias_id}")
            return result

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update alias {alias_id}: {e}")
            raise MemoryServiceError(f"Failed to update alias: {e}") from None

    def delete_alias(self, session: Session, alias_id: int) -> bool:
        """Delete an alias."""
        try:
            success = self.alias_repo.delete(session, alias_id)
            if success:
                logger.info(f"Deleted alias {alias_id}")
            else:
                raise NotFoundError(f"Alias with ID {alias_id} not found")
            return success

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete alias {alias_id}: {e}")
            raise MemoryServiceError(f"Failed to delete alias: {e}") from None

    async def delete_alias_async(self, session: AsyncSession, alias_id: int) -> bool:
        """Delete an alias (async)."""
        try:
            success = await self.alias_repo.delete_async(session, alias_id)
            if success:
                logger.info(f"Deleted alias {alias_id}")
            else:
                raise NotFoundError(f"Alias with ID {alias_id} not found")
            return success

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete alias {alias_id}: {e}")
            raise MemoryServiceError(f"Failed to delete alias: {e}") from None

    # Bidirectional alias functionality
    def query_alias(
        self,
        session: Session,
        query: str,
        user_id: Optional[str] = None,
        exact_match: bool = True,
    ) -> List[str]:
        """Query aliases bidirectionally and return matching targets.

        Args:
            session: Database session
            query: The term to look up
            user_id: Optional user ID for filtering
            exact_match: Whether to use exact matching or partial matching

        Returns:
            List of matching target terms
        """
        try:
            results = []

            if exact_match:
                # Get aliases by exact source match
                aliases_db = self.alias_repo.get_by_source(session, query, user_id)

                for alias_db in aliases_db:
                    alias = self._db_to_pydantic_alias(alias_db)
                    mapping = alias.get_mapping(query)
                    if mapping:
                        results.append(mapping)

                # Also check if query matches any target (for bidirectional aliases)
                if user_id:
                    filters = {
                        "user_id": user_id,
                        "target": query,
                        "bidirectional": True,
                    }
                else:
                    filters = {"target": query, "bidirectional": True}

                target_aliases_db = self.alias_repo.get_all(session, filters=filters)
                for alias_db in target_aliases_db:
                    results.append(alias_db.source)
            else:
                # Use search functionality for partial matching
                aliases_db = self.alias_repo.search_aliases(
                    session, query, user_id, bidirectional=True
                )

                for alias_db in aliases_db:
                    alias = self._db_to_pydantic_alias(alias_db)
                    # For partial matches, return both source and target
                    if query.lower() in alias.source.lower():
                        results.append(alias.target)
                    elif alias.bidirectional and query.lower() in alias.target.lower():
                        results.append(alias.source)

            # Remove duplicates while preserving order
            seen = set()
            unique_results = []
            for result in results:
                if result not in seen:
                    seen.add(result)
                    unique_results.append(result)

            logger.info(f"Query '{query}' returned {len(unique_results)} alias matches")
            return unique_results

        except Exception as e:
            logger.error(f"Failed to query alias '{query}': {e}")
            raise MemoryServiceError(f"Failed to query alias: {e}") from None

    async def query_alias_async(
        self,
        session: AsyncSession,
        query: str,
        user_id: Optional[str] = None,
        exact_match: bool = True,
    ) -> List[str]:
        """Query aliases bidirectionally and return matching targets (async).

        Args:
            session: Database session
            query: The term to look up
            user_id: Optional user ID for filtering
            exact_match: Whether to use exact matching or partial matching

        Returns:
            List of matching target terms
        """
        try:
            results = []

            if exact_match:
                # Get aliases by exact source match
                aliases_db = await self.alias_repo.get_by_source_async(
                    session, query, user_id
                )

                for alias_db in aliases_db:
                    alias = self._db_to_pydantic_alias(alias_db)
                    mapping = alias.get_mapping(query)
                    if mapping:
                        results.append(mapping)

                # Also check if query matches any target (for bidirectional aliases)
                if user_id:
                    filters = {
                        "user_id": user_id,
                        "target": query,
                        "bidirectional": True,
                    }
                else:
                    filters = {"target": query, "bidirectional": True}

                target_aliases_db = await self.alias_repo.get_all_async(
                    session, filters=filters
                )
                for alias_db in target_aliases_db:
                    results.append(alias_db.source)
            else:
                # Use search functionality for partial matching
                aliases_db = await self.alias_repo.search_aliases_async(
                    session, query, user_id, bidirectional=True
                )

                for alias_db in aliases_db:
                    alias = self._db_to_pydantic_alias(alias_db)
                    # For partial matches, return both source and target
                    if query.lower() in alias.source.lower():
                        results.append(alias.target)
                    elif alias.bidirectional and query.lower() in alias.target.lower():
                        results.append(alias.source)

            # Remove duplicates while preserving order
            seen = set()
            unique_results = []
            for result in results:
                if result not in seen:
                    seen.add(result)
                    unique_results.append(result)

            logger.info(f"Query '{query}' returned {len(unique_results)} alias matches")
            return unique_results

        except Exception as e:
            logger.error(f"Failed to query alias '{query}': {e}")
            raise MemoryServiceError(f"Failed to query alias: {e}") from None

    def get_word_aliases(
        self,
        session: Session,
        user_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Alias]:
        """Get word-to-word aliases (no spaces in source or target)."""
        try:
            all_aliases = self.get_aliases(
                session, user_id, skip=skip, limit=limit * 2
            )  # Get more to filter
            word_aliases = [alias for alias in all_aliases if alias.is_word_alias()]
            return word_aliases[:limit]

        except Exception as e:
            logger.error(f"Failed to get word aliases: {e}")
            raise MemoryServiceError(f"Failed to get word aliases: {e}") from None

    async def get_word_aliases_async(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Alias]:
        """Get word-to-word aliases (no spaces in source or target) (async)."""
        try:
            all_aliases = await self.get_aliases_async(
                session, user_id, skip=skip, limit=limit * 2
            )  # Get more to filter
            word_aliases = [alias for alias in all_aliases if alias.is_word_alias()]
            return word_aliases[:limit]

        except Exception as e:
            logger.error(f"Failed to get word aliases: {e}")
            raise MemoryServiceError(f"Failed to get word aliases: {e}") from None

    def get_phrase_aliases(
        self,
        session: Session,
        user_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Alias]:
        """Get phrase aliases (contains spaces in source or target)."""
        try:
            all_aliases = self.get_aliases(
                session, user_id, skip=skip, limit=limit * 2
            )  # Get more to filter
            phrase_aliases = [alias for alias in all_aliases if alias.is_phrase_alias()]
            return phrase_aliases[:limit]

        except Exception as e:
            logger.error(f"Failed to get phrase aliases: {e}")
            raise MemoryServiceError(f"Failed to get phrase aliases: {e}") from None

    async def get_phrase_aliases_async(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Alias]:
        """Get phrase aliases (contains spaces in source or target) (async)."""
        try:
            all_aliases = await self.get_aliases_async(
                session, user_id, skip=skip, limit=limit * 2
            )  # Get more to filter
            phrase_aliases = [alias for alias in all_aliases if alias.is_phrase_alias()]
            return phrase_aliases[:limit]

        except Exception as e:
            logger.error(f"Failed to get phrase aliases: {e}")
            raise MemoryServiceError(f"Failed to get phrase aliases: {e}") from None

    def find_alias_mappings(
        self, session: Session, text: str, user_id: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Find all alias mappings within a text.

        Args:
            session: Database session
            text: Text to search for alias mappings
            user_id: Optional user ID for filtering

        Returns:
            Dictionary mapping found terms to their aliases
        """
        try:
            mappings = {}

            # Get all aliases for the user
            aliases = self.get_aliases(session, user_id)

            # Check for word aliases (exact word matching)
            words = text.split()
            for word in words:
                word_clean = word.strip('.,!?;:"()[]{}').lower()
                for alias in aliases:
                    if alias.is_word_alias():
                        mapping = alias.get_mapping(word_clean)
                        if mapping:
                            if word_clean not in mappings:
                                mappings[word_clean] = []
                            if mapping not in mappings[word_clean]:
                                mappings[word_clean].append(mapping)

            # Check for phrase aliases
            text_lower = text.lower()
            for alias in aliases:
                if alias.is_phrase_alias():
                    if alias.source.lower() in text_lower:
                        if alias.source not in mappings:
                            mappings[alias.source] = []
                        if alias.target not in mappings[alias.source]:
                            mappings[alias.source].append(alias.target)

                    if alias.bidirectional and alias.target.lower() in text_lower:
                        if alias.target not in mappings:
                            mappings[alias.target] = []
                        if alias.source not in mappings[alias.target]:
                            mappings[alias.target].append(alias.source)

            logger.info(f"Found {len(mappings)} alias mappings in text")
            return mappings

        except Exception as e:
            logger.error(f"Failed to find alias mappings: {e}")
            raise MemoryServiceError(f"Failed to find alias mappings: {e}") from None

    async def find_alias_mappings_async(
        self, session: AsyncSession, text: str, user_id: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """Find all alias mappings within a text (async).

        Args:
            session: Database session
            text: Text to search for alias mappings
            user_id: Optional user ID for filtering

        Returns:
            Dictionary mapping found terms to their aliases
        """
        try:
            mappings = {}

            # Get all aliases for the user
            aliases = await self.get_aliases_async(session, user_id)

            # Check for word aliases (exact word matching)
            words = text.split()
            for word in words:
                word_clean = word.strip('.,!?;:"()[]{}').lower()
                for alias in aliases:
                    if alias.is_word_alias():
                        mapping = alias.get_mapping(word_clean)
                        if mapping:
                            if word_clean not in mappings:
                                mappings[word_clean] = []
                            if mapping not in mappings[word_clean]:
                                mappings[word_clean].append(mapping)

            # Check for phrase aliases
            text_lower = text.lower()
            for alias in aliases:
                if alias.is_phrase_alias():
                    if alias.source.lower() in text_lower:
                        if alias.source not in mappings:
                            mappings[alias.source] = []
                        if alias.target not in mappings[alias.source]:
                            mappings[alias.source].append(alias.target)

                    if alias.bidirectional and alias.target.lower() in text_lower:
                        if alias.target not in mappings:
                            mappings[alias.target] = []
                        if alias.source not in mappings[alias.target]:
                            mappings[alias.target].append(alias.source)

            logger.info(f"Found {len(mappings)} alias mappings in text")
            return mappings

        except Exception as e:
            logger.error(f"Failed to find alias mappings: {e}")
            raise MemoryServiceError(f"Failed to find alias mappings: {e}") from None

    # Note methods
    def create_note(self, session: Session, note: Note) -> Note:
        """Create a new note."""
        try:
            # Validate the note
            note.model_validate(note.model_dump())

            # Ensure user exists if specified
            self.ensure_user(session, note.user_id)

            # Create database record
            note_db = self.note_repo.create(
                session,
                user_id=note.user_id,
                title=note.title,
                content=note.content,
                category=note.category,
                tags=note.tags,
            )

            # Convert back to Pydantic model
            result = self._db_to_pydantic_note(note_db)

            # Index the note for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    search_service.index_memory(session, result)
                session.commit()
            except Exception as e:
                logger.warning(f"Failed to index note {result.id} for search: {e}")

            logger.info(f"Created note: {result.title}")
            return result

        except Exception as e:
            logger.error(f"Failed to create note: {e}")
            raise MemoryServiceError(f"Failed to create note: {e}") from None

    async def create_note_async(self, session: AsyncSession, note: Note) -> Note:
        """Create a new note (async)."""
        try:
            # Validate the note
            note.model_validate(note.model_dump())

            # Ensure user exists if specified
            await self.ensure_user_async(session, note.user_id)

            # Create database record
            note_db = await self.note_repo.create_async(
                session,
                user_id=note.user_id,
                title=note.title,
                content=note.content,
                category=note.category,
                tags=note.tags,
            )

            # Convert back to Pydantic model
            result = self._db_to_pydantic_note(note_db)

            # Index the note for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    await search_service.index_memory_async(session, result)
                await session.commit()
            except Exception as e:
                logger.warning(f"Failed to index note {result.id} for search: {e}")

            logger.info(f"Created note: {result.title}")
            return result

        except Exception as e:
            logger.error(f"Failed to create note: {e}")
            raise MemoryServiceError(f"Failed to create note: {e}") from None

    def get_notes(
        self,
        session: Session,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Note]:
        """Get notes with optional filtering."""
        try:
            if query:
                # Use repository search method for query
                notes_db = self.note_repo.search_notes(
                    session, query, user_id, category
                )
            else:
                # Use regular get_all with filters
                filters = {}
                if user_id:
                    filters["user_id"] = user_id
                if category:
                    filters["category"] = category

                notes_db = self.note_repo.get_all(session, skip, limit, filters)

            return [self._db_to_pydantic_note(note_db) for note_db in notes_db]

        except Exception as e:
            logger.error(f"Failed to get notes: {e}")
            raise MemoryServiceError(f"Failed to get notes: {e}") from None

    async def get_notes_async(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Note]:
        """Get notes with optional filtering (async)."""
        try:
            if query:
                # Use repository search method for query
                notes_db = await self.note_repo.search_notes_async(
                    session, query, user_id, category
                )
            else:
                # Use regular get_all with filters
                filters = {}
                if user_id:
                    filters["user_id"] = user_id
                if category:
                    filters["category"] = category

                notes_db = await self.note_repo.get_all_async(
                    session, skip, limit, filters
                )

            return [self._db_to_pydantic_note(note_db) for note_db in notes_db]

        except Exception as e:
            logger.error(f"Failed to get notes: {e}")
            raise MemoryServiceError(f"Failed to get notes: {e}") from None

    def update_note(self, session: Session, note_id: int, **kwargs) -> Optional[Note]:
        """Update a note."""
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.utcnow()

            note_db = self.note_repo.update(session, note_id, **kwargs)
            if not note_db:
                raise NotFoundError(f"Note with ID {note_id} not found")

            result = self._db_to_pydantic_note(note_db)
            logger.info(f"Updated note {note_id}")
            return result

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update note {note_id}: {e}")
            raise MemoryServiceError(f"Failed to update note: {e}") from None

    async def update_note_async(
        self, session: AsyncSession, note_id: int, **kwargs
    ) -> Optional[Note]:
        """Update a note (async)."""
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.utcnow()

            note_db = await self.note_repo.update_async(session, note_id, **kwargs)
            if not note_db:
                raise NotFoundError(f"Note with ID {note_id} not found")

            result = self._db_to_pydantic_note(note_db)
            logger.info(f"Updated note {note_id}")
            return result

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update note {note_id}: {e}")
            raise MemoryServiceError(f"Failed to update note: {e}") from None

    def delete_note(self, session: Session, note_id: int) -> bool:
        """Delete a note."""
        try:
            success = self.note_repo.delete(session, note_id)
            if success:
                logger.info(f"Deleted note {note_id}")
            else:
                raise NotFoundError(f"Note with ID {note_id} not found")
            return success

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete note {note_id}: {e}")
            raise MemoryServiceError(f"Failed to delete note: {e}") from None

    async def delete_note_async(self, session: AsyncSession, note_id: int) -> bool:
        """Delete a note (async)."""
        try:
            success = await self.note_repo.delete_async(session, note_id)
            if success:
                logger.info(f"Deleted note {note_id}")
            else:
                raise NotFoundError(f"Note with ID {note_id} not found")
            return success

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete note {note_id}: {e}")
            raise MemoryServiceError(f"Failed to delete note: {e}") from None

    # Observation methods
    def create_observation(
        self, session: Session, observation: Observation
    ) -> Observation:
        """Create a new observation."""
        try:
            # Validate the observation
            observation.model_validate(observation.model_dump())

            # Ensure user exists if specified
            self.ensure_user(session, observation.user_id)

            # Create database record
            observation_db = self.observation_repo.create(
                session,
                user_id=observation.user_id,
                content=observation.content,
                entity_type=observation.entity_type,
                entity_id=observation.entity_id,
                context=observation.context,
                tags=observation.tags,
            )

            # Convert back to Pydantic model
            result = self._db_to_pydantic_observation(observation_db)

            # Index the observation for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    search_service.index_memory(session, result)
                session.commit()
            except Exception as e:
                logger.warning(
                    f"Failed to index observation {result.id} for search: {e}"
                )

            logger.info(
                f"Created observation for {result.entity_type}:{result.entity_id}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to create observation: {e}")
            raise MemoryServiceError(f"Failed to create observation: {e}") from None

    async def create_observation_async(
        self, session: AsyncSession, observation: Observation
    ) -> Observation:
        """Create a new observation (async)."""
        try:
            # Validate the observation
            observation.model_validate(observation.model_dump())

            # Ensure user exists if specified
            await self.ensure_user_async(session, observation.user_id)

            # Create database record
            observation_db = await self.observation_repo.create_async(
                session,
                user_id=observation.user_id,
                content=observation.content,
                entity_type=observation.entity_type,
                entity_id=observation.entity_id,
                context=observation.context,
                tags=observation.tags,
            )

            # Convert back to Pydantic model
            result = self._db_to_pydantic_observation(observation_db)

            # Index the observation for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    await search_service.index_memory_async(session, result)
                await session.commit()
            except Exception as e:
                logger.warning(
                    f"Failed to index observation {result.id} for search: {e}"
                )

            logger.info(
                f"Created observation for {result.entity_type}:{result.entity_id}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to create observation: {e}")
            raise MemoryServiceError(f"Failed to create observation: {e}") from None

    def get_observations(
        self,
        session: Session,
        user_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Observation]:
        """Get observations with optional filtering."""
        try:
            if entity_type and entity_id:
                # Get observations for specific entity
                observations_db = self.observation_repo.get_by_entity(
                    session, entity_type, entity_id, user_id
                )
            elif query:
                # Use repository search method for query
                observations_db = self.observation_repo.search_observations(
                    session, query, user_id, entity_type
                )
            else:
                # Use regular get_all with filters
                filters = {}
                if user_id:
                    filters["user_id"] = user_id
                if entity_type:
                    filters["entity_type"] = entity_type

                observations_db = self.observation_repo.get_all(
                    session, skip, limit, filters
                )

            return [
                self._db_to_pydantic_observation(obs_db) for obs_db in observations_db
            ]

        except Exception as e:
            logger.error(f"Failed to get observations: {e}")
            raise MemoryServiceError(f"Failed to get observations: {e}") from None

    async def get_observations_async(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Observation]:
        """Get observations with optional filtering (async)."""
        try:
            if entity_type and entity_id:
                # Get observations for specific entity
                observations_db = await self.observation_repo.get_by_entity_async(
                    session, entity_type, entity_id, user_id
                )
            elif query:
                # Use repository search method for query
                observations_db = await self.observation_repo.search_observations_async(
                    session, query, user_id, entity_type
                )
            else:
                # Use regular get_all with filters
                filters = {}
                if user_id:
                    filters["user_id"] = user_id
                if entity_type:
                    filters["entity_type"] = entity_type

                observations_db = await self.observation_repo.get_all_async(
                    session, skip, limit, filters
                )

            return [
                self._db_to_pydantic_observation(obs_db) for obs_db in observations_db
            ]

        except Exception as e:
            logger.error(f"Failed to get observations: {e}")
            raise MemoryServiceError(f"Failed to get observations: {e}") from None

    def update_observation(
        self, session: Session, observation_id: int, **kwargs
    ) -> Optional[Observation]:
        """Update an observation."""
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.utcnow()

            observation_db = self.observation_repo.update(
                session, observation_id, **kwargs
            )
            if not observation_db:
                raise NotFoundError(f"Observation with ID {observation_id} not found")

            result = self._db_to_pydantic_observation(observation_db)
            logger.info(f"Updated observation {observation_id}")
            return result

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update observation {observation_id}: {e}")
            raise MemoryServiceError(f"Failed to update observation: {e}") from None

    async def update_observation_async(
        self, session: AsyncSession, observation_id: int, **kwargs
    ) -> Optional[Observation]:
        """Update an observation (async)."""
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.utcnow()

            observation_db = await self.observation_repo.update_async(
                session, observation_id, **kwargs
            )
            if not observation_db:
                raise NotFoundError(f"Observation with ID {observation_id} not found")

            result = self._db_to_pydantic_observation(observation_db)
            logger.info(f"Updated observation {observation_id}")
            return result

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update observation {observation_id}: {e}")
            raise MemoryServiceError(f"Failed to update observation: {e}") from None

    def delete_observation(self, session: Session, observation_id: int) -> bool:
        """Delete an observation."""
        try:
            success = self.observation_repo.delete(session, observation_id)
            if success:
                logger.info(f"Deleted observation {observation_id}")
            else:
                raise NotFoundError(f"Observation with ID {observation_id} not found")
            return success

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete observation {observation_id}: {e}")
            raise MemoryServiceError(f"Failed to delete observation: {e}") from None

    async def delete_observation_async(
        self, session: AsyncSession, observation_id: int
    ) -> bool:
        """Delete an observation (async)."""
        try:
            success = await self.observation_repo.delete_async(session, observation_id)
            if success:
                logger.info(f"Deleted observation {observation_id}")
            else:
                raise NotFoundError(f"Observation with ID {observation_id} not found")
            return success

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete observation {observation_id}: {e}")
            raise MemoryServiceError(f"Failed to delete observation: {e}") from None

    # Hint methods
    def create_hint(self, session: Session, hint: Hint) -> Hint:
        """Create a new hint."""
        try:
            # Validate the hint
            hint.model_validate(hint.model_dump())

            # Ensure user exists if specified
            self.ensure_user(session, hint.user_id)

            # Create database record
            hint_db = self.hint_repo.create(
                session,
                user_id=hint.user_id,
                content=hint.content,
                category=hint.category,
                priority=hint.priority,
                workflow_context=hint.workflow_context,
                tags=hint.tags,
            )

            # Convert back to Pydantic model
            result = self._db_to_pydantic_hint(hint_db)

            # Index the hint for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    search_service.index_memory(session, result)
                session.commit()
            except Exception as e:
                logger.warning(f"Failed to index hint {result.id} for search: {e}")

            logger.info(f"Created hint in category {result.category}")
            return result

        except Exception as e:
            logger.error(f"Failed to create hint: {e}")
            raise MemoryServiceError(f"Failed to create hint: {e}") from None

    async def create_hint_async(self, session: AsyncSession, hint: Hint) -> Hint:
        """Create a new hint (async)."""
        try:
            # Validate the hint
            hint.model_validate(hint.model_dump())

            # Ensure user exists if specified
            await self.ensure_user_async(session, hint.user_id)

            # Create database record
            hint_db = await self.hint_repo.create_async(
                session,
                user_id=hint.user_id,
                content=hint.content,
                category=hint.category,
                priority=hint.priority,
                workflow_context=hint.workflow_context,
                tags=hint.tags,
            )

            # Convert back to Pydantic model
            result = self._db_to_pydantic_hint(hint_db)

            # Index the hint for search
            try:
                search_service = self._get_search_service()
                if search_service:
                    await search_service.index_memory_async(session, result)
                await session.commit()
            except Exception as e:
                logger.warning(f"Failed to index hint {result.id} for search: {e}")

            logger.info(f"Created hint in category {result.category}")
            return result

        except Exception as e:
            logger.error(f"Failed to create hint: {e}")
            raise MemoryServiceError(f"Failed to create hint: {e}") from None

    def get_hints(
        self,
        session: Session,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Hint]:
        """Get hints with optional filtering."""
        try:
            if category and not query:
                # Get hints by category (sorted by priority)
                hints_db = self.hint_repo.get_by_category(session, category, user_id)
            elif query:
                # Use repository search method for query
                hints_db = self.hint_repo.search_hints(
                    session, query, user_id, category
                )
            else:
                # Use regular get_all with filters
                filters = {}
                if user_id:
                    filters["user_id"] = user_id
                if category:
                    filters["category"] = category

                hints_db = self.hint_repo.get_all(
                    session, skip, limit, filters, order_by="-priority"
                )

            return [self._db_to_pydantic_hint(hint_db) for hint_db in hints_db]

        except Exception as e:
            logger.error(f"Failed to get hints: {e}")
            raise MemoryServiceError(f"Failed to get hints: {e}") from None

    async def get_hints_async(
        self,
        session: AsyncSession,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Hint]:
        """Get hints with optional filtering (async)."""
        try:
            if category and not query:
                # Get hints by category (sorted by priority)
                hints_db = await self.hint_repo.get_by_category_async(
                    session, category, user_id
                )
            elif query:
                # Use repository search method for query
                hints_db = await self.hint_repo.search_hints_async(
                    session, query, user_id, category
                )
            else:
                # Use regular get_all with filters
                filters = {}
                if user_id:
                    filters["user_id"] = user_id
                if category:
                    filters["category"] = category

                hints_db = await self.hint_repo.get_all_async(
                    session, skip, limit, filters, order_by="-priority"
                )

            return [self._db_to_pydantic_hint(hint_db) for hint_db in hints_db]

        except Exception as e:
            logger.error(f"Failed to get hints: {e}")
            raise MemoryServiceError(f"Failed to get hints: {e}") from None

    def update_hint(self, session: Session, hint_id: int, **kwargs) -> Optional[Hint]:
        """Update a hint."""
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.utcnow()

            hint_db = self.hint_repo.update(session, hint_id, **kwargs)
            if not hint_db:
                raise NotFoundError(f"Hint with ID {hint_id} not found")

            result = self._db_to_pydantic_hint(hint_db)
            logger.info(f"Updated hint {hint_id}")
            return result

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update hint {hint_id}: {e}")
            raise MemoryServiceError(f"Failed to update hint: {e}") from None

    async def update_hint_async(
        self, session: AsyncSession, hint_id: int, **kwargs
    ) -> Optional[Hint]:
        """Update a hint (async)."""
        try:
            # Update timestamp
            kwargs["updated_at"] = datetime.utcnow()

            hint_db = await self.hint_repo.update_async(session, hint_id, **kwargs)
            if not hint_db:
                raise NotFoundError(f"Hint with ID {hint_id} not found")

            result = self._db_to_pydantic_hint(hint_db)
            logger.info(f"Updated hint {hint_id}")
            return result

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update hint {hint_id}: {e}")
            raise MemoryServiceError(f"Failed to update hint: {e}") from None

    def delete_hint(self, session: Session, hint_id: int) -> bool:
        """Delete a hint."""
        try:
            success = self.hint_repo.delete(session, hint_id)
            if success:
                logger.info(f"Deleted hint {hint_id}")
            else:
                raise NotFoundError(f"Hint with ID {hint_id} not found")
            return success

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete hint {hint_id}: {e}")
            raise MemoryServiceError(f"Failed to delete hint: {e}") from None

    async def delete_hint_async(self, session: AsyncSession, hint_id: int) -> bool:
        """Delete a hint (async)."""
        try:
            success = await self.hint_repo.delete_async(session, hint_id)
            if success:
                logger.info(f"Deleted hint {hint_id}")
            else:
                raise NotFoundError(f"Hint with ID {hint_id} not found")
            return success

        except NotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete hint {hint_id}: {e}")
            raise MemoryServiceError(f"Failed to delete hint: {e}") from None

    # Search across all memory types with semantic and exact search capabilities
    def search_memories(
        self,
        session: Session,
        query: str,
        user_id: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        search_type: str = "combined",
        limit: int = 100,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search across all memory types using semantic and exact search.

        Args:
            session: Database session
            query: Search query text
            user_id: Optional user ID for filtering
            memory_types: List of memory types to search ('alias', 'note', 'observation', 'hint')
            search_type: Type of search ('semantic', 'exact', 'combined')
            limit: Maximum number of results to return
            similarity_threshold: Minimum cosine similarity threshold for semantic search

        Returns:
            List of search results with similarity scores and metadata
        """
        try:
            if search_type == "semantic":
                search_service = self._get_search_service()
            if search_service:
                return search_service.semantic_search(
                    session, query, memory_types, user_id, limit, similarity_threshold
                )
            elif search_type == "exact":
                search_service = self._get_search_service()
            if search_service:
                return search_service.exact_search(
                    session, query, memory_types, user_id, limit
                )
            elif search_type == "combined":
                search_service = self._get_search_service()
            if search_service:
                return search_service.combined_search(
                    session,
                    query,
                    memory_types,
                    user_id,
                    limit,
                    semantic_weight=0.7,
                    similarity_threshold=similarity_threshold,
                )
            else:
                raise ValueError(
                    f"Invalid search_type: {search_type}. Must be 'semantic', 'exact', or 'combined'"
                )

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise MemoryServiceError(f"Failed to search memories: {e}") from None

    async def search_memories_async(
        self,
        session: AsyncSession,
        query: str,
        user_id: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        search_type: str = "combined",
        limit: int = 100,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Search across all memory types using semantic and exact search (async).

        Args:
            session: Database session
            query: Search query text
            user_id: Optional user ID for filtering
            memory_types: List of memory types to search ('alias', 'note', 'observation', 'hint')
            search_type: Type of search ('semantic', 'exact', 'combined')
            limit: Maximum number of results to return
            similarity_threshold: Minimum cosine similarity threshold for semantic search

        Returns:
            List of search results with similarity scores and metadata
        """
        try:
            if search_type == "semantic":
                search_service = self._get_search_service()
            if search_service:
                return await search_service.semantic_search_async(
                    session, query, memory_types, user_id, limit, similarity_threshold
                )
            elif search_type == "exact":
                search_service = self._get_search_service()
            if search_service:
                return await search_service.exact_search_async(
                    session, query, memory_types, user_id, limit
                )
            elif search_type == "combined":
                search_service = self._get_search_service()
            if search_service:
                return await search_service.combined_search_async(
                    session,
                    query,
                    memory_types,
                    user_id,
                    limit,
                    semantic_weight=0.7,
                    similarity_threshold=similarity_threshold,
                )
            else:
                raise ValueError(
                    f"Invalid search_type: {search_type}. Must be 'semantic', 'exact', or 'combined'"
                )

        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise MemoryServiceError(f"Failed to search memories: {e}") from None

    # Legacy search method for backward compatibility
    def search_memories_legacy(
        self,
        session: Session,
        query: str,
        user_id: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[BaseMemory]:
        """Legacy search method using repository-level text search."""
        try:
            results = []

            # Default to all memory types if not specified
            if not memory_types:
                memory_types = ["alias", "note", "observation", "hint"]

            # Search aliases
            if "alias" in memory_types:
                aliases = self.get_aliases(session, user_id, query, skip, limit)
                results.extend(aliases)

            # Search notes
            if "note" in memory_types:
                notes = self.get_notes(session, user_id, None, query, skip, limit)
                results.extend(notes)

            # Search observations
            if "observation" in memory_types:
                observations = self.get_observations(
                    session, user_id, None, None, query, skip, limit
                )
                results.extend(observations)

            # Search hints
            if "hint" in memory_types:
                hints = self.get_hints(session, user_id, None, query, skip, limit)
                results.extend(hints)

            # Sort by updated_at descending
            results.sort(key=lambda x: x.updated_at, reverse=True)

            # Apply limit to combined results
            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to search memories (legacy): {e}")
            raise MemoryServiceError(
                f"Failed to search memories (legacy): {e}"
            ) from None

    async def search_memories_legacy_async(
        self,
        session: AsyncSession,
        query: str,
        user_id: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[BaseMemory]:
        """Legacy search method using repository-level text search (async)."""
        try:
            results = []

            # Default to all memory types if not specified
            if not memory_types:
                memory_types = ["alias", "note", "observation", "hint"]

            # Search aliases
            if "alias" in memory_types:
                aliases = await self.get_aliases_async(
                    session, user_id, query, skip, limit
                )
                results.extend(aliases)

            # Search notes
            if "note" in memory_types:
                notes = await self.get_notes_async(
                    session, user_id, None, query, skip, limit
                )
                results.extend(notes)

            # Search observations
            if "observation" in memory_types:
                observations = await self.get_observations_async(
                    session, user_id, None, None, query, skip, limit
                )
                results.extend(observations)

            # Search hints
            if "hint" in memory_types:
                hints = await self.get_hints_async(
                    session, user_id, None, query, skip, limit
                )
                results.extend(hints)

            # Sort by updated_at descending
            results.sort(key=lambda x: x.updated_at, reverse=True)

            # Apply limit to combined results
            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to search memories (legacy): {e}")
            raise MemoryServiceError(
                f"Failed to search memories (legacy): {e}"
            ) from None

    # Helper methods for converting database models to Pydantic models
    def _db_to_pydantic_alias(self, alias_db: AliasDB) -> Alias:
        """Convert database alias to Pydantic alias."""
        return Alias(
            id=alias_db.id,
            user_id=alias_db.user_id,
            source=alias_db.source,
            target=alias_db.target,
            bidirectional=alias_db.bidirectional,
            tags=alias_db.tags or [],
            created_at=alias_db.created_at,
            updated_at=alias_db.updated_at,
        )

    def _db_to_pydantic_note(self, note_db: NoteDB) -> Note:
        """Convert database note to Pydantic note."""
        return Note(
            id=note_db.id,
            user_id=note_db.user_id,
            title=note_db.title,
            content=note_db.content,
            category=note_db.category,
            tags=note_db.tags or [],
            created_at=note_db.created_at,
            updated_at=note_db.updated_at,
        )

    def _db_to_pydantic_observation(self, observation_db: ObservationDB) -> Observation:
        """Convert database observation to Pydantic observation."""
        return Observation(
            id=observation_db.id,
            user_id=observation_db.user_id,
            content=observation_db.content,
            entity_type=observation_db.entity_type,
            entity_id=observation_db.entity_id,
            context=observation_db.context or {},
            tags=observation_db.tags or [],
            created_at=observation_db.created_at,
            updated_at=observation_db.updated_at,
        )

    # Advanced search utility methods
    def search_memories_with_filters(
        self,
        session: Session,
        query: str,
        user_id: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        search_type: str = "combined",
        limit: int = 100,
        similarity_threshold: float = 0.3,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Advanced search with additional filtering options.

        Args:
            session: Database session
            query: Search query text
            user_id: Optional user ID for filtering
            memory_types: List of memory types to search
            search_type: Type of search ('semantic', 'exact', 'combined')
            limit: Maximum number of results to return
            similarity_threshold: Minimum cosine similarity threshold
            date_from: Filter results from this date
            date_to: Filter results to this date
            tags: Filter by tags (any of these tags)
            categories: Filter by categories (for notes/hints)

        Returns:
            List of filtered search results
        """
        try:
            # Get initial search results
            results = self.search_memories(
                session,
                query,
                user_id,
                memory_types,
                search_type,
                limit * 2,
                similarity_threshold,
            )

            # Apply additional filters
            filtered_results = []
            for result in results:
                metadata = result.get("metadata", {})

                # Date filtering
                if date_from or date_to:
                    created_at_str = metadata.get("created_at")
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_str.replace("Z", "+00:00")
                            )
                            if date_from and created_at < date_from:
                                continue
                            if date_to and created_at > date_to:
                                continue
                        except (ValueError, AttributeError):
                            continue

                # Tag filtering
                if tags:
                    result_tags = metadata.get("tags", [])
                    if not any(tag in result_tags for tag in tags):
                        continue

                # Category filtering (for notes and hints)
                if categories:
                    content = result.get("content", {})
                    result_category = content.get("category")
                    if result_category not in categories:
                        continue

                filtered_results.append(result)

            # Apply final limit
            return filtered_results[:limit]

        except Exception as e:
            logger.error(f"Failed to search memories with filters: {e}")
            raise MemoryServiceError(
                f"Failed to search memories with filters: {e}"
            ) from None

    async def search_memories_with_filters_async(
        self,
        session: AsyncSession,
        query: str,
        user_id: Optional[str] = None,
        memory_types: Optional[List[str]] = None,
        search_type: str = "combined",
        limit: int = 100,
        similarity_threshold: float = 0.3,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Advanced search with additional filtering options (async)."""
        try:
            # Get initial search results
            results = await self.search_memories_async(
                session,
                query,
                user_id,
                memory_types,
                search_type,
                limit * 2,
                similarity_threshold,
            )

            # Apply additional filters
            filtered_results = []
            for result in results:
                metadata = result.get("metadata", {})

                # Date filtering
                if date_from or date_to:
                    created_at_str = metadata.get("created_at")
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(
                                created_at_str.replace("Z", "+00:00")
                            )
                            if date_from and created_at < date_from:
                                continue
                            if date_to and created_at > date_to:
                                continue
                        except (ValueError, AttributeError):
                            continue

                # Tag filtering
                if tags:
                    result_tags = metadata.get("tags", [])
                    if not any(tag in result_tags for tag in tags):
                        continue

                # Category filtering (for notes and hints)
                if categories:
                    content = result.get("content", {})
                    result_category = content.get("category")
                    if result_category not in categories:
                        continue

                filtered_results.append(result)

            # Apply final limit
            return filtered_results[:limit]

        except Exception as e:
            logger.error(f"Failed to search memories with filters: {e}")
            raise MemoryServiceError(
                f"Failed to search memories with filters: {e}"
            ) from None

    def reindex_all_memories(self, session: Session) -> Dict[str, int]:
        """Reindex all memories for search functionality.

        Args:
            session: Database session

        Returns:
            Dictionary with counts of reindexed memories by type
        """
        try:
            search_service = self._get_search_service()
            if search_service:
                return search_service.reindex_all_memories(session)
            return False
        except Exception as e:
            logger.error(f"Failed to reindex all memories: {e}")
            raise MemoryServiceError(f"Failed to reindex all memories: {e}") from None

    async def reindex_all_memories_async(self, session: AsyncSession) -> Dict[str, int]:
        """Reindex all memories for search functionality (async)."""
        try:
            search_service = self._get_search_service()
            if search_service:
                return await search_service.reindex_all_memories_async(session)
            return False
        except Exception as e:
            logger.error(f"Failed to reindex all memories: {e}")
            raise MemoryServiceError(f"Failed to reindex all memories: {e}") from None

    def _db_to_pydantic_hint(self, hint_db: HintDB) -> Hint:
        """Convert database hint to Pydantic hint."""
        return Hint(
            id=hint_db.id,
            user_id=hint_db.user_id,
            content=hint_db.content,
            category=hint_db.category,
            priority=hint_db.priority,
            workflow_context=hint_db.workflow_context,
            tags=hint_db.tags or [],
            created_at=hint_db.created_at,
            updated_at=hint_db.updated_at,
        )


# Global memory service instance
_memory_service: Optional[MemoryService] = None


def get_memory_service() -> MemoryService:
    """Get the global memory service instance."""
    global _memory_service
    if _memory_service is None:
        _memory_service = MemoryService()
    return _memory_service


def set_memory_service(service: MemoryService) -> None:
    """Set the global memory service instance."""
    global _memory_service
    _memory_service = service
