"""Repository pattern implementation for data access layer."""

import logging
from abc import ABC
from datetime import datetime, UTC
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.sql import Select

from .models import AliasDB, Base, EmbeddingDB, HintDB, NoteDB, ObservationDB, User

logger = logging.getLogger(__name__)

# Type variable for generic repository
T = TypeVar("T", bound=Base)


class BaseRepository(Generic[T], ABC):
    """Base repository class with common CRUD operations."""

    def __init__(self, model_class: Type[T]):
        """Initialize repository with model class."""
        self.model_class = model_class

    # Synchronous methods
    def create(self, session: Session, **kwargs) -> T:
        """Create a new record."""
        instance = self.model_class(**kwargs)
        session.add(instance)
        session.flush()
        session.refresh(instance)
        return instance

    def get_by_id(self, session: Session, record_id: int) -> Optional[T]:
        """Get a record by ID."""
        return (
            session.query(self.model_class)
            .filter(self.model_class.id == record_id)
            .first()
        )

    def get_all(
        self,
        session: Session,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
    ) -> List[T]:
        """Get all records with optional filtering and pagination."""
        query = session.query(self.model_class)

        # Apply filters
        if filters:
            query = self._apply_filters(query, filters)

        # Apply ordering
        if order_by:
            if order_by.startswith("-"):
                # Descending order
                field_name = order_by[1:]
                if hasattr(self.model_class, field_name):
                    query = query.order_by(desc(getattr(self.model_class, field_name)))
            else:
                # Ascending order
                if hasattr(self.model_class, order_by):
                    query = query.order_by(getattr(self.model_class, order_by))
        else:
            # Default ordering by created_at if available
            if hasattr(self.model_class, "created_at"):
                query = query.order_by(desc(self.model_class.created_at))

        return query.offset(skip).limit(limit).all()

    def update(self, session: Session, record_id: int, **kwargs) -> Optional[T]:
        """Update a record by ID."""
        instance = self.get_by_id(session, record_id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

            # Update timestamp if available
            if hasattr(instance, "updated_at"):
                instance.updated_at = datetime.now(UTC)

            session.flush()
            session.refresh(instance)
        return instance

    def delete(self, session: Session, record_id: int) -> bool:
        """Delete a record by ID."""
        instance = self.get_by_id(session, record_id)
        if instance:
            session.delete(instance)
            return True
        return False

    def count(self, session: Session, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count records with optional filtering."""
        query = session.query(func.count(self.model_class.id))
        if filters:
            query = self._apply_filters(query, filters)
        return query.scalar()

    # Asynchronous methods
    async def create_async(self, session: AsyncSession, **kwargs) -> T:
        """Create a new record asynchronously."""
        instance = self.model_class(**kwargs)
        session.add(instance)
        await session.flush()
        await session.refresh(instance)
        return instance

    async def get_by_id_async(
        self, session: AsyncSession, record_id: int
    ) -> Optional[T]:
        """Get a record by ID asynchronously."""
        result = await session.get(self.model_class, record_id)
        return result

    async def get_all_async(
        self,
        session: AsyncSession,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
    ) -> List[T]:
        """Get all records with optional filtering and pagination asynchronously."""
        from sqlalchemy import select

        query = select(self.model_class)

        # Apply filters
        if filters:
            query = self._apply_filters_async(query, filters)

        # Apply ordering
        if order_by:
            if order_by.startswith("-"):
                # Descending order
                field_name = order_by[1:]
                if hasattr(self.model_class, field_name):
                    query = query.order_by(desc(getattr(self.model_class, field_name)))
            else:
                # Ascending order
                if hasattr(self.model_class, order_by):
                    query = query.order_by(getattr(self.model_class, order_by))
        else:
            # Default ordering by created_at if available
            if hasattr(self.model_class, "created_at"):
                query = query.order_by(desc(self.model_class.created_at))

        query = query.offset(skip).limit(limit)
        result = await session.execute(query)
        return result.scalars().all()

    async def update_async(
        self, session: AsyncSession, record_id: int, **kwargs
    ) -> Optional[T]:
        """Update a record by ID asynchronously."""
        instance = await self.get_by_id_async(session, record_id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)

            # Update timestamp if available
            if hasattr(instance, "updated_at"):
                instance.updated_at = datetime.now(UTC)

            await session.flush()
            await session.refresh(instance)
        return instance

    async def delete_async(self, session: AsyncSession, record_id: int) -> bool:
        """Delete a record by ID asynchronously."""
        instance = await self.get_by_id_async(session, record_id)
        if instance:
            await session.delete(instance)
            return True
        return False

    async def count_async(
        self, session: AsyncSession, filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count records with optional filtering asynchronously."""
        from sqlalchemy import select

        query = select(func.count(self.model_class.id))
        if filters:
            query = self._apply_filters_async(query, filters)
        result = await session.execute(query)
        return result.scalar()

    def _apply_filters(self, query, filters: Dict[str, Any]):
        """Apply filters to a synchronous query."""
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                column = getattr(self.model_class, key)
                if isinstance(value, list):
                    query = query.filter(column.in_(value))
                elif isinstance(value, dict):
                    # Handle range queries, etc.
                    if "gte" in value:
                        query = query.filter(column >= value["gte"])
                    if "lte" in value:
                        query = query.filter(column <= value["lte"])
                    if "gt" in value:
                        query = query.filter(column > value["gt"])
                    if "lt" in value:
                        query = query.filter(column < value["lt"])
                    if "like" in value:
                        query = query.filter(column.like(f"%{value['like']}%"))
                else:
                    query = query.filter(column == value)
        return query

    def _apply_filters_async(self, query: Select, filters: Dict[str, Any]) -> Select:
        """Apply filters to an asynchronous query."""
        for key, value in filters.items():
            if hasattr(self.model_class, key):
                column = getattr(self.model_class, key)
                if isinstance(value, list):
                    query = query.where(column.in_(value))
                elif isinstance(value, dict):
                    # Handle range queries, etc.
                    if "gte" in value:
                        query = query.where(column >= value["gte"])
                    if "lte" in value:
                        query = query.where(column <= value["lte"])
                    if "gt" in value:
                        query = query.where(column > value["gt"])
                    if "lt" in value:
                        query = query.where(column < value["lt"])
                    if "like" in value:
                        query = query.where(column.like(f"%{value['like']}%"))
                else:
                    query = query.where(column == value)
        return query


class UserRepository(BaseRepository[User]):
    """Repository for User model."""

    def __init__(self):
        super().__init__(User)

    def get_by_username(self, session: Session, username: str) -> Optional[User]:
        """Get user by username."""
        return session.query(User).filter(User.id == username).first()

    async def get_by_username_async(
        self, session: AsyncSession, username: str
    ) -> Optional[User]:
        """Get user by username asynchronously."""
        result = await session.get(User, username)
        return result

    def get_or_create(self, session: Session, username: str) -> User:
        """Get existing user or create new one."""
        user = self.get_by_username(session, username)
        if not user:
            user = self.create(session, id=username)
        return user

    async def get_or_create_async(self, session: AsyncSession, username: str) -> User:
        """Get existing user or create new one asynchronously."""
        user = await self.get_by_username_async(session, username)
        if not user:
            user = await self.create_async(session, id=username)
        return user


class AliasRepository(BaseRepository[AliasDB]):
    """Repository for Alias model."""

    def __init__(self):
        super().__init__(AliasDB)

    def search_aliases(
        self,
        session: Session,
        query: str,
        user_id: Optional[str] = None,
        bidirectional: bool = True,
    ) -> List[AliasDB]:
        """Search aliases by source or target."""
        filters = []

        if bidirectional:
            # Search in both directions
            filters.append(
                or_(
                    AliasDB.source.like(f"%{query}%"), AliasDB.target.like(f"%{query}%")
                )
            )
        else:
            # Search only in source
            filters.append(AliasDB.source.like(f"%{query}%"))

        if user_id:
            filters.append(AliasDB.user_id == user_id)

        return session.query(AliasDB).filter(and_(*filters)).all()

    async def search_aliases_async(
        self,
        session: AsyncSession,
        query: str,
        user_id: Optional[str] = None,
        bidirectional: bool = True,
    ) -> List[AliasDB]:
        """Search aliases by source or target asynchronously."""
        from sqlalchemy import select

        filters = []

        if bidirectional:
            # Search in both directions
            filters.append(
                or_(
                    AliasDB.source.like(f"%{query}%"), AliasDB.target.like(f"%{query}%")
                )
            )
        else:
            # Search only in source
            filters.append(AliasDB.source.like(f"%{query}%"))

        if user_id:
            filters.append(AliasDB.user_id == user_id)

        query_stmt = select(AliasDB).where(and_(*filters))
        result = await session.execute(query_stmt)
        return result.scalars().all()

    def get_by_source(
        self, session: Session, source: str, user_id: Optional[str] = None
    ) -> List[AliasDB]:
        """Get aliases by exact source match."""
        filters = [AliasDB.source == source]
        if user_id:
            filters.append(AliasDB.user_id == user_id)

        return session.query(AliasDB).filter(and_(*filters)).all()

    async def get_by_source_async(
        self, session: AsyncSession, source: str, user_id: Optional[str] = None
    ) -> List[AliasDB]:
        """Get aliases by exact source match asynchronously."""
        from sqlalchemy import select

        filters = [AliasDB.source == source]
        if user_id:
            filters.append(AliasDB.user_id == user_id)

        query = select(AliasDB).where(and_(*filters))
        result = await session.execute(query)
        return result.scalars().all()


class NoteRepository(BaseRepository[NoteDB]):
    """Repository for Note model."""

    def __init__(self):
        super().__init__(NoteDB)

    def search_notes(
        self,
        session: Session,
        query: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[NoteDB]:
        """Search notes by title or content."""
        filters = [
            or_(NoteDB.title.like(f"%{query}%"), NoteDB.content.like(f"%{query}%"))
        ]

        if user_id:
            filters.append(NoteDB.user_id == user_id)

        if category:
            filters.append(NoteDB.category == category)

        return session.query(NoteDB).filter(and_(*filters)).all()

    async def search_notes_async(
        self,
        session: AsyncSession,
        query: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[NoteDB]:
        """Search notes by title or content asynchronously."""
        from sqlalchemy import select

        filters = [
            or_(NoteDB.title.like(f"%{query}%"), NoteDB.content.like(f"%{query}%"))
        ]

        if user_id:
            filters.append(NoteDB.user_id == user_id)

        if category:
            filters.append(NoteDB.category == category)

        query_stmt = select(NoteDB).where(and_(*filters))
        result = await session.execute(query_stmt)
        return result.scalars().all()

    def get_by_category(
        self, session: Session, category: str, user_id: Optional[str] = None
    ) -> List[NoteDB]:
        """Get notes by category."""
        filters = [NoteDB.category == category]
        if user_id:
            filters.append(NoteDB.user_id == user_id)

        return session.query(NoteDB).filter(and_(*filters)).all()

    async def get_by_category_async(
        self, session: AsyncSession, category: str, user_id: Optional[str] = None
    ) -> List[NoteDB]:
        """Get notes by category asynchronously."""
        from sqlalchemy import select

        filters = [NoteDB.category == category]
        if user_id:
            filters.append(NoteDB.user_id == user_id)

        query = select(NoteDB).where(and_(*filters))
        result = await session.execute(query)
        return result.scalars().all()


class ObservationRepository(BaseRepository[ObservationDB]):
    """Repository for Observation model."""

    def __init__(self):
        super().__init__(ObservationDB)

    def get_by_entity(
        self,
        session: Session,
        entity_type: str,
        entity_id: str,
        user_id: Optional[str] = None,
    ) -> List[ObservationDB]:
        """Get observations by entity."""
        filters = [
            ObservationDB.entity_type == entity_type,
            ObservationDB.entity_id == entity_id,
        ]
        if user_id:
            filters.append(ObservationDB.user_id == user_id)

        return session.query(ObservationDB).filter(and_(*filters)).all()

    async def get_by_entity_async(
        self,
        session: AsyncSession,
        entity_type: str,
        entity_id: str,
        user_id: Optional[str] = None,
    ) -> List[ObservationDB]:
        """Get observations by entity asynchronously."""
        from sqlalchemy import select

        filters = [
            ObservationDB.entity_type == entity_type,
            ObservationDB.entity_id == entity_id,
        ]
        if user_id:
            filters.append(ObservationDB.user_id == user_id)

        query = select(ObservationDB).where(and_(*filters))
        result = await session.execute(query)
        return result.scalars().all()

    def search_observations(
        self,
        session: Session,
        query: str,
        user_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> List[ObservationDB]:
        """Search observations by content."""
        filters = [ObservationDB.content.like(f"%{query}%")]

        if user_id:
            filters.append(ObservationDB.user_id == user_id)

        if entity_type:
            filters.append(ObservationDB.entity_type == entity_type)

        return session.query(ObservationDB).filter(and_(*filters)).all()

    async def search_observations_async(
        self,
        session: AsyncSession,
        query: str,
        user_id: Optional[str] = None,
        entity_type: Optional[str] = None,
    ) -> List[ObservationDB]:
        """Search observations by content asynchronously."""
        from sqlalchemy import select

        filters = [ObservationDB.content.like(f"%{query}%")]

        if user_id:
            filters.append(ObservationDB.user_id == user_id)

        if entity_type:
            filters.append(ObservationDB.entity_type == entity_type)

        query_stmt = select(ObservationDB).where(and_(*filters))
        result = await session.execute(query_stmt)
        return result.scalars().all()


class HintRepository(BaseRepository[HintDB]):
    """Repository for Hint model."""

    def __init__(self):
        super().__init__(HintDB)

    def get_by_category(
        self, session: Session, category: str, user_id: Optional[str] = None
    ) -> List[HintDB]:
        """Get hints by category."""
        filters = [HintDB.category == category]
        if user_id:
            filters.append(HintDB.user_id == user_id)

        return (
            session.query(HintDB)
            .filter(and_(*filters))
            .order_by(desc(HintDB.priority))
            .all()
        )

    async def get_by_category_async(
        self, session: AsyncSession, category: str, user_id: Optional[str] = None
    ) -> List[HintDB]:
        """Get hints by category asynchronously."""
        from sqlalchemy import select

        filters = [HintDB.category == category]
        if user_id:
            filters.append(HintDB.user_id == user_id)

        query = select(HintDB).where(and_(*filters)).order_by(desc(HintDB.priority))
        result = await session.execute(query)
        return result.scalars().all()

    def search_hints(
        self,
        session: Session,
        query: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[HintDB]:
        """Search hints by content."""
        filters = [HintDB.content.like(f"%{query}%")]

        if user_id:
            filters.append(HintDB.user_id == user_id)

        if category:
            filters.append(HintDB.category == category)

        return (
            session.query(HintDB)
            .filter(and_(*filters))
            .order_by(desc(HintDB.priority))
            .all()
        )

    async def search_hints_async(
        self,
        session: AsyncSession,
        query: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[HintDB]:
        """Search hints by content asynchronously."""
        from sqlalchemy import select

        filters = [HintDB.content.like(f"%{query}%")]

        if user_id:
            filters.append(HintDB.user_id == user_id)

        if category:
            filters.append(HintDB.category == category)

        query_stmt = (
            select(HintDB).where(and_(*filters)).order_by(desc(HintDB.priority))
        )
        result = await session.execute(query_stmt)
        return result.scalars().all()


class EmbeddingRepository(BaseRepository[EmbeddingDB]):
    """Repository for Embedding model."""

    def __init__(self):
        super().__init__(EmbeddingDB)

    def get_by_memory(
        self, session: Session, memory_type: str, memory_id: int
    ) -> Optional[EmbeddingDB]:
        """Get embedding by memory type and ID."""
        return (
            session.query(EmbeddingDB)
            .filter(
                and_(
                    EmbeddingDB.memory_type == memory_type,
                    EmbeddingDB.memory_id == memory_id,
                )
            )
            .first()
        )

    async def get_by_memory_async(
        self, session: AsyncSession, memory_type: str, memory_id: int
    ) -> Optional[EmbeddingDB]:
        """Get embedding by memory type and ID asynchronously."""
        from sqlalchemy import select

        query = select(EmbeddingDB).where(
            and_(
                EmbeddingDB.memory_type == memory_type,
                EmbeddingDB.memory_id == memory_id,
            )
        )
        result = await session.execute(query)
        return result.scalar_one_or_none()

    def get_by_memory_type(
        self, session: Session, memory_type: str
    ) -> List[EmbeddingDB]:
        """Get all embeddings by memory type."""
        return (
            session.query(EmbeddingDB)
            .filter(EmbeddingDB.memory_type == memory_type)
            .all()
        )

    async def get_by_memory_type_async(
        self, session: AsyncSession, memory_type: str
    ) -> List[EmbeddingDB]:
        """Get all embeddings by memory type asynchronously."""
        from sqlalchemy import select

        query = select(EmbeddingDB).where(EmbeddingDB.memory_type == memory_type)
        result = await session.execute(query)
        return result.scalars().all()

    def get_by_type(self, session: Session, memory_type: str) -> List[EmbeddingDB]:
        """Get all embeddings by memory type (alias for get_by_memory_type)."""
        return self.get_by_memory_type(session, memory_type)

    async def get_by_type_async(
        self, session: AsyncSession, memory_type: str
    ) -> List[EmbeddingDB]:
        """Get all embeddings by memory type asynchronously (alias for get_by_memory_type_async)."""
        return await self.get_by_memory_type_async(session, memory_type)

    def delete_all(self, session: Session) -> int:
        """Delete all embeddings and return count of deleted records."""
        count = session.query(EmbeddingDB).count()
        session.query(EmbeddingDB).delete()
        return count

    async def delete_all_async(self, session: AsyncSession) -> int:
        """Delete all embeddings and return count of deleted records asynchronously."""
        from sqlalchemy import delete, select

        # Count existing records
        count_query = select(func.count(EmbeddingDB.id))
        result = await session.execute(count_query)
        count = result.scalar()

        # Delete all records
        delete_query = delete(EmbeddingDB)
        await session.execute(delete_query)

        return count
