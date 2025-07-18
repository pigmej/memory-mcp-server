"""HTTP streaming server for the Memory MCP Server."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Config, get_config
from ..database.connection import get_database_manager
from ..models.alias import Alias
from ..models.hint import Hint
from ..models.note import Note
from ..models.observation import Observation
from ..services.memory_service import MemoryService, MemoryServiceError, NotFoundError
from ..services.search_service import SearchService


# Import utilities locally to avoid circular imports
def _get_utility_classes():
    """Lazy import of utility classes to avoid circular dependencies."""
    try:
        from ..utils.errors import (
            BaseMemoryError,
            DatabaseError,
            ProtocolError,
            ValidationError,
            create_error_context,
        )
        from ..utils.logging import get_logger
        from ..utils.middleware import setup_middleware

        return (
            BaseMemoryError,
            ProtocolError,
            ValidationError,
            DatabaseError,
            create_error_context,
            get_logger,
            setup_middleware,
        )
    except ImportError:
        # Fallback to basic classes if utils not available
        class BaseMemoryError(Exception):
            pass

        class ProtocolError(BaseMemoryError):
            pass

        class ValidationError(BaseMemoryError):
            pass

        class DatabaseError(BaseMemoryError):
            pass

        def create_error_context(**kwargs):
            return None

        def get_logger(name):
            return logging.getLogger(name)

        def setup_middleware(app):
            pass  # No-op fallback

        return (
            BaseMemoryError,
            ProtocolError,
            ValidationError,
            DatabaseError,
            create_error_context,
            get_logger,
            setup_middleware,
        )


(
    BaseMemoryError,
    ProtocolError,
    ValidationError,
    DatabaseError,
    create_error_context,
    get_logger,
    setup_middleware,
) = _get_utility_classes()

logger = get_logger(__name__)


# Request/Response models
class CreateAliasRequest(BaseModel):
    source: str = Field(..., description="Source term or phrase")
    target: str = Field(..., description="Target term or phrase")
    user_id: Optional[str] = Field(None, description="User ID for data separation")
    bidirectional: bool = Field(True, description="Whether the alias works both ways")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class CreateNoteRequest(BaseModel):
    title: str = Field(..., description="Note title")
    content: str = Field(..., description="Note content")
    user_id: Optional[str] = Field(None, description="User ID for data separation")
    category: Optional[str] = Field(None, description="Note category")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class CreateObservationRequest(BaseModel):
    content: str = Field(..., description="Observation content")
    entity_type: str = Field(..., description="Type of entity being observed")
    entity_id: str = Field(..., description="ID of the entity being observed")
    user_id: Optional[str] = Field(None, description="User ID for data separation")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class CreateHintRequest(BaseModel):
    content: str = Field(..., description="Hint content")
    category: str = Field(..., description="Hint category")
    user_id: Optional[str] = Field(None, description="User ID for data separation")
    priority: int = Field(1, description="Hint priority (1-10)")
    workflow_context: Optional[str] = Field(None, description="Workflow context")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    user_id: Optional[str] = Field(None, description="User ID for filtering")
    memory_types: Optional[List[str]] = Field(
        None, description="Memory types to search"
    )
    semantic: bool = Field(True, description="Use semantic search")
    limit: int = Field(10, description="Maximum number of results")


class StreamingSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    user_id: Optional[str] = Field(None, description="User ID for filtering")
    memory_types: Optional[List[str]] = Field(
        None, description="Memory types to search"
    )
    semantic: bool = Field(True, description="Use semantic search")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryHTTPServer:
    """HTTP streaming server for memory operations."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the HTTP server."""
        self.config = config or get_config()
        self.app = FastAPI(
            title="Memory MCP Server",
            description="HTTP API for memory operations with streaming support",
            version="0.1.0",
        )
        self.memory_service = MemoryService()
        self.search_service = SearchService(self.config)
        self.db_manager = get_database_manager(self.config)

        self._setup_middleware()
        self._setup_routes()

    def get_asgi_app(self):
        """Get the ASGI app for this HTTP server."""
        return self.app

    def _setup_middleware(self) -> None:
        """Set up CORS and other middleware."""
        # Setup error handling and logging middleware
        setup_middleware(self.app)

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.server.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self) -> None:
        """Set up API routes."""

        # Health check endpoints
        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint."""
            return {"status": "healthy", "timestamp": datetime.now(UTC)}

        @self.app.get("/health/detailed")
        async def detailed_health_check():
            """Detailed health check with component status."""
            from ..utils import get_health_checker

            health_checker = get_health_checker()
            component_health = health_checker.check_all_health()

            # Add database connectivity check
            try:
                async with self.db_manager.get_async_session() as session:
                    # Simple query to test database connectivity
                    from sqlalchemy import text

                    await session.execute(text("SELECT 1"))
                    component_health["database"] = True
            except Exception as e:
                logger.error(f"Database health check failed: {e}")
                component_health["database"] = False

            # Add search service health check
            try:
                # Test embedding model loading
                test_embedding = self.search_service.generate_embeddings("test")
                component_health["search_service"] = len(test_embedding) > 0
            except Exception as e:
                logger.warning(f"Search service health check failed: {e}")
                component_health["search_service"] = False

            overall_status = "healthy" if all(component_health.values()) else "degraded"

            return {
                "status": overall_status,
                "timestamp": datetime.now(UTC),
                "components": component_health,
            }

        @self.app.get("/metrics")
        async def get_metrics():
            """Get system metrics and error statistics."""
            from ..utils import get_error_handler

            error_handler = get_error_handler()
            error_stats = error_handler.get_error_stats()

            # Get memory statistics
            try:
                async with self.db_manager.get_async_session() as session:
                    stats = await self._get_memory_stats(session)
            except Exception as e:
                logger.error(f"Failed to get memory stats: {e}")
                stats = {"error": "Failed to retrieve stats"}

            return {
                "timestamp": datetime.now(UTC),
                "error_statistics": error_stats,
                "memory_statistics": stats,
                "system_info": {
                    "config": {
                        "embedding_model": self.search_service.model_name,
                        "database_url": self.config.database.url.split("://")[0]
                        + "://[REDACTED]",
                        "log_level": self.config.logging.level,
                    }
                },
            }

        # Alias endpoints
        @self.app.post("/aliases", response_model=Alias)
        async def create_alias(
            request: CreateAliasRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Create a new alias."""
            try:
                alias = Alias(
                    source=request.source,
                    target=request.target,
                    user_id=request.user_id,
                    bidirectional=request.bidirectional,
                    tags=request.tags,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
                result = await self.memory_service.create_alias_async(session, alias)
                return result
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to create alias: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.get("/aliases", response_model=List[Alias])
        async def get_aliases(
            user_id: Optional[str] = Query(None),
            query: Optional[str] = Query(None),
            skip: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=1000),
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Get aliases with optional filtering."""
            try:
                aliases = await self.memory_service.get_aliases_async(
                    session, user_id=user_id, query=query, skip=skip, limit=limit
                )
                return aliases
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to get aliases: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.get("/aliases/query/{query_term}")
        async def query_alias(
            query_term: str,
            user_id: Optional[str] = Query(None),
            exact_match: bool = Query(True),
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Query aliases bidirectionally."""
            try:
                results = await self.memory_service.query_alias_async(
                    session, query_term, user_id=user_id, exact_match=exact_match
                )
                return {"query": query_term, "results": results}
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to query alias: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        # Note endpoints
        @self.app.post("/notes", response_model=Note)
        async def create_note(
            request: CreateNoteRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Create a new note."""
            try:
                note = Note(
                    title=request.title,
                    content=request.content,
                    user_id=request.user_id,
                    category=request.category,
                    tags=request.tags,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
                result = await self.memory_service.create_note_async(session, note)
                return result
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to create note: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.get("/notes", response_model=List[Note])
        async def get_notes(
            user_id: Optional[str] = Query(None),
            category: Optional[str] = Query(None),
            query: Optional[str] = Query(None),
            skip: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=1000),
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Get notes with optional filtering."""
            try:
                notes = await self.memory_service.get_notes_async(
                    session,
                    user_id=user_id,
                    category=category,
                    query=query,
                    skip=skip,
                    limit=limit,
                )
                return notes
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to get notes: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        # Observation endpoints
        @self.app.post("/observations", response_model=Observation)
        async def create_observation(
            request: CreateObservationRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Create a new observation."""
            try:
                observation = Observation(
                    content=request.content,
                    entity_type=request.entity_type,
                    entity_id=request.entity_id,
                    user_id=request.user_id,
                    context=request.context,
                    tags=request.tags,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
                result = await self.memory_service.create_observation_async(
                    session, observation
                )
                return result
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to create observation: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.get("/observations", response_model=List[Observation])
        async def get_observations(
            user_id: Optional[str] = Query(None),
            entity_type: Optional[str] = Query(None),
            entity_id: Optional[str] = Query(None),
            skip: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=1000),
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Get observations with optional filtering."""
            try:
                observations = await self.memory_service.get_observations_async(
                    session,
                    user_id=user_id,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    skip=skip,
                    limit=limit,
                )
                return observations
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to get observations: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        # Hint endpoints
        @self.app.post("/hints", response_model=Hint)
        async def create_hint(
            request: CreateHintRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Create a new hint."""
            try:
                hint = Hint(
                    content=request.content,
                    category=request.category,
                    user_id=request.user_id,
                    priority=request.priority,
                    workflow_context=request.workflow_context,
                    tags=request.tags,
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
                result = await self.memory_service.create_hint_async(session, hint)
                return result
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to create hint: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.get("/hints", response_model=List[Hint])
        async def get_hints(
            user_id: Optional[str] = Query(None),
            category: Optional[str] = Query(None),
            workflow_context: Optional[str] = Query(None),
            skip: int = Query(0, ge=0),
            limit: int = Query(100, ge=1, le=1000),
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Get hints with optional filtering."""
            try:
                hints = await self.memory_service.get_hints_async(
                    session,
                    user_id=user_id,
                    category=category,
                    workflow_context=workflow_context,
                    skip=skip,
                    limit=limit,
                )
                return hints
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to get hints: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        # Search endpoints
        @self.app.post("/search")
        async def search_memories(
            request: SearchRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Search across all memory types."""
            try:
                if request.semantic:
                    results = await self.search_service.semantic_search_async(
                        session,
                        request.query,
                        user_id=request.user_id,
                        memory_types=request.memory_types,
                        limit=request.limit,
                    )
                else:
                    results = await self.search_service.exact_search_async(
                        session,
                        request.query,
                        user_id=request.user_id,
                        memory_types=request.memory_types,
                        limit=request.limit,
                    )
                return {"query": request.query, "results": results}
            except Exception as e:
                logger.error(f"Failed to search memories: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.post("/search/stream")
        async def search_memories_stream(
            request: StreamingSearchRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Search across all memory types with streaming response."""
            try:
                return StreamingResponse(
                    self._stream_search_results(session, request),
                    media_type="application/x-ndjson",
                )
            except Exception as e:
                logger.error(f"Failed to stream search results: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        # Update/Delete endpoints for all memory types
        @self.app.put("/aliases/{alias_id}", response_model=Alias)
        async def update_alias(
            alias_id: int,
            request: CreateAliasRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Update an existing alias."""
            try:
                result = await self.memory_service.update_alias_async(
                    session,
                    alias_id,
                    source=request.source,
                    target=request.target,
                    bidirectional=request.bidirectional,
                    tags=request.tags,
                )
                if not result:
                    raise HTTPException(status_code=404, detail="Alias not found")
                return result
            except NotFoundError:
                raise HTTPException(status_code=404, detail="Alias not found") from None
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to update alias: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.delete("/aliases/{alias_id}")
        async def delete_alias(
            alias_id: int, session: AsyncSession = Depends(self._get_db_session)
        ):
            """Delete an alias."""
            try:
                success = await self.memory_service.delete_alias_async(
                    session, alias_id
                )
                if not success:
                    raise HTTPException(status_code=404, detail="Alias not found")
                return {"message": "Alias deleted successfully"}
            except NotFoundError:
                raise HTTPException(status_code=404, detail="Alias not found") from None
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to delete alias: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.put("/notes/{note_id}", response_model=Note)
        async def update_note(
            note_id: int,
            request: CreateNoteRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Update an existing note."""
            try:
                result = await self.memory_service.update_note_async(
                    session,
                    note_id,
                    title=request.title,
                    content=request.content,
                    category=request.category,
                    tags=request.tags,
                )
                if not result:
                    raise HTTPException(status_code=404, detail="Note not found")
                return result
            except NotFoundError:
                raise HTTPException(status_code=404, detail="Note not found") from None
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to update note: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.delete("/notes/{note_id}")
        async def delete_note(
            note_id: int, session: AsyncSession = Depends(self._get_db_session)
        ):
            """Delete a note."""
            try:
                success = await self.memory_service.delete_note_async(session, note_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Note not found")
                return {"message": "Note deleted successfully"}
            except NotFoundError:
                raise HTTPException(status_code=404, detail="Note not found") from None
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to delete note: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.put("/observations/{observation_id}", response_model=Observation)
        async def update_observation(
            observation_id: int,
            request: CreateObservationRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Update an existing observation."""
            try:
                result = await self.memory_service.update_observation_async(
                    session,
                    observation_id,
                    content=request.content,
                    entity_type=request.entity_type,
                    entity_id=request.entity_id,
                    context=request.context,
                    tags=request.tags,
                )
                if not result:
                    raise HTTPException(status_code=404, detail="Observation not found")
                return result
            except NotFoundError:
                raise HTTPException(
                    status_code=404, detail="Observation not found"
                ) from None
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to update observation: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.delete("/observations/{observation_id}")
        async def delete_observation(
            observation_id: int, session: AsyncSession = Depends(self._get_db_session)
        ):
            """Delete an observation."""
            try:
                success = await self.memory_service.delete_observation_async(
                    session, observation_id
                )
                if not success:
                    raise HTTPException(status_code=404, detail="Observation not found")
                return {"message": "Observation deleted successfully"}
            except NotFoundError:
                raise HTTPException(
                    status_code=404, detail="Observation not found"
                ) from None
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to delete observation: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.put("/hints/{hint_id}", response_model=Hint)
        async def update_hint(
            hint_id: int,
            request: CreateHintRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Update an existing hint."""
            try:
                result = await self.memory_service.update_hint_async(
                    session,
                    hint_id,
                    content=request.content,
                    category=request.category,
                    priority=request.priority,
                    workflow_context=request.workflow_context,
                    tags=request.tags,
                )
                if not result:
                    raise HTTPException(status_code=404, detail="Hint not found")
                return result
            except NotFoundError:
                raise HTTPException(status_code=404, detail="Hint not found") from None
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to update hint: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.delete("/hints/{hint_id}")
        async def delete_hint(
            hint_id: int, session: AsyncSession = Depends(self._get_db_session)
        ):
            """Delete a hint."""
            try:
                success = await self.memory_service.delete_hint_async(session, hint_id)
                if not success:
                    raise HTTPException(status_code=404, detail="Hint not found")
                return {"message": "Hint deleted successfully"}
            except NotFoundError:
                raise HTTPException(status_code=404, detail="Hint not found") from None
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to delete hint: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        # Bulk operations endpoints
        @self.app.post("/memories/search/stream")
        async def search_all_memories_stream(
            request: StreamingSearchRequest,
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Search across all memory types with streaming response."""
            try:
                return StreamingResponse(
                    self._stream_unified_search_results(session, request),
                    media_type="application/x-ndjson",
                )
            except Exception as e:
                logger.error(f"Failed to stream unified search results: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.get("/memories/stats")
        async def get_memory_stats(
            user_id: Optional[str] = Query(None),
            session: AsyncSession = Depends(self._get_db_session),
        ):
            """Get statistics about stored memories."""
            try:
                stats = await self._get_memory_stats(session, user_id)
                return stats
            except Exception as e:
                logger.error(f"Failed to get memory stats: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        # User management endpoints
        @self.app.get("/users")
        async def get_users(session: AsyncSession = Depends(self._get_db_session)):
            """Get all users."""
            try:
                users = await self.memory_service.get_users_async(session)
                return {"users": users}
            except MemoryServiceError as e:
                raise HTTPException(status_code=400, detail=str(e)) from None
            except Exception as e:
                logger.error(f"Failed to get users: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

        @self.app.get("/users/{user_id}/stats")
        async def get_user_stats(
            user_id: str, session: AsyncSession = Depends(self._get_db_session)
        ):
            """Get statistics for a specific user."""
            try:
                stats = await self._get_memory_stats(session, user_id)
                return {"user_id": user_id, "stats": stats}
            except Exception as e:
                logger.error(f"Failed to get user stats: {e}")
                raise HTTPException(
                    status_code=500, detail="Internal server error"
                ) from None

    async def _get_db_session(self):
        """Get database session dependency."""
        async with self.db_manager.get_async_session() as session:
            yield session

    async def _stream_search_results(
        self, session: AsyncSession, request: StreamingSearchRequest
    ) -> AsyncGenerator[str, None]:
        """Stream search results as NDJSON."""
        try:
            # Send initial metadata
            metadata = {
                "type": "metadata",
                "query": request.query,
                "timestamp": datetime.now(UTC).isoformat(),
                "semantic": request.semantic,
            }
            yield json.dumps(metadata) + "\n"

            if request.semantic:
                results = await self.search_service.semantic_search_async(
                    session,
                    request.query,
                    user_id=request.user_id,
                    memory_types=request.memory_types,
                )
            else:
                results = await self.search_service.exact_search_async(
                    session,
                    request.query,
                    user_id=request.user_id,
                    memory_types=request.memory_types,
                )

            # Stream results one by one
            for result in results:
                result_data = {"type": "result", "data": result}
                yield json.dumps(result_data, default=str) + "\n"
                # Small delay to simulate streaming
                await asyncio.sleep(0.01)

            # Send completion marker
            completion = {
                "type": "complete",
                "total_results": len(results),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            yield json.dumps(completion) + "\n"

        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            yield json.dumps(error_data) + "\n"

    async def start_server(self, host: str = None, port: int = None) -> None:
        """Start the HTTP server."""
        import uvicorn

        host = host or self.config.server.host
        port = port or self.config.server.http_port

        logger.info(f"Starting HTTP server on {host}:{port}")

        # Initialize database
        await self.db_manager.initialize_database_async()

        # Configure uvicorn
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level=self.config.logging.level.lower(),
            access_log=True,
        )

        server = uvicorn.Server(config)
        await server.serve()

    async def _stream_unified_search_results(
        self, session: AsyncSession, request: StreamingSearchRequest
    ) -> AsyncGenerator[str, None]:
        """Stream unified search results across all memory types as NDJSON."""
        try:
            # Send initial metadata
            metadata = {
                "type": "metadata",
                "query": request.query,
                "timestamp": datetime.now(UTC).isoformat(),
                "semantic": request.semantic,
                "memory_types": request.memory_types
                or ["alias", "note", "observation", "hint"],
            }
            yield json.dumps(metadata) + "\n"

            # Search each memory type and stream results
            memory_types = request.memory_types or [
                "alias",
                "note",
                "observation",
                "hint",
            ]
            total_results = 0

            for memory_type in memory_types:
                try:
                    if memory_type == "alias":
                        results = await self.memory_service.get_aliases_async(
                            session, user_id=request.user_id, query=request.query
                        )
                    elif memory_type == "note":
                        results = await self.memory_service.get_notes_async(
                            session, user_id=request.user_id, query=request.query
                        )
                    elif memory_type == "observation":
                        results = await self.memory_service.get_observations_async(
                            session, user_id=request.user_id, query=request.query
                        )
                    elif memory_type == "hint":
                        results = await self.memory_service.get_hints_async(
                            session, user_id=request.user_id, query=request.query
                        )
                    else:
                        continue

                    # Stream results for this memory type
                    for result in results:
                        result_data = {
                            "type": "result",
                            "memory_type": memory_type,
                            "data": result.model_dump()
                            if hasattr(result, "model_dump")
                            else result,
                        }
                        yield json.dumps(result_data, default=str) + "\n"
                        total_results += 1
                        # Small delay to simulate streaming
                        await asyncio.sleep(0.01)

                except Exception as e:
                    error_data = {
                        "type": "error",
                        "memory_type": memory_type,
                        "error": str(e),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    yield json.dumps(error_data) + "\n"

            # Send completion marker
            completion = {
                "type": "complete",
                "total_results": total_results,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            yield json.dumps(completion) + "\n"

        except Exception as e:
            error_data = {
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            yield json.dumps(error_data) + "\n"

    async def _get_memory_stats(
        self, session: AsyncSession, user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        try:
            stats = {
                "aliases": 0,
                "notes": 0,
                "observations": 0,
                "hints": 0,
                "total": 0,
                "user_id": user_id,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            # Count aliases
            aliases = await self.memory_service.get_aliases_async(
                session, user_id=user_id, limit=10000
            )
            stats["aliases"] = len(aliases)

            # Count notes
            notes = await self.memory_service.get_notes_async(
                session, user_id=user_id, limit=10000
            )
            stats["notes"] = len(notes)

            # Count observations
            observations = await self.memory_service.get_observations_async(
                session, user_id=user_id, limit=10000
            )
            stats["observations"] = len(observations)

            # Count hints
            hints = await self.memory_service.get_hints_async(
                session, user_id=user_id, limit=10000
            )
            stats["hints"] = len(hints)

            # Calculate total
            stats["total"] = (
                stats["aliases"]
                + stats["notes"]
                + stats["observations"]
                + stats["hints"]
            )

            return stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}


def create_http_server(config: Optional[Config] = None) -> MemoryHTTPServer:
    """Create and configure the HTTP server."""
    return MemoryHTTPServer(config)
