"""MCP protocol implementation using FastMCP."""

import logging
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from ..database.connection import get_database_manager
from ..models.alias import Alias
from ..models.hint import Hint
from ..models.note import Note
from ..models.observation import Observation
from ..services.memory_service import MemoryService, MemoryServiceError
from ..services.search_service import SearchService, SearchServiceError


# Import utilities locally to avoid circular imports
def _get_utility_classes():
    """Lazy import of utility classes to avoid circular dependencies."""
    try:
        from ..utils.errors import BaseMemoryError, ProtocolError, create_error_context
        from ..utils.logging import get_logger

        return BaseMemoryError, ProtocolError, create_error_context, get_logger
    except ImportError:
        # Fallback to basic classes if utils not available
        class BaseMemoryError(Exception):
            pass

        class ProtocolError(BaseMemoryError):
            pass

        def create_error_context(**kwargs):
            return None

        def get_logger(name):
            return logging.getLogger(name)

        return BaseMemoryError, ProtocolError, create_error_context, get_logger


BaseMemoryError, ProtocolError, create_error_context, get_logger = (
    _get_utility_classes()
)

logger = get_logger(__name__)


class MemoryMCPServer:
    """MCP server implementation for memory management."""

    def __init__(self, config=None):
        """Initialize the MCP server with services."""
        from ..config import get_config

        if config is None:
            config = get_config()

        self.config = config
        self.memory_service = MemoryService()
        self.search_service = SearchService(config)
        self.db_manager = get_database_manager()
        self.mcp = FastMCP("Memory MCP Server")
        self._register_tools()
        self._register_resources()

    def get_asgi_app(self, path="/mcp"):
        """Get the ASGI app for this MCP server."""
        # FastMCP provides an HTTP app via the http_app property
        return self.mcp.http_app(path)

    def _register_tools(self):
        """Register all MCP tools."""

        @self.mcp.tool()
        async def create_alias(
            source: str,
            target: str,
            user_id: Optional[str] = None,
            bidirectional: bool = True,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Create a new alias mapping.

            Args:
                source: Source term or phrase
                target: Target term or phrase
                user_id: User ID for data separation
                bidirectional: Whether the alias works in both directions
                tags: Tags for categorization
            """
            try:
                if tags is None:
                    tags = []

                async with self.db_manager.get_async_session() as session:
                    alias = Alias(
                        source=source,
                        target=target,
                        user_id=user_id,
                        bidirectional=bidirectional,
                        tags=tags,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )

                    result = await self.memory_service.create_alias_async(
                        session, alias
                    )
                    return {
                        "success": True,
                        "alias": result.model_dump(),
                        "message": f"Created alias: {result.source} -> {result.target}",
                    }
            except MemoryServiceError as e:
                logger.error(f"Failed to create alias: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to create alias",
                }
            except Exception as e:
                logger.error(f"Unexpected error creating alias: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def create_note(
            title: str,
            content: str,
            user_id: Optional[str] = None,
            category: Optional[str] = None,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Create a new note.

            Args:
                title: Title of the note
                content: Content of the note
                user_id: User ID for data separation
                category: Category for organizing notes
                tags: Tags for categorization
            """
            try:
                if tags is None:
                    tags = []

                async with self.db_manager.get_async_session() as session:
                    note = Note(
                        title=title,
                        content=content,
                        user_id=user_id,
                        category=category,
                        tags=tags,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )

                    result = await self.memory_service.create_note_async(session, note)
                    return {
                        "success": True,
                        "note": result.model_dump(),
                        "message": f"Created note: {result.title}",
                    }
            except MemoryServiceError as e:
                logger.error(f"Failed to create note: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to create note",
                }
            except Exception as e:
                logger.error(f"Unexpected error creating note: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def create_observation(
            content: str,
            entity_type: str,
            entity_id: str,
            user_id: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Create a new observation.

            Args:
                content: Observation content
                entity_type: Type of entity being observed
                entity_id: ID of the entity being observed
                user_id: User ID for data separation
                context: Additional context
                tags: Tags for categorization
            """
            try:
                if tags is None:
                    tags = []

                async with self.db_manager.get_async_session() as session:
                    observation = Observation(
                        content=content,
                        entity_type=entity_type,
                        entity_id=entity_id,
                        user_id=user_id,
                        context=context,
                        tags=tags,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )

                    result = await self.memory_service.create_observation_async(
                        session, observation
                    )
                    return {
                        "success": True,
                        "observation": result.model_dump(),
                        "message": f"Created observation for {result.entity_type}:{result.entity_id}",
                    }
            except MemoryServiceError as e:
                logger.error(f"Failed to create observation: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to create observation",
                }
            except Exception as e:
                logger.error(f"Unexpected error creating observation: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def create_hint(
            content: str,
            category: str,
            user_id: Optional[str] = None,
            priority: int = 1,
            workflow_context: Optional[str] = None,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Create a new hint for LLM interactions.

            Args:
                content: Hint content
                category: Category of the hint
                user_id: User ID for data separation
                priority: Priority level (1-10)
                workflow_context: Workflow context for the hint
                tags: Tags for categorization
            """
            try:
                if tags is None:
                    tags = []

                async with self.db_manager.get_async_session() as session:
                    hint = Hint(
                        content=content,
                        category=category,
                        user_id=user_id,
                        priority=priority,
                        workflow_context=workflow_context,
                        tags=tags,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )

                    result = await self.memory_service.create_hint_async(session, hint)
                    return {
                        "success": True,
                        "hint": result.model_dump(),
                        "message": f"Created hint in category: {result.category}",
                    }
            except MemoryServiceError as e:
                logger.error(f"Failed to create hint: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to create hint",
                }
            except Exception as e:
                logger.error(f"Unexpected error creating hint: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def search_memories(
            query: str,
            user_id: Optional[str] = None,
            memory_types: Optional[List[str]] = None,
            limit: int = 10,
            semantic: bool = True,
            similarity_threshold: float = 0.3,
        ) -> Dict[str, Any]:
            """Search across all memory types using semantic or exact search.

            Args:
                query: Search query
                user_id: User ID for filtering results
                memory_types: Types of memories to search (alias, note, observation, hint)
                limit: Maximum number of results
                semantic: Use semantic search
                similarity_threshold: Minimum similarity threshold
            """
            try:
                async with self.db_manager.get_async_session() as session:
                    if semantic:
                        results = await self.search_service.semantic_search_async(
                            session=session,
                            query=query,
                            memory_types=memory_types,
                            user_id=user_id,
                            limit=limit,
                            similarity_threshold=similarity_threshold,
                        )
                    else:
                        results = await self.search_service.exact_search_async(
                            session=session,
                            query=query,
                            memory_types=memory_types,
                            user_id=user_id,
                            limit=limit,
                        )

                    # Format results as plain text for better display
                    if not results:
                        return f"No results found for query: '{query}'"

                    result_text = (
                        f"Search Results for '{query}' ({len(results)} found):\n"
                    )
                    result_text += "=" * 60 + "\n\n"

                    for i, result in enumerate(results, 1):
                        memory_type = result.get("memory_type", "unknown")
                        content = result.get("content", {})
                        metadata = result.get("metadata", {})
                        similarity = result.get("similarity", 0.0)

                        result_text += f"{i}. {memory_type.upper()} (ID: {metadata.get('id', 'N/A')})\n"

                        if semantic:
                            result_text += f"   Similarity: {similarity:.3f}\n"

                        # Add type-specific content
                        if memory_type == "alias":
                            arrow = "↔" if content.get("bidirectional", True) else "→"
                            result_text += f"   {content.get('source', '')} {arrow} {content.get('target', '')}\n"
                        elif memory_type == "note":
                            result_text += (
                                f"   Title: {content.get('title', 'Untitled')}\n"
                            )
                            if content.get("category"):
                                result_text += (
                                    f"   Category: {content.get('category')}\n"
                                )
                            preview = content.get("content", "")[:150]
                            if len(content.get("content", "")) > 150:
                                preview += "..."
                            result_text += f"   Content: {preview}\n"
                        elif memory_type == "observation":
                            result_text += f"   Entity: {content.get('entity_type', '')}:{content.get('entity_id', '')}\n"
                            preview = content.get("content", "")[:150]
                            if len(content.get("content", "")) > 150:
                                preview += "..."
                            result_text += f"   Content: {preview}\n"
                        elif memory_type == "hint":
                            result_text += (
                                f"   Category: {content.get('category', 'General')}\n"
                            )
                            result_text += (
                                f"   Priority: {content.get('priority', 1)}/5\n"
                            )
                            if content.get("workflow_context"):
                                result_text += (
                                    f"   Workflow: {content.get('workflow_context')}\n"
                                )
                            preview = content.get("content", "")[:150]
                            if len(content.get("content", "")) > 150:
                                preview += "..."
                            result_text += f"   Content: {preview}\n"

                        # Add tags if present
                        tags = metadata.get("tags", [])
                        if tags:
                            result_text += f"   Tags: {', '.join(tags)}\n"

                        # Add creation date
                        if metadata.get("created_at"):
                            result_text += f"   Created: {metadata.get('created_at')}\n"

                        result_text += "\n"

                    return result_text
            except SearchServiceError as e:
                logger.error(f"Failed to search memories: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to search memories",
                }
            except Exception as e:
                logger.error(f"Unexpected error searching memories: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def query_alias(
            query: str, user_id: Optional[str] = None, exact_match: bool = True
        ) -> Dict[str, Any]:
            """Query aliases bidirectionally to find mappings.

            Args:
                query: Term to look up
                user_id: User ID for filtering
                exact_match: Use exact matching
            """
            try:
                async with self.db_manager.get_async_session() as session:
                    results = await self.memory_service.query_alias_async(
                        session=session,
                        query=query,
                        user_id=user_id,
                        exact_match=exact_match,
                    )

                    return {
                        "success": True,
                        "query": query,
                        "mappings": results,
                        "count": len(results),
                        "exact_match": exact_match,
                    }
            except MemoryServiceError as e:
                logger.error(f"Failed to query alias: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to query alias",
                }
            except Exception as e:
                logger.error(f"Unexpected error querying alias: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def get_memories(
            memory_type: str,
            user_id: Optional[str] = None,
            limit: int = 10,
            skip: int = 0,
            category: Optional[str] = None,
            query: Optional[str] = None,
            entity_id: Optional[str] = None,
            entity_type: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Get memories of a specific type with optional filtering.

            Args:
                memory_type: Type of memory (alias, note, observation, hint)
                user_id: User ID for filtering
                limit: Maximum number of results
                skip: Number of results to skip
                category: Category filter (for notes and hints)
                query: Query filter (for notes and hints)
                entity_id: Entity ID filter (for observations)
                entity_type: Entity type filter (for observations)
            """
            try:
                async with self.db_manager.get_async_session() as session:
                    if memory_type == "alias":
                        results = await self.memory_service.get_aliases_async(
                            session=session,
                            user_id=user_id,
                            skip=skip,
                            limit=limit,
                        )
                    elif memory_type == "note":
                        results = await self.memory_service.get_notes_async(
                            session=session,
                            user_id=user_id,
                            category=category,
                            query=query,
                            skip=skip,
                            limit=limit,
                        )
                    elif memory_type == "observation":
                        results = await self.memory_service.get_observations_async(
                            session=session,
                            user_id=user_id,
                            entity_id=entity_id,
                            entity_type=entity_type,
                            skip=skip,
                            limit=limit,
                        )
                    elif memory_type == "hint":
                        results = await self.memory_service.get_hints_async(
                            session=session,
                            user_id=user_id,
                            category=category,
                            query=query,
                            skip=skip,
                            limit=limit,
                        )
                    else:
                        return {
                            "success": False,
                            "error": f"Unknown memory type: {memory_type}",
                            "message": "Invalid memory type",
                        }

                    return {
                        "success": True,
                        "memory_type": memory_type,
                        "memories": [memory.model_dump() for memory in results],
                        "count": len(results),
                        "skip": skip,
                        "limit": limit,
                    }
            except MemoryServiceError as e:
                logger.error(f"Failed to get memories: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to get memories",
                }
            except Exception as e:
                logger.error(f"Unexpected error getting memories: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def get_users() -> Dict[str, Any]:
            """Get all user IDs in the system."""
            try:
                async with self.db_manager.get_async_session() as session:
                    users = await self.memory_service.get_users_async(session)

                    return {"success": True, "users": users, "count": len(users)}
            except MemoryServiceError as e:
                logger.error(f"Failed to get users: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to get users",
                }
            except Exception as e:
                logger.error(f"Unexpected error getting users: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def update_memory(
            memory_type: str,
            memory_id: int,
            title: Optional[str] = None,
            content: Optional[str] = None,
            source: Optional[str] = None,
            target: Optional[str] = None,
            entity_type: Optional[str] = None,
            entity_id: Optional[str] = None,
            category: Optional[str] = None,
            priority: Optional[int] = None,
            workflow_context: Optional[str] = None,
            bidirectional: Optional[bool] = None,
            context: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            """Update a memory item.

            Args:
                memory_type: Type of memory (alias, note, observation, hint)
                memory_id: ID of the memory to update
                title: New title (for notes)
                content: New content (for notes, observations, hints)
                source: New source (for aliases)
                target: New target (for aliases)
                entity_type: New entity type (for observations)
                entity_id: New entity ID (for observations)
                category: New category (for notes, hints)
                priority: New priority (for hints)
                workflow_context: New workflow context (for hints)
                bidirectional: New bidirectional setting (for aliases)
                context: New context (for observations)
                tags: New tags (for all types)
            """
            try:
                # Build updates dict from non-None parameters
                updates = {}
                if title is not None:
                    updates["title"] = title
                if content is not None:
                    updates["content"] = content
                if source is not None:
                    updates["source"] = source
                if target is not None:
                    updates["target"] = target
                if entity_type is not None:
                    updates["entity_type"] = entity_type
                if entity_id is not None:
                    updates["entity_id"] = entity_id
                if category is not None:
                    updates["category"] = category
                if priority is not None:
                    updates["priority"] = priority
                if workflow_context is not None:
                    updates["workflow_context"] = workflow_context
                if bidirectional is not None:
                    updates["bidirectional"] = bidirectional
                if context is not None:
                    updates["context"] = context
                if tags is not None:
                    updates["tags"] = tags

                async with self.db_manager.get_async_session() as session:
                    if memory_type == "alias":
                        result = await self.memory_service.update_alias_async(
                            session, memory_id, **updates
                        )
                    elif memory_type == "note":
                        result = await self.memory_service.update_note_async(
                            session, memory_id, **updates
                        )
                    elif memory_type == "observation":
                        result = await self.memory_service.update_observation_async(
                            session, memory_id, **updates
                        )
                    elif memory_type == "hint":
                        result = await self.memory_service.update_hint_async(
                            session, memory_id, **updates
                        )
                    else:
                        return {
                            "success": False,
                            "error": f"Unknown memory type: {memory_type}",
                            "message": "Invalid memory type",
                        }

                    if result:
                        return {
                            "success": True,
                            "memory": result.model_dump(),
                            "message": f"Updated {memory_type} {memory_id}",
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"{memory_type.title()} not found",
                            "message": f"{memory_type.title()} with ID {memory_id} not found",
                        }
            except MemoryServiceError as e:
                logger.error(f"Failed to update memory: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to update memory",
                }
            except Exception as e:
                logger.error(f"Unexpected error updating memory: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

        @self.mcp.tool()
        async def delete_memory(memory_type: str, memory_id: int) -> Dict[str, Any]:
            """Delete a memory item.

            Args:
                memory_type: Type of memory (alias, note, observation, hint)
                memory_id: ID of the memory to delete
            """
            try:
                async with self.db_manager.get_async_session() as session:
                    if memory_type == "alias":
                        success = await self.memory_service.delete_alias_async(
                            session, memory_id
                        )
                    elif memory_type == "note":
                        success = await self.memory_service.delete_note_async(
                            session, memory_id
                        )
                    elif memory_type == "observation":
                        success = await self.memory_service.delete_observation_async(
                            session, memory_id
                        )
                    elif memory_type == "hint":
                        success = await self.memory_service.delete_hint_async(
                            session, memory_id
                        )
                    else:
                        return {
                            "success": False,
                            "error": f"Unknown memory type: {memory_type}",
                            "message": "Invalid memory type",
                        }

                    if success:
                        return {
                            "success": True,
                            "message": f"Deleted {memory_type} {memory_id}",
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"{memory_type.title()} not found",
                            "message": f"{memory_type.title()} with ID {memory_id} not found",
                        }
            except MemoryServiceError as e:
                logger.error(f"Failed to delete memory: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "message": "Failed to delete memory",
                }
            except Exception as e:
                logger.error(f"Unexpected error deleting memory: {e}")
                return {
                    "success": False,
                    "error": "Internal server error",
                    "message": "An unexpected error occurred",
                }

    def _register_resources(self):
        """Register MCP resources for exposing stored data."""

        @self.mcp.resource("memory://aliases/{user_id}")
        async def get_user_aliases(user_id: str) -> str:
            """Get all aliases for a specific user."""
            try:
                async with self.db_manager.get_async_session() as session:
                    aliases = await self.memory_service.get_aliases_async(
                        session, user_id=user_id, limit=1000
                    )

                    content = f"# Aliases for user: {user_id}\n\n"
                    for alias in aliases:
                        direction = "↔" if alias.bidirectional else "→"
                        content += f"- {alias.source} {direction} {alias.target}\n"
                        if alias.tags:
                            content += f"  Tags: {', '.join(alias.tags)}\n"

                    return content
            except Exception as e:
                logger.error(f"Failed to get user aliases resource: {e}")
                return f"Error loading aliases for user {user_id}: {str(e)}"

        @self.mcp.resource("memory://notes/{user_id}")
        async def get_user_notes(user_id: str) -> str:
            """Get all notes for a specific user."""
            try:
                async with self.db_manager.get_async_session() as session:
                    notes = await self.memory_service.get_notes_async(
                        session, user_id=user_id, limit=1000
                    )

                    content = f"# Notes for user: {user_id}\n\n"
                    for note in notes:
                        content += f"## {note.title}\n"
                        if note.category:
                            content += f"**Category:** {note.category}\n"
                        content += f"{note.content}\n"
                        if note.tags:
                            content += f"**Tags:** {', '.join(note.tags)}\n"
                        content += f"**Created:** {note.created_at}\n\n"

                    return content
            except Exception as e:
                logger.error(f"Failed to get user notes resource: {e}")
                return f"Error loading notes for user {user_id}: {str(e)}"

        @self.mcp.resource("memory://observations/{user_id}")
        async def get_user_observations(user_id: str) -> str:
            """Get all observations for a specific user."""
            try:
                async with self.db_manager.get_async_session() as session:
                    observations = await self.memory_service.get_observations_async(
                        session, user_id=user_id, limit=1000
                    )

                    content = f"# Observations for user: {user_id}\n\n"
                    for obs in observations:
                        content += f"## {obs.entity_type}:{obs.entity_id}\n"
                        content += f"{obs.content}\n"
                        if obs.context:
                            content += f"**Context:** {obs.context}\n"
                        if obs.tags:
                            content += f"**Tags:** {', '.join(obs.tags)}\n"
                        content += f"**Created:** {obs.created_at}\n\n"

                    return content
            except Exception as e:
                logger.error(f"Failed to get user observations resource: {e}")
                return f"Error loading observations for user {user_id}: {str(e)}"

        @self.mcp.resource("memory://hints/{user_id}")
        async def get_user_hints(user_id: str) -> str:
            """Get all hints for a specific user."""
            try:
                async with self.db_manager.get_async_session() as session:
                    hints = await self.memory_service.get_hints_async(
                        session, user_id=user_id, limit=1000
                    )

                    content = f"# Hints for user: {user_id}\n\n"

                    # Group by category
                    categories = {}
                    for hint in hints:
                        if hint.category not in categories:
                            categories[hint.category] = []
                        categories[hint.category].append(hint)

                    for category, category_hints in categories.items():
                        content += f"## {category}\n\n"
                        for hint in sorted(
                            category_hints, key=lambda h: h.priority, reverse=True
                        ):
                            content += (
                                f"- **Priority {hint.priority}:** {hint.content}\n"
                            )
                            if hint.workflow_context:
                                content += f"  *Context: {hint.workflow_context}*\n"
                            if hint.tags:
                                content += f"  *Tags: {', '.join(hint.tags)}*\n"
                        content += "\n"

                    return content
            except Exception as e:
                logger.error(f"Failed to get user hints resource: {e}")
                return f"Error loading hints for user {user_id}: {str(e)}"

        @self.mcp.resource("memory://all/{user_id}")
        async def get_all_user_memories(user_id: str) -> str:
            """Get all memories for a specific user."""
            try:
                content = f"# All Memories for user: {user_id}\n\n"

                # Get aliases
                aliases_content = await get_user_aliases(user_id)
                content += aliases_content + "\n"

                # Get notes
                notes_content = await get_user_notes(user_id)
                content += notes_content + "\n"

                # Get observations
                observations_content = await get_user_observations(user_id)
                content += observations_content + "\n"

                # Get hints
                hints_content = await get_user_hints(user_id)
                content += hints_content + "\n"

                return content
            except Exception as e:
                logger.error(f"Failed to get all user memories resource: {e}")
                return f"Error loading all memories for user {user_id}: {str(e)}"

    def get_mcp_server(self) -> FastMCP:
        """Get the configured FastMCP server instance."""
        return self.mcp


def create_mcp_server() -> MemoryMCPServer:
    """Create and configure the MCP server."""
    return MemoryMCPServer()
