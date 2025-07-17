"""Database layer for the Memory MCP Server."""

from .connection import (
    DatabaseManager,
    get_async_db_session,
    get_database_manager,
    get_db_session,
    initialize_database,
    initialize_database_async,
    set_database_manager,
)
from .models import (
    AliasDB,
    Base,
    EmbeddingDB,
    HintDB,
    JSONType,
    NoteDB,
    ObservationDB,
    User,
)
from .repositories import (
    AliasRepository,
    BaseRepository,
    EmbeddingRepository,
    HintRepository,
    NoteRepository,
    ObservationRepository,
    UserRepository,
)

__all__ = [
    # Models
    "Base",
    "JSONType",
    "User",
    "AliasDB",
    "NoteDB",
    "ObservationDB",
    "HintDB",
    "EmbeddingDB",
    # Connection management
    "DatabaseManager",
    "get_database_manager",
    "set_database_manager",
    "get_db_session",
    "get_async_db_session",
    "initialize_database",
    "initialize_database_async",
    # Repositories
    "BaseRepository",
    "UserRepository",
    "AliasRepository",
    "NoteRepository",
    "ObservationRepository",
    "HintRepository",
    "EmbeddingRepository",
]
