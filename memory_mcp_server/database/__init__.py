"""Database layer for the Memory MCP Server."""

from .models import (
    Base,
    JSONType,
    User,
    AliasDB,
    NoteDB,
    ObservationDB,
    HintDB,
    EmbeddingDB,
)
from .connection import (
    DatabaseManager,
    get_database_manager,
    set_database_manager,
    get_db_session,
    get_async_db_session,
    initialize_database,
    initialize_database_async,
)
from .repositories import (
    BaseRepository,
    UserRepository,
    AliasRepository,
    NoteRepository,
    ObservationRepository,
    HintRepository,
    EmbeddingRepository,
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
