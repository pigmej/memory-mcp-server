"""Database connection and session management utilities."""

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncGenerator, Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from ..config import Config, get_config
from .models import Base

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize database manager with configuration."""
        self.config = config or get_config()
        self._engine: Optional[Engine] = None
        self._async_engine = None
        self._session_factory: Optional[sessionmaker] = None
        self._async_session_factory = None
        self._initialized = False

    @property
    def engine(self) -> Engine:
        """Get the synchronous database engine."""
        if self._engine is None:
            self._create_engine()
        return self._engine

    @property
    def async_engine(self):
        """Get the asynchronous database engine."""
        if self._async_engine is None:
            self._create_async_engine()
        return self._async_engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get the synchronous session factory."""
        if self._session_factory is None:
            self._create_session_factory()
        return self._session_factory

    @property
    def async_session_factory(self):
        """Get the asynchronous session factory."""
        if self._async_session_factory is None:
            self._create_async_session_factory()
        return self._async_session_factory

    def _create_engine(self) -> None:
        """Create the synchronous database engine."""
        logger.info(f"Creating database engine with URL: {self.config.database.url}")

        self._engine = create_engine(
            self.config.database.url,
            echo=self.config.database.echo,
            pool_size=self.config.database.pool_size,
            max_overflow=self.config.database.max_overflow,
        )

        # Enable foreign key constraints for SQLite
        if self.config.database.url.startswith("sqlite"):

            @event.listens_for(self._engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    def _create_async_engine(self) -> None:
        """Create the asynchronous database engine."""
        # Convert sync URL to async URL for SQLite
        async_url = self.config.database.url
        if async_url.startswith("sqlite:///"):
            async_url = async_url.replace("sqlite:///", "sqlite+aiosqlite:///")

        logger.info(f"Creating async database engine with URL: {async_url}")

        self._async_engine = create_async_engine(
            async_url,
            echo=self.config.database.echo,
        )

    def _create_session_factory(self) -> None:
        """Create the synchronous session factory."""
        self._session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
        )

    def _create_async_session_factory(self) -> None:
        """Create the asynchronous session factory."""
        self._async_session_factory = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            autocommit=False,
            autoflush=False,
        )

    def initialize_database(self) -> None:
        """Initialize the database by creating all tables."""
        if self._initialized:
            logger.info("Database already initialized")
            return

        logger.info("Initializing database tables")
        try:
            Base.metadata.create_all(bind=self.engine)
            self._initialized = True
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def initialize_database_async(self) -> None:
        """Initialize the database asynchronously by creating all tables."""
        if self._initialized:
            logger.info("Database already initialized")
            return

        logger.info("Initializing database tables (async)")
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self._initialized = True
            logger.info("Database tables created successfully (async)")
        except Exception as e:
            logger.error(f"Failed to initialize database (async): {e}")
            raise

    def drop_all_tables(self) -> None:
        """Drop all database tables. Use with caution!"""
        logger.warning("Dropping all database tables")
        try:
            Base.metadata.drop_all(bind=self.engine)
            self._initialized = False
            logger.info("All database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise

    async def drop_all_tables_async(self) -> None:
        """Drop all database tables asynchronously. Use with caution!"""
        logger.warning("Dropping all database tables (async)")
        try:
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            self._initialized = False
            logger.info("All database tables dropped successfully (async)")
        except Exception as e:
            logger.error(f"Failed to drop database tables (async): {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a synchronous database session with automatic cleanup."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an asynchronous database session with automatic cleanup."""
        session = self.async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            logger.info("Synchronous database engine disposed")

        if self._async_engine:
            # Note: async engine disposal should be done in async context
            logger.info("Async database engine marked for disposal")

    async def close_async(self) -> None:
        """Close database connections asynchronously."""
        if self._async_engine:
            await self._async_engine.dispose()
            logger.info("Async database engine disposed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(config: Optional[Config] = None) -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(config)
    return _db_manager


def set_database_manager(manager: DatabaseManager) -> None:
    """Set the global database manager instance."""
    global _db_manager
    _db_manager = manager


# Convenience functions for session management
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session using the global database manager."""
    with get_database_manager().get_session() as session:
        yield session


@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session using the global database manager."""
    async with get_database_manager().get_async_session() as session:
        yield session


def initialize_database(config: Optional[Config] = None) -> None:
    """Initialize the database using the global database manager."""
    manager = get_database_manager(config)
    manager.initialize_database()


async def initialize_database_async(config: Optional[Config] = None) -> None:
    """Initialize the database asynchronously using the global database manager."""
    manager = get_database_manager(config)
    await manager.initialize_database_async()
