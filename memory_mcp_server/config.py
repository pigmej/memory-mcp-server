"""Configuration management for the Memory MCP Server."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""

    url: str = Field(default="sqlite:///memory_mcp.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQLAlchemy query logging")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum connection overflow")

    model_config = {"env_prefix": "DB_"}


class ServerConfig(BaseSettings):
    """Server configuration settings."""

    host: str = Field(default="localhost", description="Server host")
    http_port: int = Field(default=8000, description="HTTP server port")
    web_port: int = Field(default=8080, description="Web UI server port")
    stdio_enabled: bool = Field(default=True, description="Enable STDIO protocol")
    http_enabled: bool = Field(default=True, description="Enable HTTP protocol")
    web_enabled: bool = Field(default=True, description="Enable Web UI")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")

    model_config = {"env_prefix": "SERVER_"}


class EmbeddingConfig(BaseSettings):
    """Embedding and RAG configuration settings."""

    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model name"
    )
    similarity_threshold: float = Field(
        default=0.3, description="Minimum similarity threshold for semantic search"
    )
    max_results: int = Field(default=10, description="Maximum search results")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings")

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v

    model_config = {"env_prefix": "EMBEDDING_"}


class LoggingConfig(BaseSettings):
    """Logging configuration settings."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size: int = Field(
        default=10485760, description="Max log file size in bytes"
    )  # 10MB
    backup_count: int = Field(default=5, description="Number of backup log files")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()

    model_config = {"env_prefix": "LOG_"}


class Config(BaseSettings):
    """Main configuration class that combines all configuration sections."""

    # Environment and paths
    environment: str = Field(default="development", description="Environment name")
    data_dir: Path = Field(
        default=Path.home() / ".memory_mcp", description="Data directory"
    )
    config_file: Optional[Path] = Field(
        default=None, description="Configuration file path"
    )

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Feature flags (currently not implemented)
    # enable_web_ui: bool = Field(default=True, description="Enable web UI")
    # enable_rag: bool = Field(default=True, description="Enable RAG capabilities")
    # enable_user_separation: bool = Field(default=True, description="Enable user data separation")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_data_dir()
        self._update_database_path()
        self._setup_logging()

    def _ensure_data_dir(self) -> None:
        """Ensure the data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _update_database_path(self) -> None:
        """Update database path to use data directory if using SQLite."""
        if self.database.url.startswith("sqlite:///") and not os.path.isabs(
            self.database.url[10:]
        ):
            db_name = self.database.url.split("///")[-1]
            self.database.url = f"sqlite:///{self.data_dir / db_name}"

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        try:
            from .utils.logging import setup_logging

            # Set log file path relative to data directory if not absolute
            if self.logging.file_path and not os.path.isabs(self.logging.file_path):
                self.logging.file_path = str(self.data_dir / self.logging.file_path)

            setup_logging(self.logging)
        except ImportError:
            # Fallback to basic logging if utils not available
            import logging

            logging.basicConfig(
                level=getattr(logging, self.logging.level), format=self.logging.format
            )

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from a YAML or JSON file."""
        import json

        import yaml

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported configuration file format: {config_path.suffix}"
                )

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to a YAML file."""
        import yaml

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file or environment."""
    if config_path and config_path.exists():
        config = Config.from_file(config_path)
    else:
        config = Config()

    set_config(config)
    return config
