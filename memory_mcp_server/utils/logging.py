"""Comprehensive logging configuration and utilities."""

import logging
import logging.handlers
import sys
from pathlib import Path

from ..config import LoggingConfig


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        """Format log record with colors."""
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )

        return super().format(record)


def setup_logging(config: LoggingConfig) -> None:
    """Setup comprehensive logging configuration.

    Args:
        config: Logging configuration object
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Set logging level
    root_logger.setLevel(getattr(logging, config.level))

    # Create formatters
    console_formatter = ColoredFormatter(config.format)
    file_formatter = logging.Formatter(config.format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, config.level))
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if configured)
    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Use rotating file handler to prevent huge log files
        file_handler = logging.handlers.RotatingFileHandler(
            filename=file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(getattr(logging, config.level))
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels for third-party libraries
    _configure_third_party_loggers(config.level)

    # Log the logging configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {config.level}")
    if config.file_path:
        logger.info(f"Log file: {config.file_path}")


def _configure_third_party_loggers(level: str) -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from third-party libraries
    third_party_loggers = {
        "sqlalchemy.engine": "WARNING",
        "sqlalchemy.pool": "WARNING",
        "sqlalchemy.dialects": "WARNING",
        "fastapi": "INFO",
        "uvicorn": "INFO",
        "uvicorn.access": "WARNING",
        "sentence_transformers": "WARNING",
        "transformers": "WARNING",
        "torch": "WARNING",
        "urllib3": "WARNING",
        "requests": "WARNING",
    }

    for logger_name, logger_level in third_party_loggers.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, logger_level))

    # If debug mode, show SQLAlchemy queries
    if level == "DEBUG":
        logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )


def log_function_call(func):
    """Decorator to log function calls with parameters and results."""
    import functools
    import inspect

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)

        # Log function entry
        if logger.isEnabledFor(logging.DEBUG):
            # Get function signature for better logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Filter out sensitive parameters
            safe_args = {}
            for name, value in bound_args.arguments.items():
                if any(
                    sensitive in name.lower()
                    for sensitive in ["password", "token", "key", "secret"]
                ):
                    safe_args[name] = "[REDACTED]"
                else:
                    safe_args[name] = str(value)[:100]  # Truncate long values

            logger.debug(f"Calling {func.__name__} with args: {safe_args}")

        try:
            result = func(*args, **kwargs)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise

    return wrapper


def log_async_function_call(func):
    """Decorator to log async function calls with parameters and results."""
    import functools
    import inspect

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)

        # Log function entry
        if logger.isEnabledFor(logging.DEBUG):
            # Get function signature for better logging
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Filter out sensitive parameters
            safe_args = {}
            for name, value in bound_args.arguments.items():
                if any(
                    sensitive in name.lower()
                    for sensitive in ["password", "token", "key", "secret"]
                ):
                    safe_args[name] = "[REDACTED]"
                else:
                    safe_args[name] = str(value)[:100]  # Truncate long values

            logger.debug(f"Calling {func.__name__} with args: {safe_args}")

        try:
            result = await func(*args, **kwargs)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise

    return wrapper
