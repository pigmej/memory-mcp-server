"""Comprehensive error handling utilities and custom exceptions."""

import logging
import traceback
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ErrorCategory(str, Enum):
    """Categories of errors for better classification."""

    VALIDATION = "validation"
    DATABASE = "database"
    SEARCH = "search"
    PROTOCOL = "protocol"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    INTERNAL = "internal"
    EXTERNAL = "external"


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorContext(BaseModel):
    """Context information for errors."""

    user_id: Optional[str] = None
    request_id: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    additional_data: Dict[str, Any] = Field(default_factory=dict)


class ErrorDetails(BaseModel):
    """Detailed error information."""

    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    details: Optional[str] = None
    context: Optional[ErrorContext] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    traceback: Optional[str] = None
    suggestions: List[str] = Field(default_factory=list)
    retry_after: Optional[int] = None  # Seconds to wait before retry


class BaseMemoryError(Exception):
    """Base exception for all memory server errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        details: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        suggestions: Optional[List[str]] = None,
        retry_after: Optional[int] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details
        self.context = context or ErrorContext()
        self.suggestions = suggestions or []
        self.retry_after = retry_after
        self.cause = cause
        self.timestamp = datetime.utcnow()

        # Generate unique error ID
        import uuid

        self.error_id = str(uuid.uuid4())[:8]

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "context": self.context.model_dump() if self.context else None,
            "timestamp": self.timestamp.isoformat(),
            "suggestions": self.suggestions,
            "retry_after": self.retry_after,
        }

    def to_error_details(self) -> ErrorDetails:
        """Convert to ErrorDetails model."""
        return ErrorDetails(
            error_id=self.error_id,
            category=self.category,
            severity=self.severity,
            message=self.message,
            details=self.details,
            context=self.context,
            timestamp=self.timestamp,
            traceback=traceback.format_exc() if self.cause else None,
            suggestions=self.suggestions,
            retry_after=self.retry_after,
        )


class ValidationError(BaseMemoryError):
    """Error for data validation failures."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs,
        )
        self.field = field


class DatabaseError(BaseMemoryError):
    """Error for database operations."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        self.operation = operation


class SearchError(BaseMemoryError):
    """Error for search operations."""

    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.SEARCH,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        self.query = query


class ProtocolError(BaseMemoryError):
    """Error for protocol-related issues."""

    def __init__(self, message: str, protocol: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.PROTOCOL,
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )
        self.protocol = protocol


class ConfigurationError(BaseMemoryError):
    """Error for configuration issues."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )
        self.config_key = config_key


class NotFoundError(BaseMemoryError):
    """Error for resource not found."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            **kwargs,
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


class RateLimitError(BaseMemoryError):
    """Error for rate limiting."""

    def __init__(self, message: str, retry_after: int = 60, **kwargs):
        super().__init__(
            message=message,
            category=ErrorCategory.EXTERNAL,
            severity=ErrorSeverity.MEDIUM,
            retry_after=retry_after,
            **kwargs,
        )


class ErrorHandler:
    """Centralized error handling and logging."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}  # Track error frequencies

    def handle_error(
        self,
        error: Union[Exception, BaseMemoryError],
        context: Optional[ErrorContext] = None,
        log_traceback: bool = True,
    ) -> ErrorDetails:
        """Handle and log an error.

        Args:
            error: The exception that occurred
            context: Additional context information
            log_traceback: Whether to log the full traceback

        Returns:
            ErrorDetails object with comprehensive error information
        """
        # Convert to BaseMemoryError if needed
        if isinstance(error, BaseMemoryError):
            memory_error = error
        else:
            memory_error = self._convert_to_memory_error(error)

        # Add context if provided
        if context:
            memory_error.context = context

        # Track error frequency
        error_key = f"{memory_error.category.value}:{memory_error.message[:50]}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Log the error
        self._log_error(memory_error, log_traceback)

        return memory_error.to_error_details()

    def _convert_to_memory_error(self, error: Exception) -> BaseMemoryError:
        """Convert a generic exception to BaseMemoryError."""
        error_type = type(error).__name__

        # Map common exceptions to appropriate categories
        if "validation" in error_type.lower() or "pydantic" in str(type(error)).lower():
            return ValidationError(
                message=str(error),
                cause=error,
                suggestions=["Check input data format and required fields"],
            )
        elif "database" in error_type.lower() or "sql" in error_type.lower():
            return DatabaseError(
                message=str(error),
                cause=error,
                suggestions=["Check database connection and schema"],
            )
        elif "connection" in error_type.lower() or "timeout" in error_type.lower():
            return BaseMemoryError(
                message=str(error),
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.HIGH,
                cause=error,
                suggestions=["Check network connectivity", "Retry the operation"],
            )
        elif "permission" in error_type.lower() or "access" in error_type.lower():
            return BaseMemoryError(
                message=str(error),
                category=ErrorCategory.AUTHORIZATION,
                severity=ErrorSeverity.HIGH,
                cause=error,
                suggestions=["Check user permissions", "Verify authentication"],
            )
        else:
            return BaseMemoryError(
                message=str(error),
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.MEDIUM,
                cause=error,
                suggestions=["Contact system administrator if problem persists"],
            )

    def _log_error(self, error: BaseMemoryError, log_traceback: bool = True) -> None:
        """Log an error with appropriate level and detail."""
        # Determine log level based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif error.severity == ErrorSeverity.HIGH:
            log_level = logging.ERROR
        elif error.severity == ErrorSeverity.MEDIUM:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

        # Create log message
        log_msg = f"[{error.error_id}] {error.category.value.upper()}: {error.message}"

        if error.context and error.context.operation:
            log_msg += f" (Operation: {error.context.operation})"

        if error.details:
            log_msg += f" - Details: {error.details}"

        # Log the error
        self.logger.log(log_level, log_msg)

        # Log traceback for high severity errors or when explicitly requested
        if log_traceback and (
            error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]
            or error.cause
        ):
            if error.cause:
                self.logger.log(
                    log_level, f"[{error.error_id}] Caused by: {error.cause}"
                )
            self.logger.log(log_level, f"[{error.error_id}] Traceback:", exc_info=True)

        # Log suggestions if available
        if error.suggestions:
            self.logger.info(
                f"[{error.error_id}] Suggestions: {', '.join(error.suggestions)}"
            )

    def get_error_stats(self) -> Dict[str, int]:
        """Get error frequency statistics."""
        return self.error_counts.copy()

    def reset_error_stats(self) -> None:
        """Reset error frequency statistics."""
        self.error_counts.clear()


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def handle_error(
    error: Union[Exception, BaseMemoryError],
    context: Optional[ErrorContext] = None,
    log_traceback: bool = True,
) -> ErrorDetails:
    """Handle an error using the global error handler."""
    return get_error_handler().handle_error(error, context, log_traceback)


def create_error_context(
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    operation: Optional[str] = None,
    component: Optional[str] = None,
    **additional_data,
) -> ErrorContext:
    """Create an error context object."""
    return ErrorContext(
        user_id=user_id,
        request_id=request_id,
        operation=operation,
        component=component,
        additional_data=additional_data,
    )


def with_error_handling(
    operation: str, component: Optional[str] = None, user_id: Optional[str] = None
):
    """Decorator to add comprehensive error handling to functions."""

    def decorator(func):
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = create_error_context(
                operation=operation, component=component, user_id=user_id
            )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_details = handle_error(e, context)
                # Re-raise as BaseMemoryError for consistent handling
                if not isinstance(e, BaseMemoryError):
                    raise BaseMemoryError(
                        message=error_details.message,
                        category=error_details.category,
                        severity=error_details.severity,
                        details=error_details.details,
                        context=context,
                        suggestions=error_details.suggestions,
                    ) from e
                raise

        return wrapper

    return decorator


def with_async_error_handling(
    operation: str, component: Optional[str] = None, user_id: Optional[str] = None
):
    """Decorator to add comprehensive error handling to async functions."""

    def decorator(func):
        import functools

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            context = create_error_context(
                operation=operation, component=component, user_id=user_id
            )

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_details = handle_error(e, context)
                # Re-raise as BaseMemoryError for consistent handling
                if not isinstance(e, BaseMemoryError):
                    raise BaseMemoryError(
                        message=error_details.message,
                        category=error_details.category,
                        severity=error_details.severity,
                        details=error_details.details,
                        context=context,
                        suggestions=error_details.suggestions,
                    ) from e
                raise

        return wrapper

    return decorator
