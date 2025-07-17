"""Resilience utilities for graceful degradation and error recovery."""

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, List, Optional, Type

from .errors import BaseMemoryError, ErrorCategory, ErrorSeverity


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            OSError,
        ]


class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception


class CircuitBreakerState:
    """State management for circuit breaker."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self.state == "OPEN"

    def is_half_open(self) -> bool:
        """Check if circuit breaker is half-open."""
        return self.state == "HALF_OPEN"

    def should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self.state == "OPEN" and self.last_failure_time:
            return time.time() - self.last_failure_time >= self.config.recovery_timeout
        return False

    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None

    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = "OPEN"
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"

    def attempt_reset(self):
        """Attempt to reset the circuit breaker to half-open."""
        if self.should_attempt_reset():
            self.state = "HALF_OPEN"


class GracefulDegradation:
    """Utilities for graceful degradation when services fail."""

    @staticmethod
    def fallback_search(query: str, memory_objects: List[Any]) -> List[Any]:
        """Fallback search using simple text matching when semantic search fails."""
        if not query or not memory_objects:
            return []

        query_lower = query.lower()
        results = []

        for obj in memory_objects:
            # Extract searchable text based on object type
            searchable_text = ""

            if hasattr(obj, "source") and hasattr(obj, "target"):  # Alias
                searchable_text = f"{obj.source} {obj.target}".lower()
            elif hasattr(obj, "title") and hasattr(obj, "content"):  # Note
                searchable_text = f"{obj.title} {obj.content}".lower()
            elif hasattr(obj, "content"):  # Observation or Hint
                searchable_text = obj.content.lower()

            # Simple text matching
            if query_lower in searchable_text:
                results.append(obj)

        return results

    @staticmethod
    def fallback_embedding_generation(text: str) -> List[float]:
        """Fallback embedding generation using simple text features."""
        # Simple hash-based embedding as fallback
        import hashlib

        # Create a simple feature vector based on text characteristics
        features = [
            len(text),  # Length
            text.count(" "),  # Word count approximation
            text.count("."),  # Sentence count approximation
            len(set(text.lower())),  # Unique character count
        ]

        # Add hash-based features for some semantic representation
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert hash bytes to normalized floats
        hash_features = [b / 255.0 for b in hash_bytes[:8]]  # Use first 8 bytes

        return features + hash_features

    @staticmethod
    def safe_database_operation(operation: Callable, fallback_value: Any = None) -> Any:
        """Safely execute database operation with fallback."""
        try:
            return operation()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Database operation failed, using fallback: {e}")
            return fallback_value


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic with exponential backoff."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if this exception is retryable
                    if not any(
                        isinstance(e, exc_type)
                        for exc_type in config.retryable_exceptions
                    ):
                        raise

                    # Don't retry on the last attempt
                    if attempt == config.max_attempts - 1:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base**attempt),
                        config.max_delay,
                    )

                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        import random

                        delay *= 0.5 + random.random() * 0.5

                    logger = logging.getLogger(func.__module__)
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    time.sleep(delay)

            # All attempts failed
            raise last_exception

        return wrapper

    return decorator


def async_retry_with_backoff(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic with exponential backoff for async functions."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if this exception is retryable
                    if not any(
                        isinstance(e, exc_type)
                        for exc_type in config.retryable_exceptions
                    ):
                        raise

                    # Don't retry on the last attempt
                    if attempt == config.max_attempts - 1:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        config.base_delay * (config.exponential_base**attempt),
                        config.max_delay,
                    )

                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        import random

                        delay *= 0.5 + random.random() * 0.5

                    logger = logging.getLogger(func.__module__)
                    logger.warning(
                        f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)

            # All attempts failed
            raise last_exception

        return wrapper

    return decorator


def circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Decorator to implement circuit breaker pattern."""
    if config is None:
        config = CircuitBreakerConfig()

    # Store circuit breaker state per function
    state = CircuitBreakerState(config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)

            # Check if circuit breaker should attempt reset
            if state.should_attempt_reset():
                state.attempt_reset()
                logger.info(
                    f"Circuit breaker for {func.__name__} attempting reset (HALF_OPEN)"
                )

            # If circuit is open, fail fast
            if state.is_open():
                logger.warning(
                    f"Circuit breaker for {func.__name__} is OPEN, failing fast"
                )
                raise BaseMemoryError(
                    message=f"Circuit breaker is open for {func.__name__}",
                    category=ErrorCategory.EXTERNAL,
                    severity=ErrorSeverity.HIGH,
                    suggestions=[
                        "Wait for circuit breaker to reset",
                        "Check service health",
                    ],
                )

            try:
                result = func(*args, **kwargs)
                state.record_success()
                if state.is_half_open():
                    logger.info(f"Circuit breaker for {func.__name__} reset to CLOSED")
                return result
            except config.expected_exception:
                state.record_failure()
                if state.is_open():
                    logger.error(
                        f"Circuit breaker for {func.__name__} opened due to failures"
                    )
                raise

        return wrapper

    return decorator


def async_circuit_breaker(config: Optional[CircuitBreakerConfig] = None):
    """Decorator to implement circuit breaker pattern for async functions."""
    if config is None:
        config = CircuitBreakerConfig()

    # Store circuit breaker state per function
    state = CircuitBreakerState(config)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)

            # Check if circuit breaker should attempt reset
            if state.should_attempt_reset():
                state.attempt_reset()
                logger.info(
                    f"Circuit breaker for {func.__name__} attempting reset (HALF_OPEN)"
                )

            # If circuit is open, fail fast
            if state.is_open():
                logger.warning(
                    f"Circuit breaker for {func.__name__} is OPEN, failing fast"
                )
                raise BaseMemoryError(
                    message=f"Circuit breaker is open for {func.__name__}",
                    category=ErrorCategory.EXTERNAL,
                    severity=ErrorSeverity.HIGH,
                    suggestions=[
                        "Wait for circuit breaker to reset",
                        "Check service health",
                    ],
                )

            try:
                result = await func(*args, **kwargs)
                state.record_success()
                if state.is_half_open():
                    logger.info(f"Circuit breaker for {func.__name__} reset to CLOSED")
                return result
            except config.expected_exception:
                state.record_failure()
                if state.is_open():
                    logger.error(
                        f"Circuit breaker for {func.__name__} opened due to failures"
                    )
                raise

        return wrapper

    return decorator


def with_graceful_degradation(fallback_func: Optional[Callable] = None):
    """Decorator to add graceful degradation with fallback function."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(func.__module__)
                logger.warning(f"Primary function {func.__name__} failed: {e}")

                if fallback_func:
                    logger.info(f"Using fallback function for {func.__name__}")
                    try:
                        return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback function also failed: {fallback_error}")
                        raise e from None  # Raise original error
                else:
                    raise

        return wrapper

    return decorator


def with_async_graceful_degradation(fallback_func: Optional[Callable] = None):
    """Decorator to add graceful degradation with fallback function for async functions."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(func.__module__)
                logger.warning(f"Primary function {func.__name__} failed: {e}")

                if fallback_func:
                    logger.info(f"Using fallback function for {func.__name__}")
                    try:
                        if asyncio.iscoroutinefunction(fallback_func):
                            return await fallback_func(*args, **kwargs)
                        else:
                            return fallback_func(*args, **kwargs)
                    except Exception as fallback_error:
                        logger.error(f"Fallback function also failed: {fallback_error}")
                        raise e from None  # Raise original error
                else:
                    raise

        return wrapper

    return decorator


class HealthChecker:
    """Health checking utilities for system components."""

    def __init__(self):
        self.component_health = {}
        self.logger = logging.getLogger(__name__)

    def register_component(self, name: str, health_check_func: Callable[[], bool]):
        """Register a component with its health check function."""
        self.component_health[name] = health_check_func

    def check_component_health(self, name: str) -> bool:
        """Check health of a specific component."""
        if name not in self.component_health:
            return False

        try:
            return self.component_health[name]()
        except Exception as e:
            self.logger.error(f"Health check failed for {name}: {e}")
            return False

    def check_all_health(self) -> dict:
        """Check health of all registered components."""
        results = {}
        for name in self.component_health:
            results[name] = self.check_component_health(name)
        return results

    def is_system_healthy(self) -> bool:
        """Check if the entire system is healthy."""
        health_results = self.check_all_health()
        return all(health_results.values())


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker
