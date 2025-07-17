"""Middleware for error handling and logging in FastAPI applications."""

import logging
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .errors import (
    BaseMemoryError,
    ErrorCategory,
    ErrorSeverity,
    create_error_context,
    handle_error,
)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to handle errors consistently across all endpoints."""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and handle any errors."""
        # Generate request ID for tracing
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        # Log request
        start_time = time.time()
        self.logger.info(
            f"[{request_id}] {request.method} {request.url.path} - Started"
        )

        try:
            response = await call_next(request)

            # Log successful response
            duration = time.time() - start_time
            self.logger.info(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Completed {response.status_code} in {duration:.3f}s"
            )

            return response

        except BaseMemoryError as e:
            # Handle our custom errors
            duration = time.time() - start_time

            context = create_error_context(
                request_id=request_id,
                operation=f"{request.method} {request.url.path}",
                component="http_api",
            )

            error_details = handle_error(e, context)

            # Map error severity to HTTP status codes
            status_code = self._get_status_code_from_error(e)

            self.logger.error(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Error {status_code} in {duration:.3f}s: {e.message}"
            )

            return JSONResponse(
                status_code=status_code,
                content={
                    "error": {
                        "id": error_details.error_id,
                        "category": error_details.category.value,
                        "message": error_details.message,
                        "details": error_details.details,
                        "suggestions": error_details.suggestions,
                        "timestamp": error_details.timestamp.isoformat(),
                    }
                },
            )

        except Exception as e:
            # Handle unexpected errors
            duration = time.time() - start_time

            context = create_error_context(
                request_id=request_id,
                operation=f"{request.method} {request.url.path}",
                component="http_api",
            )

            error_details = handle_error(e, context)

            self.logger.error(
                f"[{request_id}] {request.method} {request.url.path} - "
                f"Unexpected error in {duration:.3f}s: {str(e)}"
            )

            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "id": error_details.error_id,
                        "category": error_details.category.value,
                        "message": "Internal server error",
                        "details": str(e) if self._should_expose_details() else None,
                        "suggestions": error_details.suggestions,
                        "timestamp": error_details.timestamp.isoformat(),
                    }
                },
            )

    def _get_status_code_from_error(self, error: BaseMemoryError) -> int:
        """Map error categories and severities to HTTP status codes."""
        if error.category == ErrorCategory.VALIDATION:
            return 400
        elif error.category == ErrorCategory.AUTHENTICATION:
            return 401
        elif error.category == ErrorCategory.AUTHORIZATION:
            return 403
        elif (
            error.category == ErrorCategory.DATABASE
            and "not found" in error.message.lower()
        ):
            return 404
        elif error.category == ErrorCategory.EXTERNAL and error.retry_after:
            return 429  # Rate limited
        elif error.category == ErrorCategory.CONFIGURATION:
            return 503  # Service unavailable
        elif error.severity == ErrorSeverity.CRITICAL:
            return 503  # Service unavailable
        elif error.severity == ErrorSeverity.HIGH:
            return 500  # Internal server error
        else:
            return 400  # Bad request

    def _should_expose_details(self) -> bool:
        """Determine if error details should be exposed to clients."""
        # In production, you might want to hide error details
        # This could be controlled by environment variables
        import os

        return os.getenv("ENVIRONMENT", "development").lower() != "production"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log detailed request information."""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request details."""
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request details in debug mode
        if self.logger.isEnabledFor(logging.DEBUG):
            headers = dict(request.headers)
            # Remove sensitive headers
            for sensitive_header in ["authorization", "cookie", "x-api-key"]:
                if sensitive_header in headers:
                    headers[sensitive_header] = "[REDACTED]"

            self.logger.debug(
                f"[{request_id}] Request details: "
                f"Headers: {headers}, "
                f"Query params: {dict(request.query_params)}, "
                f"Client: {request.client.host if request.client else 'unknown'}"
            )

        response = await call_next(request)

        # Log response details in debug mode
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"[{request_id}] Response: {response.status_code}, "
                f"Headers: {dict(response.headers)}"
            )

        return response


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the FastAPI application."""
    # Add middleware in reverse order (last added is executed first)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
