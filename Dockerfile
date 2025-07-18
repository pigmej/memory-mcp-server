# Use Python 3.13 slim image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create data directory for SQLite database
RUN mkdir -p /app/data

# Expose ports
EXPOSE 8000 8001 8002 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:///app/data/memory_mcp_server.db

# Initialize database on startup and run combined server
CMD ["uv", "run", "memory-mcp-server", "combined", "--host", "0.0.0.0"]