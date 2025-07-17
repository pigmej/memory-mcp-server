# Memory MCP Server

A lightweight memory MCP (Model Context Protocol) server that provides persistent storage and retrieval of user-specific information including aliases, notes, observations, and hints. The server supports multiple access protocols and includes a web UI for data management with configurable semantic search capabilities.

## Features

### **Memory Types**
- **Aliases**: Bidirectional mappings between terms and phrases
- **Notes**: Personal information and reminders with categories
- **Observations**: Contextual information about entities
- **Hints**: Guidance for LLM interactions and workflows

### **Multiple Access Methods**
- **MCP STDIO**: Direct MCP protocol for AI clients
- **MCP HTTP API**: HTTP Streaming MCP endpoints
- **HTTP REST API**: Traditional REST endpoints
- **Web UI**: Browser-based management interface
- **Combined Server**: All interfaces in one server

### **Advanced Capabilities**
- **Semantic Search**: Configurable embedding models for intelligent search
- **User Separation**: Keep data separate by user ID
- **RAG Support**: Vector embeddings with similarity search
- **Export/Import**: JSON and CSV export capabilities
- **Real-time Search**: Search across all memory types simultaneously

## ⚠️ Important Notice

I use this MCP server on a daily basis, but I also use multiple AI tools to work on the codebase, so your mileage may vary. The server is actively developed and tested in my workflow, but different setups and use cases might encounter issues.

## Installation

```bash
uvx --from git+https://github.com/pigmej/memory-mcp-server.git memory-mcp-server
```

Or install globally with uv:
```bash
uv tool install --from git+https://github.com/pigmej/memory-mcp-server.git memory-mcp-server
```

For local development:
```bash
git clone https://github.com/pigmej/memory-mcp-server.git
cd memory-mcp-server
uv sync
uvx --from . memory-mcp-server
```

## Quick Start

### 1. Initialize the Server
```bash
memory-mcp-server init
```

### 2. Choose Your Server Mode

#### **Combined Server (Recommended)**
Run all interfaces on one port:
```bash
memory-mcp-server combined
```
- Web UI: `http://localhost:8080/ui/`
- MCP API: `http://localhost:8080/mcp/`
- REST API: `http://localhost:8080/api/`

#### **Individual Servers**
Run specific interfaces separately:
```bash
# MCP STDIO (for AI clients)
memory-mcp-server stdio

# HTTP REST API only
memory-mcp-server http --port 8000

# Web UI only
memory-mcp-server web --port 8080

# MCP HTTP API (standalone)
memory-mcp-server mcp --port 8001
```

## Configuration

### **Configuration File**
Create a `config.yaml` file:
```yaml
# Database settings
database:
  url: "sqlite:///memory_mcp.db"
  echo: false

# Server settings
server:
  host: "localhost"
  web_port: 8080
  http_port: 8000

# Embedding/Search settings
embedding:
  model_name: "all-MiniLM-L6-v2"  # Fast, lightweight
  # model_name: "all-mpnet-base-v2"  # Better quality
  similarity_threshold: 0.3
  max_results: 10
  cache_embeddings: true

# Logging
logging:
  level: "INFO"
  # file_path: "memory_mcp.log"
```

### **Environment Variables**
```bash
# Database
DB_URL=sqlite:///memory_mcp.db

# Server
SERVER_HOST=0.0.0.0
SERVER_WEB_PORT=8080

# Embedding model
EMBEDDING_MODEL_NAME=all-mpnet-base-v2
EMBEDDING_SIMILARITY_THRESHOLD=0.5
EMBEDDING_MAX_RESULTS=20

# Logging
LOG_LEVEL=INFO
```

### **Command Line Options**
```bash
# Custom config file
memory-mcp-server --config myconfig.yaml combined

# Custom host/port
memory-mcp-server combined --host 0.0.0.0 --port 8080

# Debug mode
memory-mcp-server --debug combined
```

## Usage Examples

### **Web Interface**
1. Open `http://localhost:8080` in your browser
2. Create aliases, notes, observations, and hints
3. Use the search interface to find information
4. Export data as JSON or CSV

### **MCP HTTP API**
```bash
# Create an alias
curl -X POST http://localhost:8080/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "create_alias",
    "arguments": {
      "source": "AI",
      "target": "Artificial Intelligence",
      "bidirectional": true
    }
  }'

# Search memories
curl -X POST http://localhost:8080/mcp/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "search_memories",
    "arguments": {
      "query": "machine learning",
      "limit": 10
    }
  }'

# List available tools
curl http://localhost:8080/mcp/tools/list
```

### **REST API**
```bash
# Get all aliases
curl http://localhost:8080/api/aliases

# Create a note
curl -X POST http://localhost:8080/api/notes \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Meeting Notes",
    "content": "Discussed project timeline and deliverables",
    "category": "Work"
  }'

# Search across all types
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "project", "limit": 5}'
```

## Embedding Models

The server supports various sentence-transformer models:

### **Fast & Lightweight**
- `all-MiniLM-L6-v2` (default) - 384 dimensions, fast
- `all-MiniLM-L12-v2` - 384 dimensions, better quality

### **High Quality**
- `all-mpnet-base-v2` - 768 dimensions, best quality
- `all-distilroberta-v1` - 768 dimensions, good balance

### **Multilingual**
- `paraphrase-multilingual-MiniLM-L12-v2` - 384 dimensions
- `paraphrase-multilingual-mpnet-base-v2` - 768 dimensions

Configure via `embedding.model_name` in config or `EMBEDDING_MODEL_NAME` environment variable.

## Architecture

The server uses a clean, modular architecture:

```
memory_mcp_server/
├── assembly.py              # ASGI app composition layer
├── cli.py                   # Command-line interface
├── config.py                # Configuration management
├── models/                  # Pydantic data models
│   ├── alias.py            # Alias model
│   ├── note.py             # Note model
│   ├── observation.py      # Observation model
│   └── hint.py             # Hint model
├── database/                # SQLAlchemy models and database layer
│   ├── connection.py       # Database connection management
│   ├── models.py           # SQLAlchemy models
│   └── repositories.py     # Data access layer
├── services/                # Business logic layer
│   ├── memory_service.py   # Core memory operations
│   └── search_service.py   # Search and RAG capabilities
├── protocols/               # Protocol handlers
│   ├── mcp_server.py       # MCP protocol implementation
│   └── http_server.py      # HTTP REST API
└── web/                     # Web UI components
    ├── server.py           # Web server
    ├── templates/          # Jinja2 templates
    └── static/             # CSS and static files
```

### **Design Principles**
- **Separation of Concerns**: Each component has a single responsibility
- **ASGI Composition**: Individual servers return ASGI apps for flexible deployment
- **Configuration-Driven**: Extensive configuration options
- **Protocol Agnostic**: Same business logic serves multiple protocols

## Development

### **Setup**
```bash
git clone <repository>
cd memory-mcp-server
uv sync --dev
```

### **Testing**
```bash
pytest
pytest --cov=memory_mcp_server
```

### **Code Quality**
```bash
ruff format memory_mcp_server/
ruff check memory_mcp_server/
mypy memory_mcp_server/
```

## License

MIT License