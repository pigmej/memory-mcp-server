#!/usr/bin/env python3
"""Integration tests for MCP protocols and HTTP API endpoints."""

import asyncio
import json
import subprocess
from datetime import UTC, datetime
from typing import Any, Dict, Optional

import httpx
import pytest

from memory_mcp_server.config import Config
from memory_mcp_server.protocols.http_server import MemoryHTTPServer
from memory_mcp_server.web.server import WebUIServer


class MCPClient:
    """Simple MCP client for testing STDIO protocol."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.request_id = 0

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def send_request_sync(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and get response synchronously."""
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method
        }
        if params is not None:
            request["params"] = params

        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json)
        self.process.stdin.flush()

        # Read response, skipping any log lines
        max_attempts = 10
        for _ in range(max_attempts):
            response_line = self.process.stdout.readline()
            if not response_line:
                raise Exception("No response received")

            line = response_line.strip()
            if not line:
                continue

            # Skip log lines (they don't start with '{')
            if not line.startswith('{'):
                continue

            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

        raise Exception("Failed to get valid JSON response after multiple attempts")

    def initialize_sync(self) -> bool:
        """Initialize the MCP connection synchronously."""
        response = self.send_request_sync("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {}
            },
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        })

        return "result" in response and response["result"].get("protocolVersion") == "2024-11-05"


@pytest.mark.asyncio
class TestMCPProtocolIntegration:
    """Integration tests for MCP protocol functionality."""

    @pytest.fixture
    async def db_manager(self):
        """Create a database manager for testing."""
        from memory_mcp_server.config import Config
        from memory_mcp_server.database.connection import get_database_manager

        config = Config()
        db_manager = get_database_manager(config)
        await db_manager.initialize_database_async()

        yield db_manager

        # Cleanup: close all connections
        try:
            if hasattr(db_manager, 'engine') and db_manager.engine:
                await db_manager.engine.dispose()
        except Exception:
            pass  # Ignore cleanup errors

    async def test_mcp_server_initialization(self):
        """Test MCP server initialization and basic functionality."""
        from memory_mcp_server.config import Config
        from memory_mcp_server.protocols.mcp_server import MemoryMCPServer

        config = Config()
        server = MemoryMCPServer(config)

        # Test that server was created successfully
        assert server is not None
        assert server.mcp is not None

        # Test that tools are registered
        tools = await server.mcp.get_tools()
        assert len(tools) > 0

        # Check for expected tools - handle different return formats
        if isinstance(tools, dict):
            tool_names = list(tools.keys())
        elif isinstance(tools, list) and len(tools) > 0:
            if isinstance(tools[0], str):
                tool_names = tools
            else:
                tool_names = [tool.name for tool in tools]
        else:
            tool_names = []
        expected_tools = [
            "create_alias", "query_alias", "search_memories", "get_memories",
            "create_note", "create_observation", "create_hint",
            "get_users", "update_memory", "delete_memory"
        ]

        # Check that at least some expected tools are present
        found_tools = [tool for tool in expected_tools if tool in tool_names]
        assert len(found_tools) >= 3, f"Should find at least 3 expected tools, found: {found_tools}"

    async def test_mcp_tool_execution(self, db_manager):
        """Test MCP tool execution functionality by testing the underlying services."""
        from memory_mcp_server.config import Config
        from memory_mcp_server.models.alias import Alias
        from memory_mcp_server.protocols.mcp_server import MemoryMCPServer

        config = Config()
        server = MemoryMCPServer(config)

        # Test creating an alias through the memory service (which the MCP tools use)
        async with db_manager.get_async_session() as session:
            alias = Alias(
                source="AI",
                target="Artificial Intelligence",
                user_id="test_user",
                bidirectional=True,
                tags=[],
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC)
            )

            result = await server.memory_service.create_alias_async(session, alias)

            # Check that the alias was created successfully
            assert result is not None
            assert result.source == "AI"
            assert result.target == "Artificial Intelligence"
            assert result.user_id == "test_user"

            # Test querying the alias
            query_results = await server.memory_service.query_alias_async(
                session, "AI", user_id="test_user"
            )

            assert len(query_results) > 0
            assert "Artificial Intelligence" in query_results

    async def test_mcp_search_functionality(self, db_manager):
        """Test MCP search functionality by testing the underlying services."""
        from memory_mcp_server.config import Config
        from memory_mcp_server.models.alias import Alias
        from memory_mcp_server.models.note import Note
        from memory_mcp_server.protocols.mcp_server import MemoryMCPServer

        config = Config()
        server = MemoryMCPServer(config)

        # Create some test data first using the services directly
        async with db_manager.get_async_session() as session:
            # Create alias
            alias = Alias(
                source="ML",
                target="Machine Learning",
                user_id="test_user",
                bidirectional=True,
                tags=[],
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC)
            )
            await server.memory_service.create_alias_async(session, alias)

            # Create note
            note = Note(
                title="ML Basics",
                content="Machine learning is a subset of AI",
                user_id="test_user",
                category="education",
                tags=[],
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC)
            )
            await server.memory_service.create_note_async(session, note)

            # Test search functionality using the search service
            search_results = await server.search_service.exact_search_async(
                session=session,
                query="machine learning",
                user_id="test_user",
                limit=10
            )

            assert len(search_results) > 0
            # Check that we found relevant results
            found_ml = any("ML" in str(result) or "Machine Learning" in str(result)
                          for result in search_results)
            assert found_ml, f"Should find ML-related results in: {search_results}"


@pytest.mark.asyncio
class TestHTTPAPIIntegration:
    """Integration tests for HTTP API endpoints."""

    @pytest.fixture
    async def http_server_config(self):
        """Create a test HTTP server configuration with a unique port."""
        import socket

        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]

        config = Config()
        config.server.host = "127.0.0.1"
        config.server.http_port = port
        return config

    @pytest.fixture
    async def running_http_server(self, http_server_config):
        """Start HTTP server for testing."""
        http_server = MemoryHTTPServer(http_server_config)

        # Start server in background
        server_task = asyncio.create_task(http_server.start_server())

        # Wait for server to start
        await asyncio.sleep(2)

        try:
            yield http_server_config
        finally:
            # Proper cleanup
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass
            # Give time for cleanup
            await asyncio.sleep(0.5)

    async def test_health_endpoint(self, running_http_server):
        """Test health check endpoint."""
        config = running_http_server
        base_url = f"http://{config.server.host}:{config.server.http_port}"

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            assert response.status_code == 200

            health_data = response.json()
            assert "status" in health_data
            assert health_data["status"] == "healthy"

    async def test_alias_crud_endpoints(self, running_http_server):
        """Test alias CRUD operations via HTTP API."""
        config = running_http_server
        base_url = f"http://{config.server.host}:{config.server.http_port}"

        async with httpx.AsyncClient() as client:
            # Create alias
            alias_data = {
                "source": "test_source",
                "target": "test_target",
                "user_id": "test_user",
                "bidirectional": True,
                "tags": ["test"]
            }

            response = await client.post(f"{base_url}/aliases", json=alias_data)
            assert response.status_code == 200

            created_alias = response.json()
            assert created_alias["source"] == "test_source"
            assert created_alias["target"] == "test_target"
            alias_id = created_alias["id"]

            # Get aliases
            response = await client.get(f"{base_url}/aliases?user_id=test_user")
            assert response.status_code == 200

            aliases = response.json()
            assert len(aliases) >= 1
            assert any(alias["id"] == alias_id for alias in aliases)

            # Query alias
            response = await client.get(f"{base_url}/aliases/query/test_source?user_id=test_user")
            assert response.status_code == 200

            query_result = response.json()
            assert len(query_result) >= 1

            # Update alias (using the full alias data structure)
            update_data = {
                "source": "test_source",
                "target": "test_target",
                "user_id": "test_user",
                "bidirectional": True,
                "tags": ["test", "updated"]
            }
            response = await client.put(f"{base_url}/aliases/{alias_id}", json=update_data)
            assert response.status_code == 200

            updated_alias = response.json()
            assert "updated" in updated_alias["tags"]

            # Delete alias
            response = await client.delete(f"{base_url}/aliases/{alias_id}")
            assert response.status_code == 200

    async def test_note_crud_endpoints(self, running_http_server):
        """Test note CRUD operations via HTTP API."""
        config = running_http_server
        base_url = f"http://{config.server.host}:{config.server.http_port}"

        async with httpx.AsyncClient() as client:
            # Create note
            note_data = {
                "title": "Test Note",
                "content": "This is a test note content",
                "user_id": "test_user",
                "category": "test",
                "tags": ["test"]
            }

            response = await client.post(f"{base_url}/notes", json=note_data)
            assert response.status_code == 200

            created_note = response.json()
            assert created_note["title"] == "Test Note"
            note_id = created_note["id"]

            # Get notes
            response = await client.get(f"{base_url}/notes?user_id=test_user")
            assert response.status_code == 200

            notes = response.json()
            assert len(notes) >= 1
            assert any(note["id"] == note_id for note in notes)

            # Update note (using the full note data structure)
            update_data = {
                "title": "Test Note",
                "content": "Updated test note content",
                "user_id": "test_user",
                "category": "test",
                "tags": ["test", "updated"]
            }
            response = await client.put(f"{base_url}/notes/{note_id}", json=update_data)
            assert response.status_code == 200

            updated_note = response.json()
            assert updated_note["content"] == "Updated test note content"
            assert "updated" in updated_note["tags"]

            # Delete note
            response = await client.delete(f"{base_url}/notes/{note_id}")
            assert response.status_code == 200

    async def test_search_endpoints(self, running_http_server):
        """Test search functionality via HTTP API."""
        config = running_http_server
        base_url = f"http://{config.server.host}:{config.server.http_port}"

        async with httpx.AsyncClient() as client:
            # First create some test data
            alias_data = {
                "source": "ML",
                "target": "Machine Learning",
                "user_id": "test_user",
                "bidirectional": True,
                "tags": ["technology"]
            }
            await client.post(f"{base_url}/aliases", json=alias_data)

            note_data = {
                "title": "ML Basics",
                "content": "Machine learning is a subset of AI",
                "user_id": "test_user",
                "category": "education",
                "tags": ["ML", "AI"]
            }
            await client.post(f"{base_url}/notes", json=note_data)

            # Test search
            search_data = {
                "query": "machine learning",
                "user_id": "test_user",
                "semantic": False,
                "limit": 10
            }

            response = await client.post(f"{base_url}/search", json=search_data)
            assert response.status_code == 200

            search_results = response.json()
            assert "results" in search_results
            assert len(search_results["results"]) >= 1

            # Test memory stats endpoint
            response = await client.get(f"{base_url}/memories/stats?user_id=test_user")
            assert response.status_code == 200

            stats = response.json()
            # The stats response has individual counts by type
            assert "aliases" in stats or "notes" in stats or "observations" in stats or "hints" in stats
            # Check that we have at least some data
            total_count = sum(stats.get(key, 0) for key in ["aliases", "notes", "observations", "hints"])
            assert total_count >= 1

    async def test_streaming_endpoints(self, running_http_server):
        """Test streaming search endpoints."""
        config = running_http_server
        base_url = f"http://{config.server.host}:{config.server.http_port}"

        async with httpx.AsyncClient() as client:
            # Create test data first
            alias_data = {
                "source": "AI",
                "target": "Artificial Intelligence",
                "user_id": "test_user",
                "bidirectional": True
            }
            await client.post(f"{base_url}/aliases", json=alias_data)

            # Test streaming search
            search_data = {
                "query": "artificial intelligence",
                "user_id": "test_user",
                "memory_types": ["alias", "note"]
            }

            async with client.stream("POST", f"{base_url}/search/stream", json=search_data) as response:
                assert response.status_code == 200

                chunks_received = 0
                async for chunk in response.aiter_text():
                    if chunk.strip():
                        try:
                            data = json.loads(chunk.strip())
                            assert "type" in data
                            chunks_received += 1
                        except json.JSONDecodeError:
                            # Some chunks might be partial, that's okay
                            pass

                assert chunks_received > 0, "Should receive at least one streaming chunk"

    async def test_users_and_stats_endpoints(self, running_http_server):
        """Test users and statistics endpoints."""
        config = running_http_server
        base_url = f"http://{config.server.host}:{config.server.http_port}"

        async with httpx.AsyncClient() as client:
            # Create some test data first
            alias_data = {
                "source": "test",
                "target": "testing",
                "user_id": "test_user",
                "bidirectional": True
            }
            await client.post(f"{base_url}/aliases", json=alias_data)

            # Test users endpoint
            response = await client.get(f"{base_url}/users")
            assert response.status_code == 200

            users_response = response.json()
            # The users endpoint returns an object with a 'users' key
            if isinstance(users_response, dict) and "users" in users_response:
                users = users_response["users"]
            else:
                users = users_response
            assert isinstance(users, list)
            assert "test_user" in users

            # Test memory stats
            response = await client.get(f"{base_url}/memories/stats?user_id=test_user")
            assert response.status_code == 200

            stats = response.json()
            assert "total" in stats
            assert stats["total"] >= 1
            assert "aliases" in stats
            assert "notes" in stats


@pytest.mark.asyncio
class TestWebUIIntegration:
    """Integration tests for Web UI functionality."""

    @pytest.fixture
    async def web_server_config(self):
        """Create a test web server configuration with a unique port."""
        import socket

        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]

        config = Config()
        config.server.host = "127.0.0.1"
        config.server.web_port = port
        return config

    async def test_web_ui_accessibility(self, web_server_config):
        """Test that web UI pages are accessible."""
        web_server = WebUIServer(web_server_config)

        # Use uvicorn to run the server
        import uvicorn
        server = uvicorn.Server(uvicorn.Config(
            app=web_server.get_asgi_app(),
            host=web_server_config.server.host,
            port=web_server_config.server.web_port,
            log_level="error"
        ))

        server_task = asyncio.create_task(server.serve())

        try:
            # Wait for server to start
            await asyncio.sleep(3)

            base_url = f"http://{web_server_config.server.host}:{web_server_config.server.web_port}"

            async with httpx.AsyncClient() as client:
                # Test main page
                response = await client.get(base_url)
                assert response.status_code == 200
                assert "text/html" in response.headers.get("content-type", "")

                # Test aliases page
                response = await client.get(f"{base_url}/aliases")
                assert response.status_code == 200
                assert "text/html" in response.headers.get("content-type", "")

                # Test notes page
                response = await client.get(f"{base_url}/notes")
                assert response.status_code == 200
                assert "text/html" in response.headers.get("content-type", "")

                # Test search page
                response = await client.get(f"{base_url}/search")
                assert response.status_code == 200
                assert "text/html" in response.headers.get("content-type", "")

        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
