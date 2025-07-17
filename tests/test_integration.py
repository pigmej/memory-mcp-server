#!/usr/bin/env python3
"""Integration tests for MCP protocols and HTTP API endpoints."""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import httpx
import pytest

from memory_mcp_server.config import Config
from memory_mcp_server.protocols.http_server import create_http_server
from memory_mcp_server.protocols.mcp_server import MemoryMCPServer


class MCPClient:
    """Simple MCP client for testing STDIO protocol."""
    
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.request_id = 0
    
    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id
    
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and get response."""
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
        
        # Read response
        response_line = self.process.stdout.readline()
        if not response_line:
            raise Exception("No response received")
        
        return json.loads(response_line.strip())
    
    async def initialize(self) -> bool:
        """Initialize the MCP connection."""
        response = await self.send_request("initialize", {
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
    
    async def test_mcp_stdio_initialization(self):
        """Test MCP STDIO server initialization."""
        process = subprocess.Popen(
            [sys.executable, "-m", "memory_mcp_server.cli", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            # Give server time to start
            await asyncio.sleep(1)
            
            client = MCPClient(process)
            
            # Test initialization
            initialized = await client.initialize()
            assert initialized, "MCP server should initialize successfully"
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    async def test_mcp_tools_listing(self):
        """Test MCP tools listing functionality."""
        process = subprocess.Popen(
            [sys.executable, "-m", "memory_mcp_server.cli", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            await asyncio.sleep(1)
            client = MCPClient(process)
            
            # Initialize first
            await client.initialize()
            
            # List tools
            response = await client.send_request("tools/list", {})
            
            assert "result" in response, "Tools list should return result"
            assert "tools" in response["result"], "Result should contain tools array"
            
            tools = response["result"]["tools"]
            assert len(tools) > 0, "Should have at least one tool"
            
            # Check for expected tools
            tool_names = [tool["name"] for tool in tools]
            expected_tools = [
                "create_alias", "query_alias", "update_alias", "delete_alias",
                "create_note", "get_note", "update_note", "delete_note",
                "create_observation", "get_observation", "update_observation", "delete_observation",
                "create_hint", "get_hint", "update_hint", "delete_hint",
                "search_memories", "get_users", "get_memory_stats"
            ]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Tool {expected_tool} should be available"
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    async def test_mcp_resources_listing(self):
        """Test MCP resources listing functionality."""
        process = subprocess.Popen(
            [sys.executable, "-m", "memory_mcp_server.cli", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            await asyncio.sleep(1)
            client = MCPClient(process)
            
            # Initialize first
            await client.initialize()
            
            # List resources
            response = await client.send_request("resources/list")
            
            assert "result" in response, "Resources list should return result"
            assert "resources" in response["result"], "Result should contain resources array"
            
            resources = response["result"]["resources"]
            # Resources might be empty initially, but the structure should be correct
            assert isinstance(resources, list), "Resources should be a list"
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
    
    async def test_mcp_tool_execution(self):
        """Test MCP tool execution functionality."""
        process = subprocess.Popen(
            [sys.executable, "-m", "memory_mcp_server.cli", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            await asyncio.sleep(1)
            client = MCPClient(process)
            
            # Initialize first
            await client.initialize()
            
            # Test creating an alias
            response = await client.send_request("tools/call", {
                "name": "create_alias",
                "arguments": {
                    "source": "AI",
                    "target": "Artificial Intelligence",
                    "user_id": "test_user",
                    "bidirectional": True
                }
            })
            
            assert "result" in response, "Tool execution should return result"
            assert isinstance(response["result"], list), "Result should be a list of content items"
            assert len(response["result"]) > 0, "Should have at least one content item"
            
            # Test querying the alias
            response = await client.send_request("tools/call", {
                "name": "query_alias",
                "arguments": {
                    "query": "AI",
                    "user_id": "test_user"
                }
            })
            
            assert "result" in response, "Alias query should return result"
            
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


@pytest.mark.asyncio
class TestHTTPAPIIntegration:
    """Integration tests for HTTP API endpoints."""
    
    @pytest.fixture
    async def http_server_config(self):
        """Create a test HTTP server configuration."""
        config = Config()
        config.server.host = "127.0.0.1"
        config.server.http_port = 8001  # Use different port for testing
        return config
    
    @pytest.fixture
    async def running_http_server(self, http_server_config):
        """Start HTTP server for testing."""
        http_server = create_http_server(http_server_config)
        server_task = asyncio.create_task(http_server.start_server())
        
        # Wait for server to start
        await asyncio.sleep(2)
        
        yield http_server_config
        
        # Cleanup
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass
    
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
            
            # Update alias
            update_data = {"tags": ["test", "updated"]}
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
            
            # Get specific note
            response = await client.get(f"{base_url}/notes/{note_id}")
            assert response.status_code == 200
            
            note = response.json()
            assert note["id"] == note_id
            
            # Update note
            update_data = {"content": "Updated test note content"}
            response = await client.put(f"{base_url}/notes/{note_id}", json=update_data)
            assert response.status_code == 200
            
            updated_note = response.json()
            assert updated_note["content"] == "Updated test note content"
            
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
            
            # Test unified memory search
            response = await client.post(f"{base_url}/memories/search", json=search_data)
            assert response.status_code == 200
            
            unified_results = response.json()
            assert isinstance(unified_results, list)
            assert len(unified_results) >= 1
    
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
            
            users = response.json()
            assert isinstance(users, list)
            assert "test_user" in users
            
            # Test memory stats
            response = await client.get(f"{base_url}/memories/stats?user_id=test_user")
            assert response.status_code == 200
            
            stats = response.json()
            assert "total_memories" in stats
            assert stats["total_memories"] >= 1
            assert "by_type" in stats


@pytest.mark.asyncio
class TestWebUIIntegration:
    """Integration tests for Web UI functionality."""
    
    @pytest.fixture
    async def web_server_config(self):
        """Create a test web server configuration."""
        config = Config()
        config.server.host = "127.0.0.1"
        config.server.web_port = 8003  # Use different port for testing
        return config
    
    async def test_web_ui_accessibility(self, web_server_config):
        """Test that web UI pages are accessible."""
        # Import web server here to avoid circular imports
        from memory_mcp_server.web.server import create_web_server
        
        web_server = create_web_server(web_server_config)
        server_task = asyncio.create_task(web_server.start_server())
        
        try:
            # Wait for server to start
            await asyncio.sleep(2)
            
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