"""Assembly layer for combining different ASGI apps."""

import logging
from typing import Optional

from starlette.applications import Starlette
from starlette.routing import Mount

from .config import Config
from .protocols.http_server import MemoryHTTPServer
from .protocols.mcp_server import MemoryMCPServer
from .web.server import WebUIServer

logger = logging.getLogger(__name__)


class CombinedServer:
    """Combined server that assembles multiple ASGI apps."""

    def __init__(self, config: Config):
        self.config = config
        self.app = None

        # Initialize individual servers
        self.web_server = WebUIServer(config)
        self.mcp_server = MemoryMCPServer(config)
        self.http_server = MemoryHTTPServer(config)

        # Assemble the apps
        self._assemble_apps()

    def _assemble_apps(self):
        """Assemble all ASGI apps with proper routing."""
        logger.info("Assembling ASGI apps...")

        # Get individual ASGI apps
        web_app = self.web_server.get_asgi_app()
        mcp_app = self.mcp_server.get_asgi_app(
            "/"
        )  # / because later we mount it under /mcp
        http_app = self.http_server.get_asgi_app()

        self.app = Starlette(
            routes=[
                Mount("/mcp", app=mcp_app),
                Mount("/api", app=http_app),
                Mount("/ui", app=web_app),
            ],
            lifespan=mcp_app.lifespan,
        )

        logger.info("ASGI apps assembled successfully")

    def get_asgi_app(self):
        """Get the combined ASGI app."""
        return self.app

    async def start_server(
        self, host: Optional[str] = None, port: Optional[int] = None
    ):
        """Start the combined server."""
        import uvicorn

        from .database.connection import get_database_manager

        # Initialize database
        db_manager = get_database_manager(self.config)
        await db_manager.initialize_database_async()

        # Use config values if not provided
        server_host = host or self.config.server.host
        server_port = port or self.config.server.web_port

        logger.info(f"Starting combined server on {server_host}:{server_port}")
        logger.info(f"Web UI: http://{server_host}:{server_port}/")
        logger.info(f"MCP API: http://{server_host}:{server_port}/mcp/")
        logger.info(f"HTTP API: http://{server_host}:{server_port}/api/")
        logger.info(f"Server Info: http://{server_host}:{server_port}/server-info")

        # Configure uvicorn
        config = uvicorn.Config(
            app=self.app,
            host=server_host,
            port=server_port,
            log_level=self.config.logging.level.lower(),
            access_log=True,
        )

        server = uvicorn.Server(config)
        await server.serve()


def create_combined_server(config: Config) -> CombinedServer:
    """Create a combined server instance."""
    return CombinedServer(config)
