"""Command-line interface for the Memory MCP Server."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .config import Config, load_config
from .protocols.mcp_server import MemoryMCPServer


@click.group()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.pass_context
def cli(ctx: click.Context, config: Optional[Path], debug: bool) -> None:
    """Memory MCP Server - A lightweight memory server with MCP protocol support."""
    ctx.ensure_object(dict)

    # Load configuration
    app_config = load_config(config)
    if debug:
        app_config.logging.level = "DEBUG"

    # Initialize logging and error handling
    try:
        from .utils.logging import setup_logging

        setup_logging(app_config.logging)
    except ImportError:
        # Fallback to basic logging
        logging.basicConfig(
            level=getattr(logging, app_config.logging.level),
            format=app_config.logging.format,
        )

    logger = logging.getLogger(__name__)
    logger.info(f"Memory MCP Server starting with config: {app_config.environment}")
    logger.debug(f"Configuration loaded from: {config or 'defaults'}")

    ctx.obj["config"] = app_config


@cli.command()
@click.pass_context
def stdio(ctx: click.Context) -> None:
    """Start the STDIO MCP server."""
    config = ctx.obj["config"]

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
        stream=sys.stderr,  # Log to stderr to avoid interfering with STDIO protocol
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Memory MCP Server in STDIO mode...")
    logger.info(f"Data directory: {config.data_dir}")

    try:
        # Create and run the STDIO server
        asyncio.run(_run_stdio_server(config))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--host", default=None, help="Server host")
@click.option("--port", default=None, type=int, help="Server port")
@click.pass_context
def http(ctx: click.Context, host: Optional[str], port: Optional[int]) -> None:
    """Start the HTTP streaming server."""
    config = ctx.obj["config"]

    if host:
        config.server.host = host
    if port:
        config.server.http_port = port

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Memory MCP Server in HTTP mode...")
    logger.info(f"Host: {config.server.host}")
    logger.info(f"Port: {config.server.http_port}")
    logger.info(f"Data directory: {config.data_dir}")

    try:
        # Create and run the HTTP server
        asyncio.run(_run_http_server(config, host, port))
    except KeyboardInterrupt:
        logger.info("HTTP server stopped by user")
    except Exception as e:
        logger.error(f"HTTP server error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--host", default=None, help="Server host")
@click.option("--port", default=None, type=int, help="Server port")
@click.pass_context
def web(ctx: click.Context, host: Optional[str], port: Optional[int]) -> None:
    """Start the web UI server."""
    config = ctx.obj["config"]

    if host:
        config.server.host = host
    if port:
        config.server.web_port = port

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Memory MCP Server Web UI...")
    logger.info(f"Host: {config.server.host}")
    logger.info(f"Port: {config.server.web_port}")
    logger.info(f"Data directory: {config.data_dir}")

    try:
        # Create and run the web server
        asyncio.run(_run_web_server(config, host, port))
    except KeyboardInterrupt:
        logger.info("Web server stopped by user")
    except Exception as e:
        logger.error(f"Web server error: {e}")
        sys.exit(1)


@cli.command()
@click.option("--host", default=None, help="Server host")
@click.option("--port", default=None, type=int, help="Server port")
@click.pass_context
def combined(ctx: click.Context, host: Optional[str], port: Optional[int]) -> None:
    """Start the combined web UI + MCP HTTP server."""
    config = ctx.obj["config"]

    if host:
        config.server.host = host
    if port:
        config.server.web_port = port

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Memory MCP Server in Combined Mode...")
    logger.info("This includes both Web UI and MCP HTTP API")
    logger.info(f"Host: {config.server.host}")
    logger.info(f"Port: {config.server.web_port}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Web UI: http://{config.server.host}:{config.server.web_port}/")
    logger.info(f"MCP API: http://{config.server.host}:{config.server.web_port}/mcp/")
    logger.info(
        f"API Info: http://{config.server.host}:{config.server.web_port}/mcp-info"
    )

    try:
        # Create and run the combined server
        asyncio.run(_run_combined_server(config, host, port))
    except KeyboardInterrupt:
        logger.info("Combined server stopped by user")
    except Exception as e:
        logger.error(f"Combined server error: {e}")
        sys.exit(1)


@cli.command()
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize the database and configuration."""
    config = ctx.obj["config"]

    click.echo("Initializing Memory MCP Server...")
    click.echo(f"Data directory: {config.data_dir}")
    click.echo(f"Database URL: {config.database.url}")

    # Create data directory
    config.data_dir.mkdir(parents=True, exist_ok=True)

    # Save default configuration
    config_file = config.data_dir / "config.yaml"
    config.save_to_file(config_file)
    click.echo(f"Configuration saved to: {config_file}")

    # TODO: Initialize database
    click.echo("Database initialization not yet implemented")

    click.echo("Initialization complete!")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show server status and configuration."""
    config = ctx.obj["config"]

    click.echo("Memory MCP Server Status")
    click.echo("=" * 30)
    click.echo(f"Environment: {config.environment}")
    click.echo(f"Data directory: {config.data_dir}")
    click.echo(f"Database URL: {config.database.url}")
    click.echo(f"HTTP enabled: {config.server.http_enabled}")
    click.echo(f"STDIO enabled: {config.server.stdio_enabled}")
    click.echo(f"Web UI enabled: {config.server.web_enabled}")
    click.echo(f"RAG enabled: {config.enable_rag}")
    click.echo(f"User separation: {config.enable_user_separation}")


def main() -> None:
    """Main entry point for the CLI."""
    cli()


async def _run_stdio_server(config: Config) -> None:
    """Run the STDIO MCP server."""
    logger = logging.getLogger(__name__)

    try:
        # Initialize database
        from .database.connection import get_database_manager

        db_manager = get_database_manager(config)
        await db_manager.initialize_database_async()
        logger.info("Database initialized successfully")

        # Create and start the MCP server
        server = MemoryMCPServer()
        logger.info("MCP server created, starting STDIO transport...")

        # Run the FastMCP server with STDIO transport
        await server.mcp.run_stdio_async()

    except Exception as e:
        logger.error(f"Failed to start STDIO server: {e}")
        raise


async def _run_http_server(
    config: Config, host: Optional[str] = None, port: Optional[int] = None
) -> None:
    """Run the HTTP streaming server."""
    logger = logging.getLogger(__name__)

    try:
        # Import HTTP server
        from .protocols.http_server import create_http_server

        # Create HTTP server
        http_server = create_http_server(config)
        logger.info("HTTP server created, starting...")

        # Start the server
        await http_server.start_server(host, port)

    except Exception as e:
        logger.error(f"Failed to start HTTP server: {e}")
        raise


async def _run_web_server(
    config: Config, host: Optional[str] = None, port: Optional[int] = None
) -> None:
    """Run the web UI server."""
    logger = logging.getLogger(__name__)

    try:
        # Import web server
        from .web.server import create_web_server

        # Create web server
        web_server = create_web_server(config)
        logger.info("Web server created, starting...")

        # Start the server
        await web_server.start_server(host, port)

    except Exception as e:
        logger.error(f"Failed to start web server: {e}")
        raise


async def _run_combined_server(
    config: Config, host: Optional[str] = None, port: Optional[int] = None
) -> None:
    """Run the combined web UI + MCP API + HTTP API server."""
    logger = logging.getLogger(__name__)

    try:
        # Import assembly layer
        from .assembly import create_combined_server

        # Create combined server using assembly layer
        combined_server = create_combined_server(config)
        logger.info(
            "Combined server created (Web UI + MCP API + HTTP API), starting..."
        )

        # Start the server
        await combined_server.start_server(host, port)

    except Exception as e:
        logger.error(f"Failed to start combined server: {e}")
        raise


def stdio_main() -> None:
    """Entry point for STDIO server."""
    config = load_config()

    # Setup logging to stderr to avoid interfering with STDIO protocol
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
        stream=sys.stderr,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Memory MCP Server in STDIO mode...")
    logger.info(f"Data directory: {config.data_dir}")

    try:
        asyncio.run(_run_stdio_server(config))
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


def http_main() -> None:
    """Entry point for HTTP server."""
    config = load_config()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Memory MCP Server in HTTP mode...")
    logger.info(f"Host: {config.server.host}")
    logger.info(f"Port: {config.server.http_port}")
    logger.info(f"Data directory: {config.data_dir}")

    try:
        asyncio.run(_run_http_server(config))
    except KeyboardInterrupt:
        logger.info("HTTP server stopped by user")
    except Exception as e:
        logger.error(f"HTTP server error: {e}")
        sys.exit(1)


def web_main() -> None:
    """Entry point for Web UI server."""
    config = load_config()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Memory MCP Server Web UI...")
    logger.info(f"Host: {config.server.host}")
    logger.info(f"Port: {config.server.web_port}")
    logger.info(f"Data directory: {config.data_dir}")

    try:
        asyncio.run(_run_web_server(config))
    except KeyboardInterrupt:
        logger.info("Web server stopped by user")
    except Exception as e:
        logger.error(f"Web server error: {e}")
        sys.exit(1)


def combined_main() -> None:
    """Entry point for Combined server."""
    config = load_config()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper()),
        format=config.logging.format,
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting Memory MCP Server in Combined Mode...")
    logger.info("This includes both Web UI and MCP HTTP API")
    logger.info(f"Host: {config.server.host}")
    logger.info(f"Port: {config.server.web_port}")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Web UI: http://{config.server.host}:{config.server.web_port}/")
    logger.info(f"MCP API: http://{config.server.host}:{config.server.web_port}/mcp/")
    logger.info(
        f"API Info: http://{config.server.host}:{config.server.web_port}/mcp-info"
    )

    try:
        asyncio.run(_run_combined_server(config))
    except KeyboardInterrupt:
        logger.info("Combined server stopped by user")
    except Exception as e:
        logger.error(f"Combined server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
