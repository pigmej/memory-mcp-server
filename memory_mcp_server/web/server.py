"""Web UI server implementation using FastAPI."""

import logging
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import Config
from ..database.connection import get_database_manager
from ..models.alias import Alias
from ..models.hint import Hint
from ..models.note import Note
from ..models.observation import Observation
from ..services.memory_service import MemoryService
from ..services.search_service import SearchService

logger = logging.getLogger(__name__)


class WebUIServer:
    """Web UI server for managing memory data."""

    def __init__(self, config: Config):
        self.config = config
        self.app = FastAPI(
            title="Memory MCP Server Web UI",
            description="Web interface for managing memory data",
            version="0.1.0",
        )

        # Initialize services
        self.db_manager = get_database_manager(config)
        self.memory_service = MemoryService()
        self.search_service = SearchService(config)

        # Setup templates and static files
        self._setup_templates()
        self._setup_routes()

    def get_asgi_app(self):
        """Get the ASGI app for this web UI server."""
        return self.app

    async def _get_session(self):
        """Get an async database session."""
        return self.db_manager.get_async_session()

    # Wrapper methods for memory service operations
    async def get_aliases(
        self, user_id: Optional[str] = None, query: Optional[str] = None
    ):
        """Get aliases with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.get_aliases_async(session, user_id, query)

    async def create_alias(self, alias: Alias):
        """Create alias with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.create_alias_async(session, alias)

    async def delete_alias(self, alias_id: int):
        """Delete alias with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.delete_alias_async(session, alias_id)

    async def get_notes(
        self,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        query: Optional[str] = None,
    ):
        """Get notes with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.get_notes_async(
                session, user_id, category, query
            )

    async def create_note(self, note: Note):
        """Create note with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.create_note_async(session, note)

    async def delete_note(self, note_id: int):
        """Delete note with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.delete_note_async(session, note_id)

    async def get_observations(
        self, entity_id: Optional[str] = None, user_id: Optional[str] = None
    ):
        """Get observations with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.get_observations_async(
                session, entity_id, user_id
            )

    async def create_observation(self, observation: Observation):
        """Create observation with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.create_observation_async(
                session, observation
            )

    async def delete_observation(self, observation_id: int):
        """Delete observation with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.delete_observation_async(
                session, observation_id
            )

    async def get_hints(
        self, category: Optional[str] = None, user_id: Optional[str] = None
    ):
        """Get hints with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.get_hints_async(session, category, user_id)

    async def create_hint(self, hint: Hint):
        """Create hint with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.create_hint_async(session, hint)

    async def delete_hint(self, hint_id: int):
        """Delete hint with session management."""
        async with await self._get_session() as session:
            return await self.memory_service.delete_hint_async(session, hint_id)

    async def search_memories(
        self, query: str, memory_types: Optional[List[str]] = None
    ):
        """Search memories with session management."""
        async with await self._get_session() as session:
            raw_results = await self.search_service.combined_search_async(
                session, query, memory_types=memory_types
            )

            # Transform results to format expected by web template
            formatted_results = []
            for result in raw_results:
                memory_type = result.get("memory_type", "unknown")
                content = result.get("content", {})
                metadata = result.get("metadata", {})

                # Create a flattened result object for the template
                formatted_result = {
                    "type": memory_type,
                    "id": metadata.get("id"),
                    "score": result.get("final_score", result.get("similarity", 0)),
                    "user_id": metadata.get("user_id"),
                    "created_at": metadata.get("created_at"),
                    "tags": metadata.get("tags", []),
                }

                # Add type-specific fields
                if memory_type == "alias":
                    formatted_result.update(
                        {
                            "source": content.get("source", ""),
                            "target": content.get("target", ""),
                            "bidirectional": content.get("bidirectional", True),
                        }
                    )
                elif memory_type == "note":
                    formatted_result.update(
                        {
                            "title": content.get("title", "Untitled"),
                            "content": content.get("content", ""),
                            "category": content.get("category"),
                        }
                    )
                elif memory_type == "observation":
                    formatted_result.update(
                        {
                            "content": content.get("content", ""),
                            "entity_type": content.get("entity_type", ""),
                            "entity_id": content.get("entity_id", ""),
                            "context": content.get("context", {}),
                        }
                    )
                elif memory_type == "hint":
                    formatted_result.update(
                        {
                            "content": content.get("content", ""),
                            "category": content.get("category", "General"),
                            "priority": content.get("priority", 1),
                            "workflow_context": content.get("workflow_context"),
                        }
                    )

                # Convert created_at string to datetime object if needed
                if formatted_result.get("created_at") and isinstance(
                    formatted_result["created_at"], str
                ):
                    try:
                        from datetime import datetime

                        formatted_result["created_at"] = datetime.fromisoformat(
                            formatted_result["created_at"].replace("Z", "+00:00")
                        )
                    except Exception:
                        pass  # Keep as string if parsing fails

                formatted_results.append(formatted_result)

            return formatted_results

    def _setup_templates(self):
        """Setup Jinja2 templates and static files."""
        # Get the web directory path
        web_dir = Path(__file__).parent
        templates_dir = web_dir / "templates"
        static_dir = web_dir / "static"

        # Create directories if they don't exist
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)

        # Setup Jinja2 templates
        self.templates = Jinja2Templates(directory=str(templates_dir))

        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    def _get_mount_path(self, request: Request) -> str:
        """Get the mount path from the request scope."""
        # Check if this app is mounted under a sub-path
        script_name = request.scope.get("root_path", "")

        # If we have a root_path, that's our mount path
        if script_name:
            return script_name.rstrip("/")

        # Fallback: empty string means mounted at root
        return ""

    def _setup_routes(self):
        """Setup all web routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard showing overview of all memory types."""
            try:
                # Get counts for each memory type
                aliases = await self.get_aliases()
                notes = await self.get_notes()
                observations = await self.get_observations()
                hints = await self.get_hints()

                stats = {
                    "aliases": len(aliases),
                    "notes": len(notes),
                    "observations": len(observations),
                    "hints": len(hints),
                    "total": len(aliases) + len(notes) + len(observations) + len(hints),
                }

                # Get the mount path from the request scope
                mount_path = self._get_mount_path(request)

                return self.templates.TemplateResponse(
                    "dashboard.html",
                    {"request": request, "stats": stats, "mount_path": mount_path},
                )
            except Exception as e:
                logger.error(f"Dashboard error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        # Aliases routes
        @self.app.get("/aliases", response_class=HTMLResponse)
        async def aliases_page(request: Request):
            """Aliases management page."""
            try:
                aliases = await self.get_aliases()
                mount_path = self._get_mount_path(request)
                return self.templates.TemplateResponse(
                    "aliases.html",
                    {"request": request, "aliases": aliases, "mount_path": mount_path},
                )
            except Exception as e:
                logger.error(f"Aliases page error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        @self.app.post("/aliases/create")
        async def create_alias_route(
            request: Request,
            source: str = Form(...),
            target: str = Form(...),
            user_id: Optional[str] = Form(None),
            bidirectional: bool = Form(True),
        ):
            """Create a new alias."""
            try:
                alias = Alias(
                    source=source,
                    target=target,
                    user_id=user_id,
                    bidirectional=bidirectional,
                )
                await self.create_alias(alias)
                mount_path = self._get_mount_path(request)
                return RedirectResponse(url=f"{mount_path}/aliases", status_code=303)
            except Exception as e:
                logger.error(f"Create alias error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        @self.app.post("/aliases/{alias_id}/delete")
        async def delete_alias_route(request: Request, alias_id: int):
            """Delete an alias."""
            try:
                await self.delete_alias(alias_id)
                mount_path = self._get_mount_path(request)
                return RedirectResponse(url=f"{mount_path}/aliases", status_code=303)
            except Exception as e:
                logger.error(f"Delete alias error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        # Notes routes
        @self.app.get("/notes", response_class=HTMLResponse)
        async def notes_page(request: Request):
            """Notes management page."""
            try:
                notes = await self.get_notes()
                mount_path = self._get_mount_path(request)
                return self.templates.TemplateResponse(
                    "notes.html",
                    {"request": request, "notes": notes, "mount_path": mount_path},
                )
            except Exception as e:
                logger.error(f"Notes page error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        @self.app.post("/notes/create")
        async def create_note_route(
            request: Request,
            title: str = Form(...),
            content: str = Form(...),
            category: Optional[str] = Form(None),
            user_id: Optional[str] = Form(None),
        ):
            """Create a new note."""
            try:
                note = Note(
                    title=title, content=content, category=category, user_id=user_id
                )
                await self.create_note(note)
                mount_path = self._get_mount_path(request)
                return RedirectResponse(url=f"{mount_path}/notes", status_code=303)
            except Exception as e:
                logger.error(f"Create note error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        @self.app.post("/notes/{note_id}/delete")
        async def delete_note_route(request: Request, note_id: int):
            """Delete a note."""
            try:
                await self.delete_note(note_id)
                mount_path = self._get_mount_path(request)
                return RedirectResponse(url=f"{mount_path}/notes", status_code=303)
            except Exception as e:
                logger.error(f"Delete note error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        # Observations routes
        @self.app.get("/observations", response_class=HTMLResponse)
        async def observations_page(request: Request):
            """Observations management page."""
            try:
                observations = await self.get_observations()
                mount_path = self._get_mount_path(request)
                return self.templates.TemplateResponse(
                    "observations.html",
                    {
                        "request": request,
                        "observations": observations,
                        "mount_path": mount_path,
                    },
                )
            except Exception as e:
                logger.error(f"Observations page error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        @self.app.post("/observations/create")
        async def create_observation_route(
            request: Request,
            content: str = Form(...),
            entity_type: str = Form(...),
            entity_id: str = Form(...),
            user_id: Optional[str] = Form(None),
        ):
            """Create a new observation."""
            try:
                observation = Observation(
                    content=content,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    user_id=user_id,
                )
                await self.create_observation(observation)
                mount_path = self._get_mount_path(request)
                return RedirectResponse(
                    url=f"{mount_path}/observations", status_code=303
                )
            except Exception as e:
                logger.error(f"Create observation error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        @self.app.post("/observations/{observation_id}/delete")
        async def delete_observation_route(request: Request, observation_id: int):
            """Delete an observation."""
            try:
                await self.delete_observation(observation_id)
                mount_path = self._get_mount_path(request)
                return RedirectResponse(
                    url=f"{mount_path}/observations", status_code=303
                )
            except Exception as e:
                logger.error(f"Delete observation error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        # Hints routes
        @self.app.get("/hints", response_class=HTMLResponse)
        async def hints_page(request: Request):
            """Hints management page."""
            try:
                hints = await self.get_hints()
                mount_path = self._get_mount_path(request)
                return self.templates.TemplateResponse(
                    "hints.html",
                    {"request": request, "hints": hints, "mount_path": mount_path},
                )
            except Exception as e:
                logger.error(f"Hints page error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        @self.app.post("/hints/create")
        async def create_hint_route(
            request: Request,
            content: str = Form(...),
            category: str = Form(...),
            priority: int = Form(1),
            workflow_context: Optional[str] = Form(None),
            user_id: Optional[str] = Form(None),
        ):
            """Create a new hint."""
            try:
                hint = Hint(
                    content=content,
                    category=category,
                    priority=priority,
                    workflow_context=workflow_context,
                    user_id=user_id,
                )
                await self.create_hint(hint)
                mount_path = self._get_mount_path(request)
                return RedirectResponse(url=f"{mount_path}/hints", status_code=303)
            except Exception as e:
                logger.error(f"Create hint error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        @self.app.post("/hints/{hint_id}/delete")
        async def delete_hint_route(request: Request, hint_id: int):
            """Delete a hint."""
            try:
                await self.delete_hint(hint_id)
                mount_path = self._get_mount_path(request)
                return RedirectResponse(url=f"{mount_path}/hints", status_code=303)
            except Exception as e:
                logger.error(f"Delete hint error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

        # Search routes
        @self.app.get("/search", response_class=HTMLResponse)
        async def search_page(
            request: Request, q: Optional[str] = None, types: List[str] = None
        ):
            """Search interface."""
            results = []
            if q:
                try:
                    # Get type filters from query parameters
                    memory_types = (
                        request.query_params.getlist("types")
                        if request.query_params.getlist("types")
                        else None
                    )
                    results = await self.search_memories(q, memory_types)
                except Exception as e:
                    logger.error(f"Search error: {e}")
                    results = []

            mount_path = self._get_mount_path(request)
            return self.templates.TemplateResponse(
                "search.html",
                {
                    "request": request,
                    "query": q or "",
                    "results": results,
                    "mount_path": mount_path,
                },
            )

        @self.app.get("/search/export")
        async def export_search_results(request: Request, q: str, format: str = "json"):
            """Export search results."""
            try:
                # Get type filters from query parameters
                memory_types = (
                    request.query_params.getlist("types")
                    if request.query_params.getlist("types")
                    else None
                )
                results = await self.search_memories(q, memory_types)

                if format.lower() == "csv":
                    # Return CSV format
                    import csv
                    import io

                    from fastapi.responses import StreamingResponse

                    output = io.StringIO()
                    writer = csv.writer(output)

                    # Write header
                    writer.writerow(
                        [
                            "Type",
                            "Title/Source",
                            "Content/Target",
                            "User",
                            "Created",
                            "Score",
                        ]
                    )

                    # Write data
                    for result in results:
                        writer.writerow(
                            [
                                result.get("type", ""),
                                result.get("title", result.get("source", "")),
                                result.get("content", result.get("target", ""))[
                                    :200
                                ],  # Truncate content
                                result.get("user_id", ""),
                                result.get("created_at", ""),
                                result.get("score", ""),
                            ]
                        )

                    output.seek(0)
                    return StreamingResponse(
                        io.BytesIO(output.getvalue().encode()),
                        media_type="text/csv",
                        headers={
                            "Content-Disposition": f"attachment; filename=search_results_{q[:20]}.csv"
                        },
                    )
                else:
                    # Return JSON format (default)
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        content={"query": q, "results": results, "count": len(results)},
                        headers={
                            "Content-Disposition": f"attachment; filename=search_results_{q[:20]}.json"
                        },
                    )

            except Exception as e:
                logger.error(f"Export search results error: {e}")
                raise HTTPException(status_code=500, detail=str(e)) from None

    async def start_server(
        self, host: Optional[str] = None, port: Optional[int] = None
    ):
        """Start the web server."""
        # Initialize database
        await self.db_manager.initialize_database_async()

        # Use config values if not provided
        server_host = host or self.config.server.host
        server_port = port or self.config.server.web_port

        logger.info(f"Starting web UI server on {server_host}:{server_port}")

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


def create_web_server(config: Config) -> WebUIServer:
    """Create a web UI server instance."""
    return WebUIServer(config)
