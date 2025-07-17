"""Search service for RAG capabilities and semantic search."""

import logging
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from ..database.connection import get_database_manager
from ..database.models import AliasDB, HintDB, NoteDB, ObservationDB
from ..database.repositories import EmbeddingRepository
from ..models.alias import Alias
from ..models.base import BaseMemory
from ..models.hint import Hint
from ..models.note import Note
from ..models.observation import Observation

logger = logging.getLogger(__name__)


class SearchServiceError(Exception):
    """Base exception for search service errors."""

    pass


class EmbeddingError(SearchServiceError):
    """Exception raised for embedding generation errors."""

    pass


class SearchService:
    """Service for semantic search and RAG capabilities."""

    def __init__(self, config=None):
        """Initialize the search service with configuration.

        Args:
            config: Configuration object with embedding settings, or None to use defaults
        """
        from ..config import get_config

        if config is None:
            config = get_config()

        self.config = config
        self.model_name = config.embedding.model_name
        self.similarity_threshold = config.embedding.similarity_threshold
        self.max_results = config.embedding.max_results
        self.cache_embeddings = config.embedding.cache_embeddings

        self._model = None
        self.embedding_repo = EmbeddingRepository()
        self.db_manager = get_database_manager()

        # Memory type mapping for database queries
        self.memory_type_models = {
            "alias": AliasDB,
            "note": NoteDB,
            "observation": ObservationDB,
            "hint": HintDB,
        }

        logger.info(f"SearchService initialized with model: {self.model_name}")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"Max results: {self.max_results}")
        logger.info(f"Cache embeddings: {self.cache_embeddings}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Successfully loaded embedding model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise EmbeddingError(f"Failed to load embedding model: {e}") from None
        return self._model

    def generate_embeddings(
        self, texts: Union[str, List[str]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Generate embeddings for text(s).

        Args:
            texts: Single text string or list of text strings

        Returns:
            Numpy array of embeddings or list of arrays
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                single_text = True
            else:
                single_text = False

            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                logger.warning("No valid texts provided for embedding generation")
                return np.array([]) if single_text else []

            logger.debug(f"Generating embeddings for {len(valid_texts)} texts")
            embeddings = self.model.encode(valid_texts, convert_to_numpy=True)

            if single_text:
                return embeddings[0] if len(embeddings) > 0 else np.array([])

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from None

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize embedding for database storage."""
        try:
            return pickle.dumps(embedding)
        except Exception as e:
            logger.error(f"Failed to serialize embedding: {e}")
            raise EmbeddingError(f"Failed to serialize embedding: {e}") from None

    def _deserialize_embedding(self, data: bytes) -> np.ndarray:
        """Deserialize embedding from database storage."""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize embedding: {e}")
            raise EmbeddingError(f"Failed to deserialize embedding: {e}") from None

    def _extract_text_from_memory(self, memory: BaseMemory) -> str:
        """Extract searchable text from memory object."""
        if isinstance(memory, Alias):
            return f"{memory.source} {memory.target}"
        elif isinstance(memory, Note):
            return f"{memory.title} {memory.content}"
        elif isinstance(memory, Observation):
            return memory.content
        elif isinstance(memory, Hint):
            return memory.content
        else:
            return str(memory)

    def index_memory(self, session: Session, memory: BaseMemory) -> bool:
        """Generate and store embedding for a memory object.

        Args:
            session: Database session
            memory: Memory object to index

        Returns:
            True if successful, False otherwise
        """
        try:
            if not memory.id:
                logger.warning("Cannot index memory without ID")
                return False

            # Extract text for embedding
            text = self._extract_text_from_memory(memory)
            if not text.strip():
                logger.warning(f"No text content found for memory {memory.id}")
                return False

            # Generate embedding
            embedding = self.generate_embeddings(text)
            if embedding.size == 0:
                logger.warning(f"Empty embedding generated for memory {memory.id}")
                return False

            # Serialize embedding
            embedding_data = self._serialize_embedding(embedding)

            # Determine memory type
            memory_type = memory.__class__.__name__.lower().replace("db", "")
            if memory_type.endswith("db"):
                memory_type = memory_type[:-2]

            # Store or update embedding
            existing_embedding = self.embedding_repo.get_by_memory(
                session, memory_type, memory.id
            )
            if existing_embedding:
                self.embedding_repo.update(
                    session, existing_embedding.id, embedding=embedding_data
                )
                logger.debug(f"Updated embedding for {memory_type} {memory.id}")
            else:
                self.embedding_repo.create(
                    session,
                    memory_type=memory_type,
                    memory_id=memory.id,
                    embedding=embedding_data,
                )
                logger.debug(f"Created embedding for {memory_type} {memory.id}")

            return True

        except Exception as e:
            logger.error(f"Failed to index memory {memory.id}: {e}")
            return False

    async def index_memory_async(
        self, session: AsyncSession, memory: BaseMemory
    ) -> bool:
        """Generate and store embedding for a memory object (async).

        Args:
            session: Database session
            memory: Memory object to index

        Returns:
            True if successful, False otherwise
        """
        try:
            if not memory.id:
                logger.warning("Cannot index memory without ID")
                return False

            # Extract text for embedding
            text = self._extract_text_from_memory(memory)
            if not text.strip():
                logger.warning(f"No text content found for memory {memory.id}")
                return False

            # Generate embedding
            embedding = self.generate_embeddings(text)
            if embedding.size == 0:
                logger.warning(f"Empty embedding generated for memory {memory.id}")
                return False

            # Serialize embedding
            embedding_data = self._serialize_embedding(embedding)

            # Determine memory type
            memory_type = memory.__class__.__name__.lower().replace("db", "")
            if memory_type.endswith("db"):
                memory_type = memory_type[:-2]

            # Store or update embedding
            existing_embedding = await self.embedding_repo.get_by_memory_async(
                session, memory_type, memory.id
            )
            if existing_embedding:
                await self.embedding_repo.update_async(
                    session, existing_embedding.id, embedding=embedding_data
                )
                logger.debug(f"Updated embedding for {memory_type} {memory.id}")
            else:
                await self.embedding_repo.create_async(
                    session,
                    memory_type=memory_type,
                    memory_id=memory.id,
                    embedding=embedding_data,
                )
                logger.debug(f"Created embedding for {memory_type} {memory.id}")

            return True

        except Exception as e:
            logger.error(f"Failed to index memory {memory.id}: {e}")
            return False

    def semantic_search(
        self,
        session: Session,
        query: str,
        memory_types: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search across memory types.

        Args:
            session: Database session
            query: Search query text
            memory_types: List of memory types to search ('alias', 'note', 'observation', 'hint')
            user_id: Optional user ID for filtering
            limit: Maximum number of results to return
            similarity_threshold: Minimum cosine similarity threshold

        Returns:
            List of search results with similarity scores
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided for semantic search")
                return []

            # Default to all memory types if none specified
            if memory_types is None:
                memory_types = ["alias", "note", "observation", "hint"]

            # Generate query embedding
            query_embedding = self.generate_embeddings(query)
            if query_embedding.size == 0:
                logger.warning("Failed to generate query embedding")
                return []

            results = []

            for memory_type in memory_types:
                if memory_type not in self.memory_type_models:
                    logger.warning(f"Unknown memory type: {memory_type}")
                    continue

                try:
                    # Get embeddings for this memory type
                    embeddings = self.embedding_repo.get_by_type(session, memory_type)

                    if not embeddings:
                        logger.debug(
                            f"No embeddings found for memory type: {memory_type}"
                        )
                        continue

                    # Calculate similarities
                    memory_embeddings = []
                    memory_ids = []

                    for emb_record in embeddings:
                        try:
                            emb_vector = self._deserialize_embedding(
                                emb_record.embedding
                            )
                            memory_embeddings.append(emb_vector)
                            memory_ids.append(emb_record.memory_id)
                        except Exception as e:
                            logger.warning(
                                f"Failed to deserialize embedding {emb_record.id}: {e}"
                            )
                            continue

                    if not memory_embeddings:
                        logger.debug(
                            f"No valid embeddings for memory type: {memory_type}"
                        )
                        continue

                    # Calculate cosine similarities
                    similarities = cosine_similarity(
                        [query_embedding], memory_embeddings
                    )[0]

                    # Get memory objects and filter by user if specified
                    memory_model = self.memory_type_models[memory_type]

                    for _i, (memory_id, similarity) in enumerate(
                        zip(memory_ids, similarities)
                    ):
                        if similarity < similarity_threshold:
                            continue

                        # Get the actual memory object
                        memory_obj = (
                            session.query(memory_model)
                            .filter(memory_model.id == memory_id)
                            .first()
                        )
                        if not memory_obj:
                            continue

                        # Filter by user if specified
                        if (
                            user_id
                            and hasattr(memory_obj, "user_id")
                            and memory_obj.user_id != user_id
                        ):
                            continue

                        # Create result entry
                        result = {
                            "memory_type": memory_type,
                            "memory_id": memory_id,
                            "similarity": float(similarity),
                            "content": self._extract_memory_content(
                                memory_obj, memory_type
                            ),
                            "metadata": self._extract_memory_metadata(
                                memory_obj, memory_type
                            ),
                        }
                        results.append(result)

                except Exception as e:
                    logger.error(f"Error searching memory type {memory_type}: {e}")
                    continue

            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:limit]

            logger.info(
                f"Semantic search for '{query}' returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            raise SearchServiceError(
                f"Failed to perform semantic search: {e}"
            ) from None

    async def semantic_search_async(
        self,
        session: AsyncSession,
        query: str,
        memory_types: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search across memory types (async).

        Args:
            session: Database session
            query: Search query text
            memory_types: List of memory types to search ('alias', 'note', 'observation', 'hint')
            user_id: Optional user ID for filtering
            limit: Maximum number of results to return
            similarity_threshold: Minimum cosine similarity threshold

        Returns:
            List of search results with similarity scores
        """
        try:
            # Use configured defaults if not provided
            if limit is None:
                limit = self.max_results
            if similarity_threshold is None:
                similarity_threshold = self.similarity_threshold

            if not query.strip():
                logger.warning("Empty query provided for semantic search")
                return []

            # Default to all memory types if none specified
            if memory_types is None:
                memory_types = ["alias", "note", "observation", "hint"]

            # Generate query embedding
            query_embedding = self.generate_embeddings(query)
            if query_embedding.size == 0:
                logger.warning("Failed to generate query embedding")
                return []

            results = []

            for memory_type in memory_types:
                if memory_type not in self.memory_type_models:
                    logger.warning(f"Unknown memory type: {memory_type}")
                    continue

                try:
                    # Get embeddings for this memory type
                    embeddings = await self.embedding_repo.get_by_type_async(
                        session, memory_type
                    )

                    if not embeddings:
                        logger.debug(
                            f"No embeddings found for memory type: {memory_type}"
                        )
                        continue

                    # Calculate similarities
                    memory_embeddings = []
                    memory_ids = []

                    for emb_record in embeddings:
                        try:
                            emb_vector = self._deserialize_embedding(
                                emb_record.embedding
                            )
                            memory_embeddings.append(emb_vector)
                            memory_ids.append(emb_record.memory_id)
                        except Exception as e:
                            logger.warning(
                                f"Failed to deserialize embedding {emb_record.id}: {e}"
                            )
                            continue

                    if not memory_embeddings:
                        logger.debug(
                            f"No valid embeddings for memory type: {memory_type}"
                        )
                        continue

                    # Calculate cosine similarities
                    similarities = cosine_similarity(
                        [query_embedding], memory_embeddings
                    )[0]

                    # Get memory objects and filter by user if specified
                    memory_model = self.memory_type_models[memory_type]

                    for _i, (memory_id, similarity) in enumerate(
                        zip(memory_ids, similarities)
                    ):
                        if similarity < similarity_threshold:
                            continue

                        # Get the actual memory object
                        from sqlalchemy import select

                        query = select(memory_model).where(memory_model.id == memory_id)
                        result = await session.execute(query)
                        memory_obj = result.scalar_one_or_none()
                        if not memory_obj:
                            continue

                        # Filter by user if specified
                        if (
                            user_id
                            and hasattr(memory_obj, "user_id")
                            and memory_obj.user_id != user_id
                        ):
                            continue

                        # Create result entry
                        result = {
                            "memory_type": memory_type,
                            "memory_id": memory_id,
                            "similarity": float(similarity),
                            "content": self._extract_memory_content(
                                memory_obj, memory_type
                            ),
                            "metadata": self._extract_memory_metadata(
                                memory_obj, memory_type
                            ),
                        }
                        results.append(result)

                except Exception as e:
                    logger.error(f"Error searching memory type {memory_type}: {e}")
                    continue

            # Sort by similarity score (descending) and limit results
            results.sort(key=lambda x: x["similarity"], reverse=True)
            results = results[:limit]

            logger.info(
                f"Semantic search for '{query}' returned {len(results)} results"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            raise SearchServiceError(
                f"Failed to perform semantic search: {e}"
            ) from None

    def _extract_memory_content(
        self, memory_obj: Any, memory_type: str
    ) -> Dict[str, Any]:
        """Extract content from memory object for search results."""
        if memory_type == "alias":
            return {
                "source": memory_obj.source,
                "target": memory_obj.target,
                "bidirectional": memory_obj.bidirectional,
            }
        elif memory_type == "note":
            return {
                "title": memory_obj.title,
                "content": memory_obj.content,
                "category": memory_obj.category,
            }
        elif memory_type == "observation":
            return {
                "content": memory_obj.content,
                "entity_type": memory_obj.entity_type,
                "entity_id": memory_obj.entity_id,
                "context": memory_obj.context,
            }
        elif memory_type == "hint":
            return {
                "content": memory_obj.content,
                "category": memory_obj.category,
                "priority": memory_obj.priority,
                "workflow_context": memory_obj.workflow_context,
            }
        else:
            return {}

    def _extract_memory_metadata(
        self, memory_obj: Any, memory_type: str
    ) -> Dict[str, Any]:
        """Extract metadata from memory object for search results."""
        metadata = {
            "id": memory_obj.id,
            "user_id": getattr(memory_obj, "user_id", None),
            "created_at": memory_obj.created_at.isoformat()
            if memory_obj.created_at
            else None,
            "updated_at": memory_obj.updated_at.isoformat()
            if memory_obj.updated_at
            else None,
            "tags": getattr(memory_obj, "tags", []),
        }
        return metadata

    def exact_search(
        self,
        session: Session,
        query: str,
        memory_types: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Perform exact text search across memory types.

        Args:
            session: Database session
            query: Search query text
            memory_types: List of memory types to search
            user_id: Optional user ID for filtering
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided for exact search")
                return []

            # Default to all memory types if none specified
            if memory_types is None:
                memory_types = ["alias", "note", "observation", "hint"]

            results = []

            for memory_type in memory_types:
                if memory_type not in self.memory_type_models:
                    logger.warning(f"Unknown memory type: {memory_type}")
                    continue

                try:
                    memory_model = self.memory_type_models[memory_type]
                    query_obj = session.query(memory_model)

                    # Filter by user if specified
                    if user_id:
                        query_obj = query_obj.filter(memory_model.user_id == user_id)

                    # Add text search filters based on memory type
                    if memory_type == "alias":
                        query_obj = query_obj.filter(
                            (memory_model.source.ilike(f"%{query}%"))
                            | (memory_model.target.ilike(f"%{query}%"))
                        )
                    elif memory_type == "note":
                        query_obj = query_obj.filter(
                            (memory_model.title.ilike(f"%{query}%"))
                            | (memory_model.content.ilike(f"%{query}%"))
                        )
                    elif memory_type == "observation":
                        query_obj = query_obj.filter(
                            memory_model.content.ilike(f"%{query}%")
                        )
                    elif memory_type == "hint":
                        query_obj = query_obj.filter(
                            memory_model.content.ilike(f"%{query}%")
                        )

                    # Execute query and process results
                    memory_objects = query_obj.limit(limit).all()

                    for memory_obj in memory_objects:
                        result = {
                            "memory_type": memory_type,
                            "memory_id": memory_obj.id,
                            "similarity": 1.0,  # Exact match gets full score
                            "content": self._extract_memory_content(
                                memory_obj, memory_type
                            ),
                            "metadata": self._extract_memory_metadata(
                                memory_obj, memory_type
                            ),
                        }
                        results.append(result)

                except Exception as e:
                    logger.error(
                        f"Error in exact search for memory type {memory_type}: {e}"
                    )
                    continue

            # Limit total results
            results = results[:limit]

            logger.info(f"Exact search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Failed to perform exact search: {e}")
            raise SearchServiceError(f"Failed to perform exact search: {e}") from None

    async def exact_search_async(
        self,
        session: AsyncSession,
        query: str,
        memory_types: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Perform exact text search across memory types (async).

        Args:
            session: Database session
            query: Search query text
            memory_types: List of memory types to search
            user_id: Optional user ID for filtering
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        try:
            if not query.strip():
                logger.warning("Empty query provided for exact search")
                return []

            # Default to all memory types if none specified
            if memory_types is None:
                memory_types = ["alias", "note", "observation", "hint"]

            results = []

            for memory_type in memory_types:
                if memory_type not in self.memory_type_models:
                    logger.warning(f"Unknown memory type: {memory_type}")
                    continue

                try:
                    from sqlalchemy import or_, select

                    memory_model = self.memory_type_models[memory_type]
                    query_obj = select(memory_model)

                    # Filter by user if specified
                    if user_id:
                        query_obj = query_obj.where(memory_model.user_id == user_id)

                    # Add text search filters based on memory type
                    if memory_type == "alias":
                        query_obj = query_obj.where(
                            or_(
                                memory_model.source.ilike(f"%{query}%"),
                                memory_model.target.ilike(f"%{query}%"),
                            )
                        )
                    elif memory_type == "note":
                        query_obj = query_obj.where(
                            or_(
                                memory_model.title.ilike(f"%{query}%"),
                                memory_model.content.ilike(f"%{query}%"),
                            )
                        )
                    elif memory_type == "observation":
                        query_obj = query_obj.where(
                            memory_model.content.ilike(f"%{query}%")
                        )
                    elif memory_type == "hint":
                        query_obj = query_obj.where(
                            memory_model.content.ilike(f"%{query}%")
                        )

                    # Execute query and process results
                    result = await session.execute(query_obj.limit(limit))
                    memory_objects = result.scalars().all()

                    for memory_obj in memory_objects:
                        result = {
                            "memory_type": memory_type,
                            "memory_id": memory_obj.id,
                            "similarity": 1.0,  # Exact match gets full score
                            "content": self._extract_memory_content(
                                memory_obj, memory_type
                            ),
                            "metadata": self._extract_memory_metadata(
                                memory_obj, memory_type
                            ),
                        }
                        results.append(result)

                except Exception as e:
                    logger.error(
                        f"Error in exact search for memory type {memory_type}: {e}"
                    )
                    continue

            # Limit total results
            results = results[:limit]

            logger.info(f"Exact search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Failed to perform exact search: {e}")
            raise SearchServiceError(f"Failed to perform exact search: {e}") from None

    def combined_search(
        self,
        session: Session,
        query: str,
        memory_types: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        limit: int = 10,
        semantic_weight: float = 0.7,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """Perform combined semantic and exact search with ranking.

        Args:
            session: Database session
            query: Search query text
            memory_types: List of memory types to search
            user_id: Optional user ID for filtering
            limit: Maximum number of results to return
            semantic_weight: Weight for semantic search results (0.0-1.0)
            similarity_threshold: Minimum cosine similarity threshold for semantic search

        Returns:
            List of search results with combined ranking
        """
        try:
            # Perform both searches
            semantic_results = self.semantic_search(
                session, query, memory_types, user_id, limit * 2, similarity_threshold
            )
            exact_results = self.exact_search(
                session, query, memory_types, user_id, limit * 2
            )

            # Combine and deduplicate results
            combined_results = {}

            # Add semantic results with weighted scores
            for result in semantic_results:
                key = (result["memory_type"], result["memory_id"])
                result["final_score"] = result["similarity"] * semantic_weight
                result["search_type"] = "semantic"
                combined_results[key] = result

            # Add exact results with weighted scores, merge if already exists
            exact_weight = 1.0 - semantic_weight
            for result in exact_results:
                key = (result["memory_type"], result["memory_id"])
                if key in combined_results:
                    # Combine scores for items found in both searches
                    combined_results[key]["final_score"] += exact_weight
                    combined_results[key]["search_type"] = "combined"
                else:
                    result["final_score"] = exact_weight
                    result["search_type"] = "exact"
                    combined_results[key] = result

            # Sort by final score and limit results
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x["final_score"], reverse=True)
            final_results = final_results[:limit]

            logger.info(
                f"Combined search for '{query}' returned {len(final_results)} results"
            )
            return final_results

        except Exception as e:
            logger.error(f"Failed to perform combined search: {e}")
            raise SearchServiceError(
                f"Failed to perform combined search: {e}"
            ) from None

    async def combined_search_async(
        self,
        session: AsyncSession,
        query: str,
        memory_types: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        semantic_weight: float = 0.7,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform combined semantic and exact search with ranking (async).

        Args:
            session: Database session
            query: Search query text
            memory_types: List of memory types to search
            user_id: Optional user ID for filtering
            limit: Maximum number of results to return
            semantic_weight: Weight for semantic search results (0.0-1.0)
            similarity_threshold: Minimum cosine similarity threshold for semantic search

        Returns:
            List of search results with combined ranking
        """
        try:
            # Use configured defaults if not provided
            if limit is None:
                limit = self.max_results
            if similarity_threshold is None:
                similarity_threshold = self.similarity_threshold

            # Perform both searches
            semantic_results = await self.semantic_search_async(
                session, query, memory_types, user_id, limit * 2, similarity_threshold
            )
            exact_results = await self.exact_search_async(
                session, query, memory_types, user_id, limit * 2
            )

            # Combine and deduplicate results
            combined_results = {}

            # Add semantic results with weighted scores
            for result in semantic_results:
                key = (result["memory_type"], result["memory_id"])
                result["final_score"] = result["similarity"] * semantic_weight
                result["search_type"] = "semantic"
                combined_results[key] = result

            # Add exact results with weighted scores, merge if already exists
            exact_weight = 1.0 - semantic_weight
            for result in exact_results:
                key = (result["memory_type"], result["memory_id"])
                if key in combined_results:
                    # Combine scores for items found in both searches
                    combined_results[key]["final_score"] += exact_weight
                    combined_results[key]["search_type"] = "combined"
                else:
                    result["final_score"] = exact_weight
                    result["search_type"] = "exact"
                    combined_results[key] = result

            # Sort by final score and limit results
            final_results = list(combined_results.values())
            final_results.sort(key=lambda x: x["final_score"], reverse=True)
            final_results = final_results[:limit]

            logger.info(
                f"Combined search for '{query}' returned {len(final_results)} results"
            )
            return final_results

        except Exception as e:
            logger.error(f"Failed to perform combined search: {e}")
            raise SearchServiceError(
                f"Failed to perform combined search: {e}"
            ) from None

    def reindex_all_memories(self, session: Session) -> Dict[str, int]:
        """Reindex all memories in the database.

        Args:
            session: Database session

        Returns:
            Dictionary with counts of indexed memories by type
        """
        try:
            counts = {"alias": 0, "note": 0, "observation": 0, "hint": 0}

            # Clear existing embeddings
            self.embedding_repo.delete_all(session)
            logger.info("Cleared all existing embeddings")

            # Reindex each memory type
            for memory_type, model_class in self.memory_type_models.items():
                try:
                    memories = session.query(model_class).all()

                    for memory_obj in memories:
                        # Convert to Pydantic model for indexing
                        if memory_type == "alias":
                            from ..models.alias import Alias

                            memory = Alias(
                                id=memory_obj.id,
                                user_id=memory_obj.user_id,
                                source=memory_obj.source,
                                target=memory_obj.target,
                                bidirectional=memory_obj.bidirectional,
                                tags=memory_obj.tags or [],
                                created_at=memory_obj.created_at,
                                updated_at=memory_obj.updated_at,
                            )
                        elif memory_type == "note":
                            from ..models.note import Note

                            memory = Note(
                                id=memory_obj.id,
                                user_id=memory_obj.user_id,
                                title=memory_obj.title,
                                content=memory_obj.content,
                                category=memory_obj.category,
                                tags=memory_obj.tags or [],
                                created_at=memory_obj.created_at,
                                updated_at=memory_obj.updated_at,
                            )
                        elif memory_type == "observation":
                            from ..models.observation import Observation

                            memory = Observation(
                                id=memory_obj.id,
                                user_id=memory_obj.user_id,
                                content=memory_obj.content,
                                entity_type=memory_obj.entity_type,
                                entity_id=memory_obj.entity_id,
                                context=memory_obj.context or {},
                                tags=memory_obj.tags or [],
                                created_at=memory_obj.created_at,
                                updated_at=memory_obj.updated_at,
                            )
                        elif memory_type == "hint":
                            from ..models.hint import Hint

                            memory = Hint(
                                id=memory_obj.id,
                                user_id=memory_obj.user_id,
                                content=memory_obj.content,
                                category=memory_obj.category,
                                priority=memory_obj.priority,
                                workflow_context=memory_obj.workflow_context,
                                tags=memory_obj.tags or [],
                                created_at=memory_obj.created_at,
                                updated_at=memory_obj.updated_at,
                            )
                        else:
                            continue

                        if self.index_memory(session, memory):
                            counts[memory_type] += 1

                except Exception as e:
                    logger.error(f"Failed to reindex {memory_type} memories: {e}")
                    continue

            session.commit()
            logger.info(f"Reindexed memories: {counts}")
            return counts

        except Exception as e:
            logger.error(f"Failed to reindex all memories: {e}")
            session.rollback()
            raise SearchServiceError(f"Failed to reindex all memories: {e}") from None

    async def reindex_all_memories_async(self, session: AsyncSession) -> Dict[str, int]:
        """Reindex all memories in the database (async).

        Args:
            session: Database session

        Returns:
            Dictionary with counts of indexed memories by type
        """
        try:
            counts = {"alias": 0, "note": 0, "observation": 0, "hint": 0}

            # Clear existing embeddings
            await self.embedding_repo.delete_all_async(session)
            logger.info("Cleared all existing embeddings")

            # Reindex each memory type
            for memory_type, model_class in self.memory_type_models.items():
                try:
                    from sqlalchemy import select

                    query = select(model_class)
                    result = await session.execute(query)
                    memories = result.scalars().all()

                    for memory_obj in memories:
                        # Convert to Pydantic model for indexing
                        if memory_type == "alias":
                            from ..models.alias import Alias

                            memory = Alias(
                                id=memory_obj.id,
                                user_id=memory_obj.user_id,
                                source=memory_obj.source,
                                target=memory_obj.target,
                                bidirectional=memory_obj.bidirectional,
                                tags=memory_obj.tags or [],
                                created_at=memory_obj.created_at,
                                updated_at=memory_obj.updated_at,
                            )
                        elif memory_type == "note":
                            from ..models.note import Note

                            memory = Note(
                                id=memory_obj.id,
                                user_id=memory_obj.user_id,
                                title=memory_obj.title,
                                content=memory_obj.content,
                                category=memory_obj.category,
                                tags=memory_obj.tags or [],
                                created_at=memory_obj.created_at,
                                updated_at=memory_obj.updated_at,
                            )
                        elif memory_type == "observation":
                            from ..models.observation import Observation

                            memory = Observation(
                                id=memory_obj.id,
                                user_id=memory_obj.user_id,
                                content=memory_obj.content,
                                entity_type=memory_obj.entity_type,
                                entity_id=memory_obj.entity_id,
                                context=memory_obj.context or {},
                                tags=memory_obj.tags or [],
                                created_at=memory_obj.created_at,
                                updated_at=memory_obj.updated_at,
                            )
                        elif memory_type == "hint":
                            from ..models.hint import Hint

                            memory = Hint(
                                id=memory_obj.id,
                                user_id=memory_obj.user_id,
                                content=memory_obj.content,
                                category=memory_obj.category,
                                priority=memory_obj.priority,
                                workflow_context=memory_obj.workflow_context,
                                tags=memory_obj.tags or [],
                                created_at=memory_obj.created_at,
                                updated_at=memory_obj.updated_at,
                            )
                        else:
                            continue

                        if await self.index_memory_async(session, memory):
                            counts[memory_type] += 1

                except Exception as e:
                    logger.error(f"Failed to reindex {memory_type} memories: {e}")
                    continue

            await session.commit()
            logger.info(f"Reindexed memories: {counts}")
            return counts

        except Exception as e:
            logger.error(f"Failed to reindex all memories: {e}")
            await session.rollback()
            raise SearchServiceError(f"Failed to reindex all memories: {e}") from None
