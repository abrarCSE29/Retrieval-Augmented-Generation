import uuid
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

class Qdrant_VDB:
    """
    Implements the storing and retrieving logic of vector embeddings in a Qdrant database.
    """
    
    def __init__(self, collection_name: str, vector_size: int, host: str = None, port: int = None):
        """
        Initialize the Qdrant vector database client.

        Args:
            collection_name (str): Name of the Qdrant collection to store embeddings.
            vector_size (int): Size of the embedding vectors (e.g., 768 for sentence-transformers).
            host (str, optional): Qdrant server host. Defaults to env var or 'localhost'.
            port (int, optional): Qdrant server port. Defaults to env var or 6334.
        """
        # Load environment variables from .env file
        load_dotenv()

        # Use provided host/port or fall back to env vars or defaults
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", 6334))
        self.collection_name = collection_name
        self.vector_size = vector_size

        # Initialize Qdrant client
        self.client = QdrantClient(host=self.host, port=self.port)

        # Ensure the collection exists
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self) -> None:
        """
        Create a Qdrant collection if it doesn't already exist.
        """
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE  # Common for text embeddings
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
        except UnexpectedResponse as e:
            raise Exception(f"Failed to create/check collection: {e}")

    def store_document_embeddings(
        self,
        document_id: str,
        embeddings,
        chunk_texts: List[str] = None,
        extra_payload: Dict[str, Any] = None
    ) -> List[str]:
        """
        Store embeddings for a document's chunks in Qdrant, associating them with a document_id.

        Args:
            document_id (str): Unique identifier for the document.
            embeddings (List[List[float]]): List of embedding vectors for the document's chunks.
            chunk_texts (List[str], optional): List of chunk texts corresponding to embeddings.
            extra_payload (Dict[str, Any], optional): Additional metadata to include in all points.

        Returns:
            List[str]: List of point IDs assigned to the stored embeddings.
        """

        if not embeddings.all():
            return []

        # Generate unique point IDs for each embedding
        point_ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]

        # Prepare payloads for each embedding
        payloads = []
        for i, embedding in enumerate(embeddings):
            payload = {
                "document_id": document_id,  # Associate with the document
                "chunk_index": i  # Track chunk order
            }
            if chunk_texts and i < len(chunk_texts):
                payload["text"] = chunk_texts[i]  # Store chunk text if provided
            if extra_payload:
                payload.update(extra_payload)  # Add any extra metadata
            payloads.append(payload)

        # Prepare points for Qdrant
        points = [
            PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload
            )
            for point_id, embedding, payload in zip(point_ids, embeddings, payloads)
        ]

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Stored {len(points)} embeddings for document_id {document_id} in collection {self.collection_name}")
            return point_ids
        except UnexpectedResponse as e:
            raise Exception(f"Failed to store embeddings for document_id {document_id}: {e}")