"""
vector_db.py
────────────
Wraps Qdrant (local or cloud) for storing and searching embedding vectors.
"""

import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    """
    A simple wrapper around Qdrant for storing and searching text embeddings.
    Supports both local Docker and Qdrant Cloud.
    """

    def __init__(
        self,
        url: str = None, # type: ignore
        api_key: str = None, # type: ignore
        collection: str = "docs",
        dim: int = 384,
    ):
        """
        Connect to Qdrant (local or cloud).

        Args:
            url: Qdrant server URL (default: localhost:6333)
            api_key: API key for Qdrant Cloud (not needed for local)
            collection: Name of the vector collection
            dim: Dimension of embedding vectors (384 for bge-small)
        """
        self.collection = collection
        
        # Use environment variables if provided
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY", None)
        
        # Connect to Qdrant
        if self.api_key:
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=60
            )
            print(f"✅ Connected to Qdrant Cloud: {self.url}")
        else:
            self.client = QdrantClient(
                url=self.url,
                timeout=30
            )
            print(f"✅ Connected to local Qdrant: {self.url}")

        # Create collection if it doesn't exist
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )
            print(f"✅ Created Qdrant collection: '{self.collection}'")
        else:
            print(f"📦 Using existing Qdrant collection: '{self.collection}'")

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        """Save a batch of text chunks and their vectors into Qdrant."""
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        print(f"💾 Stored {len(points)} chunks in Qdrant.")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> dict:
        """Find the top-k most similar chunks to the query vector."""
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points

        contexts = []
        sources = set()

        for point in results:
            payload = getattr(point, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}