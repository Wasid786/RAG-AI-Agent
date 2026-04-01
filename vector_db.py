"""
vector_db.py
────────────
Wraps Qdrant (a local vector database) so the rest of the app
can store and search embedding vectors without worrying about
the Qdrant client details.

Qdrant runs locally in Docker. Start it with:
    docker run -p 6333:6333 qdrant/qdrant

No API key needed — it's just a local service on port 6333.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:
    """
    A simple wrapper around Qdrant for storing and searching text embeddings.

    How it works:
      • upsert() — saves text chunks + their vectors into Qdrant.
      • search() — finds the most similar chunks to a query vector.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        dim: int = 384,          # must match EMBED_DIM in data_loader.py
    ):
        """
        Connect to the local Qdrant instance and create the collection
        if it doesn't exist yet.

        Args:
            url:        Address of the Qdrant server (default: local Docker).
            collection: Name of the vector collection to use.
            dim:        Dimension of the embedding vectors (384 for bge-small).
        """
        self.client     = QdrantClient(url=url, timeout=30)
        self.collection = collection

        # Create the collection only if it doesn't already exist
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,   # best for normalized embeddings
                ),
            )
            print(f"✅ Created Qdrant collection: '{self.collection}'")
        else:
            print(f"📦 Using existing Qdrant collection: '{self.collection}'")

    # ─────────────────────────────────────────────────────────────────────────
    def upsert(
        self,
        ids:      list[str],
        vectors:  list[list[float]],
        payloads: list[dict],
    ) -> None:
        """
        Save a batch of text chunks and their vectors into Qdrant.

        Args:
            ids:      Unique string IDs for each chunk (we use UUID5s in main.py).
            vectors:  Embedding vectors — one per chunk.
            payloads: Metadata dicts with keys "text" and "source".
        """
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        print(f"💾 Stored {len(points)} chunks in Qdrant.")

    # ─────────────────────────────────────────────────────────────────────────
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> dict:
        """
        Find the top-k most similar chunks to the query vector.

        Args:
            query_vector: Embedding of the user's question.
            top_k:        How many chunks to return.

        Returns:
            A dict with:
              "contexts" — list of matching text chunks
              "sources"  — list of source file names
        """
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points

        contexts = []
        sources  = set()   # use a set to avoid duplicate source names

        for point in results:
            payload = getattr(point, "payload", None) or {}
            text    = payload.get("text", "")
            source  = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}