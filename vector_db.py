

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


class QdrantStorage:


    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        dim: int = 384,          # must match EMBED_DIM in data_loader.py
    ):

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
            print(f" Created Qdrant collection: '{self.collection}'")
        else:
            print(f" Using existing Qdrant collection: '{self.collection}'")


    def upsert(
        self,
        ids:      list[str],
        vectors:  list[list[float]],
        payloads: list[dict],
    ) -> None:

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        print(f" Stored {len(points)} chunks in Qdrant.")


    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> dict:

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