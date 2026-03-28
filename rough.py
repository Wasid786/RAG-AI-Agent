# /////// code for reset the model  //////////////////

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient(url="http://localhost:6333")

# Delete the collection
client.delete_collection("docs")
print("✅ Deleted all PDFs from Qdrant.")

# Recreate it fresh and empty
client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
)
print("✅ Fresh empty collection created. Ready to ingest new PDFs!")

