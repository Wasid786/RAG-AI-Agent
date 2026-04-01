"""
data_loader.py
──────────────
Handles two things:
  1. Reading a PDF and splitting it into small text chunks.
  2. Converting text chunks into numbers (embeddings) using a LOCAL model
     — no API key required!

Local embedding model used: BAAI/bge-small-en-v1.5
  • Runs entirely on your machine (CPU is fine).
  • Produces 384-dimensional vectors.
  • Install once with:  pip install sentence-transformers
"""

from pathlib import Path
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

# ── Embedding model config ────────────────────────────────────────────────────
# This model runs locally — no internet needed after first download.
# It will be downloaded (~130 MB) the very first time you run this file.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM   = 384   # number of dimensions this model outputs

print(f"Loading local embedding model: {EMBED_MODEL} ...")
try:
    embedder = SentenceTransformer(EMBED_MODEL)
    print(f"✅ Embedding model ready: {EMBED_MODEL}")
except Exception as e:
    print(f"❌ Could not load embedding model: {e}")
    print("   Fix: pip install sentence-transformers")
    raise

# ── Text splitter config ──────────────────────────────────────────────────────
# chunk_size    = max characters per chunk (≈ 1 page of text)
# chunk_overlap = how many chars overlap between consecutive chunks
#                 (helps preserve context at chunk boundaries)
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


# ─────────────────────────────────────────────────────────────────────────────
def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Read a PDF file and split it into overlapping text chunks.

    Args:
        path: File path to the PDF (e.g. "uploads/my_doc.pdf").

    Returns:
        A list of text strings (chunks), ready for embedding.
    """
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise ValueError(
            f"PDF not found: {pdf_path}\n"
            "Make sure the file path is correct."
        )

    # Load every page of the PDF as a separate document object
    docs = PDFReader().load_data(file=pdf_path)
    print(f"📄 Pages loaded: {len(docs)}")

    for i, doc in enumerate(docs):
        print(f"   Page {i + 1}: {len(doc.text)} characters")

    # Combine all page texts, then split into chunks
    texts  = [doc.text for doc in docs if getattr(doc, "text", None)]
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    print(f"✂️  Created {len(chunks)} chunks from the PDF.")
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors (lists of floats).

    This runs 100% locally using the sentence-transformers library.
    No Groq API key or internet connection is needed.

    Args:
        texts: List of strings to embed.

    Returns:
        List of 384-dimensional float vectors (one per input string).
    """
    if not texts:
        return []

    # normalize_embeddings=True makes cosine-similarity searches more accurate
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    # Convert numpy array → plain Python list (needed for JSON / Qdrant)
    return embeddings.tolist()