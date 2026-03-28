import ollama
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# ── Ollama settings ────────────────────────────────────────────────────────────
# Make sure you have Ollama running locally: https://ollama.com
# Pull the embedding model once:  ollama pull nomic-embed-text
# Pull the chat model once:       ollama pull llama3
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM   = 768          # nomic-embed-text outputs 768-dimensional vectors

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path) # type: ignore
    print(f"Pages loaded: {len(docs)}")          
    for i, d in enumerate(docs):
        print(f"Page {i+1} text length: {len(d.text)}")  # and this
    texts  = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using Ollama's nomic-embed-text model."""
    embeddings = []
    for text in texts:
        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        embeddings.append(response["embedding"])
    return embeddings