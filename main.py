

import logging
import os
import uuid

import requests
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSSearchResult, RAGUpsertResult, RAGChunkAndSrc

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)


GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

CHAT_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")

# ── Qdrant Cloud settings ─────────────────────────────────────────────────────
# Get these from https://cloud.qdrant.io (free tier available)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)


# ─────────────────────────────────────────────────────────────────────────────
def groq_chat(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """
    Send a chat request to the Groq API and return the model's reply.

    Args:
        system_prompt: Instructions that tell the model how to behave.
        user_prompt:   The actual question + context to send to the model.
        temperature:   0.0 = very deterministic, 1.0 = more creative.
                       We use 0.2 for factual Q&A (low randomness).

    Returns:
        The model's answer as a plain string.
    """
    if not GROQ_API_KEY:
        raise inngest.NonRetriableError(
            "GROQ_API_KEY is missing!\n"
            "1. Go to https://console.groq.com and create a free account.\n"
            "2. Copy your API key.\n"
            "3. Add it to your .env file:  GROQ_API_KEY=your_key_here"
        )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,   # max length of the answer
    }

    logging.info(f"Calling Groq API → model: {CHAT_MODEL}")

    try:
        response = requests.post(
            GROQ_API_URL, headers=headers, json=payload, timeout=120
        )
    except requests.exceptions.Timeout:
        raise RuntimeError("Groq API timed out after 120 seconds. Try again.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error calling Groq API: {e}")

    # Handle common HTTP errors with helpful messages
    if response.status_code == 401:
        raise inngest.NonRetriableError(
            "Groq API returned 401 Unauthorized.\n"
            "Your GROQ_API_KEY in .env is invalid or expired.\n"
            "Get a new one at: https://console.groq.com"
        )

    if response.status_code == 429:
        raise RuntimeError(
            "Groq API rate limit hit (429). Please wait a moment and try again."
        )

    if response.status_code != 200:
        try:
            err = response.json().get("error", {}).get("message", response.text)
        except Exception:
            err = response.text
        raise RuntimeError(f"Groq API error {response.status_code}: {err}")

    # Parse the response and extract the answer text
    result = response.json()
    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        logging.error(f"Unexpected Groq response: {result}")
        raise RuntimeError(f"Unexpected Groq response format: {e}") from e


# Inngest client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=os.getenv("RENDER", "").lower() == "true",
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id="rag_ingest_pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    retries=0,
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        """Step 1: Read the PDF and split it into chunks."""
        pdf_path  = ctx.event.data.get("pdf_path", "")
        source_id = ctx.event.data.get("source_id", pdf_path)

        if not pdf_path:
            raise inngest.NonRetriableError(
                "Event data is missing 'pdf_path'. "
                "Make sure the frontend is sending the correct file path."
            )

        try:
            chunks = load_and_chunk_pdf(pdf_path)  # type: ignore
        except ValueError as e:
            raise inngest.NonRetriableError(str(e)) from e

        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)  # type: ignore

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        """Step 2: Embed chunks and store in Qdrant."""
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        #  Local embedding no API key needed
        vecs = embed_texts(chunks)


        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        # Use cloud config if available, otherwise falls back to local
        store = QdrantStorage(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY, # type: ignore
        )
        store.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    # Run step 1
    chunks_and_src = await ctx.step.run(
        step_id="load-and-chunk",
        handler=lambda: _load(ctx),        # type: ignore
        output_type=RAGChunkAndSrc,
    )

    # Run step 2
    ingested = await ctx.step.run(
        step_id="embed-and-upsert",
        handler=lambda: _upsert(chunks_and_src),  # type: ignore
        output_type=RAGUpsertResult,
    )

    return ingested.model_dump()


# FUNCTION 2 — Answer a question with RAG

@inngest_client.create_function(
    fn_id="rag_query_pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
    retries=0,
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search(question: str, top_k: int = 5) -> RAGSSearchResult:
        """
        Step 1: Embed the question (locally) and find the most relevant chunks.
        """
        # Embed the question using the same local model used during ingestion
        query_vec = embed_texts([question])[0]
        
        # Use cloud config if available, otherwise falls back to local
        store = QdrantStorage(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY, # type: ignore
        )
        found = store.search(query_vec, top_k)
        return RAGSSearchResult(
            contexts=found["contexts"],
            sources=found["sources"],
        )

    def _answer(question: str, found: RAGSSearchResult) -> str:
        """
        Step 2: Build a prompt from the retrieved chunks and ask Groq to answer.
        Groq is only used here — for text generation, not for embeddings.
        """
        if not found.contexts:
            return (
                "I couldn't find any relevant information in the uploaded documents "
                "to answer your question. Try uploading a PDF first, or rephrase your question."
            )

        # Format the retrieved chunks as a bullet list for the prompt
        context_block = "\n\n".join([f"- {c}" for c in found.contexts])

        system_prompt = (
            "You are a helpful assistant that answers questions using only the "
            "provided context. Be concise and accurate. "
            "If the answer is not in the context, say so honestly."
        )

        user_prompt = (
            "Use the following context to answer the question.\n"
            "If the answer cannot be found in the context, say so.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely using only the information from the context above:"
        )

        # Groq API call 
        return groq_chat(system_prompt, user_prompt, temperature=0.2)

    # Read incoming event data
    question = ctx.event.data.get("question", "")
    top_k    = int(ctx.event.data.get("top_k", 5))  # type: ignore

    if not question:
        raise inngest.NonRetriableError(
            "Event data is missing 'question'. "
            "Make sure the frontend sends a non-empty question."
        )

    # Run step 1 — local embedding + vector search
    found = await ctx.step.run(
        step_id="embed-and-search",
        handler=lambda: _search(question, top_k),  # type: ignore
        output_type=RAGSSearchResult,
    )

    # Run step 2 — Groq LLM call
    answer = await ctx.step.run(
        step_id="llm-answer",
        handler=lambda: _answer(question, found),  # type: ignore
    )

    return {
        "answer":       answer,
        "sources":      found.sources,
        "num_contexts": len(found.contexts),
    }


#  FastAPI app
app = FastAPI(title="RAG PDF API")


@app.get("/")
def home():
    """Quick status check — visit http://localhost:8000 in your browser."""
    # Determine which Qdrant we're using
    qdrant_type = "Qdrant Cloud" if QDRANT_API_KEY else "Local Qdrant"
    
    return {
        "status": " RAG API is running",
        "llm":    f"Groq → {CHAT_MODEL}",
        "embed":  "Local → BAAI/bge-small-en-v1.5 (no API key needed)",
        "vector_db": f"{qdrant_type} ({QDRANT_URL})",
    }


@app.get("/health")
def health():
    """Health check endpoint used by monitoring tools."""
    return {"status": "healthy"}


# Register Inngest functions with FastAPI
inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai],
)