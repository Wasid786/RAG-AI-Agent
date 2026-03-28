import logging
import uuid

import ollama
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

logging.basicConfig(level=logging.INFO)

# ── Ollama settings ────────────────────────────────────────────────────────────
# Make sure Ollama is running locally: https://ollama.com
# Pull the chat model once:  ollama pull llama3
CHAT_MODEL = "llama3"

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


# ================== INGEST FUNCTION ==================
@inngest_client.create_function(
    fn_id="rag_ingest_pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path  = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks    = load_and_chunk_pdf(pdf_path) # type: ignore
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id) # type: ignore

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks    = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vecs = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {"source": source_id, "text": chunks[i]}
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        step_id="load-and-chunk",
        handler=lambda: _load(ctx), # type: ignore
        output_type=RAGChunkAndSrc
    )

    ingested = await ctx.step.run(
        step_id="embed-and-upsert",
        handler=lambda: _upsert(chunks_and_src), # type: ignore
        output_type=RAGUpsertResult
    )

    return ingested.model_dump()


# ================== QUERY FUNCTION ==================
@inngest_client.create_function(
    fn_id="rag_query_pdf",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):

    def _search(question: str, top_k: int = 5) -> RAGSSearchResult:
        query_vec = embed_texts([question])[0]
        store     = QdrantStorage()
        found     = store.search(query_vec, top_k)
        return RAGSSearchResult(
            contexts=found["contexts"],
            sources=found["sources"]
        )

    def _answer(question: str, found: RAGSSearchResult) -> str:
        """Call local Ollama llama3 to answer using retrieved context."""
        context_block = "\n\n".join([f"- {c}" for c in found.contexts])

        user_content = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context above."
        )

        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You answer using only the provided context."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            options={"temperature": 0.2}
        )

        return response["message"]["content"].strip()

    question = ctx.event.data["question"]
    top_k    = int(ctx.event.data.get("top_k", 5)) # type: ignore

    found = await ctx.step.run(
        step_id="embed-and-search",
        handler=lambda: _search(question, top_k), # type: ignore
        output_type=RAGSSearchResult
    )

    # Run Ollama inference as an Inngest step so it is retried on failure
    answer = await ctx.step.run(
        step_id="llm-answer",
        handler=lambda: _answer(question, found), # type: ignore
    )

    return {
        "answer": answer,
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }


# ================== FASTAPI ==================
app = FastAPI()

@app.get("/")
def home():
    return {"status": "RAG API running"}

inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai]
)