import logging
import uuid
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

logging.basicConfig(level=logging.INFO)

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="rag_ingest_pdf",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):

    # 🔹 Step 1: Load & Chunk PDF
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        chunks = load_and_chunk_pdf(pdf_path) # type: ignore

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id # type: ignore
        )

    # 🔹 Step 2: Embed + Store in Qdrant
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
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

    # 🔹 Run Steps (Inngest)
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


# 🔹 FastAPI App
app = FastAPI()

@app.get("/")
def home():
    return {"status": "RAG API running"}

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])