import time
from pathlib import Path
import asyncio
import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #6b3a1f;
}

.stButton > button {
    background-color: #fef9c3 !important;
    color: #6b3a1f !important;
    border-radius: 6px !important;
    border: none !important;
    padding: 0.45rem 1.4rem !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}
.stButton > button:hover {
    background-color: #e8e08a !important;
}

input[type="text"], input[type="number"] {
    border-radius: 6px !important;
    border: 1px solid #d1d5db !important;
    font-size: 0.875rem !important;
}
input[type="text"]:focus, input[type="number"]:focus {
    border-color: #fef9c3 !important;
    box-shadow: 0 0 0 2px rgba(254,249,195,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Safe async runner ─────────────────────────────────────────────────────────
def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.ensure_future(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


# ── Inngest client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def _send_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


async def _send_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    event_obj = result[0]
    return event_obj if isinstance(event_obj, str) else event_obj.id


def _inngest_api_base() -> str:
    """
    Determine the Inngest API base URL based on environment.
    
    Returns:
        - Render backend URL if running on Render (production)
        - Local Inngest dev server URL for development
        - Falls back to environment variable if set
    """
    # Check if running on Render (production)
    if os.getenv("RENDER", "").lower() == "true":
        # Change this to your actual Render backend URL!
        return "https://rag-backend.onrender.com/v1"
    
    # Check for custom INNGEST_API_BASE environment variable
    custom_base = os.getenv("INNGEST_API_BASE")
    if custom_base:
        return custom_base
    
    # Default to local Inngest dev server
    return "http://127.0.0.1:8288/v1"


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status", "")
            last_status = status or last_status

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Inngest run ended with status: {status}")

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out after {timeout_s}s (last status: {last_status})")

        time.sleep(poll_interval_s)


# ══════════════════════════════════════════════════════════════════════════════
# UI — Section 1: Ingest
# ══════════════════════════════════════════════════════════════════════════════
st.title("📄 Upload a PDF")
st.caption("Upload a document to add it to the knowledge base.")

if "last_ingested" not in st.session_state:
    st.session_state.last_ingested = None

uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None and uploaded.name != st.session_state.last_ingested:
    with st.spinner(f"Ingesting {uploaded.name}..."):
        path = save_uploaded_pdf(uploaded)
        run_async(_send_ingest_event(path))
        time.sleep(0.3)

    st.session_state.last_ingested = uploaded.name
    st.success(f"✅ Done! **{uploaded.name}** is ready to query.")
    st.caption("Upload another PDF to expand the knowledge base.")

elif uploaded is not None:
    st.info(f"**{uploaded.name}** was already ingested this session.")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# UI — Section 2: Query
# ══════════════════════════════════════════════════════════════════════════════
st.title("💬 Ask a Question")
st.caption("Ask anything about the uploaded document(s).")

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
    st.session_state.last_sources = []

with st.form("rag_query_form"):
    question = st.text_input("Your question", placeholder="e.g. What is this document about?")
    top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask →")

if submitted and question.strip():
    with st.spinner("Thinking..."):
        try:
            event_id = run_async(_send_query_event(question.strip(), int(top_k)))
            output = wait_for_run_output(event_id)

            st.session_state.last_answer = output.get("answer", "")
            st.session_state.last_sources = output.get("sources", [])

        except TimeoutError:
            st.error("⏱ Timed out. Is your backend running?")
        except Exception as e:
            st.error(f"❌ Error: {e}")


# ── Display Answer ────────────────────────────────────────────────────────────
if st.session_state.last_answer:
    st.subheader("🧠 Answer")
    st.info(st.session_state.last_answer)

    if st.session_state.last_sources:
        with st.expander("📎 Sources", expanded=False):
            for s in st.session_state.last_sources:
                st.markdown(f"- `{s}`")