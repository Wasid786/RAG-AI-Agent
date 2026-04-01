#  RAG PDF Chat

A Retrieval-Augmented Generation (RAG) app that lets you upload PDFs and ask questions about them.  
Uses **HuggingFace Inference API** (free) for embeddings & chat, **Qdrant** for vector storage, **Inngest** for workflow orchestration, and **Streamlit** for the UI.

---

##  Project Structure

```
.
├── main.py            # FastAPI + Inngest backend (ingest & query functions)
├── streamlit_app.py   # Streamlit frontend UI
├── data_loader.py     # PDF loading, chunking, and HF embeddings
├── vector_db.py       # Qdrant vector DB wrapper
├── custom_types.py    # Pydantic models
├── .env               # Your secrets (never commit this)
└── README.md
```

---

##  Prerequisites

Make sure the following are installed on your system before starting:

- Python 3.10 or higher
- [Docker](https://docs.docker.com/get-docker/) (for running Qdrant)
- [Node.js 18+](https://nodejs.org/) (for running Inngest Dev Server)
- A free [HuggingFace account](https://huggingface.co/join)

---

##  Step 1 — Get a HuggingFace API Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → choose **Read** role → copy the token

---

##  Step 2 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Then open `.env` and fill in your token:

```env
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: change the chat model (default: mistralai/Mistral-7B-Instruct-v0.3)
# HF_CHAT_MODEL=HuggingFaceH4/zephyr-7b-beta

# Inngest dev server base URL (no change needed for local dev)
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

---

##  Step 3 — Create a Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate it — Linux/macOS
source .venv/bin/activate

# Activate it — Windows
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install fastapi uvicorn streamlit inngest qdrant-client \
            llama-index llama-index-readers-file \
            python-dotenv requests pydantic
```


---

##  Step 4 — Start Qdrant (Vector Database)

Run Qdrant locally using Docker:

```bash
docker pull qdrant/qdrant

docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v $(pwd)/qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

Verify it's running:

```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine","version":"..."}
```

> **Windows users:** Replace `$(pwd)` with `%cd%` in Command Prompt, or use `${PWD}` in PowerShell.

---

##  Step 5 — Start the Inngest Dev Server

Inngest orchestrates the ingest and query workflow steps.

```bash
# Install Inngest CLI (once)
npm install -g inngest-cli

# Start the dev server
inngest dev
```

The Inngest dashboard will be available at [http://localhost:8288](http://localhost:8288).

---

##  Step 6 — Start the FastAPI Backend

In a **new terminal** (with your virtual environment activated):

```bash
uvicorn main:app --reload --port 8000
```

Verify it's running:

```bash
curl http://localhost:8000/
# Expected: {"status":"RAG API running"}
```

---

##  Step 7 — Start the Streamlit Frontend

In another **new terminal** (with your virtual environment activated):

```bash
streamlit run streamlit_app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

---



```python
# Run this once in a Python shell
from qdrant_client import QdrantClient
QdrantClient("http://localhost:6333").delete_collection("docs")
```

Or via the Qdrant REST API:

```bash
curl -X DELETE http://localhost:6333/collections/docs
```

Then re-ingest your PDFs through the Streamlit UI.

---

##  Stopping All Services

```bash
# Stop Qdrant container
docker stop qdrant

# Stop FastAPI (Ctrl+C in its terminal)
# Stop Streamlit (Ctrl+C in its terminal)
# Stop Inngest dev server (Ctrl+C in its terminal)
```


---

## 🤖 Models Used

| Purpose    | Model                                        | Dim  |
|------------|----------------------------------------------|------|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`     | 384  | 
| Chat / LLM | `mistralai/Mistral-7B-Instruct-v0.3`         | —    |

Both are served via the **HuggingFace free Inference API** — no GPU or credit card required.

---

##  Common Issues

**`401 Unauthorized` from HuggingFace**
→ Your `HF_API_TOKEN` in `.env` is missing or incorrect. Re-check Step 1.

**`Model is currently loading` error**
→ HF free tier cold-starts models. The code sets `wait_for_model: true` so it retries automatically. Wait a few seconds and try again.

**Qdrant dimension mismatch error**
→ You have an old collection with a different vector size. Run the delete command in the upgrade section above.

**`Connection refused` on port 8288**
→ The Inngest dev server is not running. Complete Step 5 first.

**Streamlit shows `Timed out waiting for an answer`**
→ Make sure all four services (Qdrant, Inngest, FastAPI, Streamlit) are running simultaneously.
