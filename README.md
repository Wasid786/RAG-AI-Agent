# 📄 RAG PDF Chat

A Retrieval-Augmented Generation (RAG) app that lets you upload PDFs and ask questions about them.

| Layer | Tool |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers` (runs **100% locally**, no API key) |
| Chat / LLM | [Groq](https://console.groq.com) free inference API (`mixtral-8x7b-32768` by default) |
| Vector DB | [Qdrant](https://qdrant.tech) running locally in Docker |
| Orchestration | [Inngest](https://www.inngest.com) dev server |
| Frontend | [Streamlit](https://streamlit.io) |
| Backend | [FastAPI](https://fastapi.tiangolo.com) + Uvicorn |

---

## 📁 Project Structure

```
.
├── main.py            # FastAPI + Inngest backend (ingest & query functions)
├── streamlit_app.py   # Streamlit frontend UI
├── data_loader.py     # PDF loading, chunking, and local embeddings
├── vector_db.py       # Qdrant vector DB wrapper
├── custom_types.py    # Pydantic models for Inngest step I/O
├── requirements.txt   # All Python dependencies
├── .env               # Your secrets (never commit this)
└── README.md
```

---

## ✅ Prerequisites

Make sure the following are installed **before** starting:

- Python 3.10 or higher
- [Docker Desktop](https://docs.docker.com/get-docker/) — for running Qdrant
- [Node.js 18+](https://nodejs.org/) — for running the Inngest Dev Server
- A free [Groq account](https://console.groq.com) — for the LLM API key

---

## 🚀 Setup — Step by Step

### Step 1 — Clone / Download the Project

```bash
# If using git:
git clone <your-repo-url>
cd <your-repo-folder>

# Or just navigate to your project folder:
cd D:\Projects\RAG_llm_project
```

---

### Step 2 — Get a Groq API Key

1. Go to [https://console.groq.com](https://console.groq.com) and sign up for free
2. Click **API Keys** → **Create API Key**
3. Copy the key — you'll need it in Step 4

---

### Step 3 — Create and Activate a Virtual Environment

```bash
# Create the virtual environment
python -m venv .venv
```

```bash
# Activate — Windows (Command Prompt)
.venv\Scripts\activate

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source .venv/bin/activate
```

---

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Windows (Command Prompt)
copy NUL .env

# macOS / Linux
touch .env
```

Open `.env` and add the following:

```env
# Required — get this from https://console.groq.com
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional — change the Groq model (default: mixtral-8x7b-32768)
# Options (fastest → most capable):
#   llama3-8b-8192        very fast, good for simple Q&A
#   mixtral-8x7b-32768    fast, excellent quality  ← default
#   llama3-70b-8192       slower, highest quality
#   gemma2-9b-it          good balance
# GROQ_MODEL=mixtral-8x7b-32768

# Optional — Inngest dev server base URL (no change needed for local dev)
# INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

---

### Step 5 — Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

> **Note:** `sentence-transformers` will download the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) the **first time** you run the app. After that it runs fully offline.

---

### Step 6 — Start Qdrant (Vector Database) via Docker

Open a **new terminal** and run:

```bash
# Pull the Qdrant image (only needed once)
docker pull qdrant/qdrant

# Run Qdrant on port 6333
# Windows (Command Prompt)
docker run -d --name qdrant -p 6333:6333 -v %cd%/qdrant_data:/qdrant/storage qdrant/qdrant

# Windows (PowerShell)
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/qdrant_data:/qdrant/storage qdrant/qdrant

# macOS / Linux
docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

Verify Qdrant is running:

```bash
curl http://localhost:6333/healthz
# Expected output: {"title":"qdrant - vector search engine","version":"..."}
```

---

### Step 7 — Start the Inngest Dev Server

Open a **new terminal** and run:

```bash
# Install Inngest CLI globally (only needed once)
npm install -g inngest-cli

# Start the dev server
inngest dev
```

The Inngest dashboard will be available at [http://localhost:8288](http://localhost:8288).

---

### Step 8 — Start the FastAPI Backend

Open a **new terminal** (with your virtual environment activated) and run:

```bash
uvicorn main:app --reload --port 8000
```

Verify the backend is running:

```bash
curl http://localhost:8000/
# Expected: {"status":"✅ RAG API is running", ...}
```

---

### Step 9 — Start the Streamlit Frontend

Open a **new terminal** (with your virtual environment activated) and run:

```bash
streamlit run streamlit_app.py
```

The app will open automatically at [http://localhost:8501](http://localhost:8501).

---

## 🔄 Running Order (Every Time)

You need **4 terminals** running simultaneously in this order:

| # | Terminal | Command |
|---|---|---|
| 1 | Qdrant (Docker) | `docker start qdrant` *(if already created)* |
| 2 | Inngest Dev Server | `inngest dev` |
| 3 | FastAPI Backend | `uvicorn main:app --reload --port 8000` |
| 4 | Streamlit Frontend | `streamlit run streamlit_app.py` |

> On first run, use the full `docker run ...` command from Step 6. On subsequent runs, use `docker start qdrant`.

---

## 🤖 Models Used

| Purpose | Model | Runs where |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) | **Locally** on your machine — no API key needed |
| Chat / LLM | `mixtral-8x7b-32768` (default) | **Groq cloud** — free API key required |

---

## 🗑️ Reset the Vector Database

If you want to clear all ingested documents and start fresh:

```bash
# Option 1 — Python shell (with venv activated)
python -c "from qdrant_client import QdrantClient; QdrantClient('http://localhost:6333').delete_collection('docs')"

# Option 2 — curl
curl -X DELETE http://localhost:6333/collections/docs
```

Then re-ingest your PDFs through the Streamlit UI.

---

## 🛑 Stopping All Services

```bash
# Stop Qdrant Docker container
docker stop qdrant

# Stop FastAPI   → Ctrl+C in its terminal
# Stop Streamlit → Ctrl+C in its terminal
# Stop Inngest   → Ctrl+C in its terminal
```

---

## ⚠️ Common Issues

**`[WinError 10061] No connection could be made` / Qdrant refused connection**
→ Qdrant Docker container is not running. Run `docker start qdrant` or repeat Step 6.

**`GROQ_API_KEY is missing` error**
→ Your `.env` file is missing or the key is not set. Re-check Step 4.

**`401 Unauthorized` from Groq**
→ Your `GROQ_API_KEY` is invalid or expired. Get a new one at [https://console.groq.com](https://console.groq.com).

**`429 Too Many Requests` from Groq**
→ You've hit the free tier rate limit. Wait a few seconds and try again, or switch to a faster model like `llama3-8b-8192`.

**`Connection refused` on port 8288**
→ The Inngest dev server is not running. Complete Step 7 first.

**`Connection refused` on port 8000**
→ The FastAPI backend is not running. Complete Step 8 first.

**Streamlit shows `Timed out`**
→ Make sure all 4 services (Qdrant, Inngest, FastAPI, Streamlit) are running simultaneously.

**Qdrant dimension mismatch error**
→ You have an old collection from a different embedding model. Delete it using the reset command above.