# Live Application Link (update after deployment)
# https://your-app-url.onrender.com

# Document Q&A RAG System

An end-to-end Retrieval-Augmented Generation (RAG) application that accepts PDF documents, indexes them into a vector store, and provides grounded answers with explicit source citations via an interactive web UI.

## Live Demo

> **Deploy link:** Add your public URL here after deploying (Render, Hugging Face Spaces, etc.)

## Features

- **PDF upload & ingestion** — multi-page PDF parsing with recursive character chunking (~800 chars, 100 overlap)
- **Dense embeddings** — `sentence-transformers/all-MiniLM-L6-v2` (cosine similarity via FAISS)
- **Hybrid retrieval (bonus)** — BM25 keyword search + dense vector search fused with Reciprocal Rank Fusion (RRF)
- **Strict grounding** — answers restricted to document context; returns *"Information not found in the provided document."* when unsupported
- **Explicit citations** — `[Page X, Chunk Y]` format in every answer
- **Streaming responses (bonus)** — token-by-token answer streaming on the frontend
- **Collapsible context view** — inspect retrieved chunks used to ground each answer

## System Architecture

```
[ Upload PDF ] ──► [ PyPDF parse + chunk ] ──► [ FAISS + BM25 index ]
                                                        │
[ User Query ] ─────────────────────────────────────────┴──► [ Hybrid search + RRF ]
                                                                        │
[ Web UI ] ◄── [ LLM answer + citations ] ◄── [ Ollama LLM (phi3:mini) ]
```

### Design choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Chunking | RecursiveCharacterTextSplitter, 800 chars / 100 overlap | Keeps ~500–1000 token chunks with ~12% overlap as required |
| Embeddings | `all-MiniLM-L6-v2` via HuggingFace | Fast, lightweight, assignment-specified model |
| Vector store | FAISS (persistent) | In-memory/persistent cosine similarity search |
| Keyword search | BM25 | Captures exact terms (section numbers, legal terms) |
| Fusion | RRF (k=60) | Merges BM25 + vector rankings without score normalization |
| LLM | Ollama `phi3:mini` | Local, no API key required; strict grounding prompt |

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with:
  ```bash
  ollama pull phi3:mini
  ```

## Local Setup

### Running with Streamlit (Recommended)

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### Running with FastAPI + HTML Frontend

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — upload a PDF, then ask questions.

### Environment variables (optional)

Create a `.env` file:

```env
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=phi3:mini
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=5
UPLOADS_PATH=uploads
VECTOR_STORE_PATH=faiss_store
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/health` | GET | System status |
| `/upload` | POST | Upload & index a PDF |
| `/query` | POST | Ask a question (JSON response) |
| `/query/stream` | POST | Ask with streaming NDJSON response |
| `/documents` | GET | List indexed documents |

## Deployment (Streamlit Community Cloud)

1. Push the code to your GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and log in.
3. Click **New app**, select your repository, branch, and set the Main file path to `streamlit_app.py`.
4. Click **Advanced settings...**.
5. In the **Secrets** section, add your environment variables:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "your-groq-api-key-here"
   LLM_MODEL = "llama-3.1-8b-instant"
   EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
   ```
6. Click **Deploy**.

## Deployment (Render)

1. Push to a public GitHub repo
2. Connect to [Render](https://render.com) as a **Web Service**
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Add environment variables from `.env.example`
6. Note: Ollama must be reachable — for cloud deploy, switch `LLM_MODEL` to an API-based provider (OpenAI/Groq) or use a host with Ollama sidecar

### Docker

```bash
docker build -t document-qa-rag .
docker run -p 8000:8000 document-qa-rag
```

## Project Structure

```
document-qa-rag/
├── app.py              # FastAPI server (upload, query, stream)
├── streamlit_app.py    # Streamlit interface (Q&A UI, upload, streaming)
├── rag_pipeline.py     # RAG pipeline (ingest, retrieve, generate)
├── frontend/           # Web UI (upload, streaming, context chunks)
├── uploads/            # Uploaded PDFs (created at runtime)
├── faiss_store/        # Persisted FAISS index (created at runtime)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Walkthrough

1. Start the server and open the UI
2. Upload a multi-page PDF (technical manual, research paper, legal doc)
3. Wait for indexing to complete (status badge turns green)
4. Ask a question — watch the answer stream in with citations
5. Expand **Retrieved Context Chunks** to inspect grounding sources

## License

MIT
