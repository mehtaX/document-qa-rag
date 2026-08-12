import json
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_pipeline import get_rag_instance

rag = get_rag_instance()
_build_error: str = ""


def _load_uploads_in_background() -> None:
  global _build_error
  try:
    rag.load_existing_uploads()
  except Exception as exc:
    _build_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
  thread = threading.Thread(target=_load_uploads_in_background, daemon=True)
  thread.start()
  yield


app = FastAPI(
  title="Document Q&A RAG",
  description="Intelligent Document Q&A with Retrieval-Augmented Generation",
  version="1.0.0",
  lifespan=lifespan,
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["GET", "POST", "OPTIONS"],
  allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")


class QueryRequest(BaseModel):
  question: str


class SourceChunk(BaseModel):
  page: str | int
  chunk: str | int
  source_file: str
  content: str
  citation: str


class QueryResponse(BaseModel):
  question: str
  answer: str
  sources: list[SourceChunk]
  latency_ms: float
  generation_ms: float
  request_id: str
  step_logs: list[str] = Field(default_factory=list)


class UploadResponse(BaseModel):
  filename: str
  pages: int
  chunks: int
  total_chunks: int
  message: str


@app.get("/health")
async def health():
  return {
    "status": "ok",
    "ready": rag.is_ready(),
    "building": rag.is_building(),
    "documents": rag.list_documents(),
    "total_chunks": len(rag.chunks),
    "build_error": _build_error or rag.get_build_error(),
    "last_step": rag.step_logs[-1] if rag.step_logs else None,
  }


@app.get("/documents")
async def documents():
  return {
    "documents": rag.list_documents(),
    "total_chunks": len(rag.chunks),
    "ready": rag.is_ready(),
  }


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
  if not file.filename or not file.filename.lower().endswith(".pdf"):
    raise HTTPException(status_code=400, detail="Only PDF files are supported.")

  safe_name = Path(file.filename).name
  dest = rag.uploads_path / safe_name

  try:
    with dest.open("wb") as handle:
      shutil.copyfileobj(file.file, handle)
    result = rag.ingest_pdf(dest)
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}")

  return UploadResponse(
    filename=result["filename"],
    pages=result["pages"],
    chunks=result["chunks"],
    total_chunks=result["total_chunks"],
    message=f"Successfully indexed {result['filename']} ({result['chunks']} chunks).",
  )


@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
  if not rag.is_ready():
    raise HTTPException(
      status_code=503,
      detail="No documents indexed yet. Upload a PDF first.",
    )

  question = payload.question.strip()
  if not question:
    raise HTTPException(status_code=400, detail="Question must not be empty.")

  result = rag.ask(question)
  if result.get("error"):
    raise HTTPException(status_code=500, detail=result["error"])

  return QueryResponse(
    question=question,
    answer=result["answer"],
    sources=[SourceChunk(**item) for item in result["sources"]],
    latency_ms=result["latency_ms"],
    generation_ms=result["generation_ms"],
    request_id=result["request_id"],
    step_logs=result.get("step_logs", []),
  )


@app.post("/query/stream")
async def query_stream(payload: QueryRequest):
  if not rag.is_ready():
    raise HTTPException(
      status_code=503,
      detail="No documents indexed yet. Upload a PDF first.",
    )

  question = payload.question.strip()
  if not question:
    raise HTTPException(status_code=400, detail="Question must not be empty.")

  def event_stream():
    for event in rag.ask_stream(question):
      yield json.dumps(event) + "\n"

  return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.get("/")
async def root():
  return RedirectResponse(url="/static/index.html")
