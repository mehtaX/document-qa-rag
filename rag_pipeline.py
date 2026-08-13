from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

NOT_FOUND_MESSAGE = "Information not found in the provided document."
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant" if LLM_PROVIDER == "groq" else "phi3:mini")


def _create_llm() -> BaseChatModel:
  if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    # Use placeholder if key not found to prevent crash on startup
    return ChatGroq(model=LLM_MODEL, temperature=0, api_key=api_key or "PLACEHOLDER")

  return ChatOllama(model=LLM_MODEL, temperature=0)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
TOP_K = int(os.environ.get("TOP_K", "5"))
BM25_K = int(os.environ.get("BM25_K", "10"))
VECTOR_K = int(os.environ.get("VECTOR_K", "10"))


class DocumentQARAG:
  """RAG pipeline for document Q&A with hybrid retrieval and strict grounding."""

  def __init__(self) -> None:
    self.uploads_path = Path(os.environ.get("UPLOADS_PATH", "uploads")).resolve()
    self.vector_store_path = Path(os.environ.get("VECTOR_STORE_PATH", "faiss_store")).resolve()
    self.uploads_path.mkdir(parents=True, exist_ok=True)
    self.vector_store_path.mkdir(parents=True, exist_ok=True)

    self.chunks: List[Any] = []
    self.uploaded_files: List[Dict[str, Any]] = []
    self._index_ready = False
    self._building = False
    self._build_error: str = ""
    self.step_logs: List[str] = []

    self.embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    self.llm = _create_llm()
    self.bm25_retriever: BM25Retriever | None = None
    self.hybrid_retriever: EnsembleRetriever | None = None
    self.vectorstore: FAISS | None = None

    self.answer_prompt = ChatPromptTemplate.from_template(
      """
You are a document Q&A assistant. Answer ONLY using the provided context.

Rules:
- Use ONLY the provided context chunks.
- If the answer cannot be derived from the context, respond with exactly: "Information not found in the provided document."
- Include explicit citations in the form [Page <page>, Chunk <chunk>] for every factual claim.
- Do not hallucinate or add information not supported by the context.

Context:
{context}

Question:
{question}

Answer:
"""
    )
    self.answer_chain = self.answer_prompt | self.llm

  def _log(self, message: str) -> None:
    logger.info(message)
    self.step_logs.append(message)

  def set_api_key(self, api_key: str) -> None:
    """Dynamically update the Groq API key and rebuild the LLM generation chain."""
    os.environ["GROQ_API_KEY"] = api_key
    self.llm = _create_llm()
    self.answer_chain = self.answer_prompt | self.llm
    self._log("LLM chain reinitialized with new API key")

  def clean_text(self, text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()

  def is_ready(self) -> bool:
    return self._index_ready and len(self.chunks) > 0

  def is_building(self) -> bool:
    return self._building

  def get_build_error(self) -> str:
    return self._build_error

  def list_documents(self) -> List[Dict[str, Any]]:
    return list(self.uploaded_files)

  def ingest_pdf(self, file_path: Path) -> Dict[str, Any]:
    """Parse a PDF, chunk it, and rebuild the retrieval index."""
    file_path = file_path.resolve()
    if not file_path.exists():
      raise FileNotFoundError(f"PDF not found: {file_path}")

    filename = file_path.name
    self.chunks = [c for c in self.chunks if c.metadata.get("source_file") != filename]
    self.uploaded_files = [f for f in self.uploaded_files if f["filename"] != filename]

    self._log(f"Loading PDF: {filename}")
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()

    for page in pages:
      page.page_content = self.clean_text(page.page_content)
      page.metadata["source_file"] = file_path.name
      page.metadata.setdefault("page", page.metadata.get("page", 0))

    splitter = RecursiveCharacterTextSplitter(
      chunk_size=CHUNK_SIZE,
      chunk_overlap=CHUNK_OVERLAP,
      length_function=len,
    )
    new_chunks = splitter.split_documents(pages)

    for idx, chunk in enumerate(new_chunks):
      chunk.metadata["chunk_id"] = idx + 1
      chunk.metadata["source_file"] = file_path.name
      page_num = chunk.metadata.get("page", 0)
      if isinstance(page_num, int):
        chunk.metadata["page"] = page_num + 1

    self.chunks.extend(new_chunks)
    self.uploaded_files.append(
      {
        "filename": file_path.name,
        "pages": len(pages),
        "chunks": len(new_chunks),
        "path": str(file_path),
      }
    )
    self._log(f"Added {len(new_chunks)} chunks from {file_path.name}")
    self._rebuild_index()
    return {
      "filename": file_path.name,
      "pages": len(pages),
      "chunks": len(new_chunks),
      "total_chunks": len(self.chunks),
    }

  def ingest_from_path(self, path: Path) -> Dict[str, Any]:
    return self.ingest_pdf(path)

  def load_existing_uploads(self) -> None:
    """Load any PDFs already present in the uploads folder."""
    pdf_files = sorted(self.uploads_path.glob("*.pdf"))
    if not pdf_files:
      self._log("No existing PDFs in uploads folder")
      return

    self.chunks = []
    self.uploaded_files = []
    for pdf_path in pdf_files:
      try:
        self.ingest_pdf(pdf_path)
      except Exception as exc:
        logger.exception("Failed to ingest %s", pdf_path)
        self._build_error = str(exc)

  def _rebuild_index(self) -> None:
    if not self.chunks:
      self._index_ready = False
      return

    self._building = True
    self._build_error = ""
    try:
      self._log("Building FAISS vector index")
      self.vectorstore = FAISS.from_documents(self.chunks, self.embedding)
      self.vectorstore.save_local(str(self.vector_store_path))

      self._log("Building BM25 retriever")
      self.bm25_retriever = BM25Retriever.from_documents(self.chunks)
      self.bm25_retriever.k = BM25_K

      vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})
      self.hybrid_retriever = EnsembleRetriever(
        retrievers=[self.bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
      )
      self._index_ready = True
      self._log(f"Index ready with {len(self.chunks)} chunks")
    except Exception as exc:
      self._build_error = str(exc)
      self._index_ready = False
      logger.exception("Index build failed")
      raise
    finally:
      self._building = False

  def reciprocal_rank_fusion(self, results: Iterable[List[Any]], k: int = 60) -> List[Any]:
    fused_scores: Dict[Any, float] = defaultdict(float)
    doc_map: Dict[Any, Any] = {}

    for docs in results:
      for rank, doc in enumerate(docs):
        key = (
          doc.metadata.get("source_file"),
          doc.metadata.get("page"),
          doc.metadata.get("chunk_id"),
          doc.page_content[:80],
        )
        doc_map[key] = doc
        fused_scores[key] += 1.0 / (k + rank + 1)

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked]

  def retrieve(self, query: str) -> List[Any]:
    if not self.is_ready() or self.hybrid_retriever is None:
      raise RuntimeError("RAG index is not ready. Upload a PDF first.")

    self._log("Running hybrid retrieval (BM25 + vector)")
    start = time.perf_counter()

    bm25_docs = self.bm25_retriever.invoke(query) if self.bm25_retriever else []
    vector_docs = self.vectorstore.similarity_search(query, k=VECTOR_K) if self.vectorstore else []
    fused = self.reciprocal_rank_fusion([bm25_docs, vector_docs])

    elapsed = time.perf_counter() - start
    self._log(f"Retrieved {len(fused)} candidates in {elapsed:.2f}s")
    return fused[:TOP_K]

  def format_context(self, docs: List[Any]) -> str:
    lines = []
    for doc in docs:
      page = doc.metadata.get("page", "Unknown")
      chunk_id = doc.metadata.get("chunk_id", "Unknown")
      citation = f"[Page {page}, Chunk {chunk_id}]"
      lines.append(f"{citation}\n{doc.page_content}")
    return "\n\n".join(lines)

  def format_sources(self, docs: List[Any]) -> List[Dict[str, Any]]:
    sources = []
    for doc in docs:
      sources.append(
        {
          "page": doc.metadata.get("page", "Unknown"),
          "chunk": doc.metadata.get("chunk_id", "Unknown"),
          "source_file": doc.metadata.get("source_file", "Unknown"),
          "content": doc.page_content,
          "citation": f"[Page {doc.metadata.get('page', 'Unknown')}, Chunk {doc.metadata.get('chunk_id', 'Unknown')}]",
        }
      )
    return sources

  def ask(self, question: str) -> Dict[str, Any]:
    self.step_logs = []
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
      docs = self.retrieve(question)
      context = self.format_context(docs)
      gen_start = time.perf_counter()
      answer_raw = self.answer_chain.invoke({"context": context, "question": question})
      answer = answer_raw.content if hasattr(answer_raw, "content") else str(answer_raw)
      generation_ms = (time.perf_counter() - gen_start) * 1000

      return {
        "request_id": request_id,
        "question": question,
        "answer": answer.strip(),
        "sources": self.format_sources(docs),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "generation_ms": round(generation_ms, 2),
        "step_logs": self.step_logs.copy(),
      }
    except Exception as exc:
      logger.exception("Query failed")
      return {
        "request_id": request_id,
        "question": question,
        "answer": NOT_FOUND_MESSAGE,
        "sources": [],
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "generation_ms": 0.0,
        "step_logs": self.step_logs.copy(),
        "error": str(exc),
      }

  def ask_stream(self, question: str) -> Generator[Dict[str, Any], None, None]:
    self.step_logs = []
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
      docs = self.retrieve(question)
      context = self.format_context(docs)
      sources = self.format_sources(docs)

      yield {
        "type": "sources",
        "request_id": request_id,
        "sources": sources,
        "step_logs": self.step_logs.copy(),
      }

      full_answer = ""
      for chunk in self.answer_chain.stream({"context": context, "question": question}):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        full_answer += token
        yield {"type": "token", "token": token}

      yield {
        "type": "done",
        "request_id": request_id,
        "question": question,
        "answer": full_answer.strip(),
        "sources": sources,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "step_logs": self.step_logs.copy(),
      }
    except Exception as exc:
      logger.exception("Stream query failed")
      yield {
        "type": "error",
        "request_id": request_id,
        "message": str(exc),
        "answer": NOT_FOUND_MESSAGE,
      }


_rag_instance: DocumentQARAG | None = None


def get_rag_instance() -> DocumentQARAG:
  global _rag_instance
  if _rag_instance is None:
    _rag_instance = DocumentQARAG()
  return _rag_instance
