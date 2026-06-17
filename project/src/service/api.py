import re
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from urllib.request import urlopen

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.document_loader import DocumentLoader
from src.models.rag_chain import get_rag_chain
from src.service.config import load_config
from src.service.vector_store import VectorDB


app = FastAPI(
    title="RAG System API",
    description="API системы вопросно-ответного поиска по образовательным материалам",
    version="1.0.0",
)

SUPPORTED_DATA_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".raw"}

config = load_config()
state_lock = Lock()
vdb: VectorDB | None = None
rag_chain = None


class QuestionRequest(BaseModel):
    question: str = Field(example="Что такое теорема Пифагора?")


class PredictionResponse(BaseModel):
    answer: str = Field(example="Теорема Пифагора связывает стороны прямоугольного треугольника.")
    sources: list[str] = Field(example=["Geometry.md"])
    model_version: str = Field(example="edu-qwen-14b")
    elapsed_seconds: float = Field(example=1.23)


class HealthResponse(BaseModel):
    status: str
    ollama: str
    vector_db: str
    llm_model: str


class FilesResponse(BaseModel):
    data_path: str
    files: list[str]


class RebuildResponse(BaseModel):
    status: str
    documents_loaded: int
    vector_store_path: str


def check_ollama_connection(timeout: float = 2.0) -> str:
    api_url = f"{config['ollama_base_url'].rstrip('/')}/api/tags"
    try:
        with urlopen(api_url, timeout=timeout) as response:
            return "ok" if 200 <= response.status < 300 else "unavailable"
    except Exception:
        return "unavailable"


def data_files() -> list[str]:
    data_path = Path(config["data_path"])
    if not data_path.exists():
        return []

    files = [
        str(file_path.relative_to(data_path))
        for file_path in data_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
    ]
    return sorted(files)


def ensure_dataset_exists() -> None:
    if data_files():
        return

    try:
        from src.data.download_data import main as download_data

        download_data()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Не удалось автоматически загрузить данные: {exc}",
        ) from exc

    if not data_files():
        raise HTTPException(
            status_code=404,
            detail=f"После загрузки данные не найдены в папке: {config['data_path']}",
        )


def vector_db_status() -> str:
    return "ok" if Path(config["vector_store_path"]).exists() else "missing"


def get_vector_db() -> VectorDB:
    global vdb

    if vdb is None:
        vdb = VectorDB(config)
    return vdb


def get_active_rag_chain():
    global rag_chain

    if rag_chain is None:
        rag_chain = get_rag_chain(config, get_vector_db())
    return rag_chain


def format_source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    source = metadata.get("source_file", "Неизвестный источник")

    if "page" not in metadata:
        return source

    try:
        page = int(metadata["page"]) + 1
    except (TypeError, ValueError):
        page = metadata["page"]

    return f"{source} (стр. {page})"


def normalize_answer_markdown(answer: str) -> str:
    """Convert common raw LaTeX wrappers to Streamlit-friendly Markdown math."""

    def display_math(match: re.Match) -> str:
        expr = match.group("expr").strip()
        return f"\n\n$${expr}$$\n\n"

    def inline_math(match: re.Match) -> str:
        expr = match.group("expr").strip()
        return f"${expr}$"

    normalized = answer

    normalized = re.sub(
        r"\\\[\s*(?P<expr>.*?)\s*\\\]",
        display_math,
        normalized,
        flags=re.DOTALL,
    )
    normalized = re.sub(
        r"\\\(\s*(?P<expr>.*?)\s*\\\)",
        inline_math,
        normalized,
        flags=re.DOTALL,
    )
    normalized = re.sub(
        r"(?m)^\s*\[\s*(?P<expr>[^\n\[\]]*\\[A-Za-z]+[^\n\[\]]*)\s*\]\s*$",
        display_math,
        normalized,
    )
    normalized = re.sub(
        r"\((?P<expr>\\[A-Za-z]+[^\n]*)\)",
        inline_math,
        normalized,
    )
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()


@app.get("/")
def root():
    return {
        "name": "RAG System API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния сервиса",
)
def health():
    db_status = vector_db_status()
    ollama_status = check_ollama_connection()

    return {
        "status": "healthy" if db_status == "ok" and ollama_status == "ok" else "degraded",
        "ollama": ollama_status,
        "vector_db": db_status,
        "llm_model": config["llm_model"],
    }


@app.get(
    "/files",
    response_model=FilesResponse,
    summary="Список файлов датасета",
)
def files():
    return {
        "data_path": config["data_path"],
        "files": data_files(),
    }


@app.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Пересборка базы знаний",
)
def rebuild():
    global rag_chain, vdb

    ensure_dataset_exists()
    loader = DocumentLoader(config["data_path"])
    docs = loader.load_documents()

    if not docs:
        raise HTTPException(
            status_code=404,
            detail=f"Файлы не найдены в папке: {config['data_path']}",
        )

    with state_lock:
        vdb = VectorDB(config)
        vector_store = vdb.create_vector_db(docs)
        if vector_store is None:
            rag_chain = None
            raise HTTPException(
                status_code=500,
                detail="Не удалось создать векторную базу",
            )
        rag_chain = get_rag_chain(config, vdb)

    return {
        "status": "ok",
        "documents_loaded": len(docs),
        "vector_store_path": config["vector_store_path"],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Выполнение вопросно-ответного поиска",
)
def predict(request: QuestionRequest):
    if check_ollama_connection() != "ok":
        raise HTTPException(
            status_code=503,
            detail=f"Не удалось подключиться к Ollama по адресу {config['ollama_base_url']}",
        )

    active_rag_chain = get_active_rag_chain()

    if active_rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="База знаний не инициализирована. Сначала вызовите /rebuild.",
        )

    start_time = perf_counter()

    try:
        with state_lock:
            response = active_rag_chain.invoke({"input": request.question})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    context = response.get("context", [])
    sources = list(dict.fromkeys(format_source(doc) for doc in context))

    return {
        "answer": normalize_answer_markdown(response.get("answer", "")),
        "sources": sources,
        "model_version": config["llm_model"],
        "elapsed_seconds": round(perf_counter() - start_time, 2),
    }
