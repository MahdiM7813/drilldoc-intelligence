from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
from pathlib import Path
import os
from typing import Any, Callable

import yaml
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import json as _json

from .schemas import ErrorResponse, HealthResponse, LLMReadiness, RetrievalRequest, RetrievalResponse
from .service import GenerationBusy, InvalidQuery, RetrievalServiceProtocol, RetrievalUnavailable, Stage5RetrievalService, build_llm_enhanced_service, build_retrieval_service
from rag_system.runtime_config import data_root, resolve_data_path
from rag_system.llm_runtime import inspect_llm_readiness
from rag_system.standards.manager import StandardLibraryManager, build_standard_manager
from rag_system.standards.query_service import StandardAwareRetrievalService
from rag_system.standards.registry import DuplicateStandard, InvalidStatusTransition, StandardNotFound
from rag_system.standards.schemas import StandardEntry, StandardStatusUpdate
from rag_system.standards.storage import UploadValidationError
from rag_system.chat.repository import ChatRepository
from rag_system.chat.schemas import ChatRequest, ChatResponse
from rag_system.chat.service import ChatService

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_CONFIG = ROOT / "config" / "api.yaml"


def load_api_config(path: Path = DEFAULT_API_CONFIG) -> dict[str, Any]:
    configured_file = os.environ.get("DRILLDOC_API_CONFIG")
    config_path = Path(configured_file).expanduser().resolve() if configured_file else path.resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("API configuration must be a mapping.")
    config = dict(raw)
    config["host"] = os.environ.get("DRILLDOC_API_HOST", config.get("host", "127.0.0.1"))
    config["port"] = int(os.environ.get("DRILLDOC_API_PORT", config.get("port", 9000)))
    data = data_root()
    defaults = {
        "chat_database_path": data / "chat.sqlite3",
        "standards_registry_path": data / "standards" / "registry.json",
        "standards_upload_root": data / "standards" / "uploads",
    }
    env_names = {
        "chat_database_path": "DRILLDOC_CHAT_DB",
        "standards_registry_path": "DRILLDOC_STANDARDS_REGISTRY",
        "standards_upload_root": "DRILLDOC_STANDARDS_UPLOAD_DIR",
    }
    for key, default in defaults.items():
        config[key] = str(resolve_data_path(os.environ.get(env_names[key]), default=default))
    hybrid_value = os.environ.get("DRILLDOC_HYBRID_CONFIG", config.get("hybrid_config_path", "hybrid_retrieval.yaml"))
    hybrid_path = Path(hybrid_value).expanduser()
    config["hybrid_config_path"] = str(hybrid_path.resolve() if hybrid_path.is_absolute() else (config_path.parent / hybrid_path).resolve())
    config["current_collection_id"] = os.environ.get("DRILLDOC_COLLECTION_ID", config.get("current_collection_id", "drilldoc_default"))
    config["vector_index_path"] = str(resolve_data_path(os.environ.get("DRILLDOC_VECTOR_INDEX_DIR"), default=data / "indexes" / "chroma"))
    return config


def create_app(service: RetrievalServiceProtocol | None = None, config: dict[str, Any] | None = None,
               standard_manager: StandardLibraryManager | None = None,
               llm_readiness_check: Callable[[], dict[str, Any]] | None = None) -> FastAPI:
    settings = load_api_config()
    readiness_check = llm_readiness_check or inspect_llm_readiness
    if config:
        settings.update(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.standard_manager = standard_manager or build_standard_manager(settings)
        app.state.chat_repository = ChatRepository(settings.get("chat_database_path", str(ROOT / "storage" / "chat.sqlite3")))
        app.state.retrieval_service = service
        app.state.startup_error = None
        if app.state.retrieval_service is None:
            try:
                base_service = build_retrieval_service(settings)
                collection_id = str(settings["current_collection_id"])
                routed_service = StandardAwareRetrievalService(app.state.standard_manager.registry, {collection_id: base_service})
                app.state.retrieval_service = build_llm_enhanced_service(routed_service)
            except Exception as exc:
                app.state.startup_error = exc
        yield
        current = app.state.retrieval_service
        close = getattr(current, "close", None)
        if callable(close):
            close()

    app = FastAPI(title="DrillDoc Intelligence RAG Service", version="1.0.0", lifespan=lifespan)

    @app.exception_handler(InvalidQuery)
    async def invalid_query_handler(_: Request, exc: InvalidQuery):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RetrievalUnavailable)
    async def unavailable_handler(_: Request, exc: RetrievalUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(GenerationBusy)
    async def generation_busy_handler(_: Request, exc: GenerationBusy):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(DuplicateStandard)
    async def duplicate_handler(_: Request, exc: DuplicateStandard):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(StandardNotFound)
    async def not_found_handler(_: Request, exc: StandardNotFound):
        return JSONResponse(status_code=404, content={"detail": f"standard not found: {exc.args[0]}"})

    @app.exception_handler(InvalidStatusTransition)
    async def transition_handler(_: Request, exc: InvalidStatusTransition):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(UploadValidationError)
    async def upload_validation_handler(_: Request, exc: UploadValidationError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def internal_error_handler(_: Request, __: Exception):
        return JSONResponse(status_code=500, content={"detail": "unexpected internal error"})

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        return HealthResponse(api_ready=True, rag_ready=request.app.state.retrieval_service is not None)

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        backend = request.app.state.retrieval_service
        if backend is None:
            raise RetrievalUnavailable("retrieval backend unavailable")
        return await ChatService(request.app.state.chat_repository, backend).answer(payload.conversation_id, payload.query, payload.document_ids, payload.language)

    @app.post("/api/v1/chat/stream")
    async def chat_stream(payload: ChatRequest, request: Request):
        result = await chat(payload, request)
        async def events():
            yield f"event: start\ndata: {_json.dumps({'conversation_id':result.conversation_id}, ensure_ascii=False)}\n\n"
            yield "event: status\ndata: {\"stage\":\"retrieval\"}\n\n"
            yield "event: status\ndata: {\"stage\":\"answering\"}\n\n"
            # Deliver the already-generated grounded answer incrementally.
            for offset in range(0, len(result.answer), 96):
                yield f"event: answer_delta\ndata: {_json.dumps({'text':result.answer[offset:offset+96]}, ensure_ascii=False)}\n\n"
            for source in result.sources:
                yield f"event: citation\ndata: {_json.dumps(source.model_dump(), ensure_ascii=False)}\n\n"
            yield f"event: done\ndata: {_json.dumps(result.model_dump(), ensure_ascii=False)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    @app.post("/api/v1/conversations")
    async def create_conversation(request: Request):
        body = await request.json()
        return {"id": request.app.state.chat_repository.create_conversation(str(body.get("title") or "New chat"))}

    @app.get("/api/conversations")
    async def conversations(request: Request):
        return request.app.state.chat_repository.conversations()

    @app.patch("/api/conversations/{conversation_id}")
    async def rename_conversation(conversation_id: str, request: Request):
        body = await request.json()
        request.app.state.chat_repository.rename_conversation(conversation_id, body.get("title") or "New chat")
        return {"id": conversation_id}

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    async def delete_conversation(conversation_id: str, request: Request):
        request.app.state.chat_repository.delete_conversation(conversation_id)

    @app.get("/api/v1/conversations/{conversation_id}")
    async def get_conversation(conversation_id: str, request: Request):
        messages=request.app.state.chat_repository.messages(conversation_id)
        return {"id":conversation_id,"messages":messages}

    @app.get("/api/v1/conversations/{conversation_id}/messages")
    async def conversation_messages(conversation_id: str, request: Request):
        return request.app.state.chat_repository.messages(conversation_id)

    @app.get("/readiness", response_model=LLMReadiness)
    async def readiness() -> LLMReadiness:
        return LLMReadiness.model_validate(await asyncio.to_thread(readiness_check))

    @app.get("/ready", response_model=None)
    async def service_readiness(request: Request) -> dict[str, Any] | JSONResponse:
        if request.app.state.startup_error is not None or request.app.state.retrieval_service is None:
            return JSONResponse(status_code=503, content={"detail": "retrieval backend unavailable"})
        llm = LLMReadiness.model_validate(await asyncio.to_thread(readiness_check))
        if llm.status != "available" or not all(role.available for role in llm.roles.values()):
            return JSONResponse(status_code=503, content={"detail": "selected LLM runtime unavailable"})
        return {
            "status": "ready",
            "retrieval": "available",
            "embedding_model": "BAAI/bge-m3",
            "llm": llm.model_dump(),
        }

    @app.get("/api/v1/standards", response_model=list[StandardEntry])
    async def list_standards(request: Request) -> list[StandardEntry]:
        return request.app.state.standard_manager.list_standards()

    @app.post("/api/v1/standards/upload", response_model=StandardEntry, status_code=201)
    async def upload_standard(
        request: Request,
        file: UploadFile = File(...),
        name: str = Form(...),
        version: str = Form(...),
        revision: str | None = Form(default=None),
        metadata: str | None = Form(default=None),
    ) -> StandardEntry:
        try:
            parsed_metadata = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError as exc:
            raise UploadValidationError("metadata must be a JSON object") from exc
        if not isinstance(parsed_metadata, dict):
            raise UploadValidationError("metadata must be a JSON object")
        content = await file.read()
        try:
            return request.app.state.standard_manager.upload(
                filename=file.filename or "upload.pdf", content=content, name=name, version=version,
                revision=revision, metadata=parsed_metadata,
            )
        except (DuplicateStandard, UploadValidationError):
            raise
        except Exception as exc:
            raise RetrievalUnavailable("standard indexing failed") from exc

    @app.patch("/api/v1/standards/{standard_id}/status", response_model=StandardEntry)
    async def update_standard_status(standard_id: str, payload: StandardStatusUpdate, request: Request) -> StandardEntry:
        return request.app.state.standard_manager.set_status(standard_id, payload.status)

    @app.post(
        "/api/v1/retrieval/query",
        response_model=RetrievalResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    )
    async def retrieval_query(payload: RetrievalRequest, request: Request) -> RetrievalResponse:
        if not payload.query.strip():
            raise InvalidQuery("query must not be empty")
        backend = request.app.state.retrieval_service
        if backend is None:
            raise RetrievalUnavailable("retrieval service unavailable")
        return await backend.query(payload.query, payload.top_k, payload.filters, payload.standard_id)

    static_root = ROOT.parent / "static_app"
    if static_root.is_dir():
        app.mount("/", StaticFiles(directory=static_root, html=True), name="static-app")
    return app


app = create_app()

