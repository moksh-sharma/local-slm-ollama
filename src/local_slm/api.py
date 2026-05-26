"""FastAPI server for the Local SLM web UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from local_slm.benchmark.report import write_markdown_report
from local_slm.benchmark.runner import benchmark_model, compare_models
from local_slm.config import settings
from local_slm.models import ComparisonReport, GenerationResult
from local_slm.ollama_client import OllamaClient, hardware_note

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Local SLM", version="0.1.0")
_client: OllamaClient | None = None


def get_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client


class ChatRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)


class CompareRequest(BaseModel):
    models: str | None = None


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    client = get_client()
    models: list[str] = []
    ollama_ok = client.mock
    error: str | None = None

    try:
        models = client.list_models()
        ollama_ok = True
    except Exception as exc:  # noqa: BLE001 — surface connection errors to UI
        error = str(exc)

    return {
        "mock": client.mock,
        "ollama_ok": ollama_ok,
        "hardware": hardware_note(),
        "default_model": settings.ollama_model,
        "benchmark_models": settings.model_list,
        "models": models,
        "error": error,
    }


@app.post("/api/chat")
def api_chat(body: ChatRequest) -> GenerationResult:
    model = body.model or settings.ollama_model
    client = get_client()
    try:
        return client.generate(model, body.prompt, max_tokens=body.max_tokens)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/chat/stream")
def api_chat_stream(body: ChatRequest) -> StreamingResponse:
    model = body.model or settings.ollama_model
    client = get_client()

    def event_stream():
        try:
            for item in client.generate_stream(model, body.prompt, max_tokens=body.max_tokens):
                if isinstance(item, str):
                    yield f"data: {json.dumps({'type': 'token', 'text': item})}\n\n"
                else:
                    yield f"data: {json.dumps({'type': 'done', 'result': item.model_dump()})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/benchmark")
def api_benchmark(model: str | None = None) -> dict[str, Any]:
    client = get_client()
    tag = model or settings.ollama_model
    try:
        result = benchmark_model(client, tag)
        return result.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/compare")
def api_compare(body: CompareRequest | None = None) -> ComparisonReport:
    client = get_client()
    models = None
    if body and body.models:
        models = [m.strip() for m in body.models.split(",") if m.strip()]
    try:
        return compare_models(models, client=client)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/report/latest")
def api_report_latest() -> ComparisonReport:
    path = settings.results_dir / "latest.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No report yet. Run a comparison first.")
    return ComparisonReport.model_validate_json(path.read_text(encoding="utf-8"))


@app.get("/api/report/latest.md")
def api_report_latest_md() -> FileResponse:
    path = settings.results_dir / "latest.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No markdown report yet.")
    return FileResponse(path, media_type="text/markdown")


@app.post("/api/report/render")
def api_report_render() -> dict[str, str]:
    json_path = settings.results_dir / "latest.json"
    md_path = settings.results_dir / "latest.md"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="No report JSON found.")
    report = ComparisonReport.model_validate_json(json_path.read_text(encoding="utf-8"))
    write_markdown_report(report, md_path)
    return {"path": str(md_path)}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
