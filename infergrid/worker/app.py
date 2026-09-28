"""Worker HTTP API.

POST /generate streams SSE events:
    {"type": "token", "index": 0, "text": "Hello"}
    ...
    {"type": "done", "finish_reason": "stop", "usage": {...}}
or, on failure, {"type": "error", "message": "..."}.

Every token carries its index so the gateway knows exactly how much of an answer
the client has received; Phase 3 uses this to resume a stream on another worker.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from infergrid.common import sse
from infergrid.common.schemas import GenerateRequest, GenerationResult
from infergrid.worker.backends import Backend


def create_app(backend: Backend, worker_id: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await backend.close()

    app = FastAPI(title=f"InferGrid worker {worker_id}", lifespan=lifespan)

    @app.post("/generate")
    async def generate(req: GenerateRequest) -> StreamingResponse:
        async def events():
            result = GenerationResult()
            index = 0
            try:
                async for text in backend.generate(req, result):
                    yield sse.encode({"type": "token", "index": index, "text": text})
                    index += 1
            except Exception as exc:  # report any backend failure in-band; headers are already sent
                yield sse.encode({"type": "error", "message": str(exc) or type(exc).__name__})
                return
            yield sse.encode(
                {"type": "done", "finish_reason": result.finish_reason, "usage": result.usage.model_dump()}
            )

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "worker_id": worker_id}

    @app.get("/stats")
    async def stats() -> dict:
        return {"worker_id": worker_id, **backend.stats()}

    return app
