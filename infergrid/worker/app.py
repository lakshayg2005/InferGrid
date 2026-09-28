"""Worker HTTP API.

POST /generate streams SSE events:
    {"type": "token", "index": 0, "text": "Hello"}
    ...
    {"type": "done", "finish_reason": "stop", "usage": {...}}
or, on failure, {"type": "error", "message": "..."}.

Every token carries its index so the gateway knows exactly how much of an answer
the client has received, and can resume the stream on another worker if this one
dies. A resumed request numbers its tokens from `resume_tokens` onwards.

If the worker is already at capacity, /generate is refused outright with 503
before any work starts (load shedding), rather than queued behind everything
already running: the gateway treats any non-200 response as "try the next
worker" (see gateway/app.py), so a shed request fails over immediately instead
of waiting in line.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from infergrid.common import sse
from infergrid.common.schemas import GenerateRequest, GenerationResult
from infergrid.membership import SwimNode
from infergrid.worker.backends import Backend


def create_app(backend: Backend, worker_id: str, membership: SwimNode | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if membership:
            await membership.start()
        yield
        if membership:
            await membership.stop()
        await backend.close()

    app = FastAPI(title=f"InferGrid worker {worker_id}", lifespan=lifespan)

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        if backend.queue_depth() >= backend.max_queue:
            return JSONResponse(
                {"error": "overloaded", "worker_id": worker_id, "queue_depth": backend.queue_depth()},
                status_code=503,
            )

        async def events():
            result = GenerationResult()
            index = req.resume_tokens
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

    @app.get("/membership")
    async def membership_view() -> dict:
        return membership.snapshot() if membership else {}

    return app
