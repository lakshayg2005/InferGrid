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

If `queue_client` is given, this worker also runs a background consumer over
the "batch-jobs" topic (DESIGN.md section 3.6): it only pulls a job once
`backend.queue_depth()` is below `idle_queue_depth`, so batch work fills idle
capacity instead of competing with interactive requests for it.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from infergrid.common import sse
from infergrid.common.schemas import ChatMessage, GenerateRequest, GenerationResult
from infergrid.membership import SwimNode
from infergrid.queue.base import Message, QueueClient
from infergrid.queue.consumer import consume_with_retry
from infergrid.store.client import StoreClient
from infergrid.worker.backends import Backend


def create_app(
    backend: Backend,
    worker_id: str,
    membership: SwimNode | None = None,
    queue_client: QueueClient | None = None,
    store_client: StoreClient | None = None,
    idle_queue_depth: int = 1,
    batch_max_attempts: int = 3,
    batch_retry_delay: float = 1.0,
) -> FastAPI:
    async def process_batch_job(msg: Message) -> None:
        job = msg.value
        result_key = f"batch-result:{job['job_id']}"
        if await store_client.get(result_key) is not None:
            return  # already processed -- a redelivered duplicate is a no-op, not repeated work
        while backend.queue_depth() >= idle_queue_depth:
            await asyncio.sleep(0.2)
        req = GenerateRequest(request_id=job["job_id"], messages=[ChatMessage(**m) for m in job["messages"]],
                              max_tokens=job.get("max_tokens"))
        result = GenerationResult()
        text = "".join([piece async for piece in backend.generate(req, result)])
        await store_client.put(result_key, {"content": text, "finish_reason": result.finish_reason,
                                            "usage": result.usage.model_dump()})
        if queue_client is not None:
            tokens = result.usage.prompt_tokens + result.usage.completion_tokens
            await queue_client.publish("usage", key=job["tenant_id"], value={
                "request_id": job["job_id"], "tenant_id": job["tenant_id"], "tokens": tokens, "kind": "batch",
            })

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if membership:
            await membership.start()
        stop = asyncio.Event()
        batch_task = None
        if queue_client is not None and store_client is not None:
            batch_task = asyncio.create_task(consume_with_retry(
                queue_client, "batch-jobs", "workers", worker_id, process_batch_job,
                dlq_topic="batch-jobs.dlq", stop=stop,
                max_attempts=batch_max_attempts, base_delay=batch_retry_delay,
            ))
        yield
        stop.set()
        if batch_task is not None:
            await batch_task
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
