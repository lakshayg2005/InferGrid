"""Gateway HTTP API: an OpenAI-compatible front door to the worker pool.

The gateway is stateless, so any number of copies can run behind a load balancer.
"""

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from infergrid.common import sse
from infergrid.common.schemas import ChatCompletionRequest, GenerateRequest
from infergrid.gateway.router import RoundRobinRouter, Router

# `read` bounds the gap between two streamed tokens, not the whole response.
WORKER_TIMEOUT = httpx.Timeout(connect=2.0, read=120.0, write=10.0, pool=5.0)


class WorkerStreamError(Exception):
    """A worker failed after it had started streaming."""


class LoadTracker:
    """Requests this gateway has in flight on each worker, and totals for /stats."""

    def __init__(self, workers: Sequence[str]):
        self.in_flight = {w: 0 for w in workers}
        self.routed = {w: 0 for w in workers}

    def reserve(self, worker: str) -> None:
        self.in_flight[worker] += 1

    def release(self, worker: str) -> None:
        self.in_flight[worker] -= 1


async def open_worker_stream(
    client: httpx.AsyncClient, candidates: Sequence[str], req: GenerateRequest, load: LoadTracker
) -> tuple[str, httpx.Response]:
    """Start a stream on the first candidate that accepts the request.

    Failing over is safe here because no token has reached the client yet. The caller
    must release the returned worker's load reservation when the stream ends.
    """
    errors = []
    for worker in candidates:
        # Reserve before awaiting: requests routed while this one connects must see it,
        # or a burst of simultaneous requests would all pick the same "idle" worker.
        load.reserve(worker)
        try:
            request = client.build_request("POST", f"{worker}/generate", json=req.model_dump())
            resp = await client.send(request, stream=True)
        except httpx.TransportError as exc:
            load.release(worker)
            errors.append(f"{worker}: {exc!r}")
            continue
        if resp.status_code == 200:
            load.routed[worker] += 1
            return worker, resp
        await resp.aclose()
        load.release(worker)
        errors.append(f"{worker}: HTTP {resp.status_code}")
    raise HTTPException(status_code=503, detail={"message": "no worker available", "errors": errors})


async def worker_events(resp: httpx.Response, on_close: Callable[[], None] = lambda: None) -> AsyncIterator[dict]:
    """Decode a worker's SSE stream, raising WorkerStreamError if it fails midway."""
    try:
        async for data in sse.iter_data(resp.aiter_lines()):
            event = json.loads(data)
            if event["type"] == "error":
                raise WorkerStreamError(event["message"])
            yield event
            if event["type"] == "done":
                return
        raise WorkerStreamError("worker closed the stream before finishing")
    except httpx.TransportError as exc:
        raise WorkerStreamError(f"lost connection to worker: {exc!r}") from exc
    finally:
        await resp.aclose()
        on_close()


def create_app(
    workers: Sequence[str],
    router: Router | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    workers = [w.rstrip("/") for w in workers]
    router = router or RoundRobinRouter()
    load = LoadTracker(workers)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if http_client is not None:
            yield
            return
        async with httpx.AsyncClient(timeout=WORKER_TIMEOUT) as client:
            app.state.client = client
            yield

    app = FastAPI(title="InferGrid gateway", lifespan=lifespan)
    app.state.client = http_client

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        request_id = uuid.uuid4().hex
        gen_req = GenerateRequest(request_id=request_id, messages=body.messages, max_tokens=body.max_tokens)
        candidates = router.candidates(gen_req, workers, load.in_flight)
        worker, resp = await open_worker_stream(app.state.client, candidates, gen_req, load)
        events = worker_events(resp, on_close=lambda: load.release(worker))

        completion_id = f"chatcmpl-{request_id}"
        created = int(time.time())
        headers = {"X-Request-Id": request_id, "X-InferGrid-Worker": worker}

        if body.stream:
            return StreamingResponse(
                _openai_chunks(events, completion_id, created, body.model),
                media_type="text/event-stream",
                headers=headers,
            )

        parts = []
        try:
            async for event in events:
                if event["type"] == "token":
                    parts.append(event["text"])
                else:
                    done = event
        except WorkerStreamError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": body.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(parts)},
                        "finish_reason": done["finish_reason"],
                    }
                ],
                "usage": _openai_usage(done["usage"]),
            },
            headers=headers,
        )

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "router": router.name, "workers": workers}

    @app.get("/stats")
    async def stats() -> dict:
        return {"router": router.name, "in_flight": load.in_flight, "routed": load.routed}

    return app


async def _openai_chunks(
    events: AsyncIterator[dict], completion_id: str, created: int, model: str
) -> AsyncIterator[str]:
    """Translate worker events into OpenAI `chat.completion.chunk` SSE messages."""

    def chunk(delta: dict, finish_reason: str | None = None, **extra) -> str:
        return sse.encode(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                **extra,
            }
        )

    yield chunk({"role": "assistant"})
    try:
        async for event in events:
            if event["type"] == "token":
                yield chunk({"content": event["text"]})
            else:
                yield chunk({}, event["finish_reason"], usage=_openai_usage(event["usage"]))
    except WorkerStreamError as exc:
        # Headers are already sent, so the failure is reported in-band.
        # Phase 3 replaces this with resuming the stream on another worker.
        yield sse.encode({"error": {"message": str(exc), "type": "worker_failure"}})
    yield sse.encode("[DONE]")


def _openai_usage(usage: dict) -> dict:
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        "prompt_tokens_details": {"cached_tokens": usage["cached_tokens"]},
    }
