"""Gateway HTTP API: an OpenAI-compatible front door to the worker pool.

The gateway is stateless, so any number of copies can run behind a load balancer.
"""

import asyncio
import functools
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing, asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from infergrid.common import sse
from infergrid.common.schemas import ChatCompletionRequest, GenerateRequest
from infergrid.gateway.rate_limit import RateLimiter
from infergrid.gateway.router import RoundRobinRouter, Router
from infergrid.membership import SwimNode

# `read` bounds the gap between two streamed tokens, not the whole response.
WORKER_TIMEOUT = httpx.Timeout(connect=2.0, read=120.0, write=10.0, pool=5.0)

log = logging.getLogger("infergrid.gateway")


class WorkerStreamError(Exception):
    """A worker failed after it had started streaming."""


class LoadTracker:
    """Requests this gateway has in flight on each worker, and totals for /stats."""

    def __init__(self, workers: Sequence[str]):
        self.in_flight = {w: 0 for w in workers}
        self.routed = {w: 0 for w in workers}
        self.failovers = 0
        self.hedges = 0
        self.rate_limited = 0

    def reserve(self, worker: str) -> None:
        self.in_flight[worker] += 1

    def release(self, worker: str) -> None:
        self.in_flight[worker] -= 1


async def open_worker_stream(
    client: httpx.AsyncClient, candidates: Sequence[str], req: GenerateRequest, load: LoadTracker
) -> tuple[str, httpx.Response]:
    """Start a stream on the first candidate that accepts the request.

    A worker at capacity refuses with 503 (see worker/app.py) rather than queueing,
    which lands here exactly like an unreachable or errored worker: try the next
    candidate. Failing over is safe before any token has reached the client. The
    caller must release the returned worker's load reservation when the stream ends.
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


async def _first_event(events: AsyncIterator[dict]) -> dict:
    return await events.__anext__()


async def _prepend(first: dict, rest: AsyncIterator[dict]) -> AsyncIterator[dict]:
    yield first
    async for event in rest:
        yield event


async def _discard(events: AsyncIterator[dict], task: asyncio.Task | None) -> None:
    """Abandon a losing hedge: stop waiting for it and release its connection and load slot."""
    if task and not task.done():
        task.cancel()
    try:
        await events.aclose()
    except Exception:
        pass


async def open_hedged_events(
    client: httpx.AsyncClient, candidates: Sequence[str], req: GenerateRequest, load: LoadTracker, hedge_delay: float,
) -> tuple[str, AsyncIterator[dict]]:
    """Start on the best candidate; if its first token is slow, also try the next one
    and continue with whichever answers first, discarding the other.

    This targets tail latency caused by one worker being unexpectedly slow (a stuck
    request, a burst of decode-heavy neighbours, a GC-style pause) rather than down,
    at the cost of extra worker load on the (intentionally rare) occasions a hedge
    fires. Only the first attempt is hedged; a mid-stream failure after that is
    handled by resilient_events' ordinary failover, not a fresh hedge.

    Needs a real socket to test meaningfully: httpx's in-process ASGITransport (used
    by most of this test suite) runs a mounted app to completion before returning
    anything, so it cannot show one worker's headers arriving before another's body
    finishes. See tests/test_hedging.py, which binds real loopback ports instead.
    """
    primary_worker, primary_resp = await open_worker_stream(client, candidates, req, load)
    primary_events = worker_events(primary_resp, on_close=functools.partial(load.release, primary_worker))
    primary_first = asyncio.ensure_future(_first_event(primary_events))

    remaining = [w for w in candidates if w != primary_worker]
    if not remaining:
        return primary_worker, _prepend(await primary_first, primary_events)

    done, _ = await asyncio.wait({primary_first}, timeout=hedge_delay)
    if primary_first in done:
        return primary_worker, _prepend(primary_first.result(), primary_events)

    try:
        secondary_worker, secondary_resp = await open_worker_stream(client, remaining, req, load)
    except HTTPException:
        return primary_worker, _prepend(await primary_first, primary_events)  # no second candidate available

    secondary_events = worker_events(secondary_resp, on_close=functools.partial(load.release, secondary_worker))
    secondary_first = asyncio.ensure_future(_first_event(secondary_events))
    load.hedges += 1

    done, _ = await asyncio.wait({primary_first, secondary_first}, return_when=asyncio.FIRST_COMPLETED)

    if primary_first in done and primary_first.exception() is None:
        asyncio.ensure_future(_discard(secondary_events, secondary_first))
        return primary_worker, _prepend(primary_first.result(), primary_events)
    if secondary_first in done and secondary_first.exception() is None:
        asyncio.ensure_future(_discard(primary_events, primary_first))
        return secondary_worker, _prepend(secondary_first.result(), secondary_events)

    # Whichever finished first errored; fall back to the other rather than giving up
    # on the first sign of trouble.
    if primary_first in done:
        asyncio.ensure_future(_discard(primary_events, None))
        try:
            return secondary_worker, _prepend(await secondary_first, secondary_events)
        except Exception:
            raise primary_first.exception() from None
    asyncio.ensure_future(_discard(secondary_events, None))
    try:
        return primary_worker, _prepend(await primary_first, primary_events)
    except Exception:
        raise secondary_first.exception() from None


async def resilient_events(
    client: httpx.AsyncClient,
    candidates: Sequence[str],
    req: GenerateRequest,
    worker: str,
    events: AsyncIterator[dict],
    load: LoadTracker,
    max_failovers: int,
) -> AsyncIterator[dict]:
    """Stream an answer, resuming it on another worker if the current one fails midway.

    `events` is the already-open stream for the first attempt (from open_worker_stream
    or, if hedging is enabled, from open_hedged_events). The new worker on a failover
    receives the text the client already has and continues from the next token index;
    tokens are deduplicated by index, so the client sees every token exactly once even
    if a worker repeats some.
    """
    delivered: list[str] = []
    tried = [worker]
    failovers = 0
    while True:
        try:
            # aclosing() makes sure the worker connection and its load reservation are
            # released as soon as we stop reading, not whenever garbage collection runs.
            async with aclosing(events) as ev:
                async for event in ev:
                    if event["type"] == "token":
                        if event["index"] < len(delivered):
                            continue  # already delivered
                        if event["index"] > len(delivered):
                            raise WorkerStreamError(f"token {len(delivered)} missing, got {event['index']}")
                        delivered.append(event["text"])
                    yield event
            return
        except WorkerStreamError as exc:
            remaining = [w for w in candidates if w not in tried]
            if failovers >= max_failovers or not remaining:
                raise
            resume = req.model_copy(update={"resume_text": "".join(delivered), "resume_tokens": len(delivered)})
            try:
                new_worker, resp = await open_worker_stream(client, remaining, resume, load)
            except HTTPException:
                raise exc from None
            tried.extend(remaining[: remaining.index(new_worker) + 1])
            failovers += 1
            load.failovers += 1
            log.warning("request %s: %s failed at token %d (%s); resumed on %s",
                        req.request_id, worker, len(delivered), exc, new_worker)
            worker = new_worker
            events = worker_events(resp, on_close=functools.partial(load.release, worker))


def create_app(
    workers: Sequence[str],
    router: Router | None = None,
    http_client: httpx.AsyncClient | None = None,
    max_failovers: int = 2,
    membership: SwimNode | None = None,
    rate_limit: RateLimiter | None = None,
    hedge_delay_ms: float | None = None,
) -> FastAPI:
    workers = [w.rstrip("/") for w in workers]
    router = router or RoundRobinRouter()
    load = LoadTracker(workers)

    def alive_workers() -> list[str]:
        """The configured workers, filtered to ones SWIM currently reports alive.

        This is what stops the gateway from wasting a connection attempt -- and the
        seconds-long wait for it to time out or be refused -- on a worker that is
        already known to be dead; see DESIGN.md section 3.4. Falls back to the full
        list if membership has no alive workers yet (e.g. still converging right
        after startup) so a slow bootstrap never looks like a total outage.
        """
        if membership is None:
            return workers
        alive = set(membership.alive_http_urls())
        return [w for w in workers if w in alive] or workers

    def sync_capacities() -> None:
        """Push each alive worker's real capacity (gossiped over SWIM) into the router.

        Only ConsistentHashRouter uses this (duck-typed, so other routers are
        unaffected); see its docstring for why this matters over an average-based
        bound. A no-op without membership, since there is nothing to sync from.
        """
        if membership is None or not hasattr(router, "set_capacities"):
            return
        router.set_capacities({m.metadata["http_url"]: m.metadata["capacity"]
                               for m in membership.alive_members()
                               if "http_url" in m.metadata and "capacity" in m.metadata})

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if membership is not None:
            await membership.start()
        try:
            if http_client is not None:
                app.state.client = http_client
                yield
            else:
                async with httpx.AsyncClient(timeout=WORKER_TIMEOUT) as client:
                    app.state.client = client
                    yield
        finally:
            if membership is not None:
                await membership.stop()

    app = FastAPI(title="InferGrid gateway", lifespan=lifespan)
    app.state.client = http_client

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        if rate_limit is not None:
            tenant = request.headers.get("x-tenant-id", "default")
            allowed, retry_after = rate_limit.allow(tenant)
            if not allowed:
                load.rate_limited += 1
                return JSONResponse({"error": {"message": f"rate limit exceeded for tenant {tenant!r}",
                                                "type": "rate_limit_exceeded"}},
                                    status_code=429, headers={"Retry-After": str(retry_after)})

        request_id = uuid.uuid4().hex
        gen_req = GenerateRequest(request_id=request_id, messages=body.messages, max_tokens=body.max_tokens)
        sync_capacities()
        candidates = router.candidates(gen_req, alive_workers(), load.in_flight)

        if hedge_delay_ms:
            worker, events = await open_hedged_events(app.state.client, candidates, gen_req, load, hedge_delay_ms / 1000)
        else:
            worker, resp = await open_worker_stream(app.state.client, candidates, gen_req, load)
            events = worker_events(resp, on_close=functools.partial(load.release, worker))
        events = resilient_events(app.state.client, candidates, gen_req, worker, events, load, max_failovers)

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
        return {"status": "ok", "router": router.name, "workers": workers, "alive_workers": alive_workers()}

    @app.get("/stats")
    async def stats() -> dict:
        return {"router": router.name, "in_flight": load.in_flight, "routed": load.routed,
                "failovers": load.failovers, "hedges": load.hedges, "rate_limited": load.rate_limited,
                "alive_workers": alive_workers()}

    @app.get("/membership")
    async def membership_view() -> dict:
        return membership.snapshot() if membership else {}

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
        # Every failover attempt failed. Headers are already sent, so report it in-band.
        yield sse.encode({"error": {"message": str(exc), "type": "worker_failure"}})
    yield sse.encode("[DONE]")


def _openai_usage(usage: dict) -> dict:
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        "prompt_tokens_details": {"cached_tokens": usage["cached_tokens"]},
    }
