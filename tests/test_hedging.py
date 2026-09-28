"""Hedged requests need real streaming timing to test meaningfully.

httpx's in-process ASGITransport (used by the rest of this suite) runs a mounted
app to completion and buffers the whole response before handing anything back, so
it cannot show one worker's headers arriving before another's body finishes. These
tests bind real loopback TCP sockets instead, so the gateway genuinely races two
in-flight HTTP streams the way it would against real workers.
"""

import asyncio
import time

import httpx
import uvicorn

from infergrid.gateway.app import create_app
from infergrid.gateway.router import RoundRobinRouter
from infergrid.worker.app import create_app as create_worker_app
from infergrid.worker.backends import SimBackend, SimConfig

FAST = SimConfig(prefill_ms_per_token=0, decode_ms_per_token=0)
BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 5}


class SlowBackend(SimBackend):
    """Like SimBackend, but waits `delay` seconds before its first token."""

    def __init__(self, delay: float, config: SimConfig | None = None):
        super().__init__(config or FAST)
        self.delay = delay

    async def generate(self, req, result):
        gen = super().generate(req, result)
        first = await gen.__anext__()
        await asyncio.sleep(self.delay)
        yield first
        async for piece in gen:
            yield piece


class RealWorker:
    """A worker served over a real loopback socket, on whatever port the OS picks."""

    def __init__(self, backend, worker_id: str):
        app = create_worker_app(backend, worker_id)
        self._server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="critical"))

    async def __aenter__(self) -> str:
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.005)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        return self.url

    async def __aexit__(self, *exc) -> None:
        self._server.should_exit = True
        await self._task


async def test_hedging_over_real_streaming_uses_whichever_worker_answers_first():
    async with RealWorker(SlowBackend(delay=0.3), "w1") as w1, RealWorker(SimBackend(FAST), "w2") as w2:
        async with httpx.AsyncClient(timeout=5.0) as worker_client:
            gateway = create_app([w1, w2], router=RoundRobinRouter(), http_client=worker_client, hedge_delay_ms=30)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway") as client:
                start = time.perf_counter()
                resp = await client.post("/v1/chat/completions", json=BODY)
                elapsed = time.perf_counter() - start
                await asyncio.sleep(0.05)  # let the abandoned hedge's cleanup task finish
                stats = (await client.get("/stats")).json()

    assert resp.status_code == 200
    assert resp.headers["x-infergrid-worker"] == w2  # the fast worker won, not the round-robin pick (w1)
    assert elapsed < 0.2, "should not have waited for the slow worker's 0.3s delay"
    assert stats["hedges"] == 1
    assert stats["in_flight"] == {w1: 0, w2: 0}  # the loser's connection and slot were released


async def test_no_hedge_fires_when_the_primary_is_fast_over_real_streaming():
    async with RealWorker(SimBackend(FAST), "w1") as w1, RealWorker(SimBackend(FAST), "w2") as w2:
        async with httpx.AsyncClient(timeout=5.0) as worker_client:
            gateway = create_app([w1, w2], router=RoundRobinRouter(), http_client=worker_client, hedge_delay_ms=200)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway") as client:
                resp = await client.post("/v1/chat/completions", json=BODY)
                stats = (await client.get("/stats")).json()

    assert resp.status_code == 200
    assert resp.headers["x-infergrid-worker"] == w1
    assert stats["hedges"] == 0
