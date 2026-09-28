import asyncio
import json

import httpx
import pytest

from infergrid.common.schemas import ChatMessage, GenerateRequest
from infergrid.gateway.app import create_app
from infergrid.gateway.rate_limit import RateLimiter
from infergrid.gateway.router import RoundRobinRouter
from infergrid.worker.app import create_app as create_worker_app
from infergrid.worker.backends import Backend, BackendError, SimBackend, SimConfig
from infergrid.worker.backends.sim import reference_answer

FAST = SimConfig(prefill_ms_per_token=0, decode_ms_per_token=0)
DEAD_WORKER = "http://127.0.0.1:1"  # nothing listens here, so connecting fails immediately
BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 5}


class CrashingBackend(SimBackend):
    """Dies after streaming `crash_after` tokens of every answer it starts."""

    def __init__(self, crash_after: int):
        super().__init__(FAST)
        self.crash_after = crash_after

    async def generate(self, req, result):
        sent = 0
        async for piece in super().generate(req, result):
            if sent == self.crash_after:
                raise BackendError("simulated crash")
            sent += 1
            yield piece


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


def cluster(worker_names: list[str], extra_urls: list[str] = (), crashing: dict[str, int] | None = None,
            backends: dict[str, Backend] | None = None, max_failovers: int = 2, rate_limit: RateLimiter | None = None,
            hedge_delay_ms: float | None = None) -> httpx.AsyncClient:
    """A gateway talking to in-process workers, with no real network involved.

    `crashing` maps worker names to the token after which that worker dies.
    `backends` supplies a specific backend instance for a worker name, overriding
    the default (and `crashing`), for tests that need a slow or overloaded worker.
    """
    crashing = crashing or {}
    backends = backends or {}

    def backend(name: str) -> Backend:
        if name in backends:
            return backends[name]
        return CrashingBackend(crashing[name]) if name in crashing else SimBackend(FAST)

    mounts = {f"http://{name}": httpx.ASGITransport(app=create_worker_app(backend(name), name))
              for name in worker_names}
    worker_client = httpx.AsyncClient(mounts=mounts)
    urls = [*extra_urls, *(f"http://{name}" for name in worker_names)]
    gateway = create_app(urls, router=RoundRobinRouter(), http_client=worker_client, max_failovers=max_failovers,
                         rate_limit=rate_limit, hedge_delay_ms=hedge_delay_ms)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway")


def expected_text(body: dict) -> str:
    pieces, _ = reference_answer([ChatMessage(**m) for m in body["messages"]], body["max_tokens"], FAST)
    return "".join(pieces)


def streamed_text(resp: httpx.Response) -> tuple[str, list[dict]]:
    payloads = sse_payloads(resp.text)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if "choices" in c)
    return text, chunks


def sse_payloads(text: str) -> list[str]:
    return [line[6:] for line in text.splitlines() if line.startswith("data: ")]


async def test_non_streaming_completion():
    async with cluster(["w1"]) as client:
        resp = await client.post("/v1/chat/completions", json=BODY)
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"]
    assert data["choices"][0]["finish_reason"] == "length"
    assert data["usage"]["completion_tokens"] == 5
    assert resp.headers["x-infergrid-worker"] == "http://w1"


async def test_streaming_completion_uses_openai_chunk_format():
    async with cluster(["w1"]) as client:
        resp = await client.post("/v1/chat/completions", json={**BODY, "stream": True})
    payloads = sse_payloads(resp.text)
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(p) for p in payloads[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    content = [c["choices"][0]["delta"].get("content") for c in chunks[1:-1]]
    assert len(content) == 5 and all(content)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 5


async def test_requests_are_spread_across_workers():
    async with cluster(["w1", "w2"]) as client:
        served = {(await client.post("/v1/chat/completions", json=BODY)).headers["x-infergrid-worker"]
                  for _ in range(4)}
    assert served == {"http://w1", "http://w2"}


async def test_stats_track_routed_requests_and_release_load():
    async with cluster(["w1", "w2"]) as client:
        await client.post("/v1/chat/completions", json=BODY)
        await client.post("/v1/chat/completions", json={**BODY, "stream": True})
        stats = (await client.get("/stats")).json()
    assert sum(stats["routed"].values()) == 2
    assert stats["in_flight"] == {"http://w1": 0, "http://w2": 0}


async def test_fails_over_when_a_worker_is_down():
    async with cluster(["w1"], extra_urls=[DEAD_WORKER]) as client:
        for _ in range(3):  # round-robin puts the dead worker first on some requests
            resp = await client.post("/v1/chat/completions", json=BODY)
            assert resp.status_code == 200
            assert resp.headers["x-infergrid-worker"] == "http://w1"


async def test_returns_503_when_no_worker_is_available():
    async with cluster([], extra_urls=[DEAD_WORKER]) as client:
        resp = await client.post("/v1/chat/completions", json=BODY)
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 503
    assert stats["in_flight"] == {DEAD_WORKER: 0}  # failed attempts release their reservation


@pytest.mark.parametrize("body", [{"messages": []}, {"messages": [{"role": "robot", "content": "x"}]}])
async def test_rejects_invalid_requests(body):
    async with cluster(["w1"]) as client:
        resp = await client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 422


LONG = {"messages": [{"role": "user", "content": "tell me a story"}], "max_tokens": 12}


async def test_stream_resumes_on_another_worker_after_a_mid_answer_crash():
    # Round-robin sends the first request to w1, which dies after 3 of 12 tokens.
    async with cluster(["w1", "w2"], crashing={"w1": 3}) as client:
        resp = await client.post("/v1/chat/completions", json={**LONG, "stream": True})
        stats = (await client.get("/stats")).json()
    text, chunks = streamed_text(resp)
    assert text == expected_text(LONG)  # nothing lost, nothing repeated
    assert chunks[-1]["usage"]["completion_tokens"] == 12
    assert stats["failovers"] == 1
    assert stats["in_flight"] == {"http://w1": 0, "http://w2": 0}


async def test_non_streaming_request_also_survives_a_crash():
    async with cluster(["w1", "w2"], crashing={"w1": 5}) as client:
        resp = await client.post("/v1/chat/completions", json=LONG)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == expected_text(LONG)


async def test_resumes_across_several_crashes():
    # w1 dies after 2 tokens; w2 resumes from token 2 and dies at its 3rd token (index 4); w3 finishes.
    async with cluster(["w1", "w2", "w3"], crashing={"w1": 2, "w2": 2}) as client:
        resp = await client.post("/v1/chat/completions", json={**LONG, "stream": True})
        stats = (await client.get("/stats")).json()
    text, _ = streamed_text(resp)
    assert text == expected_text(LONG)
    assert stats["failovers"] == 2


async def test_reports_failure_when_failover_is_disabled():
    async with cluster(["w1", "w2"], crashing={"w1": 3}, max_failovers=0) as client:
        resp = await client.post("/v1/chat/completions", json={**LONG, "stream": True})
    _, chunks = streamed_text(resp)
    assert chunks[-1]["error"]["type"] == "worker_failure"


async def test_reports_failure_when_every_worker_crashes():
    async with cluster(["w1", "w2"], crashing={"w1": 3, "w2": 1}) as client:
        resp = await client.post("/v1/chat/completions", json={**LONG, "stream": True})
        stats = (await client.get("/stats")).json()
    _, chunks = streamed_text(resp)
    assert chunks[-1]["error"]["type"] == "worker_failure"
    assert stats["in_flight"] == {"http://w1": 0, "http://w2": 0}


# -- load shedding --------------------------------------------------------------

async def test_worker_sheds_load_once_its_queue_is_full():
    backend = SimBackend(SimConfig(max_concurrency=1, max_queue=1, decode_ms_per_token=50, prefill_ms_per_token=0))
    app = create_worker_app(backend, "w1")
    req = GenerateRequest(request_id="1", messages=[ChatMessage(role="user", content="hi")], max_tokens=5)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://w1") as client:
        first = asyncio.create_task(client.post("/generate", json=req.model_dump()))
        await asyncio.sleep(0.01)  # let the first request occupy the single concurrency slot
        second = await client.post("/generate", json={**req.model_dump(), "request_id": "2"})
        assert second.status_code == 503
        assert (await first).status_code == 200


async def test_gateway_fails_over_to_another_worker_when_one_sheds_load():
    overloaded = SimBackend(SimConfig(max_queue=0))  # refuses every request outright
    async with cluster(["w1", "w2"], backends={"w1": overloaded}) as client:
        resp = await client.post("/v1/chat/completions", json=BODY)
    assert resp.status_code == 200
    assert resp.headers["x-infergrid-worker"] == "http://w2"


# -- rate limiting ----------------------------------------------------------------

def test_token_bucket_refuses_once_empty_and_refills_over_time():
    from infergrid.gateway.rate_limit import TokenBucket
    bucket = TokenBucket(capacity=2, rate=100)  # refills a token every 10 ms
    assert bucket.try_take() and bucket.try_take()
    assert not bucket.try_take()
    assert bucket.retry_after() > 0


async def test_token_bucket_refills_after_waiting():
    from infergrid.gateway.rate_limit import TokenBucket
    bucket = TokenBucket(capacity=1, rate=100)
    assert bucket.try_take()
    assert not bucket.try_take()
    await asyncio.sleep(0.02)
    assert bucket.try_take()


async def test_rate_limit_rejects_once_the_burst_is_used_and_reports_retry_after():
    # A near-zero refill rate means the burst really is exhausted for the test's
    # duration, rather than racing against how long the requests themselves take.
    limiter = RateLimiter(capacity=2, rate=1e-6)
    async with cluster(["w1"], rate_limit=limiter) as client:
        r1 = await client.post("/v1/chat/completions", json=BODY)
        r2 = await client.post("/v1/chat/completions", json=BODY)
        r3 = await client.post("/v1/chat/completions", json=BODY)
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3.status_code == 429
    assert float(r3.headers["retry-after"]) > 0
    assert r3.json()["error"]["type"] == "rate_limit_exceeded"


async def test_rate_limit_tracks_each_tenant_separately():
    limiter = RateLimiter(capacity=1, rate=1e-6)
    async with cluster(["w1"], rate_limit=limiter) as client:
        a1 = await client.post("/v1/chat/completions", json=BODY, headers={"X-Tenant-Id": "tenant-a"})
        b1 = await client.post("/v1/chat/completions", json=BODY, headers={"X-Tenant-Id": "tenant-b"})
        a2 = await client.post("/v1/chat/completions", json=BODY, headers={"X-Tenant-Id": "tenant-a"})
    assert a1.status_code == 200
    assert b1.status_code == 200
    assert a2.status_code == 429


async def test_no_rate_limit_by_default():
    async with cluster(["w1"]) as client:
        for _ in range(10):
            assert (await client.post("/v1/chat/completions", json=BODY)).status_code == 200


# -- hedged requests --------------------------------------------------------------
#
# httpx's in-process ASGITransport (used everywhere in this file) runs a mounted
# worker app to completion before open_worker_stream ever sees a response, so a
# slow worker's whole answer -- sleep included -- finishes before the gateway can
# react to it. There is no way to make that race-sensitive with this transport, so
# the test that checks hedging actually picks the faster worker lives in
# test_hedging.py, which binds real loopback sockets instead. These tests cover
# what does hold regardless of transport: no hedge fires when none should.


async def test_no_hedge_fires_when_the_primary_answers_before_the_delay():
    async with cluster(["w1", "w2"], hedge_delay_ms=200) as client:
        resp = await client.post("/v1/chat/completions", json=BODY)
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200
    assert stats["hedges"] == 0


async def test_hedging_is_off_by_default():
    async with cluster(["w1", "w2"]) as client:
        resp = await client.post("/v1/chat/completions", json=BODY)
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200
    assert stats["hedges"] == 0


async def test_hedging_with_a_single_worker_never_fires():
    slow = SlowBackend(delay=0.05)
    async with cluster(["w1"], backends={"w1": slow}, hedge_delay_ms=10) as client:
        resp = await client.post("/v1/chat/completions", json=BODY)
        stats = (await client.get("/stats")).json()
    assert resp.status_code == 200
    assert stats["hedges"] == 0
