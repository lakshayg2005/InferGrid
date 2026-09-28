import json

import httpx
import pytest

from infergrid.gateway.app import create_app
from infergrid.gateway.router import RoundRobinRouter
from infergrid.worker.app import create_app as create_worker_app
from infergrid.worker.backends import SimBackend, SimConfig

FAST = SimConfig(prefill_ms_per_token=0, decode_ms_per_token=0)
DEAD_WORKER = "http://127.0.0.1:1"  # nothing listens here, so connecting fails immediately
BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 5}


def cluster(worker_names: list[str], extra_urls: list[str] = ()) -> httpx.AsyncClient:
    """A gateway talking to in-process workers, with no real network involved."""
    mounts = {f"http://{name}": httpx.ASGITransport(app=create_worker_app(SimBackend(FAST), name))
              for name in worker_names}
    worker_client = httpx.AsyncClient(mounts=mounts)
    urls = [*extra_urls, *(f"http://{name}" for name in worker_names)]
    gateway = create_app(urls, router=RoundRobinRouter(), http_client=worker_client)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://gateway")


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
