"""Gateway integration test for the semantic cache: proves a hit actually skips
the worker entirely (not just that SemanticCache's own logic works in
isolation -- tests/test_semantic_cache.py already covers that). In-process
over real request/response cycles via httpx's ASGITransport, same pattern as
tests/test_batch.py.
"""

import asyncio

import httpx
import pytest

from infergrid.gateway.app import create_app as create_gateway_app
from infergrid.semantic_cache import HashEmbedder, SemanticCache
from infergrid.store.app import create_app as create_store_app
from infergrid.store.client import StoreClient
from infergrid.store.node import StoreNode
from infergrid.store.transport import HttpTransport
from infergrid.worker.app import create_app as create_worker_app
from infergrid.worker.backends import SimBackend, SimConfig

FAST = SimConfig(prefill_ms_per_token=0, decode_ms_per_token=0)
BODY = {"messages": [{"role": "user", "content": "how do I reset my password?"}], "max_tokens": 10}


class CountingBackend(SimBackend):
    def __init__(self):
        super().__init__(FAST)
        self.calls = 0

    async def generate(self, req, result):
        self.calls += 1
        async for piece in super().generate(req, result):
            yield piece


@pytest.fixture
async def gateway():
    store_node = StoreNode("http://store", ["http://store"], HttpTransport(None), n_replicas=1, w=1, r=1)
    store_app = create_store_app(store_node)

    shared = httpx.AsyncClient(mounts={"http://store": httpx.ASGITransport(app=store_app)})
    store_client = StoreClient(shared, "http://store")
    cache = SemanticCache(store_client, HashEmbedder(), threshold=0.9)

    backend = CountingBackend()
    worker_client = httpx.AsyncClient(mounts={"http://worker-1": httpx.ASGITransport(app=create_worker_app(backend, "worker-1"))})
    gateway_app = create_gateway_app(["http://worker-1"], http_client=worker_client, semantic_cache=cache)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway") as client:
        yield client, backend


async def test_first_request_is_a_miss_and_reaches_the_worker(gateway):
    client, backend = gateway
    resp = await client.post("/v1/chat/completions", json=BODY)
    assert resp.status_code == 200
    assert resp.headers["x-infergrid-worker"] == "http://worker-1"
    assert backend.calls == 1


async def test_a_repeated_prompt_is_served_from_cache_without_touching_the_worker(gateway):
    client, backend = gateway
    first = await client.post("/v1/chat/completions", json=BODY)
    content = first.json()["choices"][0]["message"]["content"]
    await asyncio.sleep(0.05)  # let the fire-and-forget cache write finish before asking again

    second = await client.post("/v1/chat/completions", json=BODY)
    assert second.status_code == 200
    assert second.headers["x-infergrid-worker"] == "semantic-cache"
    assert second.json()["choices"][0]["message"]["content"] == content
    assert backend.calls == 1  # the worker was never contacted the second time


async def test_a_different_tenant_does_not_share_the_first_tenants_cache(gateway):
    client, backend = gateway
    await client.post("/v1/chat/completions", json=BODY, headers={"x-tenant-id": "acme"})
    resp = await client.post("/v1/chat/completions", json=BODY, headers={"x-tenant-id": "globex"})
    assert resp.headers["x-infergrid-worker"] == "http://worker-1"
    assert backend.calls == 2


async def test_stats_report_cache_hits_and_misses(gateway):
    client, _ = gateway
    await client.post("/v1/chat/completions", json=BODY)
    await asyncio.sleep(0.05)  # let the fire-and-forget cache write finish before asking again
    await client.post("/v1/chat/completions", json=BODY)
    stats = (await client.get("/stats")).json()
    assert stats["cache_misses"] == 1
    assert stats["cache_hits"] == 1
