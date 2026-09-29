"""End-to-end test of the Phase 5 batch-inference vertical slice: gateway ->
queue broker -> a worker's background batch consumer -> the Phase 4 store for
results, plus the billing consumer applying usage events. All in-process over
real request/response cycles via httpx's ASGITransport (see
tests/test_store_http.py and tests/test_gateway.py for the same pattern).

FastAPI's lifespan (where the worker's batch consumer task starts) does not run
automatically under ASGITransport, so the worker app's lifespan is driven
directly via `app.router.lifespan_context`, the same thing a real ASGI server
does on startup/shutdown.
"""

import asyncio

import httpx
import pytest

from infergrid.billing.consumer import apply_usage_event
from infergrid.gateway.app import create_app as create_gateway_app
from infergrid.queue.base import Message
from infergrid.queue.broker.app import create_app as create_broker_app
from infergrid.queue.broker.core import Broker
from infergrid.queue.client import BrokerClient
from infergrid.store.app import create_app as create_store_app
from infergrid.store.client import StoreClient
from infergrid.store.node import StoreNode
from infergrid.store.transport import HttpTransport
from infergrid.worker.app import create_app as create_worker_app
from infergrid.worker.backends import SimBackend, SimConfig

FAST = SimConfig(prefill_ms_per_token=0, decode_ms_per_token=0)


class CountingBackend(SimBackend):
    """Counts real generation calls, to prove idempotency skips a duplicate."""

    def __init__(self, config=FAST):
        super().__init__(config)
        self.calls = 0

    async def generate(self, req, result):
        self.calls += 1
        async for piece in super().generate(req, result):
            yield piece


class AlwaysFailsBackend(SimBackend):
    def __init__(self):
        super().__init__(FAST)

    async def generate(self, req, result):
        raise RuntimeError("poison job")
        yield  # pragma: no cover -- makes this an async generator


class System:
    def __init__(self, gateway, queue_client, store_client, worker_app, shared):
        self.gateway = gateway
        self.queue_client = queue_client
        self.store_client = store_client
        self.worker_app = worker_app
        self.shared = shared


def build_system(backend, **worker_kwargs) -> System:
    broker = Broker(num_partitions=4)
    broker_app = create_broker_app(broker)

    store_node = StoreNode("http://store", ["http://store"], HttpTransport(None), n_replicas=1, w=1, r=1)
    store_app = create_store_app(store_node)

    shared = httpx.AsyncClient(mounts={
        "http://broker": httpx.ASGITransport(app=broker_app),
        "http://store": httpx.ASGITransport(app=store_app),
    })
    queue_client = BrokerClient(shared, "http://broker")
    store_client = StoreClient(shared, "http://store")

    worker_app = create_worker_app(backend, "worker-1", queue_client=queue_client, store_client=store_client,
                                   **worker_kwargs)
    worker_client = httpx.AsyncClient(mounts={"http://worker-1": httpx.ASGITransport(app=worker_app)})
    gateway_app = create_gateway_app(["http://worker-1"], http_client=worker_client,
                                     queue_client=queue_client, store_client=store_client)
    gateway = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gateway")

    return System(gateway, queue_client, store_client, worker_app, shared)


BODY = {"requests": [
    {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 5},
    {"messages": [{"role": "user", "content": "world"}], "max_tokens": 5},
]}


async def test_submitting_a_batch_returns_a_batch_id_and_one_job_id_per_request():
    system = build_system(SimBackend(FAST))
    resp = await system.gateway.post("/v1/batches", json=BODY)
    assert resp.status_code == 202
    body = resp.json()
    assert len(body["job_ids"]) == 2
    assert body["batch_id"]


async def test_a_worker_processes_queued_jobs_and_results_become_visible():
    backend = SimBackend(FAST)
    system = build_system(backend)
    resp = await system.gateway.post("/v1/batches", json=BODY)
    batch_id = resp.json()["batch_id"]

    async with system.worker_app.router.lifespan_context(system.worker_app):
        async def wait_for_completion():
            while True:
                status = (await system.gateway.get(f"/v1/batches/{batch_id}")).json()
                if status["done"] == status["total"]:
                    return status
                await asyncio.sleep(0.05)

        status = await asyncio.wait_for(wait_for_completion(), timeout=10)

    assert status["done"] == 2
    assert all(job["status"] == "done" for job in status["jobs"])
    assert all(job["content"] for job in status["jobs"])


async def test_batches_of_an_unknown_id_404():
    system = build_system(SimBackend(FAST))
    resp = await system.gateway.get("/v1/batches/does-not-exist")
    assert resp.status_code == 404


async def test_a_redelivered_duplicate_job_is_not_reprocessed():
    """Idempotency: the worker checks the store for an existing result before
    doing any real work, so redelivering the same job_id (as at-least-once
    delivery legitimately can) must not run generation twice."""
    backend = CountingBackend()
    system = build_system(backend)
    job = {"job_id": "dup-1", "batch_id": "b1", "tenant_id": "acme",
           "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
    await system.queue_client.publish("batch-jobs", key="acme", value=job)
    await system.queue_client.publish("batch-jobs", key="acme", value=job)  # simulated redelivery

    async with system.worker_app.router.lifespan_context(system.worker_app):
        async def wait_for_result():
            while await system.store_client.get("batch-result:dup-1") is None:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(wait_for_result(), timeout=10)
        await asyncio.sleep(0.3)  # give the second delivery a chance to be (wrongly) reprocessed

    assert backend.calls == 1


async def test_a_poison_job_ends_up_on_the_dead_letter_topic_not_stuck_forever():
    system = build_system(AlwaysFailsBackend(), batch_max_attempts=2, batch_retry_delay=0.05)
    job = {"job_id": "poison-1", "batch_id": "b1", "tenant_id": "acme",
           "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
    await system.queue_client.publish("batch-jobs", key="acme", value=job)

    async with system.worker_app.router.lifespan_context(system.worker_app):
        async def wait_for_dlq():
            while not await system.queue_client.poll("batch-jobs.dlq", group="watch", consumer_id="w"):
                await asyncio.sleep(0.05)

        await asyncio.wait_for(wait_for_dlq(), timeout=10)

    [dead] = await system.queue_client.poll("batch-jobs.dlq", group="watch2", consumer_id="w2")
    assert dead.key == "acme"
    assert dead.value["job_id"] == "poison-1"
    assert "poison job" in dead.value["_error"]
    assert await system.store_client.get("batch-result:poison-1") is None


async def test_interactive_and_batch_requests_both_meter_usage_idempotently():
    backend = SimBackend(FAST)
    system = build_system(backend)

    # one interactive request
    resp = await system.gateway.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        headers={"x-tenant-id": "acme"},
    )
    assert resp.status_code == 200

    # one batch job
    batch_resp = await system.gateway.post(
        "/v1/batches", json={"requests": [{"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}]},
        headers={"x-tenant-id": "acme"},
    )
    batch_id = batch_resp.json()["batch_id"]

    async with system.worker_app.router.lifespan_context(system.worker_app):
        async def wait_for_batch_done():
            while (await system.gateway.get(f"/v1/batches/{batch_id}")).json()["done"] < 1:
                await asyncio.sleep(0.05)

        await asyncio.wait_for(wait_for_batch_done(), timeout=10)
        await asyncio.sleep(0.1)  # let the fire-and-forget interactive usage event land

    applied = 0
    while True:
        messages = await system.queue_client.poll("usage", group="billing", consumer_id="billing-1")
        if not messages:
            break
        for msg in messages:
            await apply_usage_event(system.store_client, msg)
            await system.queue_client.commit("usage", "billing", msg)
            applied += 1

    assert applied == 2  # one interactive event, one batch event
    total = await system.store_client.get("usage:acme")
    assert total is not None and total > 0

    # Redelivering the same usage event (same request_id) must not double the total --
    # this is the dedup guarantee that actually prevents double billing on redelivery.
    before = total
    dummy_event = {"request_id": "re-check", "tenant_id": "acme", "tokens": 999, "kind": "interactive"}
    msg = Message("usage", 0, 0, "acme", dummy_event)
    await apply_usage_event(system.store_client, msg)
    await apply_usage_event(system.store_client, msg)  # duplicate application of the same request_id
    after = await system.store_client.get("usage:acme")
    assert after == before + 999  # applied once, not twice
