"""Smoke test for the store's real HTTP layer (app.py + transport.py): everything
in test_store.py exercises StoreNode's logic directly through a fake in-process
transport, so nothing there catches a JSON-serialization or routing mistake in
the FastAPI layer itself. This wires real `HttpTransport` instances to real
`create_app()` instances, mounted in-process via httpx's ASGITransport (real
request/response cycles, no sockets -- the same trick tests/test_gateway.py uses
for its multi-worker tests).
"""

import httpx
import pytest

from infergrid.store.app import create_app
from infergrid.store.node import StoreNode
from infergrid.store.transport import HttpTransport


@pytest.fixture
async def cluster():
    # httpx needs every mount up front to build one AsyncClient, but each app needs
    # its StoreNode, and each node's HttpTransport needs that same client -- so
    # nodes are built with a placeholder transport and wired to the real client
    # (and its mounts, built from their apps) only once it exists.
    addrs = [f"http://node{i}" for i in range(3)]
    nodes: dict[str, StoreNode] = {}
    transports: dict[str, HttpTransport] = {}
    apps = {}
    for addr in addrs:
        transport = HttpTransport(None)
        node = StoreNode(addr, addrs, transport, n_replicas=3, w=2, r=2)
        nodes[addr] = node
        transports[addr] = transport
        apps[addr] = create_app(node)

    client = httpx.AsyncClient(mounts={addr: httpx.ASGITransport(app=app) for addr, app in apps.items()})
    for transport in transports.values():
        transport.http_client = client

    for node in nodes.values():
        await node.start()
    try:
        yield nodes, client
    finally:
        for node in nodes.values():
            await node.stop()
        await client.aclose()


async def test_put_over_http_is_readable_from_another_node_over_http(cluster):
    nodes, client = cluster
    addrs = list(nodes)
    resp = await client.put(f"{addrs[0]}/kv/greeting", json={"value": "hello"})
    assert resp.status_code == 200

    resp = await client.get(f"{addrs[1]}/kv/greeting")
    assert resp.status_code == 200
    assert resp.json() == {"value": "hello"}


async def test_get_missing_key_over_http_is_404(cluster):
    nodes, client = cluster
    addr = next(iter(nodes))
    resp = await client.get(f"{addr}/kv/nope")
    assert resp.status_code == 404


async def test_delete_over_http_then_get_is_404(cluster):
    nodes, client = cluster
    addrs = list(nodes)
    await client.put(f"{addrs[0]}/kv/k", json={"value": "v"})
    resp = await client.delete(f"{addrs[0]}/kv/k")
    assert resp.status_code == 200
    resp = await client.get(f"{addrs[1]}/kv/k")
    assert resp.status_code == 404


async def test_stats_endpoint_reports_key_count(cluster):
    nodes, client = cluster
    addrs = list(nodes)
    await client.put(f"{addrs[0]}/kv/a", json={"value": 1})
    await client.put(f"{addrs[0]}/kv/b", json={"value": 2})
    totals = 0
    for addr in addrs:
        resp = await client.get(f"{addr}/stats")
        totals += resp.json()["keys"]
    assert totals >= 2  # replicated across n_replicas=3 nodes, so likely > 2
