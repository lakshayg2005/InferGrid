"""Tests for create_app(..., dynamic_workers=True): a gateway that routes to
whatever SWIM currently reports alive, not a fixed --workers list -- what lets
an autoscaled-up worker (infergrid/autoscale/, DESIGN.md section 3.9) become
routable with no gateway restart. Real SWIM nodes over loopback UDP (fast
settings, same as tests/test_swim.py), a worker mounted in-process via
ASGITransport.
"""

import asyncio

import httpx
import pytest

from infergrid.gateway.app import create_app as create_gateway_app
from infergrid.membership import SwimNode
from infergrid.worker.app import create_app as create_worker_app
from infergrid.worker.backends import SimBackend, SimConfig

FAST_SWIM = dict(protocol_period=0.03, ping_timeout=0.03, indirect_count=2, suspicion_timeout=0.2,
                 gossip_retransmits=8)
FAST_BACKEND = SimConfig(prefill_ms_per_token=0, decode_ms_per_token=0)
_next_port = iter(range(19500, 19999))


def swim_addr() -> str:
    return f"127.0.0.1:{next(_next_port)}"


async def wait_until(cond, timeout: float = 5.0, step: float = 0.01) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if cond():
            return True
        await asyncio.sleep(step)
    return False


async def test_a_worker_never_in_the_static_list_is_routed_to_once_swim_reports_it():
    gateway_swim = SwimNode(swim_addr(), **FAST_SWIM)
    worker_swim = SwimNode(swim_addr(), seeds=[gateway_swim.addr], metadata={"http_url": "http://worker-1"},
                           **FAST_SWIM)
    await gateway_swim.start()
    await worker_swim.start()
    try:
        assert await wait_until(lambda: "http://worker-1" in gateway_swim.alive_http_urls())

        worker_app = create_worker_app(SimBackend(FAST_BACKEND), "worker-1")
        worker_client = httpx.AsyncClient(mounts={"http://worker-1": httpx.ASGITransport(app=worker_app)})
        gateway_app = create_gateway_app([], http_client=worker_client, membership=gateway_swim,
                                         dynamic_workers=True)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gw") as client:
            resp = await client.post("/v1/chat/completions",
                                     json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
            assert resp.status_code == 200
            assert resp.headers["x-infergrid-worker"] == "http://worker-1"
    finally:
        await gateway_swim.stop()
        await worker_swim.stop()


async def test_without_membership_dynamic_workers_503s_instead_of_silently_routing_nowhere():
    gateway_app = create_gateway_app([], dynamic_workers=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gw") as client:
        resp = await client.post("/v1/chat/completions",
                                 json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
        assert resp.status_code == 503


async def test_a_worker_that_stops_gossiping_stops_being_routed_to():
    gateway_swim = SwimNode(swim_addr(), **FAST_SWIM)
    worker_swim = SwimNode(swim_addr(), seeds=[gateway_swim.addr], metadata={"http_url": "http://worker-1"},
                           **FAST_SWIM)
    await gateway_swim.start()
    await worker_swim.start()
    try:
        assert await wait_until(lambda: "http://worker-1" in gateway_swim.alive_http_urls())
        await worker_swim.stop()  # no goodbye message, like a killed process
        assert await wait_until(lambda: "http://worker-1" not in gateway_swim.alive_http_urls())

        gateway_app = create_gateway_app([], membership=gateway_swim, dynamic_workers=True)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app), base_url="http://gw") as client:
            resp = await client.post("/v1/chat/completions",
                                     json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
            assert resp.status_code == 503
    finally:
        await gateway_swim.stop()
