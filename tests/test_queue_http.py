"""Tests for the broker's real HTTP layer (broker/app.py + queue/client.py) and
the generic retry/DLQ consumer loop (queue/consumer.py) running against it over
real request/response cycles (in-process via httpx's ASGITransport, no sockets --
see tests/test_store_http.py for why this matters over the fully in-process
Broker tests in tests/test_queue_broker.py).
"""

import asyncio

import httpx
import pytest

from infergrid.queue.broker.app import create_app
from infergrid.queue.broker.core import Broker
from infergrid.queue.client import BrokerClient
from infergrid.queue.consumer import consume_with_retry


@pytest.fixture
async def client():
    broker = Broker(num_partitions=4)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(broker)), base_url="http://broker") as http:
        yield BrokerClient(http, "http://broker")


async def test_publish_then_poll_over_http(client: BrokerClient):
    await client.publish("t", key="a", value={"n": 1})
    [msg] = await client.poll("t", group="g", consumer_id="c1")
    assert msg.key == "a"
    assert msg.value == {"n": 1}


async def test_commit_over_http_stops_redelivery(client: BrokerClient):
    await client.publish("t", key="a", value={})
    [msg] = await client.poll("t", group="g", consumer_id="c1")
    await client.commit("t", "g", msg)
    assert await client.poll("t", group="g", consumer_id="c1") == []


async def test_consume_with_retry_processes_every_message_then_stops(client: BrokerClient):
    for i in range(5):
        await client.publish("jobs", key=f"k{i}", value={"i": i})

    seen = []
    stop = asyncio.Event()

    async def handler(msg):
        seen.append(msg.value["i"])
        if len(seen) == 5:
            stop.set()

    await asyncio.wait_for(
        consume_with_retry(client, "jobs", "workers", "w1", handler, stop=stop, poll_interval=0.01),
        timeout=5,
    )
    assert sorted(seen) == [0, 1, 2, 3, 4]
    # every message was committed, not just handled -- nothing left to redeliver
    assert await client.poll("jobs", group="workers", consumer_id="w2") == []


async def test_consume_with_retry_retries_a_transient_failure_then_succeeds(client: BrokerClient):
    await client.publish("jobs", key="a", value={})
    attempts = []
    stop = asyncio.Event()

    async def handler(msg):
        attempts.append(msg.attempt)
        if len(attempts) < 2:
            raise RuntimeError("transient")
        stop.set()

    await asyncio.wait_for(
        consume_with_retry(client, "jobs", "workers", "w1", handler, stop=stop,
                           poll_interval=0.01, base_delay=0.01, max_attempts=5),
        timeout=5,
    )
    assert attempts == [0, 1]  # the retry carries attempt+1, and it was accepted the second time
    assert await client.poll("jobs.dlq", group="watch", consumer_id="c1") == []


async def test_consume_with_retry_sends_a_poison_message_to_the_dead_letter_topic(client: BrokerClient):
    await client.publish("jobs", key="poison", value={"payload": 42})
    stop = asyncio.Event()
    attempts = []

    async def always_fails(msg):
        attempts.append(msg.attempt)
        raise ValueError("bad payload")

    async def run_until_dlq():
        task = asyncio.create_task(
            consume_with_retry(client, "jobs", "workers", "w1", always_fails, stop=stop,
                               poll_interval=0.01, base_delay=0.01, max_attempts=3)
        )
        while not await client.poll("jobs.dlq", group="watch", consumer_id="watcher"):
            await asyncio.sleep(0.01)
        stop.set()
        await task

    await asyncio.wait_for(run_until_dlq(), timeout=5)
    assert attempts == [0, 1, 2]  # gave up only after max_attempts=3 real tries
    [dead] = await client.poll("jobs.dlq", group="watch2", consumer_id="watcher2")
    assert dead.key == "poison"
    assert dead.value["payload"] == 42
    assert dead.value["_failed_topic"] == "jobs"
    assert "bad payload" in dead.value["_error"]


async def test_consume_with_retry_leaves_the_group_when_stopped():
    broker = Broker(num_partitions=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(broker)), base_url="http://b") as http:
        client = BrokerClient(http, "http://b")
        await client.publish("t", key="a", value={})
        stop = asyncio.Event()

        async def handler(msg):
            stop.set()

        await asyncio.wait_for(
            consume_with_retry(client, "t", "g", "c1", handler, stop=stop, poll_interval=0.01), timeout=5
        )
        assert broker.stats()["t"]["groups"]["g"]["members"] == []
