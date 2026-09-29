"""Broker HTTP API: the network boundary around `broker/core.py::Broker`.

Every InferGrid process that produces or consumes messages (the gateway, a
worker, the billing consumer) talks to this over HTTP through
`infergrid.queue.client.BrokerClient`, exactly like store nodes talk to each
other's `/kv` API and workers talk to the gateway's `/generate` -- the actual
partitioning/consumer-group/rebalancing logic never leaves this one process.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from infergrid.queue.base import Message
from infergrid.queue.broker.core import Broker


class PublishBody(BaseModel):
    key: str
    value: dict[str, Any]
    attempt: int = 0


class PollBody(BaseModel):
    group: str
    consumer_id: str
    max_messages: int = 10


class CommitBody(BaseModel):
    group: str
    partition: int
    offset: int


class LeaveBody(BaseModel):
    group: str
    consumer_id: str


def create_app(broker: Broker) -> FastAPI:
    app = FastAPI(title="InferGrid queue broker")

    @app.post("/topics/{topic}/publish")
    async def publish(topic: str, body: PublishBody) -> dict:
        msg = broker.publish(topic, body.key, body.value, body.attempt)
        return {"partition": msg.partition, "offset": msg.offset}

    @app.post("/topics/{topic}/poll")
    async def poll(topic: str, body: PollBody) -> dict:
        messages = broker.poll(topic, body.group, body.consumer_id, body.max_messages)
        return {"messages": [
            {"partition": m.partition, "offset": m.offset, "key": m.key, "value": m.value, "attempt": m.attempt}
            for m in messages
        ]}

    @app.post("/topics/{topic}/commit")
    async def commit(topic: str, body: CommitBody) -> dict:
        broker.commit(topic, body.group, Message(topic, body.partition, body.offset, "", {}))
        return {"ok": True}

    @app.post("/topics/{topic}/leave")
    async def leave(topic: str, body: LeaveBody) -> dict:
        broker.leave(topic, body.group, body.consumer_id)
        return {"ok": True}

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/stats")
    async def stats() -> dict:
        return broker.stats()

    return app
