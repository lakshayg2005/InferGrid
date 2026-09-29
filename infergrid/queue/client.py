"""HTTP client for `infergrid.queue.broker`, implementing the `QueueClient` protocol."""

from __future__ import annotations

import httpx

from infergrid.queue.base import Message


class BrokerClient:
    def __init__(self, http_client: httpx.AsyncClient, base_url: str, timeout: float = 5.0):
        self.http_client = http_client
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def publish(self, topic: str, key: str, value: dict, attempt: int = 0) -> None:
        resp = await self.http_client.post(
            f"{self.base_url}/topics/{topic}/publish",
            json={"key": key, "value": value, "attempt": attempt},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    async def poll(self, topic: str, group: str, consumer_id: str, max_messages: int = 10) -> list[Message]:
        resp = await self.http_client.post(
            f"{self.base_url}/topics/{topic}/poll",
            json={"group": group, "consumer_id": consumer_id, "max_messages": max_messages},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return [Message(topic, m["partition"], m["offset"], m["key"], m["value"], m["attempt"])
                for m in resp.json()["messages"]]

    async def commit(self, topic: str, group: str, msg: Message) -> None:
        resp = await self.http_client.post(
            f"{self.base_url}/topics/{topic}/commit",
            json={"group": group, "partition": msg.partition, "offset": msg.offset},
            timeout=self.timeout,
        )
        resp.raise_for_status()

    async def leave(self, topic: str, group: str, consumer_id: str) -> None:
        resp = await self.http_client.post(
            f"{self.base_url}/topics/{topic}/leave",
            json={"group": group, "consumer_id": consumer_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
