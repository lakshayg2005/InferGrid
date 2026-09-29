"""Types shared by every queue implementation.

Application code (the gateway's batch API, a worker's batch consumer, the billing
consumer) is written against `QueueClient` alone, never against a specific
implementation -- exactly like `worker/backends/base.py::Backend` lets the same
worker code run against a simulated or a real model. Two implementations exist:
`infergrid.queue.client.BrokerClient` talks to `infergrid.queue.broker`, this
project's own from-scratch broker; `infergrid.queue.redpanda.RedpandaClient` talks
to a real Kafka-API broker (Redpanda) via `aiokafka`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Message:
    topic: str
    partition: int
    offset: int
    key: str
    value: dict[str, Any]
    attempt: int = 0


class QueueClient(Protocol):
    async def publish(self, topic: str, key: str, value: dict[str, Any], attempt: int = 0) -> None:
        """Append a message, routed to a partition by hashing `key`."""

    async def poll(self, topic: str, group: str, consumer_id: str, max_messages: int = 10) -> list[Message]:
        """This consumer's share of `topic`'s partitions, from where `group` left off.

        A message stays "in flight" (not redelivered to this consumer or reassigned
        to another) until `commit`, except once `consumer_id` has gone quiet long
        enough to look crashed, at which point its partitions -- and whatever they
        had in flight -- are handed to another consumer in the group. That is what
        makes this at-least-once: a crash after `poll` but before `commit` means
        the message is delivered again, to whoever picks up the partition next.
        """

    async def commit(self, topic: str, group: str, msg: Message) -> None:
        """Mark `msg` (and everything before it on its partition) as processed."""

    async def leave(self, topic: str, group: str, consumer_id: str) -> None:
        """Give up this consumer's partitions immediately, for a graceful shutdown."""
