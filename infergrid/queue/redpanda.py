"""A `QueueClient` backed by a real Kafka-API broker (Redpanda, or Kafka itself)
via `aiokafka`, instead of `infergrid.queue.broker`'s from-scratch one.

**Not exercised by this project's test suite or by any script here, and not
run in this development environment (no Docker available when it was
written).** Every other piece of InferGrid that claims to work has a real,
repeatable test or chaos script behind that claim (see DESIGN.md throughout);
this file deliberately does not make that claim. It exists to show what
swapping the from-scratch broker for a real one looks like behind the same
`QueueClient` interface the rest of the application already codes against --
validate it yourself against a running broker (`docker compose up -d`, see
`docker-compose.yml`) before trusting it.

Needs the optional `kafka` extra: `pip install -e ".[kafka]"`.

Two real semantic differences from `infergrid.queue.broker`, both consequences
of aiokafka wrapping a real Kafka client rather than this project's simplified
single-in-flight-message-per-partition model:
- `poll` here returns whatever aiokafka's internal fetcher has already
  buffered for the partitions currently assigned to this consumer (real
  pipelined fetching), not "at most one new message per partition since the
  last commit."
- Partition assignment and rebalancing are entirely the broker's and
  aiokafka's job (`group_id` alone triggers Kafka's real consumer-group
  protocol) -- this class does not implement `_rebalance`-style logic itself.
"""

from __future__ import annotations

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.structs import TopicPartition
except ImportError as exc:  # pragma: no cover -- optional dependency
    raise ImportError("infergrid.queue.redpanda needs the 'kafka' extra: pip install -e \".[kafka]\"") from exc

import json

from infergrid.queue.base import Message


class RedpandaClient:
    def __init__(self, bootstrap_servers: str):
        self.bootstrap_servers = bootstrap_servers
        self._producer: AIOKafkaProducer | None = None
        self._consumers: dict[tuple[str, str], AIOKafkaConsumer] = {}  # (topic, group) -> live consumer

    async def _producer_client(self) -> AIOKafkaProducer:
        if self._producer is None:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                key_serializer=lambda k: k.encode(),
                value_serializer=lambda v: json.dumps(v).encode(),
            )
            await self._producer.start()
        return self._producer

    async def publish(self, topic: str, key: str, value: dict, attempt: int = 0) -> None:
        producer = await self._producer_client()
        # `attempt` travels inside the value (Kafka has no first-class retry-count
        # field); infergrid.queue.broker.core.Broker tracks it as a Message field
        # instead, purely because it owns the log format and can afford to.
        await producer.send_and_wait(topic, key=key, value={**value, "_attempt": attempt})

    async def _consumer_for(self, topic: str, group: str, consumer_id: str) -> AIOKafkaConsumer:
        key = (topic, group)
        if key not in self._consumers:
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=group,
                client_id=consumer_id,
                enable_auto_commit=False,  # commit() below is the only thing that advances an offset
                auto_offset_reset="earliest",
                key_deserializer=lambda k: k.decode() if k else "",
                value_deserializer=lambda v: json.loads(v.decode()),
            )
            await consumer.start()
            self._consumers[key] = consumer
        return self._consumers[key]

    async def poll(self, topic: str, group: str, consumer_id: str, max_messages: int = 10) -> list[Message]:
        consumer = await self._consumer_for(topic, group, consumer_id)
        batches = await consumer.getmany(timeout_ms=200, max_records=max_messages)
        return [
            Message(topic, record.partition, record.offset, record.key, record.value,
                    record.value.pop("_attempt", 0))
            for records in batches.values() for record in records
        ]

    async def commit(self, topic: str, group: str, msg: Message) -> None:
        consumer = await self._consumer_for(topic, group, "")
        await consumer.commit({TopicPartition(topic, msg.partition): msg.offset + 1})

    async def leave(self, topic: str, group: str, consumer_id: str) -> None:
        key = (topic, group)
        consumer = self._consumers.pop(key, None)
        if consumer is not None:
            await consumer.stop()

    async def close(self) -> None:
        if self._producer is not None:
            await self._producer.stop()
        for consumer in self._consumers.values():
            await consumer.stop()
        self._consumers.clear()
