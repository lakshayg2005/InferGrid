"""Generic at-least-once processing loop: poll, handle, and on failure either
retry with exponential backoff and jitter or -- past `max_attempts` -- give up
and forward to a dead-letter topic instead of blocking everything behind it.

This is application-level policy, not something either `QueueClient`
implementation provides for you (real Kafka doesn't have a built-in retry
topic either): `consume_with_retry` is written once, against the `QueueClient`
protocol, and used identically by the worker's batch consumer and the billing
consumer, whichever broker backs them.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from infergrid.queue.base import Message, QueueClient

log = logging.getLogger("infergrid.queue.consumer")

Handler = Callable[[Message], Awaitable[None]]


async def consume_with_retry(
    client: QueueClient,
    topic: str,
    group: str,
    consumer_id: str,
    handler: Handler,
    *,
    dlq_topic: str | None = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    poll_interval: float = 0.2,
    max_messages: int = 10,
    stop: asyncio.Event | None = None,
) -> None:
    """Runs until `stop` is set (or forever, for a long-lived worker process)."""
    dlq_topic = dlq_topic or f"{topic}.dlq"
    try:
        while stop is None or not stop.is_set():
            messages = await client.poll(topic, group, consumer_id, max_messages)
            if not messages:
                await asyncio.sleep(poll_interval)
                continue
            # Different partitions are independent and safe to process concurrently;
            # the broker never hands out a second in-flight message from the *same*
            # partition until the first is committed, so this can't reorder a key.
            await asyncio.gather(
                *(_handle_one(client, topic, group, msg, handler, dlq_topic, max_attempts, base_delay)
                  for msg in messages)
            )
    finally:
        await client.leave(topic, group, consumer_id)


async def _handle_one(
    client: QueueClient, topic: str, group: str, msg: Message, handler: Handler,
    dlq_topic: str, max_attempts: int, base_delay: float,
) -> None:
    try:
        await handler(msg)
        await client.commit(topic, group, msg)
    except Exception as exc:
        if msg.attempt + 1 >= max_attempts:
            log.warning("%s: %r exhausted %d attempts (%s) -- sending to %s",
                        topic, msg.key, msg.attempt + 1, exc, dlq_topic)
            await client.publish(dlq_topic, msg.key,
                                  {**msg.value, "_error": str(exc) or type(exc).__name__, "_failed_topic": topic})
            await client.commit(topic, group, msg)
        else:
            delay = base_delay * (2**msg.attempt) + random.uniform(0, base_delay)
            log.info("%s: %r failed (attempt %d): %s -- retrying in %.1fs",
                      topic, msg.key, msg.attempt + 1, exc, delay)
            # Commit the original now, freeing its partition immediately, rather than
            # holding it in flight for `delay`: the retry is a brand new message.
            await client.commit(topic, group, msg)
            await asyncio.sleep(delay)
            await client.publish(topic, msg.key, msg.value, attempt=msg.attempt + 1)
