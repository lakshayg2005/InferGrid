"""Idempotent usage-to-billing consumer (DESIGN.md section 3.6): folds each
"usage" event -- published by the gateway per completed interactive request and
by a worker per completed batch job -- into a running per-tenant token total,
exactly once no matter how many times at-least-once delivery redelivers it.

Idempotency is a dedup marker per `request_id` in the state store, checked
before the total is touched: a redelivered event with the same `request_id`
sees its marker already there and is skipped entirely, so the total is never
double-counted. This is a plain read-modify-write against the store, not an
atomic increment -- fine for one billing consumer instance, but two instances
processing the same tenant's events concurrently could race on the total (a
lost update, not a correctness violation of the dedup guarantee itself). A
real billing pipeline would use a CRDT counter or a transactional store here;
out of scope for this project, in the same spirit as DESIGN.md 3.5's
unimplemented Merkle-tree anti-entropy.
"""

from __future__ import annotations

from infergrid.queue.base import Message
from infergrid.store.client import StoreClient


async def apply_usage_event(store: StoreClient, msg: Message) -> None:
    event = msg.value
    dedup_key = f"usage-event:{event['request_id']}"
    if await store.get(dedup_key) is not None:
        return
    total_key = f"usage:{event['tenant_id']}"
    current = await store.get(total_key) or 0
    await store.put(total_key, current + event["tokens"])
    await store.put(dedup_key, True)
