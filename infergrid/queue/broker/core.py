"""InferGrid's own Kafka-compatible broker, written from scratch to demonstrate the
real mechanics DESIGN.md section 3.6 asks for: topic partitioning, consumer groups
that share a topic's partitions and rebalance when a member disappears, and
at-least-once delivery (a message is redelivered, not lost, if the consumer that
had it crashes before committing).

A teaching-scale implementation, like `membership/swim.py` and `store/node.py`:
one process, one in-memory copy of every log, no replication across broker nodes
and no persistence to disk. Phase 4's `infergrid/store/` already demonstrates
replication and durability; this module's job is the messaging model on top of
it, not re-deriving replication a second time. A production swap is
`infergrid.queue.redpanda.RedpandaClient`, the same `QueueClient` interface
backed by a real (replicated, persistent) Kafka-API broker.

**Delivery model.** Each partition allows at most one *in-flight* (delivered but
uncommitted) message per consumer group at a time: `poll` skips a partition that
already has one outstanding, and a retry only becomes available again once it is
committed. This is a deliberate simplification of Kafka's pipelined fetch (many
records in flight per partition), traded for a trivially easy-to-reason-about
implementation -- and it is not the bottleneck it might sound like, since
different partitions are still delivered and processed independently and
concurrently.

**Rebalancing.** A consumer that has not polled or explicitly left within
`session_timeout` looks crashed: its group membership is dropped, partitions are
reassigned round-robin across whoever is left, and anything it had in flight is
freed for redelivery to the new owner. This is the same mechanism a real Kafka
consumer group uses to recover a job an abandoned worker never finished.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from infergrid.common.hashring import ring_hash
from infergrid.queue.base import Message

DEFAULT_PARTITIONS = 8
DEFAULT_SESSION_TIMEOUT = 10.0


@dataclass
class _Record:
    key: str
    value: dict
    attempt: int


@dataclass
class _InFlight:
    offset: int
    consumer_id: str
    delivered_at: float


@dataclass
class _Group:
    committed: dict[int, int] = field(default_factory=dict)  # partition -> next offset to deliver
    in_flight: dict[int, _InFlight] = field(default_factory=dict)  # partition -> outstanding message
    members: dict[str, float] = field(default_factory=dict)  # consumer_id -> last seen (monotonic)
    assignment: dict[str, list[int]] = field(default_factory=dict)  # consumer_id -> partitions


class Broker:
    def __init__(self, num_partitions: int = DEFAULT_PARTITIONS, session_timeout: float = DEFAULT_SESSION_TIMEOUT):
        self.num_partitions = num_partitions
        self.session_timeout = session_timeout
        self._logs: dict[str, list[list[_Record]]] = {}
        self._groups: dict[tuple[str, str], _Group] = {}  # (topic, group) -> state

    def publish(self, topic: str, key: str, value: dict, attempt: int = 0) -> Message:
        log = self._partitions_for(topic)
        partition = ring_hash(key) % self.num_partitions
        offset = len(log[partition])
        log[partition].append(_Record(key, value, attempt))
        return Message(topic, partition, offset, key, value, attempt)

    def poll(self, topic: str, group: str, consumer_id: str, max_messages: int = 10) -> list[Message]:
        log = self._partitions_for(topic)
        g = self._group_for(topic, group)
        now = time.monotonic()
        g.members[consumer_id] = now
        self._expire_stale(g, now)
        self._rebalance(g)

        out: list[Message] = []
        for partition in g.assignment.get(consumer_id, []):
            if len(out) >= max_messages or partition in g.in_flight:
                continue
            offset = g.committed.get(partition, 0)
            partition_log = log[partition]
            if offset < len(partition_log):
                rec = partition_log[offset]
                g.in_flight[partition] = _InFlight(offset, consumer_id, now)
                out.append(Message(topic, partition, offset, rec.key, rec.value, rec.attempt))
        return out

    def commit(self, topic: str, group: str, msg: Message) -> None:
        g = self._group_for(topic, group)
        current = g.in_flight.get(msg.partition)
        if current is not None and current.offset == msg.offset:
            del g.in_flight[msg.partition]
        g.committed[msg.partition] = max(g.committed.get(msg.partition, 0), msg.offset + 1)

    def leave(self, topic: str, group: str, consumer_id: str) -> None:
        g = self._group_for(topic, group)
        if g.members.pop(consumer_id, None) is not None:
            self._free_in_flight_for(g, consumer_id)
            self._rebalance(g)

    def stats(self) -> dict:
        return {
            topic: {"partitions": [len(p) for p in log],
                    "groups": {group: {"committed": g.committed, "members": sorted(g.members)}
                               for (t, group), g in self._groups.items() if t == topic}}
            for topic, log in self._logs.items()
        }

    def _partitions_for(self, topic: str) -> list[list[_Record]]:
        if topic not in self._logs:
            self._logs[topic] = [[] for _ in range(self.num_partitions)]
        return self._logs[topic]

    def _group_for(self, topic: str, group: str) -> _Group:
        self._partitions_for(topic)
        return self._groups.setdefault((topic, group), _Group())

    def _expire_stale(self, g: _Group, now: float) -> None:
        stale = [c for c, last_seen in g.members.items() if now - last_seen > self.session_timeout]
        for consumer_id in stale:
            del g.members[consumer_id]
            self._free_in_flight_for(g, consumer_id)

    def _free_in_flight_for(self, g: _Group, consumer_id: str) -> None:
        for partition, entry in list(g.in_flight.items()):
            if entry.consumer_id == consumer_id:
                del g.in_flight[partition]  # redelivered to whoever this partition is reassigned to

    def _rebalance(self, g: _Group) -> None:
        members = sorted(g.members)
        if not members:
            g.assignment = {}
            return
        assignment: dict[str, list[int]] = {c: [] for c in members}
        for partition in range(self.num_partitions):
            assignment[members[partition % len(members)]].append(partition)
        g.assignment = assignment
