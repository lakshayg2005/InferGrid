"""Dynamo-style leaderless replicated key-value store (DeCandia et al., 2007).

See DESIGN.md section 3.5 for the full design rationale. In one paragraph: every
key maps to a preference list of `n_replicas` nodes on a hash ring (same ring as
`gateway/router.py`'s cache-aware routing); a write succeeds once `w` of them have
it, a read is answered from the newest of `r` of them, and `w + r > n_replicas`
guarantees every read overlaps every write in at least one replica. There is no
leader and no partition owner: any node can coordinate any key's read or write.

Two mechanisms keep the store available during a failure instead of just
consistent when everything is healthy:

- **Sloppy quorum + hinted handoff.** If a preference-list node is down, the
  coordinator writes to the next *alive* node on the ring instead and marks that
  copy as a hint "for" the down node. The write still counts toward `w`, so a
  single down replica never blocks writes. The node holding the hint keeps
  retrying delivery to its rightful owner in the background once membership
  reports it alive again, at which point the hint is handed off and dropped.
- **Read repair.** A read fans out to `r` replicas and returns the newest
  version (by `store.clock.HybridClock`); any replica that answered with a
  stale or missing value is asynchronously brought up to date, so reads
  themselves heal the common case without waiting for a background sweep.

Concurrent writes to the same key are resolved last-write-wins by HLC version,
not merged -- simpler than Dynamo's vector clocks + client-side reconciliation,
at the cost of silently dropping one write if two coordinators race on the same
key at nearly the same instant. Acceptable here: see DESIGN.md 3.5 for why.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from infergrid.common.hashring import HashRing
from infergrid.membership import DEAD, SwimNode
from infergrid.store.clock import HybridClock, Version

log = logging.getLogger("infergrid.store")

_UNREACHABLE = object()  # distinct from "replica reachable but doesn't have this key"


@dataclass
class Entry:
    value: Any
    version: Version
    deleted: bool = False  # a tombstone: a delete is a write, so it replicates and wins races like any other


def _newer(a: Entry | None, b: Entry | None) -> Entry | None:
    if a is None:
        return b
    if b is None:
        return a
    return b if b.version > a.version else a


class Transport(Protocol):
    """How a node talks to a peer. `HttpTransport` (transport.py) is the real
    implementation; tests use an in-process double so failures can be simulated
    by simply removing a node from a dict, with no sockets or subprocesses."""

    async def get(self, peer: str, key: str) -> Entry | None: ...
    async def put(self, peer: str, key: str, entry: Entry, hint_for: str | None = None) -> bool: ...


class StoreNode:
    """One replica, reachable at `addr`, and a coordinator for any key.

    `nodes` is the full node list (up or down); the ring built from it is the
    cluster's fixed partitioning and does not change when a node's liveness
    changes -- only `alive_nodes()` does, which is what routes a given write or
    read around a down node instead of remapping who owns what.
    """

    def __init__(
        self,
        addr: str,
        nodes: list[str],
        transport: Transport,
        n_replicas: int = 3,
        w: int = 2,
        r: int = 2,
        vnodes: int = 100,
        membership: SwimNode | None = None,
        hint_retry_period: float = 2.0,
    ):
        if w + r <= n_replicas:
            raise ValueError(f"w={w} + r={r} must exceed n_replicas={n_replicas} (Dynamo's W+R>N quorum rule)")
        if n_replicas > len(nodes):
            raise ValueError(f"n_replicas={n_replicas} exceeds the {len(nodes)} nodes given")
        self.addr = addr
        self.transport = transport
        self.n_replicas = n_replicas
        self.w = w
        self.r = r
        self.membership = membership
        self.hint_retry_period = hint_retry_period
        self._ring = HashRing(nodes, vnodes=vnodes)
        self._data: dict[str, Entry] = {}
        self._hints: dict[str, dict[str, Entry]] = {}  # held-for-node-addr -> key -> entry
        self.clock = HybridClock(addr)
        self._task: asyncio.Task | None = None
        self._last_repair: asyncio.Task | None = None

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        if self.membership:
            await self.membership.start()
        self._task = asyncio.create_task(self._hint_loop(), name=f"store-hints-{self.addr}")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self.membership:
            await self.membership.stop()

    # -- client API --------------------------------------------------------

    async def put(self, key: str, value: Any, w: int | None = None) -> bool:
        return await self._write(key, Entry(value, self.clock.tick()), w)

    async def delete(self, key: str, w: int | None = None) -> bool:
        return await self._write(key, Entry(None, self.clock.tick(), deleted=True), w)

    async def get(self, key: str, r: int | None = None) -> Any | None:
        entry = await self._read(key, r)
        return None if entry is None or entry.deleted else entry.value

    # -- write path: sloppy quorum -----------------------------------------

    async def _write(self, key: str, entry: Entry, w: int | None) -> bool:
        """Write to each preference-list node; a node membership has confirmed
        DEAD is skipped without spending a timeout on it, and any node -- known
        dead or not -- whose write actually fails gets a spare substitute holding
        a hint. This reacts to real delivery failure, not just to what membership
        happens to have noticed yet, so a just-crashed node is covered too.

        Only *confirmed* DEAD skips the attempt, not merely "not ALIVE" (i.e. a
        SUSPECT node still gets a real try): SWIM's suspicion window is a real
        window of uncertainty, not evidence of failure, and a live node that is
        only transiently suspected (e.g. the coordinator's own SWIM ping/ack
        got briefly delayed by a burst of concurrent replication work -- exactly
        the load pattern this method itself creates) would otherwise be skipped
        for no reason, burning through the small spare pool a real failure needs.
        Found via scripts/store_chaos.py: two never-killed nodes went SUSPECT at
        once under real concurrent load and a write failed quorum despite both
        being perfectly reachable -- see DESIGN.md section 3.5.
        """
        w = w or self.w
        pref = self._ring.preference_list(key, self.n_replicas)
        dead = self._known_dead()
        spares = iter(n for n in self._ring.walk(key) if n not in pref)

        async def deliver(node: str) -> bool:
            if node in dead:
                return False
            return await self._store_at(node, key, entry, None)

        delivered = await asyncio.gather(*(deliver(n) for n in pref))
        acked = 0
        for node, ok in zip(pref, delivered):
            if ok:
                acked += 1
                continue
            for spare in spares:  # first untried spare willing to hold the hint
                if await self._store_at(spare, key, entry, node):
                    acked += 1
                    break
        if acked < w:
            log.warning("%s: quorum not reached for %r: acked=%d w=%d pref=%s delivered=%s known_dead=%s",
                        self.addr, key, acked, w, pref, list(zip(pref, delivered)), sorted(dead))
        return acked >= w

    async def _store_at(self, node: str, key: str, entry: Entry, hint_for: str | None) -> bool:
        if node == self.addr:
            self._accept(key, entry, hint_for)
            return True
        try:
            return await self.transport.put(node, key, entry, hint_for)
        except Exception:
            log.debug("%s: put to %s failed", self.addr, node, exc_info=True)
            return False

    def _accept(self, key: str, entry: Entry, hint_for: str | None) -> None:
        """Apply a write this node is a real or stand-in replica for."""
        self.clock.observe(entry.version)
        bucket = self._hints.setdefault(hint_for, {}) if hint_for else self._data
        current = bucket.get(key)
        if _newer(current, entry) is entry:
            bucket[key] = entry

    # -- read path: quorum + read repair ------------------------------------

    async def _read(self, key: str, r: int | None) -> Entry | None:
        """Fetch from every preference-list node reachable, topping up with spares
        until `r` have actually answered (not just been asked, mirroring the write
        path's reactive substitution) or spares run out. `_UNREACHABLE` is kept
        distinct from "no entry" so a replica that simply doesn't have the key yet
        isn't mistaken for a failed one and isn't targeted by read repair."""
        r = r or self.r
        pref = self._ring.preference_list(key, self.n_replicas)
        dead = self._known_dead()
        spares = iter(n for n in self._ring.walk(key) if n not in pref)

        async def fetch(node: str) -> Entry | None | object:
            if node in dead:
                return _UNREACHABLE
            return await self._fetch(node, key)

        results: dict[str, Entry | None | object] = dict(
            zip(pref, await asyncio.gather(*(fetch(n) for n in pref)))
        )
        reached = sum(1 for v in results.values() if v is not _UNREACHABLE)
        for spare in spares:
            if reached >= r:
                break
            results[spare] = await self._fetch(spare, key)
            if results[spare] is not _UNREACHABLE:
                reached += 1

        winner: Entry | None = None
        for v in results.values():
            if v is not _UNREACHABLE:
                winner = _newer(winner, v)

        if winner is not None:
            stale = [n for n, v in results.items() if v is _UNREACHABLE or v is None or v.version != winner.version]
            if stale:
                # Fire-and-forget so the client doesn't wait on it; kept on
                # self so tests can await convergence deterministically instead
                # of sleeping and hoping the scheduler got to it in time.
                self._last_repair = asyncio.ensure_future(self._repair(stale, key, winner))
        return winner

    async def _fetch(self, node: str, key: str) -> Entry | None | object:
        if node == self.addr:
            return self._data.get(key)
        try:
            return await self.transport.get(node, key)
        except Exception:
            log.debug("%s: get from %s failed", self.addr, node, exc_info=True)
            return _UNREACHABLE

    async def _repair(self, nodes: list[str], key: str, winner: Entry) -> None:
        await asyncio.gather(*(self._store_at(n, key, winner, None) for n in nodes), return_exceptions=True)

    # -- internal API, called by app.py's /internal endpoints ----------------

    def receive(self, key: str, entry: Entry, hint_for: str | None) -> None:
        self._accept(key, entry, hint_for)

    def receive_get(self, key: str) -> Entry | None:
        return self._data.get(key)

    # -- membership ----------------------------------------------------------

    def alive_nodes(self) -> set[str]:
        """Confirmed alive: used for hinted handoff, which should only retry
        delivery once a target is actually known to be back, and for /stats."""
        if self.membership is None:
            return set(self._ring.nodes)
        return {self.addr, *self.membership.alive_http_urls()} & self._ring.nodes

    def _known_dead(self) -> set[str]:
        """Confirmed DEAD: used to skip a write/read attempt that would almost
        certainly fail. Deliberately not "not alive_nodes()" -- see _write's
        docstring for why a merely SUSPECT node must still get a real attempt."""
        if self.membership is None:
            return set()
        return {m.metadata["http_url"] for m in self.membership.members.values()
                if m.state == DEAD and "http_url" in m.metadata} & self._ring.nodes

    # -- hinted handoff --------------------------------------------------

    async def _hint_loop(self) -> None:
        while True:
            await asyncio.sleep(self.hint_retry_period)
            try:
                await self._flush_hints()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("%s: hint flush failed", self.addr)

    async def _flush_hints(self) -> None:
        alive = self.alive_nodes()
        for target in [t for t in self._hints if t in alive]:
            pending = self._hints[target]
            for key in list(pending):
                if await self._store_at(target, key, pending[key], None):
                    del pending[key]
            if not pending:
                del self._hints[target]

    # -- introspection ---------------------------------------------------

    def stats(self) -> dict:
        return {
            "keys": len(self._data),
            "hints_held": {target: len(entries) for target, entries in self._hints.items()},
        }
