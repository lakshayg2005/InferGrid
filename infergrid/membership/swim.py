"""SWIM: Scalable Weakly-consistent Infection-style process group Membership.

Every member periodically pings one random peer over UDP (Das, Gupta and Motivala,
2002; see DESIGN.md section 3.4). A missed ping is double-checked through a few
other members (an "indirect ping", so one broken link between two members does not
look like a crash) before the peer is marked SUSPECT, and DEAD if it does not
refute the suspicion in time. Membership changes piggyback on these same ping/ack
messages, so the whole group learns about joins, suspicions and deaths without a
separate gossip round and without any central directory.

This is a teaching-scale implementation: JSON over UDP rather than a compact binary
encoding, and a fixed gossip retransmit count rather than one scaled to log(group
size). The state machine and the failure-detection guarantees are the real thing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("infergrid.membership")

ALIVE, SUSPECT, DEAD = "alive", "suspect", "dead"
_RANK = {ALIVE: 0, SUSPECT: 1, DEAD: 2}  # for tie-breaking when incarnations match


@dataclass
class Member:
    addr: str
    state: str = ALIVE
    incarnation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    changed_at: float = field(default_factory=time.monotonic)


def _is_newer(current: Member | None, state: str, incarnation: int) -> bool:
    """SWIM's merge rule: does incoming gossip about a member beat what we already have?

    A higher incarnation always wins (it is the member's own proof of a more recent
    state, since only a member can raise its own incarnation). At equal incarnations,
    worse news wins (alive < suspect < dead), since only the member itself can prove
    it is still alive by raising its incarnation to refute a suspicion.
    """
    if current is None:
        return True
    if incarnation != current.incarnation:
        return incarnation > current.incarnation
    return _RANK[state] > _RANK[current.state]


class SwimNode(asyncio.DatagramProtocol):
    """One SWIM group member, reachable at `addr` ("host:port").

    `metadata` is arbitrary JSON-able data this node advertises about itself (workers
    advertise their HTTP URL); every member learns every other alive member's
    metadata through gossip, with no central directory to query it from.
    """

    def __init__(
        self,
        addr: str,
        seeds: list[str] = (),
        metadata: dict[str, Any] | None = None,
        protocol_period: float = 0.5,
        ping_timeout: float = 0.5,
        indirect_count: int = 3,
        suspicion_timeout: float = 2.0,
        gossip_retransmits: int = 6,
    ):
        """
        The timeout defaults are deliberately more forgiving than a from-the-paper
        implementation: a member sharing a machine with a busy inference worker can
        legitimately take a while to get its ping-handling coroutine scheduled, and a
        false "dead" verdict is worse than a slightly slower true one. `tests/test_swim.py`
        overrides these with much smaller values, since it runs many members on one
        machine with nothing else competing for the event loop.
        """
        self.addr = addr
        self.protocol_period = protocol_period
        self.ping_timeout = ping_timeout
        self.indirect_count = indirect_count
        self.suspicion_timeout = suspicion_timeout
        self.gossip_retransmits = gossip_retransmits

        # A restarted process is a brand-new SwimNode with no memory of its previous
        # incarnation. If it started back at 0, SWIM's own merge rule (_is_newer)
        # would keep it stuck: peers who last heard it was DEAD at incarnation N
        # never accept an ALIVE claim at an incarnation no higher than N, and a
        # crashed-and-restarted process has no way to know what N was. Seeding from
        # the current time instead of 0 sidesteps this without persisting state to
        # disk: nanosecond resolution keeps two restarts of the same node from
        # landing on the same incarnation even seconds-scale wall-clock time would
        # not distinguish (this project's own fast-settings test suite runs many
        # SWIM rounds within a single wall-clock second, and did exactly that with
        # second resolution -- a real, reproducible flake, not a hypothetical one).
        self.incarnation = time.time_ns()
        self.members: dict[str, Member] = {addr: Member(addr, ALIVE, self.incarnation, dict(metadata or {}))}
        self._gossip: dict[tuple[str, str, int], int] = {}  # (addr, state, incarnation) -> transmits left

        self._transport: asyncio.DatagramTransport | None = None
        self._task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}  # ping seq -> future, resolved by a direct or relayed ack
        self._seq = 0
        self._seeds = list(seeds)
        self.on_change: Any = None  # optional hook(addr, state), used by tests

    # -- lifecycle -------------------------------------------------------

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        host, port = self.addr.rsplit(":", 1)
        self._transport, _ = await loop.create_datagram_endpoint(lambda: self, local_addr=(host, int(port)))
        # Seeds are only a bootstrap probe list, not added to self.members: a fake
        # placeholder entry (empty metadata, incarnation 0) would block the seed's
        # real announcement from ever being applied, since it wouldn't count as
        # "newer" than what looks like an already-known, same-incarnation member.
        # Seed our own gossip buffer, or nobody we ping would ever learn we exist.
        self._gossip[(self.addr, ALIVE, self.incarnation)] = self.gossip_retransmits
        self._task = asyncio.create_task(self._run(), name=f"swim-{self.addr}")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        if self._transport:
            self._transport.close()

    # -- queries -----------------------------------------------------------

    def alive_members(self) -> list[Member]:
        return [m for m in self.members.values() if m.state == ALIVE]

    def alive_http_urls(self) -> list[str]:
        return [m.metadata["http_url"] for m in self.alive_members() if "http_url" in m.metadata]

    def snapshot(self) -> dict:
        return {addr: {"state": m.state, "incarnation": m.incarnation, "metadata": m.metadata}
                for addr, m in sorted(self.members.items())}

    # -- protocol loop -------------------------------------------------------

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.protocol_period)
            try:
                self._check_suspicion_timeouts()
                await self._probe_one()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("%s: swim tick failed", self.addr)

    def _candidates(self) -> list[str]:
        known = [a for a in self.members if a != self.addr and self.members[a].state != DEAD]
        unseen_seeds = [s for s in self._seeds if s not in self.members]
        return known + unseen_seeds

    async def _probe_one(self) -> None:
        candidates = self._candidates()
        if not candidates:
            return
        target = random.choice(candidates)
        if await self._ping(target, self.ping_timeout):
            return
        helpers = [a for a in candidates if a != target]
        random.shuffle(helpers)
        helpers = helpers[: self.indirect_count]
        if helpers and any(await asyncio.gather(*(self._ping_req(h, target) for h in helpers))):
            return
        # Neither we nor any helper could reach it: it may still be alive behind a
        # broken link, so we only suspect it, giving it a chance to refute.
        self._update(target, SUSPECT, self.members[target].incarnation)

    def _check_suspicion_timeouts(self) -> None:
        now = time.monotonic()
        for addr, m in list(self.members.items()):
            if m.state == SUSPECT and now - m.changed_at >= self.suspicion_timeout:
                self._update(addr, DEAD, m.incarnation)

    # -- sending -------------------------------------------------------------

    async def _ping(self, target: str, timeout: float) -> bool:
        seq, fut = self._await_reply()
        self._send(target, {"type": "ping", "seq": seq, "from": self.addr})
        return await self._wait(seq, fut, timeout)

    async def _ping_req(self, helper: str, target: str) -> bool:
        # The helper's own ping to target (self.ping_timeout, in _relay_ping_req) can by
        # itself take nearly this whole budget, so this needs real margin beyond 2x for
        # the two extra network hops (us -> helper -> target -> helper -> us), or a
        # helper that would have confirmed the target in time reads as a timeout instead.
        seq, fut = self._await_reply()
        self._send(helper, {"type": "ping-req", "seq": seq, "from": self.addr, "target": target})
        return await self._wait(seq, fut, self.ping_timeout * 2.5)

    def _await_reply(self) -> tuple[int, asyncio.Future]:
        self._seq += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[self._seq] = fut
        return self._seq, fut

    async def _wait(self, seq: int, fut: asyncio.Future, timeout: float) -> bool:
        try:
            await asyncio.wait_for(fut, timeout)
            return True
        except asyncio.TimeoutError:
            return False
        # CancelledError is intentionally not caught here: swallowing it would stop
        # this node's stop() from ever being able to cancel the protocol loop.
        finally:
            self._pending.pop(seq, None)

    def _send(self, target_addr: str, message: dict) -> None:
        message["gossip"] = self._take_gossip()
        host, port = target_addr.rsplit(":", 1)
        try:
            self._transport.sendto(json.dumps(message).encode(), (host, int(port)))
        except OSError:
            pass  # unreachable right now; the next probe round will notice

    def _take_gossip(self, limit: int = 6) -> list[list]:
        items = sorted(self._gossip.items(), key=lambda kv: -kv[1])[:limit]
        out = []
        for key, left in items:
            addr, state, incarnation = key
            metadata = self.members[addr].metadata if addr in self.members else {}
            out.append([addr, state, incarnation, metadata])
            if left - 1 <= 0:
                del self._gossip[key]
            else:
                self._gossip[key] = left - 1
        return out

    # -- receiving -------------------------------------------------------

    def datagram_received(self, data: bytes, addr) -> None:  # asyncio.DatagramProtocol
        try:
            message = json.loads(data.decode())
        except ValueError:
            return
        for addr_, state, incarnation, metadata in message.get("gossip", []):
            self._update(addr_, state, incarnation, metadata)

        kind = message["type"]
        if kind == "ping":
            self._send(message["from"], {"type": "ack", "seq": message["seq"]})
        elif kind == "ack":
            fut = self._pending.get(message["seq"])
            if fut and not fut.done():
                fut.set_result(True)
        elif kind == "ping-req":
            asyncio.ensure_future(self._relay_ping_req(message))

    async def _relay_ping_req(self, message: dict) -> None:
        if await self._ping(message["target"], self.ping_timeout):
            self._send(message["from"], {"type": "ack", "seq": message["seq"]})

    def error_received(self, exc: Exception) -> None:
        log.debug("%s: udp error: %r", self.addr, exc)

    # -- state merge -----------------------------------------------------

    def _update(self, addr: str, state: str, incarnation: int, metadata: dict | None = None) -> None:
        if addr == self.addr:
            if state != ALIVE or incarnation >= self.incarnation:
                # Someone doubts us, or thinks we have a newer incarnation than we do
                # (e.g. we restarted and lost our counter). Refute by asserting a
                # provably newer incarnation than anything anyone has heard.
                self.incarnation = incarnation + 1 if state != ALIVE else max(self.incarnation, incarnation)
                self._set(addr, ALIVE, self.incarnation, self.members[addr].metadata)
            return
        current = self.members.get(addr)
        if _is_newer(current, state, incarnation):
            self._set(addr, state, incarnation, metadata or (current.metadata if current else {}))

    def _set(self, addr: str, state: str, incarnation: int, metadata: dict) -> None:
        before = self.members.get(addr)
        self.members[addr] = Member(addr, state, incarnation, metadata or {})
        self._gossip[(addr, state, incarnation)] = self.gossip_retransmits
        if before is None or before.state != state:
            log.info("%s: %s is now %s", self.addr, addr, state)
            if self.on_change:
                self.on_change(addr, state)
