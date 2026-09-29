"""These use real UDP sockets on 127.0.0.1, so protocol timings are set small
(tens of milliseconds) to keep the suite fast; production defaults are larger.
"""

import asyncio

import pytest

from infergrid.membership.swim import ALIVE, DEAD, SUSPECT, SwimNode

FAST = dict(protocol_period=0.03, ping_timeout=0.03, indirect_count=2, suspicion_timeout=0.2, gossip_retransmits=8)
_next_port = iter(range(19100, 19999))


def addr() -> str:
    return f"127.0.0.1:{next(_next_port)}"


async def wait_until(cond, timeout: float = 5.0, step: float = 0.01) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if cond():
            return True
        await asyncio.sleep(step)
    return False


class Cluster:
    """Starts and always cleans up a set of nodes, even if a test assertion fails."""

    def __init__(self):
        self.nodes: list[SwimNode] = []

    def add(self, seeds: list[str] = (), metadata: dict | None = None) -> SwimNode:
        node = SwimNode(addr(), seeds=list(seeds), metadata=metadata, **FAST)
        self.nodes.append(node)
        return node

    async def start(self) -> None:
        for node in self.nodes:
            await node.start()

    async def __aenter__(self) -> "Cluster":
        return self

    async def __aexit__(self, *exc) -> None:
        for node in self.nodes:
            await node.stop()


@pytest.fixture
async def cluster():
    c = Cluster()
    try:
        yield c
    finally:
        for node in c.nodes:
            await node.stop()


async def test_two_nodes_discover_each_other(cluster: Cluster):
    a = cluster.add(metadata={"http_url": "http://a"})
    b = cluster.add(seeds=[a.addr], metadata={"http_url": "http://b"})
    await cluster.start()

    assert await wait_until(lambda: len(a.alive_members()) == 2 and len(b.alive_members()) == 2)
    assert a.members[b.addr].metadata == {"http_url": "http://b"}
    assert b.members[a.addr].metadata == {"http_url": "http://a"}


async def test_gossip_reaches_a_node_with_no_direct_seed(cluster: Cluster):
    """C is only seeded with B, never told about A directly, and must learn about A via gossip."""
    a = cluster.add(metadata={"http_url": "http://a"})
    b = cluster.add(seeds=[a.addr])
    c = cluster.add(seeds=[b.addr])
    await cluster.start()

    assert await wait_until(lambda: len(c.alive_members()) == 3)
    assert c.members[a.addr].state == ALIVE
    assert c.members[a.addr].metadata == {"http_url": "http://a"}


async def test_a_late_joiner_learns_about_a_seed_with_no_pending_gossip_about_itself(cluster: Cluster):
    """The bug this guards against: a node's self-announcement gossip entry (seeded
    once in start()) drains after gossip_retransmits sends, same as any other gossip
    entry -- in a cluster that has been running a while, it is normally long gone.
    A node joining afterwards, seeded on that node, still pings it successfully every
    round, but a bare successful ping/ack proves nothing on its own: receiving an ack
    only resolves a pending future, it never updates self.members. Unless the seed
    happens to have *some* gossip about itself still pending at that exact moment
    -- luck, not a guarantee, exactly like the restart bug below -- the joiner's
    membership table would stay {itself} forever. Found via scripts/store_chaos.py,
    where a restarted store node's alive_nodes() stayed empty of its peers well past
    every wait budget the chaos script gave it.
    """
    a = cluster.add(metadata={"http_url": "http://a"})
    await cluster.start()
    a._gossip.clear()  # force-exhaust: a's self-announcement has already drained

    b = cluster.add(seeds=[a.addr], metadata={"http_url": "http://b"})
    await b.start()
    assert await wait_until(lambda: a.addr in b.members and b.members[a.addr].state == ALIVE), (
        "a joiner must learn about a seed even when the seed has no gossip news pending"
    )
    assert b.members[a.addr].metadata == {"http_url": "http://a"}
    assert await wait_until(lambda: b.addr in a.members and a.members[b.addr].state == ALIVE), (
        "the seed must learn about the joiner too, from the joiner's own ping"
    )


async def test_detects_a_crashed_node_and_passes_through_suspect(cluster: Cluster):
    a = cluster.add()
    b = cluster.add(seeds=[a.addr])
    c = cluster.add(seeds=[a.addr])
    await cluster.start()
    assert await wait_until(lambda: all(len(n.alive_members()) == 3 for n in (a, b, c)))

    seen: list[str] = []
    a.on_change = lambda addr_, state: seen.append(state) if addr_ == c.addr else None

    await c.stop()  # no goodbye message: this is what a killed process looks like
    assert await wait_until(lambda: a.members[c.addr].state == DEAD and b.members[c.addr].state == DEAD)

    assert seen[0] == SUSPECT, "a live node is never marked dead in one step; it must be suspected first"
    assert seen[-1] == DEAD


async def test_a_single_broken_link_does_not_cause_a_false_positive(cluster: Cluster):
    """A cannot reach C directly, but B can and relays for it (indirect ping)."""
    a = cluster.add()
    b = cluster.add(seeds=[a.addr])
    c = cluster.add(seeds=[b.addr])
    await cluster.start()
    assert await wait_until(lambda: all(len(n.alive_members()) == 3 for n in (a, b, c)))

    real_send = a._send

    def blocking_send(target_addr, message):
        if target_addr == c.addr:
            return  # drop every message a tries to send straight to c
        real_send(target_addr, message)

    a._send = blocking_send
    await asyncio.sleep(FAST["suspicion_timeout"] * 3)  # long enough that a real crash would be detected
    assert a.members[c.addr].state == ALIVE, "b's indirect ping should have vouched for c"


async def test_refutes_a_false_suspicion_by_raising_its_incarnation():
    node = SwimNode(addr(), **FAST)
    await node.start()
    try:
        before = node.incarnation
        node._update(node.addr, SUSPECT, node.incarnation)  # a peer's gossip about us, arriving locally
        assert node.incarnation == before + 1
        assert node.members[node.addr].state == ALIVE
    finally:
        await node.stop()


async def test_higher_incarnation_alive_beats_a_stale_dead_report():
    node = SwimNode(addr(), **FAST)
    await node.start()
    try:
        peer = addr()
        node._update(peer, DEAD, 0)
        assert node.members[peer].state == DEAD
        node._update(peer, ALIVE, 1)  # the peer proves it is alive with a newer incarnation
        assert node.members[peer].state == ALIVE
        node._update(peer, DEAD, 0)  # a late, stale copy of the old report must not override that
        assert node.members[peer].state == ALIVE
    finally:
        await node.stop()


async def test_alive_http_urls_only_includes_live_members_with_metadata(cluster: Cluster):
    a = cluster.add(metadata={"http_url": "http://a"})
    b = cluster.add(seeds=[a.addr], metadata={"http_url": "http://b"})
    c = cluster.add(seeds=[a.addr])  # no http_url: e.g. a pure gateway member
    await cluster.start()
    assert await wait_until(lambda: len(a.alive_members()) == 3)

    assert set(a.alive_http_urls()) == {"http://a", "http://b"}

    await b.stop()
    assert await wait_until(lambda: a.members[b.addr].state == DEAD)
    assert "http://b" not in a.alive_http_urls()


async def test_a_restarted_node_at_the_same_address_is_seen_alive_again(cluster: Cluster):
    """The bug this guards against: a node that crashed and came back with no memory
    of its previous incarnation must not be stuck DEAD forever in its peers' eyes.

    A revived node announcing itself at the same incarnation its peer last recorded
    it DEAD at is rejected by SWIM's own merge rule (equal incarnation: worse news
    wins) -- unless the peer is still actively re-gossiping "it's dead" at that exact
    moment, which triggers the revived node to self-refute by bumping its incarnation.
    That's a lucky race, not a guarantee: once a peer's retransmit budget for that
    gossip entry is spent, it stops re-announcing it, yet still holds the address as
    DEAD forever, since a DEAD member is never re-probed. `a`'s buffer is force-
    exhausted below (deleting the entry, exactly as happens once its transmit count
    naturally reaches zero) to test the real fix -- a higher starting incarnation --
    rather than the lucky path this test could otherwise pass on by accident.
    """
    a = cluster.add()
    observer = cluster.add(seeds=[a.addr])
    dead_addr = observer.addr
    await cluster.start()
    assert await wait_until(lambda: len(a.alive_members()) == 2 and len(observer.alive_members()) == 2)

    await observer.stop()
    assert await wait_until(lambda: a.members[dead_addr].state == DEAD)
    a._gossip = {k: v for k, v in a._gossip.items() if k[0] != dead_addr}  # force-exhaust: no lucky refutation

    # A brand-new SwimNode at the same address: no memory of the old incarnation.
    revived = SwimNode(dead_addr, seeds=[a.addr], **FAST)
    cluster.nodes.append(revived)
    await revived.start()
    assert await wait_until(lambda: a.members[dead_addr].state == ALIVE, timeout=5), (
        "a restarted node must eventually be seen alive again, not stuck dead forever"
    )
