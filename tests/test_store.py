"""Tests for the Dynamo-style store (infergrid/store). A shared in-process
`FakeTransport` stands in for the network: "down" just means removed from its
registry, so failures are instant and deterministic, the same trick
tests/test_swim.py uses for UDP nodes -- no sockets or subprocesses needed here.
"""

import pytest

from infergrid.store.clock import HybridClock
from infergrid.store.node import Entry, StoreNode


class FakeTransport:
    def __init__(self):
        self.nodes: dict[str, StoreNode] = {}

    async def get(self, peer, key):
        node = self.nodes.get(peer)
        if node is None:
            raise ConnectionError(f"{peer} unreachable")
        return node.receive_get(key)

    async def put(self, peer, key, entry, hint_for=None):
        node = self.nodes.get(peer)
        if node is None:
            raise ConnectionError(f"{peer} unreachable")
        node.receive(key, entry, hint_for)
        return True


def make_cluster(n=4, n_replicas=3, w=2, r=2):
    transport = FakeTransport()
    addrs = [f"node-{i}" for i in range(n)]
    nodes = {a: StoreNode(a, addrs, transport, n_replicas=n_replicas, w=w, r=r) for a in addrs}
    transport.nodes.update(nodes)
    return nodes, transport


def replicas_for(nodes: dict[str, StoreNode], key: str) -> list[str]:
    any_node = next(iter(nodes.values()))
    return any_node._ring.preference_list(key, any_node.n_replicas)


# -- basic quorum read/write ---------------------------------------------------


async def test_a_write_from_one_node_is_readable_from_a_different_node():
    nodes, _ = make_cluster()
    writer, reader = list(nodes.values())[0], list(nodes.values())[1]
    assert await writer.put("k", "v1")
    assert await reader.get("k") == "v1"


async def test_missing_key_reads_as_none():
    nodes, _ = make_cluster()
    node = next(iter(nodes.values()))
    assert await node.get("nope") is None


async def test_a_later_write_overwrites_an_earlier_one_regardless_of_coordinator():
    nodes, _ = make_cluster()
    a, b, c = list(nodes.values())[:3]
    assert await a.put("k", "old")
    assert await b.put("k", "new")
    assert await c.get("k") == "new"


async def test_delete_is_visible_from_every_node_as_a_tombstone():
    nodes, _ = make_cluster()
    a, b = list(nodes.values())[:2]
    await a.put("k", "v1")
    assert await a.delete("k")
    assert await b.get("k") is None


def test_w_plus_r_must_exceed_n_replicas():
    transport = FakeTransport()
    with pytest.raises(ValueError):
        StoreNode("a", ["a", "b", "c"], transport, n_replicas=3, w=1, r=1)


def test_n_replicas_cannot_exceed_the_node_count():
    transport = FakeTransport()
    with pytest.raises(ValueError):
        StoreNode("a", ["a", "b"], transport, n_replicas=3, w=2, r=2)


# -- sloppy quorum + hinted handoff -----------------------------------------


async def test_write_survives_one_of_three_replicas_being_down():
    nodes, transport = make_cluster(n=4, n_replicas=3, w=2, r=2)
    key = "k"
    down = replicas_for(nodes, key)[0]
    coordinator = next(a for a in nodes if a not in replicas_for(nodes, key))
    del transport.nodes[down]

    assert await nodes[coordinator].put(key, "v1")
    # a spare outside the preference list must be holding a hint for the down node
    assert any(down in n._hints and key in n._hints[down] for n in nodes.values() if n.addr in transport.nodes)


async def test_write_fails_once_too_few_replicas_are_reachable():
    nodes, transport = make_cluster(n=4, n_replicas=3, w=2, r=2)
    key = "k"
    pref = replicas_for(nodes, key)
    # take down every node except the coordinator itself: no spare, no quorum
    coordinator = pref[0]
    for addr in list(transport.nodes):
        if addr != coordinator:
            del transport.nodes[addr]
    assert await nodes[coordinator].put(key, "v1") is False


async def test_hinted_handoff_delivers_once_the_down_node_is_reachable_again():
    nodes, transport = make_cluster(n=4, n_replicas=3, w=2, r=2)
    key = "k"
    down = replicas_for(nodes, key)[0]
    coordinator = next(a for a in nodes if a not in replicas_for(nodes, key))
    del transport.nodes[down]

    assert await nodes[coordinator].put(key, "v1")
    holder = next(n for n in nodes.values() if down in n._hints and key in n._hints[down])

    transport.nodes[down] = nodes[down]  # the node comes back
    await holder._flush_hints()

    assert down not in holder._hints or key not in holder._hints.get(down, {})
    assert nodes[down].receive_get(key).value == "v1"


# -- read repair ---------------------------------------------------------------


async def test_read_repair_fixes_a_stale_replica():
    nodes, _ = make_cluster(n=4, n_replicas=3, w=2, r=2)
    key = "k"
    pref = replicas_for(nodes, key)
    coordinator = nodes[pref[0]]

    await coordinator.put(key, "v1")
    stale_replica = nodes[pref[1]]
    # simulate a replica that missed a later write (whitebox: poke its local copy)
    stale_replica._data[key] = Entry("stale", (1, 0, "nobody"))

    assert await coordinator.get(key) == "v1"
    await coordinator._last_repair
    assert stale_replica.receive_get(key).value == "v1"


async def test_read_repair_fills_in_a_replica_that_never_saw_the_write():
    nodes, transport = make_cluster(n=4, n_replicas=3, w=2, r=2)
    key = "k"
    pref = replicas_for(nodes, key)
    down = pref[0]
    coordinator = next(a for a in nodes if a not in pref)
    del transport.nodes[down]
    await nodes[coordinator].put(key, "v1")  # the down node gets a hint instead, not a direct copy

    transport.nodes[down] = nodes[down]
    assert await nodes[coordinator].get(key) == "v1"
    await nodes[coordinator]._last_repair
    assert nodes[down].receive_get(key) is not None
    assert nodes[down].receive_get(key).value == "v1"


# -- hybrid logical clock -------------------------------------------------------


def test_hlc_ticks_are_strictly_increasing_on_one_node():
    clock = HybridClock("a")
    versions = [clock.tick() for _ in range(20)]
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)


def test_hlc_observing_a_far_future_remote_version_pulls_this_node_ahead():
    a, b = HybridClock("a"), HybridClock("b")
    far_future = (a.tick()[0] + 10**12, 0, "b")
    caught_up = b.observe(far_future)
    assert caught_up > far_future
    assert b.tick() > caught_up
