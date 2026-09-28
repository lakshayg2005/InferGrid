from collections import Counter

from infergrid.common.hashring import HashRing

KEYS = [f"key-{i}" for i in range(20_000)]
NODES = ["a", "b", "c", "d"]


def owners(ring: HashRing) -> dict[str, str]:
    return {k: ring.node_for(k) for k in KEYS}


def test_walk_visits_every_node_once():
    ring = HashRing(NODES)
    assert sorted(ring.walk("anything")) == NODES


def test_same_key_always_maps_to_same_node():
    assert owners(HashRing(NODES)) == owners(HashRing(reversed(NODES)))


def test_virtual_nodes_spread_keys_evenly():
    counts = Counter(owners(HashRing(NODES, vnodes=100)).values())
    mean = len(KEYS) / len(NODES)
    assert all(0.75 * mean < c < 1.25 * mean for c in counts.values()), counts


def test_adding_a_node_moves_only_keys_it_takes_over():
    ring = HashRing(NODES)
    before = owners(ring)
    ring.add("e")
    after = owners(ring)
    moved = [k for k in KEYS if before[k] != after[k]]
    assert all(after[k] == "e" for k in moved)  # nothing moves between existing nodes
    assert 0.1 < len(moved) / len(KEYS) < 0.3  # about 1/5 of keys, unlike ~80% with hash % N


def test_removing_a_node_moves_only_its_keys():
    ring = HashRing(NODES)
    before = owners(ring)
    ring.remove("b")
    after = owners(ring)
    assert all(after[k] == before[k] for k in KEYS if before[k] != "b")
    assert "b" not in after.values()


def test_preference_list_is_distinct_and_starts_with_owner():
    ring = HashRing(NODES)
    prefs = ring.preference_list("some-key", 3)
    assert len(set(prefs)) == 3
    assert prefs[0] == ring.node_for("some-key")
    assert ring.preference_list("some-key", 10) == list(ring.walk("some-key"))


def test_empty_ring():
    assert HashRing().node_for("x") is None
