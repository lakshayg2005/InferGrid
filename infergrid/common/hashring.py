"""Consistent hashing with virtual nodes.

Nodes and keys are hashed onto the same ring of 64-bit positions. A key belongs to
the first node found walking clockwise from the key's position.

Why not `hash(key) % len(nodes)`? Because changing the number of nodes remaps almost
every key. On a ring, adding a node only takes over the keys between it and its
predecessor (about 1/N of them), and removing a node only moves that node's keys.

Each node is placed at `vnodes` positions instead of one. With a single position the
gaps between nodes are uneven, so some nodes own far more keys than others; many
positions average that out. It also means a removed node's keys spread over all
remaining nodes instead of landing on one unlucky neighbour.

The router uses the ring to pick workers; the state store (Phase 4) uses it to pick
the replicas that hold a key.
"""

import bisect
import hashlib
from collections.abc import Iterable, Iterator
from itertools import islice


def ring_hash(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")


class HashRing:
    def __init__(self, nodes: Iterable[str] = (), vnodes: int = 100):
        self.vnodes = vnodes
        self._points: list[int] = []  # sorted ring positions
        self._owners: list[str] = []  # _owners[i] is the node at _points[i]
        self._nodes: set[str] = set()
        for node in nodes:
            self.add(node)

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self._nodes)

    def __len__(self) -> int:
        return len(self._nodes)

    def add(self, node: str) -> None:
        if node in self._nodes:
            return
        self._nodes.add(node)
        for i in range(self.vnodes):
            point = ring_hash(f"{node}#{i}")
            index = bisect.bisect_left(self._points, point)
            self._points.insert(index, point)
            self._owners.insert(index, node)

    def remove(self, node: str) -> None:
        if node not in self._nodes:
            return
        self._nodes.discard(node)
        kept = [(p, o) for p, o in zip(self._points, self._owners) if o != node]
        self._points = [p for p, _ in kept]
        self._owners = [o for _, o in kept]

    def walk(self, key: int | str) -> Iterator[str]:
        """Every node exactly once, in clockwise order starting from `key`'s position."""
        if not self._points:
            return
        position = key if isinstance(key, int) else ring_hash(key)
        start = bisect.bisect_left(self._points, position)
        seen: set[str] = set()
        for offset in range(len(self._points)):
            owner = self._owners[(start + offset) % len(self._points)]
            if owner not in seen:
                seen.add(owner)
                yield owner
                if len(seen) == len(self._nodes):
                    return

    def node_for(self, key: int | str) -> str | None:
        return next(self.walk(key), None)

    def preference_list(self, key: int | str, n: int) -> list[str]:
        """The first `n` distinct nodes clockwise from `key`: the replicas that store it."""
        return list(islice(self.walk(key), n))
