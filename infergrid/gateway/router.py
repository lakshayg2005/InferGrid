"""Routing policies: which workers should serve a request, in order of preference.

`load` maps each worker to the number of requests this gateway currently has in
flight on it. With several gateways each only sees its own share of the load;
Phase 3 replaces this with load reported by the workers themselves.
"""

import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

from infergrid.common.hashring import HashRing
from infergrid.common.schemas import GenerateRequest
from infergrid.common.tokens import routing_key


class Router(ABC):
    name: str

    @abstractmethod
    def candidates(self, req: GenerateRequest, workers: Sequence[str], load: Mapping[str, int]) -> list[str]:
        """Every worker, best first. The gateway falls back down the list on failure."""


class RoundRobinRouter(Router):
    """Baseline: ignores both cache contents and load."""

    name = "round_robin"

    def __init__(self) -> None:
        self._counter = itertools.count()

    def candidates(self, req: GenerateRequest, workers: Sequence[str], load: Mapping[str, int]) -> list[str]:
        if not workers:
            return []
        start = next(self._counter) % len(workers)
        return [*workers[start:], *workers[:start]]


class LeastLoadedRouter(Router):
    """Stronger baseline: balances load perfectly but ignores cache contents."""

    name = "least_loaded"

    def __init__(self) -> None:
        self._rotation = RoundRobinRouter()

    def candidates(self, req: GenerateRequest, workers: Sequence[str], load: Mapping[str, int]) -> list[str]:
        # Rotate first so ties do not always go to the same worker; sorted() is stable.
        rotated = self._rotation.candidates(req, workers, load)
        return sorted(rotated, key=lambda w: load.get(w, 0))


class ConsistentHashRouter(Router):
    """Cache-aware routing: consistent hashing with bounded loads (Mirrokni et al., 2017).

    Each conversation maps to a fixed spot on a hash ring, so all its turns reach the
    worker that already caches its history. To stop a hot spot from overloading one
    worker, no worker may take more than ceil((1 + epsilon) * average load) requests;
    a full worker is skipped and the request continues clockwise to the next one.

    epsilon trades cache hits for balance: a small value spreads load evenly but
    moves more conversations off their home worker; math.inf disables the bound.
    """

    name = "consistent_hash"

    def __init__(self, epsilon: float = 0.25, vnodes: int = 100):
        if epsilon < 0:
            raise ValueError("epsilon must be >= 0")
        self.epsilon = epsilon
        self.vnodes = vnodes
        self._ring = HashRing(vnodes=vnodes)

    def candidates(self, req: GenerateRequest, workers: Sequence[str], load: Mapping[str, int]) -> list[str]:
        if not workers:
            return []
        self._sync_ring(workers)
        order = list(self._ring.walk(routing_key(req.messages)))
        if math.isinf(self.epsilon):
            return order

        total = sum(load.get(w, 0) for w in workers) + 1  # + 1 for the request being routed
        capacity = math.ceil((1 + self.epsilon) * total / len(workers))
        for i, worker in enumerate(order):
            if load.get(worker, 0) < capacity:
                # Workers that were skipped for being full go to the back of the fallback list.
                return [worker, *order[i + 1 :], *order[:i]]
        return order  # unreachable: some worker is always below capacity, kept for safety

    def _sync_ring(self, workers: Sequence[str]) -> None:
        wanted = set(workers)
        for node in self._ring.nodes - wanted:
            self._ring.remove(node)
        for node in wanted - self._ring.nodes:
            self._ring.add(node)


def make_router(name: str, epsilon: float = 0.25) -> Router:
    if name == RoundRobinRouter.name:
        return RoundRobinRouter()
    if name == LeastLoadedRouter.name:
        return LeastLoadedRouter()
    if name == ConsistentHashRouter.name:
        return ConsistentHashRouter(epsilon)
    raise ValueError(f"unknown router {name!r}")


ROUTERS = [RoundRobinRouter.name, LeastLoadedRouter.name, ConsistentHashRouter.name]
