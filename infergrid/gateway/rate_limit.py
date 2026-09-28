"""Per-tenant rate limiting with a token bucket.

Each tenant starts with `capacity` tokens, refilled at `rate` tokens/second up to
that same cap. A request costs one token; if none are left, the request is
rejected with 429 rather than queued, so one tenant's burst cannot add latency
for every other tenant sharing the cluster.

This gateway's buckets live in local memory. With several gateway replicas each
enforces its own share of the limit independently (capacity * replica count,
cluster-wide) rather than one exact shared limit; an exact cluster-wide limit
needs a store all replicas share (e.g. Redis), which is future work -- the same
build-it-yourself-vs-Redis trade-off as the state store in DESIGN.md section 3.5.
"""

import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    capacity: float
    rate: float  # tokens refilled per second
    tokens: float = field(init=False)
    _updated: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokens = self.capacity
        self._updated = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self._updated) * self.rate)
        self._updated = now

    def try_take(self, cost: float = 1.0) -> bool:
        self._refill()
        if self.tokens < cost:
            return False
        self.tokens -= cost
        return True

    def retry_after(self, cost: float = 1.0) -> float:
        self._refill()
        deficit = cost - self.tokens
        return round(deficit / self.rate, 3) if deficit > 0 else 0.0


class RateLimiter:
    def __init__(self, capacity: float, rate: float):
        self.capacity = capacity
        self.rate = rate
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, tenant: str) -> tuple[bool, float]:
        """(allowed, seconds until a request would be allowed if this one is refused)."""
        bucket = self._buckets.setdefault(tenant, TokenBucket(self.capacity, self.rate))
        if bucket.try_take():
            return True, 0.0
        return False, bucket.retry_after()
