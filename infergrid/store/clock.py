"""Hybrid Logical Clock (Kulkarni, Demirbas, Wang, Zou and Zhu, 2014).

Every write needs a version that lets any two replicas agree, without talking to
each other, on which of two writes happened later. A plain wall-clock timestamp
almost works, but two writes on different machines within the same clock tick (or
across a clock skew) can tie or even invert; a plain Lamport counter orders
causality correctly but throws away wall-clock meaning (a version number alone
cannot tell an operator "this write is from 3 seconds ago").

An HLC keeps both: a physical time component that never runs behind wall-clock
time, paired with a logical counter that breaks ties and absorbs skew. Every node
that sees a remote timestamp folds it in, so the clock is monotonic per node and
causally consistent cluster-wide -- concurrent writes on different nodes get
different, comparable versions without any coordination.
"""

import time

Version = tuple[int, int, str]  # (physical_ns, logical, node_id); compares lexicographically


class HybridClock:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self._pt = 0
        self._l = 0

    def tick(self) -> Version:
        """A version for a new local write."""
        return self._advance(time.time_ns(), -1)

    def observe(self, remote: Version) -> Version:
        """Fold in a version seen from another node (a replicated write, a read
        response) so this node's own next version is provably later than it."""
        remote_pt, remote_l, _ = remote
        return self._advance(remote_pt, remote_l)

    def _advance(self, other_pt: int, other_l: int) -> Version:
        now = time.time_ns()
        pt = max(self._pt, other_pt, now)
        if pt == self._pt and pt == other_pt:
            l = max(self._l, other_l) + 1
        elif pt == self._pt:
            l = self._l + 1
        elif pt == other_pt:
            l = other_l + 1
        else:
            l = 0
        self._pt, self._l = pt, l
        return (pt, l, self.node_id)
