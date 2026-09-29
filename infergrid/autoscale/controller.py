"""Turns a Holt-Winters rate forecast into a worker-count decision. Pure
decision logic, no I/O: `scripts/autoscale_demo.py` is what actually polls the
gateway for the observed rate and spawns/kills worker processes, so this stays
unit-testable without a running cluster.

Comparing predictive against a purely *reactive* policy is the point of the
demo, not just "does autoscaling help": a reactive policy calls
`target_workers()` with the rate already observed in the window that just
ended, so it only starts spinning up capacity once a spike has already
arrived. A predictive policy calls it with `forecast(1)` instead -- the same
target-sizing formula, fed the *next* window's forecast rather than the
current one -- so extra capacity is warm before the spike lands, one full
tick sooner, provided the spike is part of a pattern Holt-Winters has already
seen enough of to learn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from infergrid.autoscale.forecast import HoltWinters


@dataclass
class Controller:
    forecaster: HoltWinters
    requests_per_worker: float  # requests per tick one worker can handle before it's considered saturated
    min_workers: int = 1
    max_workers: int = 10
    headroom: float = 1.2  # provision for 20% above the target rate, not exactly it

    def target_workers(self, rate: float) -> int:
        if rate <= 0:
            return self.min_workers
        target = math.ceil((rate * self.headroom) / self.requests_per_worker)
        return max(self.min_workers, min(self.max_workers, target))

    def predictive_target(self, observed_rate: float, steps_ahead: int = 1) -> int:
        """Feed one newly observed window's rate into the model and return the
        target worker count for `steps_ahead` windows from now."""
        self.forecaster.update(observed_rate)
        return self.target_workers(self.forecaster.forecast(steps_ahead))
