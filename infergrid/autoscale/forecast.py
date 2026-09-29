"""Holt-Winters triple exponential smoothing (Winters, 1960): forecasts a time
series that has both a trend and a repeating seasonal pattern from nothing but
its own history -- three smoothing constants, no training data, no model
weights. DESIGN.md section 3.9's predictive autoscaling forecasts the next
window's request rate with this, so a scale-up can happen *before* a
recurring spike arrives, not only after queue depth already shows the damage
-- the difference between this and a purely reactive autoscaler (see
`Controller`'s docstring for how the two are compared).

Additive seasonality (a fixed number of extra requests per phase of the
cycle, not a fixed multiplier) fits request-rate spikes reasonably well and
keeps the update rule simple; multiplicative seasonality is the textbook
alternative when a season's amplitude should scale with the trend.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HoltWinters:
    season_length: int  # number of observations per repeating cycle
    alpha: float = 0.3  # level smoothing
    beta: float = 0.1  # trend smoothing
    gamma: float = 0.3  # seasonal smoothing
    level: float = 0.0
    trend: float = 0.0
    seasonal: list[float] = field(default_factory=list)  # length == season_length once seeded
    _t: int = 0  # observations absorbed by update() since seeding

    @property
    def seeded(self) -> bool:
        return bool(self.seasonal)

    def seed(self, history: list[float]) -> None:
        """The standard Holt-Winters initialization: level and trend from the
        first two seasons' averages, seasonal indices from the first season's
        deviation from its own average. Any observations beyond the first two
        seasons are then absorbed one at a time via `update`, exactly as new
        data would be."""
        m = self.season_length
        if len(history) < 2 * m:
            raise ValueError(f"need at least {2 * m} observations to seed a season of length {m}")
        season1, season2 = history[:m], history[m:2 * m]
        avg1, avg2 = sum(season1) / m, sum(season2) / m
        self.level = avg1
        self.trend = (avg2 - avg1) / m
        self.seasonal = [history[i] - avg1 for i in range(m)]
        self._t = 0
        for value in history[2 * m:]:
            self.update(value)

    def update(self, value: float) -> None:
        if not self.seeded:
            raise RuntimeError("seed() must be called with at least two full seasons before update()")
        m = self.season_length
        last_season = self.seasonal[self._t % m]
        new_level = self.alpha * (value - last_season) + (1 - self.alpha) * (self.level + self.trend)
        new_trend = self.beta * (new_level - self.level) + (1 - self.beta) * self.trend
        new_seasonal = self.gamma * (value - new_level) + (1 - self.gamma) * last_season
        self.level, self.trend = new_level, new_trend
        self.seasonal[self._t % m] = new_seasonal
        self._t += 1

    def forecast(self, steps_ahead: int) -> float:
        if not self.seeded:
            raise RuntimeError("seed() must be called before forecast()")
        m = self.season_length
        season_index = (self._t + steps_ahead - 1) % m
        return self.level + steps_ahead * self.trend + self.seasonal[season_index]
