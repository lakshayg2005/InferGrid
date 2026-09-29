"""Tests for infergrid/autoscale/forecast.py's Holt-Winters implementation."""

import pytest

from infergrid.autoscale.forecast import HoltWinters


def test_seed_needs_at_least_two_full_seasons():
    hw = HoltWinters(season_length=4)
    with pytest.raises(ValueError):
        hw.seed([1.0, 2.0, 3.0])


def test_update_and_forecast_need_seeding_first():
    hw = HoltWinters(season_length=4)
    with pytest.raises(RuntimeError):
        hw.update(1.0)
    with pytest.raises(RuntimeError):
        hw.forecast(1)


def test_a_flat_series_forecasts_flat():
    hw = HoltWinters(season_length=4, alpha=0.5, beta=0.5, gamma=0.5)
    hw.seed([10.0] * 16)
    for h in range(1, 5):
        assert hw.forecast(h) == pytest.approx(10.0, abs=1e-6)


def test_a_repeating_seasonal_pattern_is_learned():
    """No trend, a clean repeating [low, low, high, high] cycle: forecasts for
    the next cycle's phases should closely track the pattern already seen,
    which is the whole point -- a spike that recurs on a schedule should be
    visible in the forecast *before* it happens again, not only once it does."""
    pattern = [10.0, 10.0, 50.0, 50.0]
    hw = HoltWinters(season_length=4, alpha=0.6, beta=0.3, gamma=0.6)
    hw.seed(pattern * 6)  # several full cycles to let the seasonal indices converge
    forecasts = [hw.forecast(h) for h in range(1, 5)]
    assert forecasts[0] == pytest.approx(10.0, abs=3.0)
    assert forecasts[1] == pytest.approx(10.0, abs=3.0)
    assert forecasts[2] == pytest.approx(50.0, abs=3.0)  # the spike, predicted two steps ahead of it recurring
    assert forecasts[3] == pytest.approx(50.0, abs=3.0)


def test_an_upward_trend_is_extrapolated():
    series = [10.0 + i for i in range(24)]  # season_length=4, +1 per step, no seasonality (flat within a season)
    hw = HoltWinters(season_length=4, alpha=0.5, beta=0.5, gamma=0.1)
    hw.seed(series)
    # after observing a steady +1/step trend through value ~33, forecasting ahead
    # should continue climbing, not flatten out
    assert hw.forecast(1) > 30.0
    assert hw.forecast(4) > hw.forecast(1)


def test_update_after_seeding_keeps_tracking_new_observations():
    hw = HoltWinters(season_length=4, alpha=0.5, beta=0.3, gamma=0.5)
    hw.seed([10.0, 10.0, 50.0, 50.0] * 4)
    before = hw.forecast(1)
    hw.update(999.0)  # a genuine surprise, unlike anything the pattern predicted
    after = hw.forecast(1)
    assert before != after  # the model actually moved in response to new data
