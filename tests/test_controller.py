"""Tests for infergrid/autoscale/controller.py."""

from infergrid.autoscale.controller import Controller
from infergrid.autoscale.forecast import HoltWinters


def make_controller(**kwargs) -> Controller:
    return Controller(HoltWinters(season_length=4), requests_per_worker=10.0, **kwargs)


def test_a_zero_rate_scales_to_the_minimum():
    c = make_controller(min_workers=2)
    assert c.target_workers(0.0) == 2


def test_target_scales_with_rate_and_headroom():
    c = make_controller(headroom=1.0, min_workers=1, max_workers=100)
    assert c.target_workers(10.0) == 1  # exactly one worker's capacity
    assert c.target_workers(11.0) == 2  # just over -> rounds up to a second worker


def test_headroom_provisions_above_the_raw_rate():
    c = make_controller(headroom=1.5, min_workers=1, max_workers=100)
    assert c.target_workers(10.0) == 2  # 10 * 1.5 = 15 -> needs 2 workers at 10/worker, not 1


def test_target_is_clamped_to_max_workers():
    c = make_controller(max_workers=3)
    assert c.target_workers(1000.0) == 3


def test_target_is_clamped_to_min_workers_even_for_a_tiny_rate():
    c = make_controller(min_workers=2)
    assert c.target_workers(0.5) == 2


def test_predictive_target_reacts_to_a_forecast_not_just_the_raw_observation():
    """The whole point: predictive_target() should recommend scaling up based
    on what the *next* window is forecast to need, from a pattern already
    learned -- even while the just-observed rate is still low."""
    c = make_controller(min_workers=1, max_workers=100, headroom=1.0)
    # teach it a clean low-low-high-high cycle; seeding on whole seasons leaves
    # the model expecting a "low" as the next unabsorbed observation
    c.forecaster.seed([5.0, 5.0, 50.0, 50.0] * 4)
    c.forecaster.update(5.0)  # absorb that low, landing on the second "low" of the cycle
    # the window that JUST ended was low; the next one is the spike
    target = c.predictive_target(5.0)
    assert target >= 4  # ceil(50/10) == 5, forecasting the spike one step ahead of it actually recurring
