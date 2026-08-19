"""The parameter object: bounds, the Feller ratio, and immutability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from voldesk.quant.model import (
    PARAM_BOUNDS,
    PARAM_NAMES,
    HestonParams,
    MarketState,
    ParameterOutOfBoundsError,
    bounds_arrays,
    clip_to_bounds,
)


def test_feller_ratio_is_the_ratio_in_the_specification() -> None:
    r""":math:`2\kappa\theta/\sigma^2`, and the condition is ratio >= 1."""
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.05, sigma=0.4, rho=-0.5)
    assert params.feller_ratio == pytest.approx(2 * 2.0 * 0.05 / 0.16)
    assert params.satisfies_feller

    violating = HestonParams(v0=0.04, kappa=1.0, theta=0.02, sigma=0.6, rho=-0.5)
    assert violating.feller_ratio < 1.0
    assert not violating.satisfies_feller


def test_feller_violation_is_a_signal_and_not_an_exception() -> None:
    """CLAUDE.md section 5.1: record it, raise an incident, do not crash.

    A parameter set that violates Feller is perfectly constructible, perfectly valid
    within the bounds, and prices perfectly well. The whole operational story — rule R003,
    a P3 incident, a runbook — depends on the numerics *not* refusing to proceed.
    """
    violating = HestonParams(v0=0.04, kappa=1.0, theta=0.02, sigma=0.6, rho=-0.5)
    violating.validate()  # must not raise
    assert violating.is_within_bounds()
    assert violating.feller_ratio < 1.0


def test_validate_rejects_out_of_bounds_parameters() -> None:
    """Each bound in CLAUDE.md section 5.6 is enforced, and the message names the parameter."""
    base = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=0.3, rho=-0.7)
    for name in PARAM_NAMES:
        low, high = PARAM_BOUNDS[name]
        too_low = base.replace(**{name: low - abs(low) - 1.0})
        with pytest.raises(ParameterOutOfBoundsError, match=name):
            too_low.validate()
        too_high = base.replace(**{name: high + 1.0})
        with pytest.raises(ParameterOutOfBoundsError, match=name):
            too_high.validate()


def test_validate_rejects_non_finite_parameters() -> None:
    """A NaN parameter is the classic symptom of a diverged optimizer; it must not pass."""
    base = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=0.3, rho=-0.7)
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ParameterOutOfBoundsError):
            base.replace(kappa=value).validate()


def test_validation_is_not_run_on_construction() -> None:
    """Construction must stay cheap and total.

    The optimizer evaluates points on and just outside the boundary, and a
    finite-difference Jacobian steps outside it by design. Raising in ``__post_init__``
    would turn those routine numerical events into crashes, so validation is applied at
    the boundaries that mean something instead — entry to a calibration, and persistence.
    """
    out_of_bounds = HestonParams(v0=-1.0, kappa=1e9, theta=0.0, sigma=0.0, rho=2.0)
    assert not out_of_bounds.is_within_bounds()


def test_parameters_are_immutable() -> None:
    """A calibration result that could be mutated in place is a reproducibility hazard."""
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=0.3, rho=-0.7)
    with pytest.raises((AttributeError, TypeError)):
        params.kappa = 2.0  # type: ignore[misc]


def test_array_round_trip_preserves_order() -> None:
    """``to_array`` and ``from_array`` must agree on the ordering in :data:`PARAM_NAMES`.

    Everything downstream — the Jacobian columns, the Fisher information matrix, the
    eigenvector composition bars in figure 2 — is indexed by this order. A silent
    transposition here would produce a plausible, wrong identifiability result.
    """
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=0.3, rho=-0.7)
    array = params.to_array()
    assert array.tolist() == [0.04, 1.5, 0.06, 0.3, -0.7]
    assert HestonParams.from_array(array) == params

    for index, name in enumerate(PARAM_NAMES):
        assert array[index] == getattr(params, name)


def test_from_array_rejects_the_wrong_shape() -> None:
    with pytest.raises(ValueError, match="5 parameters"):
        HestonParams.from_array(np.array([1.0, 2.0]))


def test_bounds_arrays_line_up_with_the_bounds_dict() -> None:
    lower, upper = bounds_arrays()
    for index, name in enumerate(PARAM_NAMES):
        assert (lower[index], upper[index]) == PARAM_BOUNDS[name]
    assert np.all(lower < upper)


def test_clip_projects_onto_the_box() -> None:
    clipped = clip_to_bounds(HestonParams(v0=-5.0, kappa=1e6, theta=0.06, sigma=0.3, rho=0.99))
    assert clipped.is_within_bounds()
    assert clipped.v0 == PARAM_BOUNDS["v0"][0]
    assert clipped.kappa == PARAM_BOUNDS["kappa"][1]
    assert clipped.rho == PARAM_BOUNDS["rho"][1]


def test_market_state_forward_and_discount() -> None:
    market = MarketState(spot=100.0, rate=0.03, dividend=0.01)
    assert market.forward(2.0) == pytest.approx(100.0 * math.exp(0.02 * 2.0))
    assert market.discount(2.0) == pytest.approx(math.exp(-0.06))
    market.validate()


def test_market_state_rejects_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="spot"):
        MarketState(spot=-1.0).validate()
    with pytest.raises(ValueError, match="rate"):
        MarketState(spot=100.0, rate=math.nan).validate()


def test_to_dict_carries_every_parameter() -> None:
    """The dict goes into the ``CalibrationRun`` JSON column, so it must be complete."""
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=0.3, rho=-0.7)
    as_dict = params.to_dict()
    assert set(as_dict) == set(PARAM_NAMES)
    assert HestonParams(**as_dict) == params
