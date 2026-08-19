r"""Gate: the characteristic function has no branch-cut discontinuity.

CLAUDE.md section 5.2 calls this out as the single highest-risk bug in the project and
makes the test mandatory: sample :math:`\varphi` on a dense grid of :math:`u` at
:math:`T = 5` and assert that :math:`\max_k |\varphi(u_{k+1}) - \varphi(u_k)|` **scales
with the grid spacing** rather than jumping.

The distinction matters. A discontinuous function also has a finite maximum increment on
any fixed grid; what separates it from a continuous one is that refining the grid does not
shrink that increment. So the assertion is on the *ratio* under refinement, not on a
magnitude.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import EXTREME, FELLER_VIOLATING, REFERENCE
from voldesk.quant.charfunc import (
    _cf_components,
    char_func_log_return,
    char_func_log_spot,
    first_cumulant_closed_form,
    log_return_cumulants,
)
from voldesk.quant.model import HestonParams, MarketState

ALL_PARAMS = [REFERENCE, FELLER_VIOLATING, EXTREME]


@pytest.mark.gate
@pytest.mark.parametrize("params", ALL_PARAMS, ids=["reference", "feller_violating", "extreme"])
@pytest.mark.parametrize("maturity", [5.0, 10.0, 20.0])
def test_cf_has_no_jump_discontinuity(
    params: HestonParams, maturity: float, market: MarketState
) -> None:
    """GATE (CLAUDE.md section 7, "CF continuity").

    Halving the grid spacing must roughly halve the largest increment. A branch-cut jump
    of :math:`2\\pi i` in the log would leave the maximum increment flat under refinement,
    so the ratio is the discriminator.
    """
    increments = []
    spacings = []
    for n in (4_000, 8_000, 16_000):
        u = np.linspace(-250.0, 250.0, n)
        phi = char_func_log_return(u, maturity, params, market)
        increments.append(float(np.max(np.abs(np.diff(phi)))))
        spacings.append(float(u[1] - u[0]))

    for coarse, fine, du_coarse, du_fine in zip(
        increments, increments[1:], spacings, spacings[1:], strict=False
    ):
        spacing_ratio = du_coarse / du_fine
        increment_ratio = coarse / fine
        # Linear scaling would give increment_ratio == spacing_ratio == 2. A discontinuity
        # would peg it near 1. The window is generous on the upper side because the maximum
        # increment can sit on a slightly different point of the curve after refinement.
        assert 1.7 <= increment_ratio <= 2.3, (
            f"max increment scaled by {increment_ratio:.3f} when the grid was refined by "
            f"{spacing_ratio:.1f}x. Flat scaling means a branch-cut jump — check that "
            "charfunc.py uses g2 with exp(-d*T) and not g1 with exp(+d*T)."
        )


@pytest.mark.gate
@pytest.mark.parametrize("params", ALL_PARAMS, ids=["reference", "feller_violating", "extreme"])
def test_principal_square_root_keeps_d_in_the_right_half_plane(
    params: HestonParams, market: MarketState
) -> None:
    r"""The stability of the Albrecher form rests on :math:`\operatorname{Re}(d) \ge 0`.

    That is what bounds :math:`e^{-dT}` by 1 for every :math:`u` and every :math:`T`. If
    this ever failed, the logarithm argument could wander onto the branch cut and the
    continuity test above would start failing intermittently rather than reproducibly —
    the worst kind of numerical bug. Asserting it directly turns that into a clear failure.
    """
    u = np.linspace(-500.0, 500.0, 20_001)
    d, _, _ = _cf_components(u, 5.0, params)
    assert np.all(np.real(d) >= -1e-12), f"min Re(d) = {np.min(np.real(d)):.3e}"


@pytest.mark.parametrize("params", ALL_PARAMS, ids=["reference", "feller_violating", "extreme"])
@pytest.mark.parametrize("maturity", [0.1, 1.0, 5.0])
def test_cf_at_zero_is_exactly_one(
    params: HestonParams, maturity: float, market: MarketState
) -> None:
    r""":math:`\varphi(0) = \mathbb{E}[1] = 1` for any distribution, exactly."""
    value = char_func_log_return(np.array([0.0]), maturity, params, market)[0]
    assert abs(value - 1.0) < 1e-14


@pytest.mark.parametrize("params", ALL_PARAMS, ids=["reference", "feller_violating", "extreme"])
def test_cf_is_hermitian(params: HestonParams, market: MarketState) -> None:
    r"""For a real-valued random variable, :math:`\varphi(-u) = \overline{\varphi(u)}`.

    A cheap and very sharp check: an error in a sign or in the branch of the square root
    breaks this immediately.
    """
    u = np.linspace(0.5, 200.0, 500)
    phi_plus = char_func_log_return(u, 2.0, params, market)
    phi_minus = char_func_log_return(-u, 2.0, params, market)
    assert np.allclose(phi_minus, np.conj(phi_plus), rtol=0, atol=1e-12)


@pytest.mark.parametrize("params", ALL_PARAMS, ids=["reference", "feller_violating", "extreme"])
def test_cf_modulus_is_bounded_by_one(params: HestonParams, market: MarketState) -> None:
    r""":math:`|\varphi(u)| \le 1` always; a value above 1 is a divergence, not a price."""
    u = np.linspace(-400.0, 400.0, 10_001)
    for maturity in (0.05, 1.0, 10.0):
        phi = char_func_log_return(u, maturity, params, market)
        assert np.max(np.abs(phi)) <= 1.0 + 1e-10


def test_log_spot_and_log_return_differ_only_by_the_spot(market: MarketState) -> None:
    r""":math:`\varphi_{\log S}(u) = e^{iu\log S_0}\varphi_X(u)` by definition of the two."""
    u = np.linspace(-50.0, 50.0, 301)
    spot_cf = char_func_log_spot(u, 1.5, REFERENCE, market)
    return_cf = char_func_log_return(u, 1.5, REFERENCE, market)
    assert np.allclose(spot_cf, np.exp(1j * u * np.log(market.spot)) * return_cf, atol=1e-12)


def test_closed_form_c1_agrees_with_the_characteristic_function(market: MarketState) -> None:
    r"""Fang & Oosterlee's :math:`c_1` matches the implemented characteristic function.

    This is the control for the next test. If a published closed form is going to be
    called wrong, the same differentiation had better reproduce the one that is right.
    Measured agreement: 1e-9 to 2e-7 relative across maturities, the spread being the
    finite-difference truncation error rather than any disagreement.
    """
    for maturity in (0.25, 1.0, 3.0):
        c1, _, _ = log_return_cumulants(maturity, REFERENCE, market)
        closed = first_cumulant_closed_form(maturity, REFERENCE, market)
        assert abs(c1 - closed) < 1e-6 * abs(closed), (
            f"T={maturity}: numeric {c1}, closed form {closed}"
        )


def test_c2_matches_an_independent_analytic_derivation(market: MarketState) -> None:
    r"""The numerical :math:`c_2` reproduces a hand derivation of the log-return variance.

    In the special case :math:`\rho = 0,\ v_0 = \theta` the variance of :math:`X_T` can be
    written down with no characteristic function involved at all. Writing
    :math:`I_T = \int_0^T v_s\,ds`, the log-return is
    :math:`X_T = (r-q)T - \tfrac{1}{2}I_T + M_T` with :math:`M_T = \int_0^T \sqrt{v_s}\,dW^S_s`,
    so

    .. math::
        \mathrm{Var}(X_T) = \mathbb{E}[I_T] + \tfrac{1}{4}\mathrm{Var}(I_T)

    since :math:`\mathrm{Var}(M_T) = \mathbb{E}[I_T]` and :math:`\rho = 0` kills the
    covariance term. With :math:`v_0 = \theta` the CIR variance collapses to
    :math:`\mathrm{Var}(v_s) = \frac{\sigma^2\theta}{2\kappa}(1 - e^{-2\kappa s})`, and
    integrating :math:`\mathrm{Cov}(v_s, v_t) = e^{-\kappa(t-s)}\mathrm{Var}(v_s)` over the
    square gives

    .. math::
        \mathrm{Var}(I_T) = \frac{\sigma^2\theta}{\kappa^2}
            \left[T + \frac{1-e^{-2\kappa T}}{2\kappa}
                    - \frac{2\,(1-e^{-\kappa T})}{\kappa}\right]

    This is the test that settled the matter. Fang & Oosterlee's Table 11 expression for
    :math:`c_2`, transcribed as published, disagrees with both this derivation and the
    characteristic function by 0.6% at these parameters, so the cumulants are taken from
    the characteristic function instead. See :func:`log_return_cumulants`.
    """
    cases = [
        (0.06, 1.5, 0.3, 1.0),
        (0.06, 1.5, 0.3, 0.25),
        (0.04, 2.0, 0.6, 2.0),
        (0.09, 1.0, 0.5, 3.0),
    ]
    for theta, kappa, sigma, maturity in cases:
        params = HestonParams(v0=theta, kappa=kappa, theta=theta, sigma=sigma, rho=0.0)
        var_integrated = (sigma**2 * theta / kappa**2) * (
            maturity
            + (1 - np.exp(-2 * kappa * maturity)) / (2 * kappa)
            - 2 * (1 - np.exp(-kappa * maturity)) / kappa
        )
        analytic = theta * maturity + 0.25 * var_integrated
        _, c2, _ = log_return_cumulants(maturity, params, market)
        assert abs(c2 - analytic) < 1e-6 * analytic, (
            f"theta={theta} kappa={kappa} sigma={sigma} T={maturity}: "
            f"numeric {c2}, analytic {analytic}"
        )


def test_cumulants_are_stable_in_the_stencil_width(market: MarketState) -> None:
    """The cumulants must be properties of the model, not of the finite-difference step.

    This is the measurement that justifies ``_CUMULANT_STENCIL_H``. If it ever fails, the
    COS truncation range is being set by round-off rather than by the model. Measured
    spread across h in {0.1, 0.05, 0.02}: c1 2.5e-6, c2 2.4e-7, c4 1.4e-3 — c4 is the
    loosest because a fourth derivative from five points is only second-order accurate,
    and a fraction of a percent on a width is of no consequence.
    """
    for index, label, tolerance in ((0, "c1", 1e-4), (1, "c2", 1e-5), (2, "c4", 1e-2)):
        values = [
            log_return_cumulants(1.0, FELLER_VIOLATING, market, h=h)[index]
            for h in (0.1, 0.05, 0.02)
        ]
        spread = (max(values) - min(values)) / abs(max(values, key=abs))
        assert spread < tolerance, f"{label} varies by {spread:.2e} across stencil widths: {values}"
