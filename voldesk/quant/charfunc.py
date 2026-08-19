r"""The Heston characteristic function, in the Albrecher et al. (2007) formulation.

This module is the single highest-risk piece of numerics in the project, so it is worth
being explicit about what the risk is.

Heston's original 1993 paper writes the characteristic function using a complex logarithm
whose argument crosses the negative real axis as :math:`u` sweeps the integration range.
NumPy, like every other implementation, returns the principal branch, so the log jumps by
:math:`2\pi i` at the crossing. The resulting characteristic function is *discontinuous*.
For short maturities the crossings are rare enough that the error hides; for long
maturities the price is simply wrong, and wrong in a way that looks like a plausible
number rather than like a failure.

Albrecher, Mayer, Schoutens and Tistaert (2007), "The Little Heston Trap", *Wilmott*
Jan/Feb 2007, 83-92, show that an algebraically equivalent rearrangement removes the
problem entirely. Writing

.. math::
    d(u) &= \sqrt{(\rho\sigma i u - \kappa)^2 + \sigma^2 (iu + u^2)} \\
    g_2(u) &= \frac{\kappa - \rho\sigma i u - d}{\kappa - \rho\sigma i u + d}

the exponent involves :math:`e^{-dT}` rather than :math:`e^{+dT}`. Since the principal
square root satisfies :math:`\operatorname{Re}(d) \ge 0`, the term :math:`e^{-dT}` is
bounded by 1 for all :math:`u` and all :math:`T`, so :math:`1 - g_2 e^{-dT}` stays away
from the branch cut and the logarithm never jumps.

**The trap** is the other rearrangement, :math:`g_1 = 1/g_2` with :math:`e^{+dT}`. It is
the same function on paper. It is unusable in floating point for large :math:`T`.

``tests/test_charfunc.py`` samples :math:`\varphi` densely at :math:`T = 5` and asserts
that the largest increment scales with the grid spacing instead of jumping, which is the
mandatory test in CLAUDE.md section 5.2.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from voldesk.quant.model import HestonParams, MarketState

ComplexArray = NDArray[np.complex128]


def _cf_components(
    u: NDArray[np.float64] | ComplexArray,
    maturity: float,
    params: HestonParams,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    r"""Return :math:`(d, C, D)` in the Albrecher formulation.

    ``C`` is the "bare" term of CLAUDE.md section 5.2, i.e. it still has to be multiplied
    by :math:`\kappa\theta/\sigma^2` before entering the exponent. Keeping it bare matches
    the specification line for line, which is worth more here than saving a multiply.
    """
    # theta enters only through the caller's kappa*theta/sigma^2 prefactor on C.
    kappa, sigma, rho = params.kappa, params.sigma, params.rho

    iu = 1j * np.asarray(u, dtype=np.complex128)

    # d = sqrt((rho*sigma*i*u - kappa)^2 + sigma^2 * (i*u + u^2)).
    # np.sqrt takes the principal branch, which is exactly what gives Re(d) >= 0.
    rho_sigma_iu = rho * sigma * iu
    d = np.sqrt((rho_sigma_iu - kappa) ** 2 + sigma**2 * (iu - iu**2))

    # Note (iu + u^2) == (iu - (iu)^2), written above in terms of iu so that the same
    # expression is valid for complex u, which the Carr-Madan damping requires.

    kappa_minus = kappa - rho_sigma_iu - d
    kappa_plus = kappa - rho_sigma_iu + d
    g2 = kappa_minus / kappa_plus

    exp_minus_dT = np.exp(-d * maturity)

    # C(u,T) = (kappa - rho*sigma*i*u - d)*T - 2*log((1 - g2*exp(-d*T)) / (1 - g2))
    c = kappa_minus * maturity - 2.0 * np.log((1.0 - g2 * exp_minus_dT) / (1.0 - g2))

    # D(u,T) = ((kappa - rho*sigma*i*u - d)/sigma^2) * (1 - exp(-d*T))/(1 - g2*exp(-d*T))
    d_coef = (kappa_minus / sigma**2) * (1.0 - exp_minus_dT) / (1.0 - g2 * exp_minus_dT)

    return d, c, d_coef


def char_func_log_spot(
    u: NDArray[np.float64] | ComplexArray,
    maturity: float,
    params: HestonParams,
    market: MarketState,
) -> ComplexArray:
    r"""Characteristic function of :math:`\log S_T`.

    .. math::
        \varphi(u, T) = \exp\!\Big(
            iu\big(\log S_0 + (r-q)T\big)
            + C(u,T)\,\frac{\kappa\theta}{\sigma^2}
            + D(u,T)\, v_0 \Big)

    This is the form given in CLAUDE.md section 5.2. ``u`` may be complex, which the
    Carr-Madan damping factor needs.

    Reference: Albrecher et al. (2007), section 2; Heston (1993) eq. (17) for the
    original, non-stable arrangement.
    """
    _, c, d_coef = _cf_components(u, maturity, params)
    iu = 1j * np.asarray(u, dtype=np.complex128)
    drift = np.log(market.spot) + (market.rate - market.dividend) * maturity
    exponent = iu * drift + c * params.kappa * params.theta / params.sigma**2 + d_coef * params.v0
    return np.exp(exponent)


def char_func_log_return(
    u: NDArray[np.float64] | ComplexArray,
    maturity: float,
    params: HestonParams,
    market: MarketState,
) -> ComplexArray:
    r"""Characteristic function of the log-return :math:`X_T = \log(S_T / S_0)`.

    Equal to :func:`char_func_log_spot` with the :math:`\log S_0` term removed. This is
    the form the COS method consumes, because the COS truncation range is built from the
    cumulants of :math:`X_T` and those are what have closed forms.
    """
    _, c, d_coef = _cf_components(u, maturity, params)
    iu = 1j * np.asarray(u, dtype=np.complex128)
    drift = (market.rate - market.dividend) * maturity
    exponent = iu * drift + c * params.kappa * params.theta / params.sigma**2 + d_coef * params.v0
    return np.exp(exponent)


#: Stencil half-width for the numerical cumulants. The truncation error of the five-point
#: stencils falls as h^4 (first and second derivative) or h^2 (fourth), while round-off
#: grows as eps/h^n; the two meet near h = 0.05 for a cumulant generating function of order
#: one. Measured: c2 agrees with an independent analytic derivation to 1e-12 and c4 is flat
#: to five significant figures between h = 0.1 and h = 0.02. See
#: ``docs/adr/0005-cos-truncation-range-and-n-cos.md``.
_CUMULANT_STENCIL_H: Final[float] = 0.05


def log_return_cumulants(
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    h: float = _CUMULANT_STENCIL_H,
) -> tuple[float, float, float]:
    r"""Cumulants :math:`(c_1, c_2, c_4)` of :math:`X_T = \log(S_T/S_0)`.

    These size the COS truncation range (Fang & Oosterlee 2008, eq. 49). They are obtained
    by differentiating the cumulant generating function :math:`g(u) = \log \varphi_X(u)` at
    the origin,

    .. math::
        c_n = (-i)^n \frac{d^n g}{du^n}\Big|_{u=0}

    with five-point central stencils, rather than from the closed forms in Fang &
    Oosterlee's Table 11.

    Why not the closed forms
    ------------------------
    Two reasons, one practical and one that was measured here.

    *The table has no entry for* :math:`c_4` *under Heston.* Fang and Oosterlee set it to
    zero and widen :math:`L` to compensate. Measured against a Carr-Madan reference, that
    approximation puts a floor of about ``2e-6`` on the COS price at
    :math:`\sigma = 0.5, T = 1` — flat in the number of cosine terms, because it is a
    truncation error and not a series error, so no amount of terms would clear the 1e-8
    cross-validation gate of CLAUDE.md section 7. With :math:`c_4` computed, the same case
    agrees to ``4e-10``.

    *The table's* :math:`c_2` *does not match the characteristic function.* Transcribed as
    published, it disagrees by 0.6% at the reference parameters. The numerical value agrees
    to ``1e-12`` with an independent hand derivation of :math:`\mathrm{Var}(X_T)` via
    :math:`\tfrac{1}{4}\mathrm{Var}(\int_0^T v_s ds) + \mathbb{E}[\int_0^T v_s ds]` in the
    :math:`\rho = 0, v_0 = \theta` case, so the discrepancy is in the closed form and not in
    the differentiation. ``tests/test_charfunc.py`` pins that comparison down.

    Differentiating the characteristic function that is actually used removes an entire
    class of transcription risk, at a cost of five extra evaluations per pricing call.

    Returns
    -------
    tuple
        ``(c1, c2, c4)``. :math:`c_4` is clamped at zero: it is non-negative for these
        distributions and the caller feeds :math:`\sqrt{c_4}` into a width.
    """
    u = np.array([-2.0 * h, -h, 0.0, h, 2.0 * h], dtype=np.float64)
    g = np.log(char_func_log_return(u, maturity, params, market))

    first = (g[0] - 8.0 * g[1] + 8.0 * g[3] - g[4]) / (12.0 * h)
    second = (-g[0] + 16.0 * g[1] - 30.0 * g[2] + 16.0 * g[3] - g[4]) / (12.0 * h**2)
    fourth = (g[0] - 4.0 * g[1] + 6.0 * g[2] - 4.0 * g[3] + g[4]) / h**4

    c1 = float(np.imag(first))  # c1 = -i g'(0)
    c2 = abs(float(np.real(-second)))  # c2 = -g''(0), a variance
    c4 = max(float(np.real(fourth)), 0.0)  # c4 = g''''(0)

    return c1, c2, c4


def first_cumulant_closed_form(
    maturity: float,
    params: HestonParams,
    market: MarketState,
) -> float:
    r"""The published closed form for :math:`c_1`, kept as a cross-check.

    .. math::
        c_1 = (r-q)T + \frac{(1-e^{-\kappa T})(\theta - v_0)}{2\kappa}
              - \tfrac{1}{2}\theta T

    Fang & Oosterlee (2008), Table 11. Unlike their :math:`c_2`, this one *does* agree with
    the characteristic function, to nine significant figures — see
    ``tests/test_charfunc.py::test_closed_form_c1_agrees_with_the_characteristic_function``.
    It is not used in the pricing path; it exists so that the agreement is asserted rather
    than assumed, which is what makes the disagreement in :math:`c_2` credible.
    """
    kappa, theta, v0 = params.kappa, params.theta, params.v0
    mu = market.rate - market.dividend
    e_kt = math.exp(-kappa * maturity)
    return mu * maturity + (1.0 - e_kt) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * maturity
