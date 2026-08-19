r"""The COS method: Fourier-cosine expansion pricing of European options.

Reference: Fang, F. and Oosterlee, C.W. (2008), "A Novel Pricing Method for European
Options Based on Fourier-Cosine Series Expansions", *SIAM Journal on Scientific
Computing* 31(2), 826-848. Section 2.2 gives the density expansion and section 3.1 the
option-pricing formula used here.

The method expands the risk-neutral density of the log-return on a truncated interval
:math:`[a, b]`, reading the series coefficients straight off the characteristic function.
The option value is then one dot product:

.. math::
    v = e^{-rT} \sum_{k=0}^{N-1}{}' \operatorname{Re}\!\Big\{
        \varphi_X\!\Big(\frac{k\pi}{b-a}\Big)\, e^{-i k \pi \frac{a}{b-a}} \Big\}\, V_k

where the prime halves the :math:`k = 0` term and :math:`V_k` are the cosine coefficients
of the payoff, available in closed form.

Coordinates
-----------
Everything is expressed in the log-return :math:`X_T = \log(S_T / S_0)` rather than in
log-moneyness. The consequence is that :math:`[a, b]`, the frequencies :math:`u_k` and
the characteristic function values are all independent of the strike, so a whole slice is
priced with one evaluation of :math:`\varphi` and a single matrix product. Only the
payoff coefficients carry the strike, through the upper integration limit
:math:`\log(K/S_0)`.

Two choices here are load-bearing, not style.

**Puts are priced; calls come from put-call parity.** CLAUDE.md section 5.3. The put
payoff :math:`(K - S_0 e^X)^+` lives on :math:`X < \log(K/S_0)`, so the exponential inside
its coefficients is bounded by the strike. The call payoff lives on the upper tail, where
the coefficients carry :math:`e^{b}` and :math:`b` grows like :math:`\sqrt{T}`; for long
maturities that term burns through the cancellation budget and the call series loses its
significant digits while still returning a finite, plausible-looking number. The unstable
series is therefore not implemented at all — an unstable code path that exists is one that
will eventually be called.

**The truncation range comes from cumulants, not from a guess.**

.. math::
    [a, b] = \Big[c_1 - L\sqrt{c_2 + \sqrt{c_4}},\; c_1 + L\sqrt{c_2 + \sqrt{c_4}}\Big],
    \quad L = 12
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from voldesk.quant.charfunc import char_func_log_return, log_return_cumulants
from voldesk.quant.model import HestonParams, MarketState
from voldesk.quant.pricing.blackscholes import OptionType

#: Truncation width in units of the square-root cumulant scale. Fang & Oosterlee use
#: L = 10 for Heston with a computed fourth cumulant; CLAUDE.md section 5.3 fixes L = 12,
#: which buys margin for c4 being set to zero. See docs/adr/0005.
L_TRUNCATION: Final[float] = 12.0

#: Default number of cosine terms.
#:
#: CLAUDE.md section 5.3 gives 256 as the default and requires it be "a convergence-tested
#: parameter, not a magic number". The convergence test says 256 is not enough for the
#: regimes this project actually runs. On the ``steep_skew`` regime of
#: ``voldesk/quant/synthetic/generator.py`` (Feller ratio 0.25, rho = -0.9), measured
#: against a Carr-Madan quadrature reference:
#:
#:     T = 0.5   n_cos 256: 2.7e-04    512: 4.0e-07   1024: 4.9e-07
#:     T = 1.0   n_cos 256: 1.3e-03    512: 5.6e-06   1024: 4.4e-06
#:
#: A 1e-3 pricing error is four orders of magnitude above the 1e-8 cross-validation gate
#: and would dominate the residuals of any calibration in that regime. So the default is
#: 512, and the cost is measured too: a 200-option surface takes 8.0 ms rather than
#: 6.4 ms, against the 50 ms budget in PROJECT_PLAN.md Phase 1 — the extra terms are
#: nearly free because the characteristic function is shared across the whole slice. See
#: ``docs/adr/0005-cos-truncation-range-and-n-cos.md`` and
#: ``tests/test_cos.py::test_cos_convergence_in_n_cos``.
N_COS_DEFAULT: Final[int] = 512


@dataclasses.dataclass(frozen=True, slots=True)
class TruncationRange:
    r"""The interval :math:`[a, b]` on which the log-return density is expanded."""

    a: float
    b: float

    @property
    def width(self) -> float:
        r""":math:`b - a`."""
        return self.b - self.a


def truncation_range(
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    l_factor: float = L_TRUNCATION,
) -> TruncationRange:
    r"""Cumulant-based truncation range for :math:`X_T = \log(S_T/S_0)`.

    Fang & Oosterlee (2008) eq. (49). Widening ``l_factor`` reduces truncation error and
    increases the number of terms needed for a given accuracy; that trade-off is measured
    in ``tests/test_cos.py`` rather than asserted.
    """
    c1, c2, c4 = log_return_cumulants(maturity, params, market)
    scale = l_factor * math.sqrt(c2 + math.sqrt(c4))
    return TruncationRange(c1 - scale, c1 + scale)


def _chi(
    omega: NDArray[np.float64], a: float, c: float, d: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Fang & Oosterlee (2008) eq. (22).

    .. math::
        \chi_k(c,d) = \int_c^d e^{y}\cos\!\Big(k\pi\frac{y-a}{b-a}\Big) dy

    ``omega`` is :math:`k\pi/(b-a)` with shape ``(n_cos,)``; ``d`` has shape
    ``(n_strikes,)``. The result is broadcast to ``(n_cos, n_strikes)``.
    """
    w = omega[:, None]
    dd = d[None, :]
    denom = 1.0 + w**2
    exp_d = np.exp(dd)
    exp_c = math.exp(c)
    return (
        np.cos(w * (dd - a)) * exp_d
        - np.cos(w * (c - a)) * exp_c
        + w * np.sin(w * (dd - a)) * exp_d
        - w * np.sin(w * (c - a)) * exp_c
    ) / denom


def _psi(
    omega: NDArray[np.float64], a: float, c: float, d: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Fang & Oosterlee (2008) eq. (23).

    .. math::
        \psi_k(c,d) = \int_c^d \cos\!\Big(k\pi\frac{y-a}{b-a}\Big) dy

    with :math:`\psi_0(c,d) = d - c`.
    """
    w = omega[:, None]
    dd = d[None, :]
    out = np.empty((omega.size, d.size), dtype=np.float64)
    out[0, :] = d - c
    w_nz = w[1:]
    out[1:, :] = (np.sin(w_nz * (dd - a)) - np.sin(w_nz * (c - a))) / w_nz
    return out


def _put_payoff_coefficients(
    omega: NDArray[np.float64],
    truncation: TruncationRange,
    strikes: NDArray[np.float64],
    spot: float,
) -> NDArray[np.float64]:
    r"""Cosine coefficients :math:`V_k` of the put payoff in the log-return coordinate.

    The payoff is :math:`(K - S_0 e^{X})^{+}`, non-zero for
    :math:`X < \log(K/S_0)`, so the integral runs from :math:`a` to
    :math:`d = \min(\log(K/S_0),\, b)`:

    .. math::
        V_k = \frac{2}{b-a}\Big(K\,\psi_k(a, d) - S_0\,\chi_k(a, d)\Big)

    Shape ``(n_cos, n_strikes)``. Every exponential inside :math:`\chi` is
    :math:`e^{X}` with :math:`X \le \log(K/S_0)`, hence bounded by the strike — this is
    exactly why the put series stays well conditioned where the call series does not.
    """
    a, b = truncation.a, truncation.b
    d = np.minimum(np.log(strikes / spot), b)

    coeffs = (2.0 / truncation.width) * (
        strikes[None, :] * _psi(omega, a, a, d) - spot * _chi(omega, a, a, d)
    )
    # Where the truncation interval lies entirely above the strike the payoff is
    # identically zero on [a, b] and so are its coefficients.
    coeffs[:, d <= a] = 0.0
    return coeffs


def price_put(
    strikes: NDArray[np.float64] | float,
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    n_cos: int = N_COS_DEFAULT,
    l_factor: float = L_TRUNCATION,
) -> NDArray[np.float64]:
    r"""European put prices by the COS method, vectorised over strikes.

    This is the production pricing path; :func:`price_call` derives calls from it by
    parity.

    Parameters
    ----------
    strikes
        One strike or an array of them.
    maturity
        Time to expiry :math:`T` in years.
    n_cos
        Number of cosine terms. For a smooth density the error decays exponentially in
        this; ``tests/test_cos.py`` records the achieved rate rather than claiming one.

    Notes
    -----
    The characteristic function is evaluated once per frequency, independent of how many
    strikes are being priced. Pricing a 200-option slice therefore costs barely more than
    pricing one option, which is what makes the Phase 3 experiment sweeps affordable.
    """
    k_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    if maturity <= 0.0:
        return np.maximum(k_arr - market.spot, 0.0)

    truncation = truncation_range(maturity, params, market, l_factor=l_factor)
    k = np.arange(n_cos, dtype=np.float64)
    omega = k * math.pi / truncation.width

    phi = char_func_log_return(omega, maturity, params, market)
    weights = np.real(phi * np.exp(-1j * omega * truncation.a))
    weights[0] *= 0.5  # the primed sum halves the k = 0 term

    v_k = _put_payoff_coefficients(omega, truncation, k_arr, market.spot)
    values = weights @ v_k
    return np.asarray(market.discount(maturity) * values, dtype=np.float64)


def price_call(
    strikes: NDArray[np.float64] | float,
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    n_cos: int = N_COS_DEFAULT,
    l_factor: float = L_TRUNCATION,
) -> NDArray[np.float64]:
    r"""European call prices, recovered from the put by put-call parity.

    .. math::
        C = P + S_0 e^{-qT} - K e^{-rT}

    CLAUDE.md section 5.3 requires this route.
    """
    k_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    puts = price_put(k_arr, maturity, params, market, n_cos=n_cos, l_factor=l_factor)
    fwd_pv = market.spot * math.exp(-market.dividend * maturity)
    return np.asarray(puts + fwd_pv - k_arr * market.discount(maturity), dtype=np.float64)


def price(
    strikes: NDArray[np.float64] | float,
    maturity: float,
    params: HestonParams,
    market: MarketState,
    option_type: OptionType = "call",
    *,
    n_cos: int = N_COS_DEFAULT,
    l_factor: float = L_TRUNCATION,
) -> NDArray[np.float64]:
    """Dispatch to :func:`price_call` or :func:`price_put`."""
    fn = price_call if option_type == "call" else price_put
    return fn(strikes, maturity, params, market, n_cos=n_cos, l_factor=l_factor)


def price_surface(
    strikes_by_maturity: dict[float, NDArray[np.float64]],
    params: HestonParams,
    market: MarketState,
    option_type: OptionType = "call",
    *,
    n_cos: int = N_COS_DEFAULT,
) -> dict[float, NDArray[np.float64]]:
    """Price a whole surface, one maturity slice at a time.

    The truncation range and the characteristic function both depend on maturity, so work
    is shared within a slice but not across slices.
    """
    return {
        maturity: price(strikes, maturity, params, market, option_type, n_cos=n_cos)
        for maturity, strikes in strikes_by_maturity.items()
    }
