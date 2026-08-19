r"""Black-Scholes prices, greeks, and implied-volatility inversion by Brent's method.

Two roles. First, the :math:`\sigma \to 0` limit of Heston must reproduce Black-Scholes,
which is gate 1 of CLAUDE.md section 7 and the cheapest way to catch a sign error in the
characteristic function. Second, calibration happens in implied-volatility space
(CLAUDE.md section 5.5), so every model and market price has to be inverted to a
volatility, and that inversion has to fail loudly when no volatility exists.

CLAUDE.md section 5.4 is unambiguous about the failure mode: a price outside the
no-arbitrage bounds returns ``NaN`` together with a reject reason. It is never clamped to
the nearest feasible value. Clamping would silently move a bad quote onto the boundary
of the feasible set, where it looks like data.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.stats import norm

from voldesk.quant.reject_reasons import RejectReason

OptionType = Literal["call", "put"]

#: Brent search interval for the implied volatility, CLAUDE.md section 5.4.
IV_LOWER: float = 1e-6
IV_UPPER: float = 5.0

#: Absolute tolerance on the price when deciding whether a quote sits on a no-arbitrage
#: boundary. Prices are of order 1e0 to 1e2, so 1e-12 is comfortably below round-off in
#: the bound computation itself while still catching genuine violations.
_BOUND_TOL: float = 1e-12


def _d1_d2(
    spot: float,
    strike: NDArray[np.float64] | float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: NDArray[np.float64] | float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k = np.asarray(strike, dtype=np.float64)
    s = np.asarray(vol, dtype=np.float64)
    sqrt_t = math.sqrt(maturity)
    d1 = (np.log(spot / k) + (rate - dividend + 0.5 * s**2) * maturity) / (s * sqrt_t)
    return d1, d1 - s * sqrt_t


def bs_price(
    spot: float,
    strike: NDArray[np.float64] | float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: NDArray[np.float64] | float,
    option_type: OptionType = "call",
) -> NDArray[np.float64]:
    r"""Black-Scholes-Merton price with a continuous dividend yield.

    .. math::
        C &= S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2) \\
        P &= K e^{-rT} N(-d_2) - S_0 e^{-qT} N(-d_1)

    Vectorised over ``strike`` and ``vol``. Reference: Merton, R.C. (1973), "Theory of
    Rational Option Pricing", *Bell Journal of Economics* 4(1), 141-183, eq. (18).
    """
    k = np.asarray(strike, dtype=np.float64)
    df_r = math.exp(-rate * maturity)
    df_q = math.exp(-dividend * maturity)

    if maturity <= 0.0:
        intrinsic = (
            np.maximum(spot - k, 0.0) if option_type == "call" else np.maximum(k - spot, 0.0)
        )
        return np.asarray(intrinsic, dtype=np.float64)

    d1, d2 = _d1_d2(spot, k, maturity, rate, dividend, vol)
    if option_type == "call":
        price = spot * df_q * norm.cdf(d1) - k * df_r * norm.cdf(d2)
    else:
        price = k * df_r * norm.cdf(-d2) - spot * df_q * norm.cdf(-d1)
    return np.asarray(price, dtype=np.float64)


def bs_vega(
    spot: float,
    strike: NDArray[np.float64] | float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    r""":math:`\partial V / \partial \sigma = S_0 e^{-qT} \phi(d_1)\sqrt{T}`.

    Identical for calls and puts. Used to convert price residuals into volatility
    residuals when a full inversion would be wasteful, and to sanity-check the
    conditioning of the inversion near the wings, where vega collapses.
    """
    if maturity <= 0.0:
        return np.zeros_like(np.asarray(strike, dtype=np.float64))
    d1, _ = _d1_d2(spot, strike, maturity, rate, dividend, vol)
    return np.asarray(spot * math.exp(-dividend * maturity) * norm.pdf(d1) * math.sqrt(maturity))


def no_arbitrage_bounds(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    option_type: OptionType,
) -> tuple[float, float]:
    r"""Model-free lower and upper bounds on a European option price.

    .. math::
        \max(S_0 e^{-qT} - K e^{-rT},\, 0) \le C \le S_0 e^{-qT} \\
        \max(K e^{-rT} - S_0 e^{-qT},\, 0) \le P \le K e^{-rT}

    These hold under no assumption beyond no-arbitrage and non-negative payoffs. A price
    outside them is not a price with an unusual volatility; it is an arbitrage, and no
    Black-Scholes volatility reproduces it.
    """
    df_r = math.exp(-rate * maturity)
    df_q = math.exp(-dividend * maturity)
    fwd_pv = spot * df_q
    strike_pv = strike * df_r
    if option_type == "call":
        return max(fwd_pv - strike_pv, 0.0), fwd_pv
    return max(strike_pv - fwd_pv, 0.0), strike_pv


@dataclasses.dataclass(frozen=True, slots=True)
class ImpliedVolResult:
    """Outcome of one implied-volatility inversion.

    ``vol`` is ``NaN`` exactly when ``reject_reason`` is set. There is no third state and
    no clamped value, which is the point of returning a record rather than a float.
    """

    vol: float
    reject_reason: RejectReason | None = None
    iterations: int = 0

    @property
    def ok(self) -> bool:
        """Whether a volatility was found."""
        return self.reject_reason is None


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    option_type: OptionType = "call",
    *,
    lower: float = IV_LOWER,
    upper: float = IV_UPPER,
) -> ImpliedVolResult:
    r"""Invert Black-Scholes for :math:`\sigma` by Brent's method on ``[lower, upper]``.

    The price is monotonically increasing in volatility (vega > 0 for :math:`T > 0`), so
    the root is unique wherever it exists and a bracketing method is both safe and fast.
    Brent is used rather than Newton because vega vanishes in the deep wings, where Newton
    diverges precisely on the quotes that matter most for this project.

    Failure is explicit, per CLAUDE.md section 5.4:

    - price outside the no-arbitrage bounds -> :attr:`RejectReason.IV_NO_SOLUTION`
    - price inside the bounds but outside the volatility bracket ->
      :attr:`RejectReason.IV_NOT_BRACKETED`
    - non-finite input or expired option -> :attr:`RejectReason.MISSING_FIELD` /
      :attr:`RejectReason.EXPIRED`

    In every failing case ``vol`` is ``NaN``. Nothing is clamped.
    """
    if not all(math.isfinite(x) for x in (price, spot, strike, maturity, rate, dividend)):
        return ImpliedVolResult(math.nan, RejectReason.MISSING_FIELD)
    if maturity <= 0.0:
        return ImpliedVolResult(math.nan, RejectReason.EXPIRED)
    if price < 0.0:
        return ImpliedVolResult(math.nan, RejectReason.NEGATIVE_PRICE)

    low_bound, high_bound = no_arbitrage_bounds(spot, strike, maturity, rate, dividend, option_type)
    if price < low_bound - _BOUND_TOL:
        return ImpliedVolResult(math.nan, RejectReason.IV_NO_SOLUTION)
    if price > high_bound + _BOUND_TOL:
        return ImpliedVolResult(math.nan, RejectReason.IV_NO_SOLUTION)

    def objective(vol: float) -> float:
        return float(bs_price(spot, strike, maturity, rate, dividend, vol, option_type)) - price

    f_low = objective(lower)
    f_high = objective(upper)

    # A price at — or numerically indistinguishable from — the intrinsic boundary has no
    # meaningful implied volatility: the objective is flat there to machine precision, and
    # every volatility in a neighbourhood of `lower` reproduces the price exactly. Brent
    # would happily return `lower` itself, which reads like an answer and is really a clamp
    # in disguise, and CLAUDE.md section 5.4 rules those out. So it is reported as a
    # bracketing failure, which is what it is, and gets its own reject reason so the
    # support console can distinguish it from an actual arbitrage.
    if f_low >= 0.0 or f_high < 0.0:
        return ImpliedVolResult(math.nan, RejectReason.IV_NOT_BRACKETED)

    root, result = brentq(objective, lower, upper, xtol=1e-12, rtol=1e-14, full_output=True)
    if not result.converged:
        return ImpliedVolResult(math.nan, RejectReason.IV_NOT_BRACKETED)
    return ImpliedVolResult(float(root), None, int(result.iterations))


def vol_resolution(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend: float,
    vol: float,
) -> float:
    r"""Smallest volatility change that moves the price by one representable amount.

    .. math::
        \delta\sigma_{\min} \approx \frac{\varepsilon\, V}{\mathcal{V}}

    where :math:`\mathcal{V}` is vega, :math:`V` the price and :math:`\varepsilon` the
    double-precision epsilon. Below this, two volatilities produce the *same* float64
    price and no inversion — Brent, Newton or otherwise — can tell them apart.

    This is not a limitation of the implementation, it is a property of the problem, and
    it is severe exactly where this project's subject matter lives. A deep in-the-money
    call at three-month maturity has a vega of order :math:`10^{-9}` on a price of order
    :math:`25`, giving a resolution of about :math:`10^{-6}` in volatility — a hundred
    times the size of a quoted tick in vol terms.

    It is the concrete, mechanical reason CLAUDE.md section 5.5 fits in volatility space
    with spread-inverse weights instead of in price space: such a quote carries almost no
    information about the volatility, and a price-space objective would nonetheless give
    it the largest weight in the sum.
    """
    vega = float(bs_vega(spot, strike, maturity, rate, dividend, vol))
    price = float(bs_price(spot, strike, maturity, rate, dividend, vol, "call"))
    if vega <= 0.0:
        return math.inf
    return float(np.finfo(np.float64).eps * max(price, 1.0) / vega)


def implied_vol_surface(
    prices: NDArray[np.float64],
    spot: float,
    strikes: NDArray[np.float64],
    maturity: float,
    rate: float,
    dividend: float,
    option_type: OptionType = "call",
) -> tuple[NDArray[np.float64], list[RejectReason | None]]:
    """Invert a whole strike slice, one strike at a time.

    Returns the volatilities, ``NaN`` where the inversion failed, alongside the parallel
    list of reject reasons. The caller is expected to persist those reasons; dropping the
    second return value would violate CLAUDE.md invariant 3.
    """
    vols = np.full(len(strikes), np.nan, dtype=np.float64)
    reasons: list[RejectReason | None] = []
    for i, (price, strike) in enumerate(zip(prices, strikes, strict=True)):
        result = implied_vol(
            float(price), spot, float(strike), maturity, rate, dividend, option_type
        )
        vols[i] = result.vol
        reasons.append(result.reject_reason)
    return vols, reasons
