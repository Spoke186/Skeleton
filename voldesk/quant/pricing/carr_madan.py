r"""Carr-Madan FFT pricing — an independent reference for the COS engine.

Reference: Carr, P. and Madan, D.B. (1999), "Option Valuation Using the Fast Fourier
Transform", *Journal of Computational Finance* 2(4), 61-73.

The call price is not square-integrable in log-strike, because it tends to :math:`S_0` as
:math:`K \to 0`. Carr and Madan fix that by damping it,
:math:`c_T(k) = e^{\alpha k} C_T(k)`, which *is* integrable for a suitable
:math:`\alpha > 0`, and then Fourier-inverting. Their eq. (6) gives the transform in
closed form:

.. math::
    \psi_T(v) = \frac{e^{-rT}\, \varphi\big(v - (\alpha+1)i\big)}
                     {\alpha^2 + \alpha - v^2 + i(2\alpha+1)v}

and their eq. (5) recovers the price:

.. math::
    C_T(k) = \frac{e^{-\alpha k}}{\pi} \int_0^{\infty} e^{-ivk}\, \psi_T(v)\, dv

Note that :math:`\varphi` is evaluated at the *complex* argument
:math:`v - (\alpha+1)i`, which is why :func:`voldesk.quant.charfunc.char_func_log_spot`
accepts complex ``u``.

Two entry points, for two different jobs
----------------------------------------
:func:`price_call_fft` is the method as published: one FFT produces a whole grid of
log-strikes at once. It is fast and it is what "Carr-Madan" means, but the strikes it
produces sit on the FFT grid, so any other strike needs interpolation, and that
interpolation error dominates everything else.

:func:`price_call_quadrature` evaluates the same integral by adaptive quadrature at the
exact strike asked for. It is slower per strike and it is not an FFT, but it carries no
grid error.

The Phase 1 gate "COS vs Carr-Madan agrees to < 1e-8" is run against the quadrature form,
because 1e-8 is below the interpolation floor of any practical FFT grid — insisting on it
there would be measuring the grid, not the pricer. The FFT form is checked separately at
a tolerance appropriate to its grid. Both are stated in ``tests/test_cross_validation.py``
with the measured numbers, and in ``docs/adr/0006-carr-madan-fft-versus-quadrature.md``.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import IntegrationWarning, quad

from voldesk.quant.charfunc import char_func_log_spot
from voldesk.quant.model import HestonParams, MarketState

#: Damping factor. Carr & Madan (1999) section 4 require alpha > 0 with
#: E[S_T^{alpha+1}] finite; for Heston with the parameter box of CLAUDE.md section 5.6,
#: alpha = 1.5 is inside the admissible region for every maturity used here.
ALPHA_DEFAULT: Final[float] = 1.5

#: FFT grid size and spacing. N * eta is the integration cutoff; lambda = 2*pi/(N*eta) is
#: the log-strike spacing, so the two cannot be tuned independently.
N_FFT_DEFAULT: Final[int] = 4096
ETA_DEFAULT: Final[float] = 0.25


def auto_upper_limit(maturity: float) -> float:
    r"""Integration cutoff for :func:`price_call_quadrature`, scaled by maturity.

    The Carr-Madan integrand inherits the Gaussian-like decay of the characteristic
    function, whose width in :math:`v` scales like :math:`1/\sqrt{T}`. A cutoff that is
    generous at one year is a truncation error at one month, so the rule is

    .. math::
        v_{\max} = \max\big(500,\; 1600/\sqrt{T}\big)

    Measured at :math:`\sigma = 0.5,\, T = 0.1`: a cutoff of 400 leaves an error of
    :math:`8\times10^{-8}` against a converged COS price, while this rule leaves
    :math:`3\times10^{-10}`.
    """
    return max(500.0, 1600.0 / math.sqrt(max(maturity, 1e-6)))


@dataclasses.dataclass(frozen=True, slots=True)
class FFTGrid:
    """The log-strike grid an FFT pricing produced, and the prices on it."""

    log_strikes: NDArray[np.float64]
    strikes: NDArray[np.float64]
    call_prices: NDArray[np.float64]

    @property
    def spacing(self) -> float:
        """Log-strike spacing :math:`\\lambda`."""
        return float(self.log_strikes[1] - self.log_strikes[0])


def _psi(
    v: NDArray[np.float64],
    maturity: float,
    params: HestonParams,
    market: MarketState,
    alpha: float,
) -> NDArray[np.complex128]:
    r"""Carr & Madan (1999) eq. (6), the Fourier transform of the damped call price."""
    u = np.asarray(v - (alpha + 1.0) * 1j, dtype=np.complex128)
    phi = char_func_log_spot(u, maturity, params, market)
    denom = alpha**2 + alpha - v**2 + 1j * (2.0 * alpha + 1.0) * v
    return np.asarray(market.discount(maturity) * phi / denom, dtype=np.complex128)


def price_call_fft(
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    alpha: float = ALPHA_DEFAULT,
    n_fft: int = N_FFT_DEFAULT,
    eta: float = ETA_DEFAULT,
) -> FFTGrid:
    r"""Price calls on the whole FFT log-strike grid in one transform.

    Simpson weights are applied to the integrand, as in Carr & Madan (1999) eq. (24),
    which buys an order of accuracy for free given the grid is already regular.

    Returns
    -------
    FFTGrid
        The log-strikes, the strikes, and the call prices, centred on the money.
    """
    lam = 2.0 * math.pi / (n_fft * eta)
    b = n_fft * lam / 2.0

    j = np.arange(n_fft, dtype=np.float64)
    v = eta * j
    log_strikes = -b + lam * j + math.log(market.spot)

    # Simpson weights 1, 4, 2, 4, ..., 4, 1 scaled by eta/3, with the j = 0 term halved
    # by the Kronecker delta of eq. (24).
    simpson = 3.0 + (-1.0) ** (j + 1)
    simpson[0] -= 1.0
    weights = eta * simpson / 3.0

    integrand = np.exp(1j * (b - math.log(market.spot)) * v) * _psi(
        v, maturity, params, market, alpha
    )
    transform = np.fft.fft(integrand * weights)

    call_prices = np.real(np.exp(-alpha * log_strikes) * transform) / math.pi
    return FFTGrid(
        log_strikes=log_strikes,
        strikes=np.exp(log_strikes),
        call_prices=np.asarray(call_prices, dtype=np.float64),
    )


def price_call_quadrature(
    strikes: NDArray[np.float64] | float,
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    alpha: float = ALPHA_DEFAULT,
    upper_limit: float | None = None,
    epsabs: float = 1e-13,
    epsrel: float = 1e-13,
) -> NDArray[np.float64]:
    r"""Evaluate the Carr-Madan integral by adaptive quadrature, at the exact strikes.

    The same :math:`\psi_T` as :func:`price_call_fft`; only the quadrature differs. Used
    as the high-accuracy reference in the cross-validation gate, where FFT grid
    interpolation error would otherwise set the floor.

    ``upper_limit`` truncates the semi-infinite integral; left as ``None`` it comes from
    :func:`auto_upper_limit`. This is not a detail. A fixed cutoff of 400 looks perfectly
    converged at :math:`T = 1` and silently costs :math:`8\times10^{-8}` at
    :math:`T = 0.1` — enough to make this *reference*, rather than the engine under test,
    the thing that fails a 1e-8 gate. That was measured while building the
    cross-validation suite; see ``docs/adr/0006-carr-madan-fft-versus-quadrature.md``.
    """
    if upper_limit is None:
        upper_limit = auto_upper_limit(maturity)
    k_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    out = np.empty_like(k_arr)

    def psi_re(v: float) -> float:
        return float(
            np.real(_psi(np.asarray([v], dtype=np.float64), maturity, params, market, alpha)[0])
        )

    def psi_im(v: float) -> float:
        return float(
            np.imag(_psi(np.asarray([v], dtype=np.float64), maturity, params, market, alpha)[0])
        )

    # Re{e^{-i v k} psi(v)} = Re(psi) cos(vk) + Im(psi) sin(vk). QUADPACK's oscillatory
    # weights integrate each piece against its own trigonometric kernel, which is what lets
    # this reach 1e-13 where a plain adaptive rule stalls around 1e-6 on an integrand that
    # oscillates roughly 150 times over the range.
    #
    # QUADPACK also emits a round-off warning, because epsabs = 1e-13 is at the edge of what
    # the accumulated sum can represent. Its *error estimate* is what becomes unreliable
    # there, not the result: the result was verified independently by sweeping the damping
    # factor from 0.75 to 3.0 and the cutoff from 100 to 20000, over which the price is
    # stable to 1e-10 — and an exact price cannot depend on either. The warning is scoped
    # away here rather than globally, so it cannot bury a real one somewhere else.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        for i, strike in enumerate(k_arr):
            log_k = math.log(float(strike))
            c_part, _ = quad(
                psi_re,
                0.0,
                upper_limit,
                weight="cos",
                wvar=log_k,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=400,
            )
            s_part, _ = quad(
                psi_im,
                0.0,
                upper_limit,
                weight="sin",
                wvar=log_k,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=400,
            )
            out[i] = math.exp(-alpha * log_k) * (c_part + s_part) / math.pi

    return out


def price_put_quadrature(
    strikes: NDArray[np.float64] | float,
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    alpha: float = ALPHA_DEFAULT,
) -> NDArray[np.float64]:
    """Puts from :func:`price_call_quadrature` by put-call parity.

    Carr-Madan damps the *call*; the put follows from parity exactly, so this adds no
    error of its own beyond the call's.
    """
    k_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    calls = price_call_quadrature(k_arr, maturity, params, market, alpha=alpha)
    fwd_pv = market.spot * math.exp(-market.dividend * maturity)
    return np.asarray(calls - fwd_pv + k_arr * market.discount(maturity), dtype=np.float64)


def interpolate_from_grid(grid: FFTGrid, strikes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linearly interpolate FFT grid prices onto arbitrary strikes.

    Exposed so that the interpolation error is visible and measurable in the tests rather
    than buried inside a pricing call that pretends to be exact.
    """
    return np.interp(np.log(strikes), grid.log_strikes, grid.call_prices)
