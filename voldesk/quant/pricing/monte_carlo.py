r"""Monte Carlo pricing with the Andersen (2008) Quadratic-Exponential scheme.

Reference: Andersen, L.B.G. (2008), "Simple and Efficient Simulation of the Heston
Stochastic Volatility Model", *Journal of Computational Finance* 11(3), 1-42.

This is the independent reference for the COS engine. Independent is the operative word:
it shares no code path with COS beyond the parameter object — no characteristic function,
no Fourier transform, no cumulants. If both agree, the agreement means something. Incident
rule R004 fires when they stop agreeing in production.

Why QE and not Euler
--------------------
A naive Euler discretisation of the variance process goes negative whenever
:math:`\kappa(\theta - v)\Delta` overshoots, and the usual repairs — full truncation,
reflection — introduce a bias that is largest exactly where this project spends its time:
low Feller ratio, where the variance genuinely visits the neighbourhood of zero. Andersen's
scheme instead matches the first two moments of the exact transition law with a
distribution chosen by a switching rule:

- when :math:`\psi = s^2/m^2 \le \psi_c`, a shifted squared Gaussian
  :math:`v_{t+\Delta} = a(b + Z)^2`;
- when :math:`\psi > \psi_c`, a mass at zero plus an exponential tail.

The switching threshold is :math:`\psi_c = 1.5`, fixed by CLAUDE.md section 5.3.

Martingale correction
---------------------
The discretised spot is not automatically a martingale, and the drift error accumulates
over steps in a way that looks exactly like a mispriced forward. Andersen section 4.2
corrects it by solving for the :math:`K_0` that makes
:math:`\mathbb{E}[S_{t+\Delta} \mid \mathcal{F}_t] = S_t e^{(r-q)\Delta}` hold exactly
under the scheme's own distribution — different closed form in each branch of the switch.
That is implemented here; without it the COS/MC comparison drifts with maturity and it is
tempting to blame the Fourier method.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from voldesk.quant.model import HestonParams, MarketState
from voldesk.quant.pricing.blackscholes import OptionType

#: Switching rule threshold, CLAUDE.md section 5.3 and Andersen (2008) section 3.2.4.
PSI_C: Final[float] = 1.5

#: Central-discretisation weights gamma1 = gamma2 = 1/2, Andersen (2008) eq. (33). The
#: alternative gamma1 = 1, gamma2 = 0 is the log-Euler variant and is measurably worse.
GAMMA_1: Final[float] = 0.5
GAMMA_2: Final[float] = 0.5


@dataclasses.dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """A Monte Carlo price with the uncertainty attached to it.

    A Monte Carlo number without its standard error cannot be compared to anything, so the
    two never travel separately here. ``tests`` compares COS against ``price`` within
    ``3 * standard_error``, which is the gate in CLAUDE.md section 7.
    """

    price: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    n_paths: int
    n_steps: int
    seed: int

    def within(self, reference: NDArray[np.float64], n_sigma: float = 3.0) -> NDArray[np.bool_]:
        """Whether ``reference`` lies within ``n_sigma`` standard errors, elementwise."""
        return np.abs(self.price - reference) <= n_sigma * self.standard_error

    def z_scores(self, reference: NDArray[np.float64]) -> NDArray[np.float64]:
        """Signed discrepancy in units of the standard error."""
        return (self.price - reference) / np.maximum(self.standard_error, 1e-300)


def _qe_variance_step(
    v: NDArray[np.float64],
    z_v: NDArray[np.float64],
    u_v: NDArray[np.float64],
    params: HestonParams,
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    r"""One QE step of the variance process.

    Andersen (2008) eqs. (17)-(18) for the moments, (23)-(24) for the quadratic branch and
    (25)-(26) for the exponential branch.

    Returns
    -------
    tuple
        ``(v_next, a_or_p, b2_or_beta, use_quadratic)`` — the branch parameters are handed
        back so the martingale correction can use the very distribution that was sampled,
        rather than an approximation to it.
    """
    kappa, theta, sigma = params.kappa, params.theta, params.sigma
    e = math.exp(-kappa * dt)

    # Exact conditional mean and variance of v_{t+dt} given v_t, Andersen eqs. (17)-(18).
    m = theta + (v - theta) * e
    s2 = (v * sigma**2 * e / kappa) * (1.0 - e) + (theta * sigma**2 / (2.0 * kappa)) * (
        1.0 - e
    ) ** 2
    psi = s2 / np.maximum(m**2, 1e-300)

    use_quadratic = psi <= PSI_C

    # --- quadratic branch: v = a (b + Z)^2 ------------------------------------------
    psi_q = np.where(use_quadratic, psi, 1.0)  # placeholder keeps the sqrt real
    inv = 2.0 / psi_q
    b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
    a = m / (1.0 + b2)
    v_quad = a * (np.sqrt(b2) + z_v) ** 2

    # --- exponential branch: P(v = 0) = p, else Exp(beta) ---------------------------
    psi_e = np.where(use_quadratic, 2.0, psi)  # placeholder keeps p in (0, 1)
    p = (psi_e - 1.0) / (psi_e + 1.0)
    beta = (1.0 - p) / np.maximum(m, 1e-300)
    # Inverse of the CDF: zero below the atom, exponential tail above it.
    v_exp = np.where(u_v <= p, 0.0, np.log(np.maximum((1.0 - p) / (1.0 - u_v), 1.0)) / beta)

    v_next = np.where(use_quadratic, v_quad, v_exp)
    return v_next, np.where(use_quadratic, a, p), np.where(use_quadratic, b2, beta), use_quadratic


def _martingale_k0(
    v: NDArray[np.float64],
    a_or_p: NDArray[np.float64],
    b2_or_beta: NDArray[np.float64],
    use_quadratic: NDArray[np.bool_],
    k1: float,
    k2: float,
    k3: float,
    k4: float,
) -> NDArray[np.float64]:
    r"""The :math:`K_0^*` that makes the discretised spot an exact martingale.

    Andersen (2008) section 4.2, eqs. (32)-(34). Writing
    :math:`A = K_2 + \tfrac{1}{2}K_4`, the requirement is

    .. math::
        K_0^* = -\log \mathbb{E}\big[e^{A v_{t+\Delta}}\big]
                - \big(K_1 + \tfrac{1}{2}K_3\big) v_t

    and the moment generating function is available in closed form in each branch:

    .. math::
        \mathbb{E}\big[e^{A v}\big] =
        \begin{cases}
            \dfrac{\exp\!\big(\frac{A a b^2}{1 - 2Aa}\big)}{\sqrt{1 - 2Aa}},
                & \psi \le \psi_c \\[2ex]
            p + \dfrac{(1-p)\beta}{\beta - A}, & \psi > \psi_c
        \end{cases}

    The quadratic branch needs :math:`2Aa < 1` and the exponential branch :math:`A < \beta`
    for the expectation to exist. Both hold for the step sizes used here; where they would
    not, the correction is dropped for that path rather than producing a NaN, which is a
    bias of order ``dt`` in a place the scheme is already inaccurate.
    """
    a_coef = k2 + 0.5 * k4

    # quadratic branch
    a_q = a_or_p
    b2_q = b2_or_beta
    denom_q = 1.0 - 2.0 * a_coef * a_q
    safe_q = denom_q > 1e-12
    log_mgf_quad = np.where(
        safe_q,
        a_coef * a_q * b2_q / np.where(safe_q, denom_q, 1.0)
        - 0.5 * np.log(np.where(safe_q, denom_q, 1.0)),
        0.0,
    )

    # exponential branch
    p = a_or_p
    beta = b2_or_beta
    safe_e = beta > a_coef + 1e-12
    mgf_exp = np.where(safe_e, p + (1.0 - p) * beta / np.where(safe_e, beta - a_coef, 1.0), 1.0)
    log_mgf_exp = np.log(np.maximum(mgf_exp, 1e-300))

    log_mgf = np.where(use_quadratic, log_mgf_quad, log_mgf_exp)
    return -log_mgf - (k1 + 0.5 * k3) * v


def simulate_paths(
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    n_paths: int = 200_000,
    n_steps: int = 100,
    seed: int = 0,
    antithetic: bool = True,
) -> NDArray[np.float64]:
    r"""Simulate terminal spot values under the QE scheme.

    Parameters
    ----------
    n_paths
        Total number of paths. With ``antithetic=True`` this must be even; the paths come
        in mirrored pairs sharing the same uniforms through :math:`U \mapsto 1 - U`, which
        antithetises the variance draws and the spot draws consistently.
    n_steps
        Time steps. The QE scheme is not exact, so this is a convergence parameter;
        ``experiments/e4_cross_validation.py`` measures the rate.
    seed
        Every random draw in this project is seeded and the seed is persisted, per
        CLAUDE.md invariant 5.

    Returns
    -------
    ndarray
        Terminal spot prices, shape ``(n_paths,)``.
    """
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"antithetic sampling needs an even n_paths, got {n_paths}")

    dt = maturity / n_steps
    # theta enters only through the QE variance step, not through the K coefficients.
    kappa, sigma, rho = params.kappa, params.sigma, params.rho

    # Andersen (2008) eq. (33), with gamma1 = gamma2 = 1/2.
    k1 = GAMMA_1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    k2 = GAMMA_2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    k3 = GAMMA_1 * dt * (1.0 - rho**2)
    k4 = GAMMA_2 * dt * (1.0 - rho**2)

    rng = np.random.default_rng(seed)
    n_base = n_paths // 2 if antithetic else n_paths

    v = np.full(n_paths, params.v0, dtype=np.float64)
    log_s = np.full(n_paths, math.log(market.spot), dtype=np.float64)
    drift = (market.rate - market.dividend) * dt

    for _ in range(n_steps):
        # Draw uniforms once and mirror them. Working from uniforms rather than from
        # normals is what lets the antithetic pairing act on the exponential branch of the
        # QE switch as well as on the Gaussian one.
        u_v_base = rng.random(n_base)
        u_s_base = rng.random(n_base)
        if antithetic:
            u_v = np.concatenate([u_v_base, 1.0 - u_v_base])
            u_s = np.concatenate([u_s_base, 1.0 - u_s_base])
        else:
            u_v, u_s = u_v_base, u_s_base

        z_v = norm.ppf(np.clip(u_v, 1e-16, 1.0 - 1e-16))
        z_s = norm.ppf(np.clip(u_s, 1e-16, 1.0 - 1e-16))

        v_next, a_or_p, b2_or_beta, use_quadratic = _qe_variance_step(v, z_v, u_v, params, dt)
        k0 = _martingale_k0(v, a_or_p, b2_or_beta, use_quadratic, k1, k2, k3, k4)

        variance_term = np.maximum(k3 * v + k4 * v_next, 0.0)
        log_s = log_s + drift + k0 + k1 * v + k2 * v_next + np.sqrt(variance_term) * z_s
        v = v_next

    return np.exp(log_s)


def price_european(
    strikes: NDArray[np.float64] | float,
    maturity: float,
    params: HestonParams,
    market: MarketState,
    option_type: OptionType = "call",
    *,
    n_paths: int = 200_000,
    n_steps: int = 100,
    seed: int = 0,
    antithetic: bool = True,
) -> MonteCarloResult:
    r"""Price European options by simulation, with standard errors.

    With antithetic sampling the two paths in a pair are not independent, so the standard
    error is computed over *pair averages* rather than over individual payoffs. Computing
    it over the raw paths would understate the true error by pretending there are twice as
    many independent samples as there are — a mistake that would make the COS/MC gate pass
    for the wrong reason.
    """
    k_arr = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    terminal = simulate_paths(
        maturity,
        params,
        market,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
        antithetic=antithetic,
    )

    if option_type == "call":
        payoff = np.maximum(terminal[:, None] - k_arr[None, :], 0.0)
    else:
        payoff = np.maximum(k_arr[None, :] - terminal[:, None], 0.0)
    payoff *= market.discount(maturity)

    if antithetic:
        half = n_paths // 2
        samples = 0.5 * (payoff[:half] + payoff[half:])
    else:
        samples = payoff

    n_independent = samples.shape[0]
    mean = samples.mean(axis=0)
    stderr = samples.std(axis=0, ddof=1) / math.sqrt(n_independent)

    return MonteCarloResult(
        price=np.asarray(mean, dtype=np.float64),
        standard_error=np.asarray(stderr, dtype=np.float64),
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed,
    )


def forward_martingale_error(
    maturity: float,
    params: HestonParams,
    market: MarketState,
    *,
    n_paths: int = 200_000,
    n_steps: int = 100,
    seed: int = 0,
) -> tuple[float, float]:
    r"""Relative error in :math:`\mathbb{E}[S_T]` against the analytic forward.

    A direct check that the martingale correction is doing its job: the simulated mean
    terminal spot must equal :math:`S_0 e^{(r-q)T}` to within Monte Carlo noise. If this
    drifts, every price from the scheme is biased and no amount of paths will fix it.

    Returns
    -------
    tuple
        ``(relative_error, relative_standard_error)``.
    """
    terminal = simulate_paths(maturity, params, market, n_paths=n_paths, n_steps=n_steps, seed=seed)
    forward = market.forward(maturity)
    mean = float(terminal.mean())
    stderr = float(terminal.std(ddof=1) / math.sqrt(n_paths))
    return (mean - forward) / forward, stderr / forward
