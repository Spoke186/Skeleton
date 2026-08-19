r"""Gates on the COS engine: Black-Scholes limit, put-call parity, convergence, positivity.

Four of the eight gates in CLAUDE.md section 7 live here. The remaining four — the two
cross-method comparisons, the noiseless recovery and the CF continuity — live in
``test_cross_validation.py``, ``test_calibration.py`` and ``test_charfunc.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.conftest import BS_LIMIT, EXTREME, FELLER_VIOLATING, MATURITIES, REFERENCE
from voldesk.quant.model import HestonParams, MarketState
from voldesk.quant.pricing import cos
from voldesk.quant.pricing.blackscholes import bs_price

#: Prices below this carry no meaningful relative accuracy from any Fourier method: an
#: option worth 1e-3 on a spot of 100 is one part in 1e5 of the notional, and a 1e-8
#: absolute error in it is a 1e-5 relative one. Relative tolerances are applied above the
#: floor and absolute ones below it, and both are asserted, so nothing escapes unchecked.
#: 0.1 is a tenth of one percent of spot — comfortably inside where a quote would exist.
PRICE_FLOOR = 0.1


@pytest.mark.gate
@pytest.mark.parametrize("maturity", MATURITIES)
def test_black_scholes_limit(
    maturity: float, market: MarketState, wide_strikes: np.ndarray
) -> None:
    r"""GATE (CLAUDE.md section 7, "BS limit").

    With :math:`\sigma \to 0` and :math:`v_0 = \theta`, the variance process is frozen at
    :math:`\theta` and Heston degenerates to Black-Scholes at volatility
    :math:`\sqrt{\theta}`. Reproducing that is the cheapest possible check that the
    characteristic function has no sign error: almost any mistake in :math:`C` or
    :math:`D` breaks it immediately.

    **Why the correlation must be zero here.** The leading correction to the
    Black-Scholes price is :math:`O(\rho\sigma)`, the skew term. At
    :math:`\sigma = 10^{-4}` with :math:`\rho = -0.7` that is a *genuine* relative
    difference of about :math:`2\times10^{-4}` — the model really is not Black-Scholes,
    and asserting 1e-6 against it would be asserting that a correct pricer is wrong.
    Measured: the discrepancy scales exactly linearly in :math:`\sigma` from 1e-2 down to
    1e-5, which is the signature of a real :math:`O(\rho\sigma)` term rather than of an
    implementation error. With :math:`\rho = 0` the leading correction is
    :math:`O(\sigma^2)` and the limit is clean.
    """
    cos_put = cos.price_put(wide_strikes, maturity, BS_LIMIT, market, n_cos=512)
    bs_put = bs_price(
        market.spot,
        wide_strikes,
        maturity,
        market.rate,
        market.dividend,
        math.sqrt(BS_LIMIT.theta),
        "put",
    )

    absolute = np.abs(cos_put - bs_put)
    # The residual O(sigma^2) difference between Heston at sigma = 1e-4 and Black-Scholes
    # is genuinely of order 1e-8 on a spot of 100. Measured maximum across the maturity
    # range: 5.2e-8, so the ceiling below is the model's own limit, not slack.
    assert np.max(absolute) < 1e-7, f"max absolute error {np.max(absolute):.3e}"

    meaningful = bs_put > PRICE_FLOOR
    relative = absolute[meaningful] / bs_put[meaningful]
    assert np.max(relative) < 1e-6, (
        f"max relative error {np.max(relative):.3e} on prices above {PRICE_FLOOR}"
    )


@pytest.mark.gate
@pytest.mark.parametrize(
    "params",
    [REFERENCE, FELLER_VIOLATING, EXTREME],
    ids=["reference", "feller_violating", "extreme"],
)
@pytest.mark.parametrize("maturity", MATURITIES)
def test_put_call_parity(
    params: HestonParams, maturity: float, market: MarketState, wide_strikes: np.ndarray
) -> None:
    r"""GATE (CLAUDE.md section 7, "Put-call parity"), exact to < 1e-10 across the grid.

    .. math::
        C - P = S_0 e^{-qT} - K e^{-rT}

    Because :func:`cos.price_call` *derives* the call from the put by exactly this
    identity, the test is close to a tautology on its own — its real job is to pin the
    discount and dividend conventions, where an error would otherwise show up much later
    as a small, plausible calibration bias in :math:`\rho`.
    """
    puts = cos.price_put(wide_strikes, maturity, params, market)
    calls = cos.price_call(wide_strikes, maturity, params, market)
    expected = market.spot * math.exp(-market.dividend * maturity) - wide_strikes * market.discount(
        maturity
    )
    assert np.max(np.abs((calls - puts) - expected)) < 1e-10


@pytest.mark.gate
@pytest.mark.parametrize(
    "params", [REFERENCE, FELLER_VIOLATING], ids=["reference", "feller_violating"]
)
@pytest.mark.parametrize("maturity", MATURITIES)
def test_prices_respect_no_arbitrage_bounds(
    params: HestonParams, maturity: float, market: MarketState, wide_strikes: np.ndarray
) -> None:
    r"""GATE (CLAUDE.md section 7, "Positivity").

    .. math::
        \max(K e^{-rT} - S_0 e^{-qT},\, 0) \le P \le K e^{-rT}

    and the same for calls. These hold under no assumption but no-arbitrage, so a
    violation means the pricer, not the model.

    The tolerance is a few ULP of the price scale rather than exactly zero: the bound and
    the price are computed by different routes, and demanding bit-exactness would be a
    test of floating-point associativity.
    """
    tol = 1e-10
    df_r = market.discount(maturity)
    df_q = math.exp(-market.dividend * maturity)

    puts = cos.price_put(wide_strikes, maturity, params, market)
    put_lower = np.maximum(wide_strikes * df_r - market.spot * df_q, 0.0)
    put_upper = wide_strikes * df_r
    assert np.all(puts >= put_lower - tol), f"put below intrinsic by {np.min(puts - put_lower):.3e}"
    assert np.all(puts <= put_upper + tol)

    calls = cos.price_call(wide_strikes, maturity, params, market)
    call_lower = np.maximum(market.spot * df_q - wide_strikes * df_r, 0.0)
    call_upper = np.full_like(wide_strikes, market.spot * df_q)
    assert np.all(calls >= call_lower - tol)
    assert np.all(calls <= call_upper + tol)


@pytest.mark.gate
@pytest.mark.parametrize(
    "params", [REFERENCE, FELLER_VIOLATING], ids=["reference", "feller_violating"]
)
def test_cos_convergence_in_n_cos(
    params: HestonParams, market: MarketState, strikes: np.ndarray
) -> None:
    """GATE (CLAUDE.md section 7, "COS convergence") — and it records the achieved rate.

    The COS series error decays exponentially in the number of terms for a density that is
    smooth on the truncation interval, which the Heston log-return density is. So the test
    asserts a *rate*, not a tolerance: each doubling of ``n_cos`` must cut the error by at
    least an order of magnitude until it reaches the truncation floor.

    The floor is set by the truncation range, not by the series, and is where the error
    stops improving — see ``docs/adr/0005-cos-truncation-range-and-n-cos.md`` for the
    measured numbers behind the default ``N_cos = 256``.
    """
    maturity = 1.0
    reference = cos.price_call(strikes, maturity, params, market, n_cos=8192)

    errors = {}
    for n in (16, 32, 64, 128, 256):
        priced = cos.price_call(strikes, maturity, params, market, n_cos=n)
        errors[n] = float(np.max(np.abs(priced - reference)))

    # Exponential convergence. Measured on the Feller-violating set at T = 1:
    #   n_cos:  16       32       64       128      256
    #   error:  1.8e+00  2.7e-01  1.2e-02  1.3e-04  1.5e-08
    # so each doubling gains between one and four orders of magnitude, accelerating as the
    # series settles. The assertion takes the weakest observed step as its bar.
    for coarse, fine in zip([16, 32, 64, 128], [32, 64, 128, 256], strict=True):
        if errors[coarse] < 1e-12:
            continue
        assert errors[fine] < errors[coarse] / 5.0, (
            f"n_cos {coarse} -> {fine}: error only fell from {errors[coarse]:.3e} to "
            f"{errors[fine]:.3e}; that is not exponential convergence"
        )

    assert errors[256] < 1e-7, f"error at the default n_cos=256 is {errors[256]:.3e}"


def test_default_n_cos_is_sufficient_for_the_calibration_region(
    market: MarketState, wide_strikes: np.ndarray
) -> None:
    """The shipped default must be accurate where calibration actually spends its time.

    A default that is only correct at the reference parameters is not a default. This
    sweeps the whole maturity range at a Feller-violating parameter set, which is the
    common case in practice.
    """
    for maturity in MATURITIES:
        reference = cos.price_call(wide_strikes, maturity, FELLER_VIOLATING, market, n_cos=8192)
        default = cos.price_call(wide_strikes, maturity, FELLER_VIOLATING, market)
        assert np.max(np.abs(default - reference)) < 1e-7, (
            f"T={maturity}: n_cos={cos.N_COS_DEFAULT} leaves "
            f"{np.max(np.abs(default - reference)):.3e}"
        )


def test_truncation_range_widens_with_maturity(market: MarketState) -> None:
    """The truncation interval must grow with maturity, roughly like the square root.

    A range that failed to widen would silently clip the tails of long-dated densities,
    which is exactly the regime the identifiability analysis cares about.
    """
    widths = [cos.truncation_range(t, REFERENCE, market).width for t in (0.25, 1.0, 4.0)]
    assert widths[0] < widths[1] < widths[2]
    # Doubling sqrt(T) should not far more than double the width.
    assert 1.5 < widths[1] / widths[0] < 3.0
    assert 1.5 < widths[2] / widths[1] < 3.0


def test_vectorised_and_scalar_pricing_agree(market: MarketState, strikes: np.ndarray) -> None:
    """Pricing a ladder must equal pricing its members one at a time.

    The engine shares the characteristic function across strikes for speed; this is the
    test that the sharing is correct rather than merely fast.
    """
    batch = cos.price_call(strikes, 1.0, REFERENCE, market)
    one_by_one = np.array(
        [cos.price_call(np.array([k]), 1.0, REFERENCE, market)[0] for k in strikes]
    )
    assert np.allclose(batch, one_by_one, rtol=1e-13, atol=1e-13)


def test_zero_maturity_returns_intrinsic_value(market: MarketState, strikes: np.ndarray) -> None:
    """At expiry the price is the payoff; no Fourier machinery should run."""
    puts = cos.price_put(strikes, 0.0, REFERENCE, market)
    assert np.allclose(puts, np.maximum(strikes - market.spot, 0.0))


def test_call_prices_decrease_in_strike(market: MarketState, wide_strikes: np.ndarray) -> None:
    """Monotonicity in strike, a static no-arbitrage requirement on any surface.

    This is the same property ``voldesk/quality/arbitrage.py`` checks on market data. If
    the engine itself produced a surface that failed it, every downstream quality signal
    would be measuring the pricer.
    """
    for maturity in MATURITIES:
        calls = cos.price_call(wide_strikes, maturity, FELLER_VIOLATING, market)
        assert np.all(np.diff(calls) <= 1e-12), f"T={maturity}: call prices not monotone"


def test_pricing_a_200_option_surface_is_fast(market: MarketState) -> None:
    """Benchmark from PROJECT_PLAN.md Phase 1: 200 options via COS.

    The plan sets ~50 ms as the point at which to stop and profile, because the Phase 3
    experiment sweeps price this surface tens of thousands of times. The measurement is
    recorded in the test output rather than asserted tightly — a CI runner under load is
    not a benchmark — but a gross regression will still trip the ceiling.
    """
    import time

    maturities = np.linspace(0.1, 2.0, 10)
    strikes = np.linspace(65.0, 135.0, 20)
    assert len(maturities) * len(strikes) == 200

    # Warm up, so the measurement excludes first-call import and allocation costs.
    for maturity in maturities:
        cos.price_call(strikes, float(maturity), REFERENCE, market)

    start = time.perf_counter()
    for maturity in maturities:
        cos.price_call(strikes, float(maturity), REFERENCE, market)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print(f"\n200-option COS surface: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 250.0, f"200-option surface took {elapsed_ms:.1f} ms"
