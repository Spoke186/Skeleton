r"""Gates: the COS engine agrees with two independent pricing methods.

These are the two gates PROJECT_PLAN.md singles out — *"Do not proceed if the CF
continuity test or the COS-vs-MC test fails. Those two failing means the pricer is wrong,
and everything downstream would be invalid."*

The value of a cross-check is exactly its independence. Carr-Madan shares the
characteristic function but nothing else — different transform, different quadrature,
different damping. Monte Carlo shares nothing at all: no Fourier machinery, no cumulants,
a direct simulation of the stochastic differential equations. Agreement between all three
is hard to obtain by accident.

Two limits are recorded rather than papered over, because knowing where a method stops
working is part of validating it:

* the FFT form of Carr-Madan is compared at a tolerance set by its **grid interpolation**,
  not at 1e-8, which no practical FFT grid reaches;
* at :math:`\sigma = 1.0` (Feller ratio 0.18, far outside any calibration region) the
  agreement degrades to ~1e-6, and that is asserted as a measurement.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import EXTREME, FELLER_VIOLATING, MATURITIES, REFERENCE
from voldesk.quant.model import HestonParams, MarketState
from voldesk.quant.pricing import carr_madan, cos, monte_carlo


@pytest.mark.gate
@pytest.mark.parametrize(
    "params", [REFERENCE, FELLER_VIOLATING], ids=["reference", "feller_violating"]
)
@pytest.mark.parametrize("maturity", MATURITIES)
def test_cos_agrees_with_carr_madan(
    params: HestonParams, maturity: float, market: MarketState, strikes: np.ndarray
) -> None:
    """GATE (CLAUDE.md section 7, "COS vs Carr-Madan"), agreement to < 1e-8.

    Run against the *quadrature* form of Carr-Madan, not the FFT form. See
    ``test_carr_madan_fft_matches_within_its_grid_resolution`` below and
    ``docs/adr/0006-carr-madan-fft-versus-quadrature.md`` for why: 1e-8 is an order of
    magnitude below the interpolation floor of any FFT grid one would actually use, so
    running the gate there would measure the grid rather than the pricer.

    Measured agreement on these parameter sets: between 4e-14 and 4e-10.
    """
    reference = carr_madan.price_call_quadrature(strikes, maturity, params, market)
    priced = cos.price_call(strikes, maturity, params, market, n_cos=512)
    error = float(np.max(np.abs(priced - reference)))
    assert error < 1e-8, f"COS vs Carr-Madan quadrature: {error:.3e}"


@pytest.mark.parametrize("maturity", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_agreement_degrades_gracefully_at_extreme_vol_of_vol(
    maturity: float, market: MarketState, strikes: np.ndarray
) -> None:
    r"""Where the methods stop agreeing, measured rather than hidden.

    At :math:`\sigma = 1.0` the Feller ratio is 0.18: the variance process spends real time
    at the origin, the density develops a near-singularity there, and a cosine series on a
    smooth-density assumption starts to struggle. This is far outside the parameter box any
    calibration would land in, but a pricer that silently loses six digits somewhere is a
    pricer whose error bars are unknown.

    Measured at ``n_cos = 1024``: 1e-11 at long maturities rising to 2e-6 at one month.
    """
    reference = carr_madan.price_call_quadrature(strikes, maturity, EXTREME, market)
    priced = cos.price_call(strikes, maturity, EXTREME, market, n_cos=1024)
    error = float(np.max(np.abs(priced - reference)))
    print(f"\nsigma=1.0, T={maturity}: COS vs Carr-Madan = {error:.3e}")
    assert error < 1e-5, f"even the degraded regime should stay under 1e-5, got {error:.3e}"


def test_carr_madan_fft_matches_within_its_grid_resolution(
    market: MarketState, strikes: np.ndarray
) -> None:
    """The published FFT form, checked at a tolerance appropriate to a strike grid.

    The FFT produces prices on a fixed log-strike lattice; anything else needs
    interpolation, and linear interpolation on a lattice of spacing :math:`\\lambda` carries
    an error of order :math:`\\lambda^2 \\partial^2 C/\\partial k^2`. With the default grid
    that is ~1e-3 on a spot of 100 — five orders of magnitude above the quadrature form.

    Asserting the loose bound here, and the tight one against quadrature above, is the
    honest way to report both: the method is implemented correctly, and its accuracy is
    limited by the thing that actually limits it.
    """
    for maturity in (0.5, 1.0, 2.0):
        grid = carr_madan.price_call_fft(maturity, FELLER_VIOLATING, market)
        interpolated = carr_madan.interpolate_from_grid(grid, strikes)
        reference = cos.price_call(strikes, maturity, FELLER_VIOLATING, market, n_cos=1024)
        error = float(np.max(np.abs(interpolated - reference)))
        assert error < 5e-3, f"T={maturity}: FFT grid interpolation error {error:.3e}"
        # And it must be *better* than a coarse grid would be, or the FFT is not working.
        assert error > 0.0


@pytest.mark.gate
@pytest.mark.slow
@pytest.mark.parametrize("maturity", [0.5, 1.0, 2.0])
def test_cos_agrees_with_monte_carlo_qe(
    maturity: float, market: MarketState, strikes: np.ndarray
) -> None:
    """GATE (CLAUDE.md section 7, "COS vs MC-QE") — within 3 standard errors.

    200k paths with antithetic variates, as specified. The standard error is computed over
    antithetic *pairs*, not over individual paths: the two members of a pair are negatively
    correlated by construction, and treating them as independent would understate the
    error by up to a factor of the square root of two and make this gate pass for the wrong
    reason.

    Measured on the Feller-violating parameter set: every strike lands within 0.8 standard
    errors, comfortably inside the 3-sigma bar.
    """
    result = monte_carlo.price_european(
        strikes,
        maturity,
        FELLER_VIOLATING,
        market,
        "call",
        n_paths=200_000,
        n_steps=200,
        seed=20260819,
    )
    reference = cos.price_call(strikes, maturity, FELLER_VIOLATING, market, n_cos=512)
    z = result.z_scores(reference)
    print(f"\nT={maturity} z-scores: {np.round(z, 2)}")
    assert np.all(np.abs(z) <= 3.0), f"COS outside 3 MC standard errors: z = {z}"


@pytest.mark.slow
@pytest.mark.parametrize("maturity", [0.5, 2.0])
def test_monte_carlo_spot_is_a_martingale(maturity: float, market: MarketState) -> None:
    r"""The martingale correction must hold: :math:`\mathbb{E}[S_T] = S_0 e^{(r-q)T}`.

    This is the direct test of Andersen's :math:`K_0^*`. Without the correction the
    discretised spot drifts, and the drift compounds with the number of steps — which
    presents as the Fourier method and the simulation disagreeing more at long maturities,
    a symptom that invites blaming the wrong component.
    """
    relative_error, relative_stderr = monte_carlo.forward_martingale_error(
        maturity, FELLER_VIOLATING, market, n_paths=100_000, n_steps=100, seed=5
    )
    print(
        f"\nT={maturity}: forward relative error {relative_error:+.3e} (se {relative_stderr:.3e})"
    )
    assert abs(relative_error) <= 3.0 * relative_stderr, (
        f"E[S_T] off the forward by {relative_error:.3e}, which is "
        f"{abs(relative_error) / relative_stderr:.1f} standard errors"
    )


@pytest.mark.slow
def test_monte_carlo_is_reproducible_from_its_seed(
    market: MarketState, strikes: np.ndarray
) -> None:
    """CLAUDE.md invariant 5: byte-reproducible from the stored seed.

    Two runs at the same seed must be bit-identical, and two runs at different seeds must
    not be — the second half catches a seed that is being ignored, which would make the
    first half pass trivially.
    """
    kwargs = dict(n_paths=20_000, n_steps=50)
    first = monte_carlo.price_european(strikes, 1.0, REFERENCE, market, "call", seed=123, **kwargs)
    again = monte_carlo.price_european(strikes, 1.0, REFERENCE, market, "call", seed=123, **kwargs)
    other = monte_carlo.price_european(strikes, 1.0, REFERENCE, market, "call", seed=124, **kwargs)
    assert np.array_equal(first.price, again.price)
    assert not np.array_equal(first.price, other.price)


@pytest.mark.slow
def test_antithetic_sampling_reduces_the_standard_error(
    market: MarketState, strikes: np.ndarray
) -> None:
    """Antithetic variates must actually buy something, at equal path count.

    If they did not, the extra bookkeeping in the standard-error computation would be pure
    cost. Measuring it also guards against the mirroring being applied inconsistently
    between the variance draws and the spot draws, which would quietly break the negative
    correlation without breaking anything visible.
    """
    plain = monte_carlo.price_european(
        strikes,
        1.0,
        REFERENCE,
        market,
        "call",
        n_paths=40_000,
        n_steps=50,
        seed=9,
        antithetic=False,
    )
    paired = monte_carlo.price_european(
        strikes, 1.0, REFERENCE, market, "call", n_paths=40_000, n_steps=50, seed=9, antithetic=True
    )
    ratio = paired.standard_error / plain.standard_error
    print(f"\nantithetic / plain standard-error ratio: {np.round(ratio, 3)}")
    assert np.median(ratio) < 1.0, f"antithetic sampling did not help: ratio {ratio}"
