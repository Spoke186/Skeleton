"""The synthetic generator: does it produce data that is hard in the right ways?

CLAUDE.md section 4 names five microstructure defects the generator must reproduce. Each
gets a test here, because a generator that quietly stopped injecting one of them would
make the quality layer look better than it is, and would make the calibration results
optimistic in a way nothing downstream could detect.

The ground-truth separation of invariant 6 is tested too: the calibration view must not
carry the answer.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from voldesk.quant.model import MarketState
from voldesk.quant.synthetic.generator import (
    REGIMES,
    MicrostructureConfig,
    SurfaceGrid,
    default_grid,
    generate_surface,
)


@pytest.fixture
def surface(market: MarketState):
    """A stressed-regime surface with the default noise model."""
    return generate_surface(REGIMES["stressed"], market, seed=11)


def test_same_seed_gives_byte_identical_output(market: MarketState) -> None:
    """CLAUDE.md invariant 5: every experiment is byte-reproducible from its stored config."""
    first = generate_surface(REGIMES["calm"], market, seed=7)
    again = generate_surface(REGIMES["calm"], market, seed=7)
    assert first.quotes.equals(again.quotes)
    assert first.seed == again.seed == 7


def test_different_seeds_give_different_output(market: MarketState) -> None:
    """The control for the test above: a seed that is ignored would pass reproducibility."""
    assert not generate_surface(REGIMES["calm"], market, seed=7).quotes.equals(
        generate_surface(REGIMES["calm"], market, seed=8).quotes
    )


def test_calibration_view_hides_the_ground_truth(surface) -> None:
    """CLAUDE.md invariant 6, enforced at the API rather than by discipline.

    A calibrator handed the full frame could read ``true_iv`` and score beautifully. The
    separation has to be structural, or it will eventually be violated by accident.
    """
    view = surface.calibration_view()
    for forbidden in ("true_price", "true_iv", "is_stale", "is_crossed"):
        assert forbidden not in view.columns
    assert "bid" in view.columns and "ask" in view.columns


def test_truth_view_carries_the_ground_truth(surface) -> None:
    """The evaluator's counterpart to the calibration view."""
    truth = surface.truth_view()
    assert "true_iv" in truth.columns
    assert np.all(np.isfinite(truth["true_price"].to_numpy()))


# --- the five microstructure defects of CLAUDE.md section 4 ----------------------------


def test_spreads_widen_with_moneyness(surface, market: MarketState) -> None:
    """Defect 1a: the wings are quoted wider than the money."""
    quotes = surface.quotes.filter(
        (pl.col("maturity") > 0.4) & (pl.col("bid") > 0) & (pl.col("mid") > 0.5)
    ).with_columns(
        relative_spread=(pl.col("ask") - pl.col("bid")) / pl.col("mid"),
        abs_log_moneyness=(pl.col("strike") / market.spot).log().abs(),
    )
    near = quotes.filter(pl.col("abs_log_moneyness") < 0.1)["relative_spread"].mean()
    far = quotes.filter(pl.col("abs_log_moneyness") > 0.2)["relative_spread"].mean()
    assert far > near, f"wing spread {far:.4f} not wider than at-the-money {near:.4f}"


def test_spreads_widen_as_maturity_shortens(surface) -> None:
    """Defect 1b: the front month is quoted widest in relative terms."""
    quotes = surface.quotes.filter((pl.col("bid") > 0) & (pl.col("mid") > 0.5)).with_columns(
        relative_spread=(pl.col("ask") - pl.col("bid")) / pl.col("mid")
    )
    short = quotes.filter(pl.col("maturity") < 0.2)["relative_spread"].mean()
    long = quotes.filter(pl.col("maturity") > 0.9)["relative_spread"].mean()
    assert short > long, f"front-month spread {short:.4f} not wider than back {long:.4f}"


def test_quotes_are_discretised_to_the_tick(surface) -> None:
    """Defect 2: prices land on the tick lattice, which floors their information content."""
    tick = surface.config.tick
    for column in ("bid", "ask"):
        values = surface.quotes[column].to_numpy()
        remainder = np.abs(values / tick - np.round(values / tick))
        assert np.max(remainder) < 1e-9, f"{column} is not on the {tick} tick grid"


def test_some_quotes_are_stale(surface) -> None:
    """Defect 3: a fraction is priced off a lagged spot, so it looks locally arbitraged."""
    stale = surface.quotes.filter(pl.col("is_stale"))
    assert stale.height > 0
    # A stale quote was priced at a different spot from the current one.
    assert not np.allclose(stale["spot_observed"].to_numpy(), surface.market.spot)


def test_deep_out_of_the_money_options_have_no_bid(surface) -> None:
    """Defect 4: there is no two-sided market for an option worth a fraction of a tick."""
    zero_bid = surface.quotes.filter(pl.col("bid") <= 0.0)
    assert zero_bid.height > 0
    threshold = surface.config.zero_bid_price_threshold
    assert np.all(zero_bid["true_price"].to_numpy() < max(threshold, surface.config.tick))


def test_some_quotes_are_crossed(surface) -> None:
    """Defect 5: bid above ask, which must reach the quality layer rather than a mid."""
    crossed = surface.quotes.filter(pl.col("is_crossed") & (pl.col("bid") > 0))
    assert crossed.height > 0
    assert np.all(crossed["bid"].to_numpy() > crossed["ask"].to_numpy())


# --- the clean surface, used by the recovery gate --------------------------------------


def test_clean_surface_has_no_defects_at_all(market: MarketState) -> None:
    """The noiseless-recovery gate needs data with nothing wrong with it.

    Otherwise a failure to recover the parameters to 1e-3 could be blamed on the noise
    model rather than on the calibrator, which is the only thing that gate is about.
    """
    clean = generate_surface(REGIMES["stressed"], market, seed=3, clean=True)
    summary = clean.summary()
    assert summary["n_stale"] == 0
    assert summary["n_crossed"] == 0
    assert summary["n_zero_bid"] == 0
    assert np.allclose(clean.quotes["mid"].to_numpy(), clean.quotes["true_price"].to_numpy())
    assert np.allclose(clean.quotes["bid"].to_numpy(), clean.quotes["ask"].to_numpy())


def test_clean_surface_prices_are_arbitrage_free_in_strike(market: MarketState) -> None:
    """Call prices must be non-increasing in strike on every clean slice.

    The tolerance is not zero, and the number is the COS engine's absolute accuracy in the
    deep wing rather than slack. Out past 30% moneyness on a short-dated steep-skew slice
    the true call price is below 1e-6, and both the cosine series *and* the Carr-Madan
    quadrature return values that wobble around zero at the 1e-6 level, occasionally
    slightly negative. That is an unavoidable property of a truncated Fourier method at the
    edge of the density's support. It sits five orders of magnitude below one tick, so no
    such value ever becomes a quote — but it is measured here rather than hidden behind a
    clamp inside the pricer, which would make the wing look better than it is.

    The second assertion is the one with teeth: wherever the price is large enough to
    carry information, monotonicity is strict.
    """
    clean = generate_surface(REGIMES["steep_skew"], market, seed=3, clean=True)
    for maturity in clean.quotes["maturity"].unique():
        slice_ = clean.quotes.filter(pl.col("maturity") == maturity).sort("strike")
        prices = slice_["true_price"].to_numpy()
        assert np.all(np.diff(prices) <= 1e-5)
        # And strictly monotone wherever the prices are large enough to mean anything.
        meaningful = prices > 1e-4
        assert np.all(np.diff(prices[meaningful]) < 0.0)


def test_true_implied_vols_show_a_skew(market: MarketState) -> None:
    """A negative correlation must produce a downward-sloping smile.

    If it did not, either the sign of rho or the moneyness convention would be wrong, and
    every conclusion about the wings would be reversed while still looking reasonable.
    """
    clean = generate_surface(REGIMES["steep_skew"], market, seed=1, clean=True)
    slice_ = (
        clean.quotes.filter(pl.col("maturity") > 0.4)
        .sort("strike")
        .filter(pl.col("true_iv").is_not_nan())
    )
    vols = slice_["true_iv"].to_numpy()
    assert vols[0] > vols[-1], "no skew: low strikes should carry higher implied volatility"


def test_regimes_are_distinguishable(market: MarketState) -> None:
    """Calm, stressed and steep-skew must actually differ, since figure 2 compares them."""
    levels = {}
    for name, params in REGIMES.items():
        clean = generate_surface(params, market, seed=1, clean=True)
        atm = clean.quotes.filter(
            (pl.col("maturity") > 0.4) & ((pl.col("strike") - market.spot).abs() < 4.0)
        )
        levels[name] = float(atm["true_iv"].mean())
    assert levels["stressed"] > levels["calm"]
    assert len(set(round(v, 3) for v in levels.values())) == len(REGIMES)


def test_a_custom_grid_is_respected(market: MarketState) -> None:
    grid = SurfaceGrid(maturities=np.array([0.5, 1.0]), moneyness=np.linspace(0.9, 1.1, 5))
    surface = generate_surface(REGIMES["calm"], market, grid=grid, seed=2)
    assert surface.quotes.height == grid.n_quotes == 10
    assert sorted(surface.quotes["maturity"].unique().to_list()) == [0.5, 1.0]


def test_noise_level_scales_the_dispersion(market: MarketState) -> None:
    """Turning the noise knob must move the data, or the sweeps in E1 measure nothing."""
    quiet = generate_surface(
        REGIMES["calm"], market, seed=4, config=MicrostructureConfig(noise_vol_bps=5.0)
    )
    loud = generate_surface(
        REGIMES["calm"], market, seed=4, config=MicrostructureConfig(noise_vol_bps=200.0)
    )
    quiet_dev = np.abs(quiet.quotes["mid"].to_numpy() - quiet.quotes["true_price"].to_numpy())
    loud_dev = np.abs(loud.quotes["mid"].to_numpy() - loud.quotes["true_price"].to_numpy())
    assert loud_dev.sum() > 3.0 * quiet_dev.sum()


def test_default_grid_is_a_plausible_listed_chain() -> None:
    grid = default_grid()
    assert grid.n_quotes == 126
    assert grid.maturities[0] < 0.1 < grid.maturities[-1]
    assert grid.moneyness[0] < 0.7 and grid.moneyness[-1] > 1.3
