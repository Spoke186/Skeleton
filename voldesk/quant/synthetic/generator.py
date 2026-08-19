r"""Synthetic option surfaces with a realistic microstructure noise model.

CLAUDE.md section 4 makes synthetic data the *primary* source, and says why: with real
market data there is no ground truth, so parameter recovery cannot be measured — only fit
residual, which is misleading in an ill-conditioned problem. The whole scientific claim of
this project is about recovery, so the data has to be generated from parameters that are
known and then hidden from the calibrator.

That only works if the synthetic data is hard in the ways real data is hard. A clean
arbitrage-free grid would make the quality layer decorative and the calibration
unrealistically easy. So the generator reproduces the five defects named in section 4:

1. **bid-ask spreads that widen** with moneyness and with shortening maturity — the wings
   and the front month are where liquidity actually thins;
2. **tick-size discretisation** of the quoted prices, which puts a hard floor under the
   information content of a quote and matters enormously for cheap options;
3. **stale quotes**, priced off a lagged spot, which look like a local arbitrage but are
   really a timestamp problem;
4. **zero-bid deep out-of-the-money options**, which have no two-sided market at all;
5. **occasional crossed quotes** (bid above ask), which exist in real feeds and must be
   caught rather than averaged into a mid.

Ground truth travels with the surface in :attr:`SyntheticSurface.true_params`, and
CLAUDE.md invariant 6 forbids showing it to the calibrator. The evaluator sees it; the
objective function does not. :meth:`SyntheticSurface.calibration_view` returns exactly
the columns a calibrator is allowed to look at, so the separation is enforced by the API
rather than by discipline.
"""

from __future__ import annotations

import dataclasses
from typing import Final

import numpy as np
import polars as pl
from numpy.typing import NDArray

from voldesk.quant.model import HestonParams, MarketState
from voldesk.quant.pricing import cos
from voldesk.quant.pricing.blackscholes import implied_vol

#: Smallest quotable price increment. US equity options quote in $0.01 below $3 and $0.05
#: above; the single value here is the coarser of the two, which is the conservative
#: choice for a study of how much information a quote carries.
DEFAULT_TICK: Final[float] = 0.05


@dataclasses.dataclass(frozen=True, slots=True)
class MicrostructureConfig:
    """Knobs of the noise model. Every default is stated, none is hidden in the code.

    Parameters
    ----------
    base_spread_frac
        Relative half-spread at the money, at the longest maturity.
    moneyness_spread_slope
        How much the relative half-spread grows per unit of squared log-moneyness. Squared,
        not linear, because the widening is symmetric about the forward.
    maturity_spread_power
        The half-spread scales like ``maturity ** (-maturity_spread_power)``: short-dated
        options are quoted wider in relative terms.
    tick
        Price increment to round quotes to.
    stale_fraction
        Fraction of quotes priced off a lagged spot.
    stale_spot_lag_frac
        Relative size of the spot move between the lagged snapshot and the current one.
    zero_bid_price_threshold
        Quotes whose true price falls below this lose their bid entirely.
    crossed_fraction
        Fraction of quotes deliberately crossed (bid pushed above ask).
    noise_vol_bps
        Standard deviation of the idiosyncratic quote noise, in volatility basis points.
        Applied in volatility space, because that is where the calibration lives, and
        applying it in price space would silently weight the wings differently.
    """

    base_spread_frac: float = 0.01
    moneyness_spread_slope: float = 0.35
    maturity_spread_power: float = 0.35
    tick: float = DEFAULT_TICK
    stale_fraction: float = 0.03
    stale_spot_lag_frac: float = 0.004
    zero_bid_price_threshold: float = 0.10
    crossed_fraction: float = 0.005
    noise_vol_bps: float = 25.0


@dataclasses.dataclass(frozen=True, slots=True)
class SurfaceGrid:
    """The (maturity, strike) grid a surface is generated on."""

    maturities: NDArray[np.float64]
    moneyness: NDArray[np.float64]
    """Strike over spot, so the same grid is meaningful at any spot level."""

    def strikes(self, spot: float) -> NDArray[np.float64]:
        """Absolute strikes for this grid at a given spot."""
        return self.moneyness * spot

    @property
    def n_quotes(self) -> int:
        """Total number of grid points."""
        return len(self.maturities) * len(self.moneyness)


def default_grid(n_maturities: int = 6, n_strikes: int = 21) -> SurfaceGrid:
    """A representative listed-option grid: six expiries out to two years, wings to +-35%."""
    maturities = np.array([1.0 / 12, 2.0 / 12, 3.0 / 12, 6.0 / 12, 1.0, 2.0])[:n_maturities]
    moneyness = np.linspace(0.65, 1.35, n_strikes)
    return SurfaceGrid(maturities=maturities, moneyness=moneyness)


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticSurface:
    """A generated surface together with the parameters that produced it.

    Attributes
    ----------
    quotes
        One row per quoted option. Contains both the observable columns and the
        ground-truth ones; see :meth:`calibration_view`.
    true_params
        The parameters the surface was generated from. **Never** to be shown to a
        calibrator (CLAUDE.md invariant 6).
    market
        Spot, rate and dividend used for generation.
    seed
        The seed the noise was drawn with; persisted so the surface is byte-reproducible.
    """

    quotes: pl.DataFrame
    true_params: HestonParams
    market: MarketState
    config: MicrostructureConfig
    seed: int

    #: Columns a calibrator is permitted to see.
    OBSERVABLE_COLUMNS: Final[tuple[str, ...]] = (
        "maturity",
        "strike",
        "option_type",
        "bid",
        "ask",
        "mid",
        "spot_observed",
        "rate",
        "dividend",
    )

    def calibration_view(self) -> pl.DataFrame:
        """The observable columns only.

        Enforces CLAUDE.md invariant 6 at the API boundary. A calibrator handed the full
        frame could read ``true_iv`` and score suspiciously well; a calibrator handed this
        cannot, whatever anyone intended.
        """
        return self.quotes.select(self.OBSERVABLE_COLUMNS)

    def truth_view(self) -> pl.DataFrame:
        """The ground-truth columns, for the evaluator only."""
        return self.quotes.select(
            ["maturity", "strike", "option_type", "true_price", "true_iv", "is_stale", "is_crossed"]
        )

    def summary(self) -> dict[str, float | int]:
        """Counts of each injected defect, for the experiment log and the README."""
        q = self.quotes
        return {
            "n_quotes": q.height,
            "n_stale": int(q["is_stale"].sum()),
            "n_crossed": int(q["is_crossed"].sum()),
            "n_zero_bid": int((q["bid"] <= 0.0).sum()),
            "n_maturities": q["maturity"].n_unique(),
            "median_relative_spread": float(
                ((q["ask"] - q["bid"]) / q["mid"].clip(lower_bound=1e-8)).median() or 0.0  # type: ignore[arg-type]
            ),
        }


def _relative_half_spread(
    log_moneyness: NDArray[np.float64],
    maturity: float,
    config: MicrostructureConfig,
) -> NDArray[np.float64]:
    r"""Relative half-spread as a function of moneyness and maturity.

    .. math::
        \frac{\text{half-spread}}{\text{mid}} =
            s_0 \big(1 + \beta\, m^2\big)\, T^{-p}

    Quadratic in log-moneyness so the widening is symmetric about the forward, and a
    negative power of maturity so the front month is quoted widest. Both effects are
    calibrated to be of the right order rather than fitted to any particular market —
    the point is that the calibration has to cope with heteroskedastic quote quality, not
    that the heteroskedasticity matches QQQ on some date.
    """
    return (
        config.base_spread_frac
        * (1.0 + config.moneyness_spread_slope * log_moneyness**2)
        * maturity ** (-config.maturity_spread_power)
    )


def generate_surface(
    params: HestonParams,
    market: MarketState,
    *,
    grid: SurfaceGrid | None = None,
    config: MicrostructureConfig | None = None,
    seed: int = 0,
    option_type: str = "call",
    clean: bool = False,
) -> SyntheticSurface:
    """Generate one option surface from known parameters.

    Parameters
    ----------
    clean
        If true, produce arbitrage-free mid prices with no noise and no defects. Used by
        the noiseless-recovery gate of CLAUDE.md section 7, which must recover all five
        parameters to better than 1e-3 relative — a test of the calibrator, not of the
        noise model.
    seed
        All randomness is drawn from this one seed and it is stored on the result
        (CLAUDE.md invariant 5).

    Returns
    -------
    SyntheticSurface
        Quotes plus the ground truth that produced them.
    """
    grid = grid or default_grid()
    config = config or MicrostructureConfig()
    rng = np.random.default_rng(seed)

    rows: list[dict[str, object]] = []

    for maturity in grid.maturities:
        strikes = grid.strikes(market.spot)
        true_prices = cos.price(strikes, float(maturity), params, market, option_type)  # type: ignore[arg-type]

        log_moneyness = np.log(strikes / market.forward(float(maturity)))

        if clean:
            observed_prices = true_prices.copy()
            half_spread = np.zeros_like(true_prices)
            stale_mask = np.zeros(len(strikes), dtype=bool)
            crossed_mask = np.zeros(len(strikes), dtype=bool)
            spot_observed = np.full(len(strikes), market.spot)
        else:
            # --- stale quotes: priced off a spot that has since moved ----------------
            stale_mask = rng.random(len(strikes)) < config.stale_fraction
            spot_shift = rng.normal(0.0, config.stale_spot_lag_frac, size=len(strikes))
            spot_observed = np.where(stale_mask, market.spot * (1.0 + spot_shift), market.spot)

            stale_prices = true_prices.copy()
            if stale_mask.any():
                lagged_market = MarketState(
                    spot=float(market.spot), rate=market.rate, dividend=market.dividend
                )
                for index in np.flatnonzero(stale_mask):
                    lagged = MarketState(
                        spot=float(spot_observed[index]),
                        rate=lagged_market.rate,
                        dividend=lagged_market.dividend,
                    )
                    stale_prices[index] = float(
                        cos.price(
                            np.array([strikes[index]]),
                            float(maturity),
                            params,
                            lagged,
                            option_type,  # type: ignore[arg-type]
                        )[0]
                    )

            # --- idiosyncratic noise, applied in volatility space --------------------
            observed_prices = _perturb_in_vol_space(
                stale_prices,
                strikes,
                float(maturity),
                market,
                option_type,
                config.noise_vol_bps,
                rng,
            )

            half_spread = observed_prices * _relative_half_spread(
                log_moneyness, float(maturity), config
            )
            crossed_mask = rng.random(len(strikes)) < config.crossed_fraction

        bid = observed_prices - half_spread
        ask = observed_prices + half_spread

        if not clean:
            # Order matters here, and it is the order a real feed produces.
            #
            # 1. Tick discretisation first, because everything a venue publishes is on the
            #    lattice. Crossing before rounding lets the rounding undo the cross when
            #    the half-spread is finer than a tick, which produced "crossed" quotes with
            #    bid == ask and a quality layer with nothing to catch.
            bid = np.round(bid / config.tick) * config.tick
            ask = np.round(ask / config.tick) * config.tick

            # 2. Zero-bid deep out-of-the-money: an option worth a fraction of a tick has
            #    no two-sided market at all.
            bid = np.where(true_prices < config.zero_bid_price_threshold, 0.0, bid)
            bid = np.maximum(bid, 0.0)
            ask = np.maximum(ask, 0.0)

            # 3. Cross the book, by at least one whole tick, and only where a bid exists.
            #    A crossed quote on an option with no bid is not a thing.
            has_bid = bid > 0.0
            crossed_mask = crossed_mask & has_bid
            cross_ticks = np.maximum(np.round(2.5 * half_spread / config.tick), 1.0) * config.tick
            bid = np.where(crossed_mask, ask + cross_ticks, bid)

        mid = 0.5 * (bid + ask)

        true_iv = np.array(
            [
                implied_vol(
                    float(p),
                    market.spot,
                    float(k),
                    float(maturity),
                    market.rate,
                    market.dividend,
                    option_type,  # type: ignore[arg-type]
                ).vol
                for p, k in zip(true_prices, strikes, strict=True)
            ]
        )

        for i, strike in enumerate(strikes.tolist()):
            rows.append(
                {
                    "maturity": float(maturity),
                    "strike": float(strike),
                    "option_type": option_type,
                    "bid": float(bid[i]),
                    "ask": float(ask[i]),
                    "mid": float(mid[i]),
                    "spot_observed": float(spot_observed[i]),
                    "rate": market.rate,
                    "dividend": market.dividend,
                    "true_price": float(true_prices[i]),
                    "true_iv": float(true_iv[i]),
                    "is_stale": bool(stale_mask[i]),
                    "is_crossed": bool(crossed_mask[i]),
                }
            )

    return SyntheticSurface(
        quotes=pl.DataFrame(rows),
        true_params=params,
        market=market,
        config=config,
        seed=seed,
    )


def _perturb_in_vol_space(
    prices: NDArray[np.float64],
    strikes: NDArray[np.float64],
    maturity: float,
    market: MarketState,
    option_type: str,
    noise_vol_bps: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    r"""Add noise of a given size *in volatility units*, then convert back to price.

    Noise added directly to prices would be enormous in implied-volatility terms for cheap
    wing options and negligible for expensive ones — the opposite of how quote uncertainty
    actually behaves, and it would bias every conclusion about the wings, which is where
    the identifiability question lives.
    """
    from voldesk.quant.pricing.blackscholes import bs_price

    shocks = rng.normal(0.0, noise_vol_bps * 1e-4, size=len(prices))
    out = prices.copy()
    for i, (price, strike) in enumerate(zip(prices, strikes, strict=True)):
        result = implied_vol(
            float(price),
            market.spot,
            float(strike),
            maturity,
            market.rate,
            market.dividend,
            option_type,  # type: ignore[arg-type]
        )
        if not result.ok:
            continue
        bumped = max(result.vol + shocks[i], 1e-6)
        out[i] = float(
            bs_price(
                market.spot,
                float(strike),
                maturity,
                market.rate,
                market.dividend,
                bumped,
                option_type,  # type: ignore[arg-type]
            )
        )
    return out


#: Named parameter sets used across the experiments, so that "calm" and "stressed" mean the
#: same thing in every figure and every table. Referenced by
#: ``experiments/e2_sloppiness.py``.
REGIMES: Final[dict[str, HestonParams]] = {
    "calm": HestonParams(v0=0.020, kappa=2.50, theta=0.025, sigma=0.30, rho=-0.55),
    "stressed": HestonParams(v0=0.090, kappa=1.20, theta=0.070, sigma=0.75, rho=-0.75),
    "steep_skew": HestonParams(v0=0.040, kappa=1.00, theta=0.045, sigma=0.60, rho=-0.90),
}
