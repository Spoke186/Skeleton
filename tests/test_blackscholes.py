r"""Black-Scholes pricing and the implied-volatility inversion.

The inversion is the piece that decides what the calibrator sees, so its *failure*
behaviour matters at least as much as its success behaviour. CLAUDE.md section 5.4 is
categorical: a price outside the no-arbitrage bounds returns ``NaN`` with a reject reason
and is never clamped. Most of the tests below are about that.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from voldesk.quant.model import MarketState
from voldesk.quant.pricing.blackscholes import (
    bs_price,
    bs_vega,
    implied_vol,
    implied_vol_surface,
    no_arbitrage_bounds,
    vol_resolution,
)
from voldesk.quant.reject_reasons import RejectReason


def test_put_call_parity_holds_for_black_scholes(market: MarketState, strikes: np.ndarray) -> None:
    r""":math:`C - P = S_0 e^{-qT} - K e^{-rT}`, the definition the rest of the code relies on."""
    for maturity in (0.1, 1.0, 5.0):
        call = bs_price(market.spot, strikes, maturity, market.rate, market.dividend, 0.25, "call")
        put = bs_price(market.spot, strikes, maturity, market.rate, market.dividend, 0.25, "put")
        expected = market.spot * math.exp(-market.dividend * maturity) - strikes * market.discount(
            maturity
        )
        assert np.max(np.abs((call - put) - expected)) < 1e-12


def test_vega_matches_a_finite_difference(market: MarketState, strikes: np.ndarray) -> None:
    """The analytic vega must equal the numerical derivative of the price."""
    vol, h = 0.25, 1e-6
    for maturity in (0.25, 1.0, 3.0):
        up = bs_price(market.spot, strikes, maturity, market.rate, market.dividend, vol + h, "call")
        down = bs_price(
            market.spot, strikes, maturity, market.rate, market.dividend, vol - h, "call"
        )
        numeric = (up - down) / (2 * h)
        analytic = bs_vega(market.spot, strikes, maturity, market.rate, market.dividend, vol)
        assert np.allclose(numeric, analytic, rtol=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_inversion_round_trips(option_type: str, market: MarketState, strikes: np.ndarray) -> None:
    """Price at a known volatility, invert, and recover it — to the resolution vega allows.

    The tolerance is not a constant. Where vega is large the volatility comes back to
    machine precision; where vega is small the *price* is flat in volatility and no
    inversion can do better than :func:`vol_resolution` says. Asserting a fixed 1e-8
    everywhere would be asserting that float64 has more precision than it has, so the bar
    is ``max(1e-8, 100 * vol_resolution)`` and the price round-trip is checked separately
    and tightly.
    """
    for vol in (0.05, 0.15, 0.35, 0.80, 2.0):
        for maturity in (0.05, 0.5, 2.0):
            prices = bs_price(
                market.spot,
                strikes,
                maturity,
                market.rate,
                market.dividend,
                vol,
                option_type,  # type: ignore[arg-type]
            )
            for price, strike in zip(prices, strikes, strict=True):
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
                    assert result.reject_reason is RejectReason.IV_NOT_BRACKETED
                    continue

                tolerance = max(
                    1e-8,
                    100.0
                    * vol_resolution(
                        market.spot, float(strike), maturity, market.rate, market.dividend, vol
                    ),
                )
                assert abs(result.vol - vol) < tolerance, (
                    f"vol={vol} K={strike} T={maturity}: recovered {result.vol}, "
                    f"tolerance {tolerance:.3e}"
                )

                # Whatever the volatility resolution, the price must round-trip exactly.
                reprice = float(
                    bs_price(
                        market.spot,
                        float(strike),
                        maturity,
                        market.rate,
                        market.dividend,
                        result.vol,
                        option_type,  # type: ignore[arg-type]
                    )
                )
                assert abs(reprice - float(price)) < 1e-10


def test_vol_resolution_explains_where_inversion_loses_precision() -> None:
    """The vega-limited resolution, stated as a measurement.

    A deep in-the-money short-dated call has a vega around 1e-9 on a price around 25, so
    two volatilities 1e-6 apart give the *same* float64 price. This is the mechanical
    reason CLAUDE.md section 5.5 fits in volatility space with spread-inverse weights:
    such a quote carries almost no information about volatility, yet a price-space
    objective would hand it the largest weight in the sum.
    """
    spot, rate, dividend = 100.0, 0.02, 0.01

    deep_itm = vol_resolution(spot, 75.0, 0.125, rate, dividend, 0.125)
    at_the_money = vol_resolution(spot, 100.0, 1.0, rate, dividend, 0.30)

    print(f"\nvol resolution: deep ITM short-dated {deep_itm:.3e}, ATM one-year {at_the_money:.3e}")
    assert deep_itm > 1e-8, "deep ITM should be resolution-limited"
    assert at_the_money < 1e-15, "ATM should be resolution-limited only by float64 itself"
    assert deep_itm > 1e5 * at_the_money


def test_price_below_intrinsic_is_rejected_not_clamped(market: MarketState) -> None:
    """CLAUDE.md section 5.4: no volatility exists, so return NaN and say why.

    The temptation is to return the lower bound's volatility of zero, or to clamp to
    ``IV_LOWER``. Both would place an arbitrageable quote onto the boundary of the feasible
    set, where it is indistinguishable from a legitimate deep-in-the-money quote and would
    be fitted with full weight.
    """
    lower, _ = no_arbitrage_bounds(market.spot, 80.0, 1.0, market.rate, market.dividend, "call")
    result = implied_vol(lower - 1.0, market.spot, 80.0, 1.0, market.rate, market.dividend, "call")
    assert math.isnan(result.vol)
    assert result.reject_reason is RejectReason.IV_NO_SOLUTION
    assert not result.ok


def test_price_above_upper_bound_is_rejected(market: MarketState) -> None:
    """A call worth more than the discounted forward is an arbitrage, not a high volatility."""
    _, upper = no_arbitrage_bounds(market.spot, 100.0, 1.0, market.rate, market.dividend, "call")
    result = implied_vol(upper + 1.0, market.spot, 100.0, 1.0, market.rate, market.dividend, "call")
    assert math.isnan(result.vol)
    assert result.reject_reason is RejectReason.IV_NO_SOLUTION


def test_price_inside_bounds_but_outside_the_bracket_gets_its_own_reason(
    market: MarketState,
) -> None:
    """Not every failure is an arbitrage, and the two must not be conflated.

    A price sitting essentially on the intrinsic boundary is *inside* the no-arbitrage
    bounds — it is a perfectly legal quote — but its implied volatility is below the Brent
    bracket. That is a numerical limit of the search, not a defect in the market, and the
    support console has to be able to tell those apart when explaining a rejection.
    """
    lower, _ = no_arbitrage_bounds(market.spot, 60.0, 1.0, market.rate, market.dividend, "call")
    result = implied_vol(lower, market.spot, 60.0, 1.0, market.rate, market.dividend, "call")
    assert math.isnan(result.vol), (
        f"returned vol={result.vol}, which is the bottom of the search bracket dressed up "
        "as an answer — exactly the silent clamp CLAUDE.md section 5.4 forbids"
    )
    assert result.reject_reason is RejectReason.IV_NOT_BRACKETED


def test_expired_and_non_finite_inputs_get_their_own_reasons(market: MarketState) -> None:
    """Structural problems are named as such rather than surfacing as an arbitrage."""
    expired = implied_vol(5.0, market.spot, 100.0, 0.0, market.rate, market.dividend, "call")
    assert expired.reject_reason is RejectReason.EXPIRED

    missing = implied_vol(math.nan, market.spot, 100.0, 1.0, market.rate, market.dividend, "call")
    assert missing.reject_reason is RejectReason.MISSING_FIELD

    negative = implied_vol(-1.0, market.spot, 100.0, 1.0, market.rate, market.dividend, "call")
    assert negative.reject_reason is RejectReason.NEGATIVE_PRICE


def test_surface_inversion_returns_a_reason_for_every_quote(
    market: MarketState, strikes: np.ndarray
) -> None:
    """Invariant 3 at the API level: the reasons come back alongside the volatilities.

    A signature that returned only the volatilities would make silent dropping the path of
    least resistance for every caller.
    """
    prices = bs_price(market.spot, strikes, 1.0, market.rate, market.dividend, 0.3, "call")
    prices[0] = -1.0  # one deliberately impossible quote
    vols, reasons = implied_vol_surface(
        prices, market.spot, strikes, 1.0, market.rate, market.dividend, "call"
    )
    assert len(reasons) == len(strikes)
    assert math.isnan(vols[0]) and reasons[0] is RejectReason.NEGATIVE_PRICE
    assert all(r is None for r in reasons[1:])
    assert np.all(np.isfinite(vols[1:]))


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    vol=st.floats(min_value=0.02, max_value=2.5),
    moneyness=st.floats(min_value=0.6, max_value=1.6),
    maturity=st.floats(min_value=0.02, max_value=5.0),
)
def test_inversion_round_trips_on_arbitrary_inputs(
    vol: float, moneyness: float, maturity: float
) -> None:
    """Property test: whatever the inputs, price-then-invert returns the input volatility.

    Hypothesis explores the corners a hand-written parameter list does not think of — very
    short maturities, very high volatilities, deep wings — which is where an inversion
    that merely looks correct tends to break.
    """
    spot, rate, dividend = 100.0, 0.02, 0.01
    strike = spot * moneyness
    price = float(bs_price(spot, strike, maturity, rate, dividend, vol, "call"))
    result = implied_vol(price, spot, strike, maturity, rate, dividend, "call")

    if not result.ok:
        assert result.reject_reason is RejectReason.IV_NOT_BRACKETED
        return
    tolerance = max(1e-6, 100.0 * vol_resolution(spot, strike, maturity, rate, dividend, vol))
    assert abs(result.vol - vol) < tolerance


def test_no_arbitrage_bounds_are_the_textbook_ones(market: MarketState) -> None:
    """Guard the bounds themselves; every rejection decision rests on them."""
    maturity = 1.5
    df_r, df_q = market.discount(maturity), math.exp(-market.dividend * maturity)
    for strike in (60.0, 100.0, 150.0):
        call_low, call_high = no_arbitrage_bounds(
            market.spot, strike, maturity, market.rate, market.dividend, "call"
        )
        assert call_low == pytest.approx(max(market.spot * df_q - strike * df_r, 0.0))
        assert call_high == pytest.approx(market.spot * df_q)

        put_low, put_high = no_arbitrage_bounds(
            market.spot, strike, maturity, market.rate, market.dividend, "put"
        )
        assert put_low == pytest.approx(max(strike * df_r - market.spot * df_q, 0.0))
        assert put_high == pytest.approx(strike * df_r)
