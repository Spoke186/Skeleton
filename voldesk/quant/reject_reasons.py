"""Reasons a quote or a derived quantity may be discarded.

CLAUDE.md invariant 3: every filtered or rejected quote records *why*. Silent dropping
is forbidden. This enum is the vocabulary for that, and it lives in ``quant`` — with no
dependency beyond the standard library — so that the numerical layer can name a rejection
without importing the web layer.

The Django ``RejectedQuote.reject_reason`` column takes its choices from here, so the
database and the numerics cannot drift apart.
"""

from __future__ import annotations

import enum


class RejectReason(enum.StrEnum):
    """Why a quote did not reach the calibrator."""

    # --- quote-level microstructure defects (voldesk/quality/quotes.py) ---
    CROSSED_QUOTE = "crossed_quote"
    """Bid strictly above ask. The book is inconsistent; the mid is meaningless."""

    ZERO_BID = "zero_bid"
    """Bid is zero — typically deep out-of-the-money. There is no two-sided market."""

    NEGATIVE_PRICE = "negative_price"
    """Bid, ask or mid is negative."""

    WIDE_SPREAD = "wide_spread"
    """Relative bid-ask spread exceeds the configured tolerance; the mid carries little
    information and would be given spurious weight by the spread-inverse weighting."""

    STALE_QUOTE = "stale_quote"
    """Quote timestamp lags the spot snapshot by more than the tolerance, so it is priced
    off a spot that no longer holds."""

    ZERO_VOLUME = "zero_volume"
    """No trading interest behind the quote."""

    # --- no-arbitrage and static-shape violations (voldesk/quality/arbitrage.py) ---
    BELOW_INTRINSIC = "below_intrinsic"
    """Price below the option's intrinsic value — an immediate arbitrage."""

    ABOVE_UPPER_BOUND = "above_upper_bound"
    """Price above the model-free upper bound (discounted forward for a call, discounted
    strike for a put)."""

    CALL_MONOTONICITY = "call_monotonicity"
    """Call price not non-increasing in strike."""

    PUT_MONOTONICITY = "put_monotonicity"
    """Put price not non-decreasing in strike."""

    BUTTERFLY_ARBITRAGE = "butterfly_arbitrage"
    """Negative butterfly spread: the implied risk-neutral density would be negative."""

    CALENDAR_ARBITRAGE = "calendar_arbitrage"
    """Total implied variance not non-decreasing in maturity at fixed log-moneyness."""

    PARITY_VIOLATION = "parity_violation"
    """Put-call parity breached beyond the tolerance implied by the spreads."""

    # --- inversion failures (voldesk/quant/pricing/blackscholes.py) ---
    IV_NO_SOLUTION = "iv_no_solution"
    """The price lies outside the no-arbitrage bounds, so no Black-Scholes volatility
    reproduces it. CLAUDE.md section 5.4: return NaN and record this — never clamp."""

    IV_NOT_BRACKETED = "iv_not_bracketed"
    """The price is inside the bounds but outside the search interval [1e-6, 5.0]."""

    # --- structural ---
    MISSING_FIELD = "missing_field"
    """A field required to use the quote is absent or not finite."""

    EXPIRED = "expired"
    """Time to maturity is zero or negative."""


#: Reasons that indicate a defective quote rather than a defective market.
QUOTE_LEVEL_REASONS = frozenset(
    {
        RejectReason.CROSSED_QUOTE,
        RejectReason.ZERO_BID,
        RejectReason.NEGATIVE_PRICE,
        RejectReason.WIDE_SPREAD,
        RejectReason.STALE_QUOTE,
        RejectReason.ZERO_VOLUME,
        RejectReason.MISSING_FIELD,
        RejectReason.EXPIRED,
    }
)

#: Reasons that indicate a static-arbitrage violation in the surface.
ARBITRAGE_REASONS = frozenset(
    {
        RejectReason.BELOW_INTRINSIC,
        RejectReason.ABOVE_UPPER_BOUND,
        RejectReason.CALL_MONOTONICITY,
        RejectReason.PUT_MONOTONICITY,
        RejectReason.BUTTERFLY_ARBITRAGE,
        RejectReason.CALENDAR_ARBITRAGE,
        RejectReason.PARITY_VIOLATION,
    }
)
