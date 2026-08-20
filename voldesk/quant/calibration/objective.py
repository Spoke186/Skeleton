r"""The calibration objective: spread-weighted implied-volatility residuals plus Tikhonov.

CLAUDE.md section 5.5:

.. math::
    L(\Theta) = \sum_i w_i \big(\mathrm{iv}^{\text{obs}}_i
                               - \mathrm{iv}^{\text{model}}_i(\Theta)\big)^2
                + \lambda\,(\Theta - \Theta_{\text{prior}})^{\!\top} M\,
                  (\Theta - \Theta_{\text{prior}})

Three choices in that formula are doing real work.

**Fitting in volatility space, not price space.** A price-space objective weights each
quote by its price, so a deep in-the-money option — worth 25 on a spot of 100 and carrying
almost no information about volatility — dominates the sum, while the wings, which are
what the smile is *about*, contribute almost nothing. Phase 1 measured how extreme this
gets: the implied volatility of a deep in-the-money short-dated call is resolvable only to
about 1e-6 because its vega is 1e-9. Price space would hand that quote the largest weight
in the objective.

**Weights inversely proportional to the relative bid-ask spread.** A quote with a 1% spread
is a much sharper constraint than one with a 20% spread. Weighting by the inverse of the
relative spread is the crude version of weighting by inverse variance, and it is what the
data supports without inventing a quote-noise model.

**Regularising scaled parameters, not raw ones.** :math:`\kappa` ranges over
:math:`(0, 20]` and :math:`v_0` over :math:`(0, 1]`. A Tikhonov penalty on raw differences
would be almost entirely a penalty on :math:`\kappa`, and the "regularisation strength"
:math:`\lambda` would mean something different for every parameter. :math:`M` is diagonal
and makes the five contributions dimensionally comparable.

Everything here is expressed as a **residual vector**, not as a scalar. SciPy's
``least_squares`` wants residuals, the Gauss-Newton Fisher information wants the Jacobian
of residuals, and the Tikhonov term is itself a set of residuals appended to the data ones.
Building the scalar only where a scalar is needed keeps all three consistent by
construction.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from voldesk.quant.model import PARAM_NAMES, HestonParams, MarketState
from voldesk.quant.pricing import cos
from voldesk.quant.pricing.blackscholes import OptionType, implied_vol, implied_vol_fast
from voldesk.quant.reject_reasons import RejectReason

#: One volatility basis point. RMSE is always reported in these; CLAUDE.md section 5.5
#: forbids reporting raw objective values, which are not interpretable.
VOL_BP: Final[float] = 1e-4

#: Relative bid-ask spread assumed when a quote has no spread information at all (a clean
#: synthetic surface, for instance). Any constant works — it cancels out of the normalised
#: weights — but it must not be zero.
_DEFAULT_RELATIVE_SPREAD: Final[float] = 0.01

#: Ceiling on the weight any single quote may take after normalisation. A quote with a
#: near-zero spread would otherwise take over the objective entirely, which is the
#: spread-weighting scheme failing in exactly the way it was meant to prevent.
_MAX_WEIGHT_RATIO: Final[float] = 50.0


@dataclasses.dataclass(frozen=True, slots=True)
class QuoteSlice:
    """One maturity's worth of quotes, already converted to implied volatilities.

    Constructed by :func:`prepare_slice`, which does the inversion and the rejection
    bookkeeping. By the time an objective sees a slice, every quote in it is usable and
    every quote that is not is recorded in ``rejects``.

    Attributes
    ----------
    strikes
        Strikes of the surviving quotes.
    option_types
        ``"call"`` or ``"put"`` per quote — the out-of-the-money side, see
        :func:`prepare_slice`.
    implied_vols
        Observed implied volatilities.
    weights
        Normalised spread-inverse weights, averaging 1.
    rejects
        ``(strike, reason)`` for every quote that did not make it, so that
        CLAUDE.md invariant 3 can be satisfied by the caller.
    """

    maturity: float
    strikes: NDArray[np.float64]
    option_types: tuple[OptionType, ...]
    implied_vols: NDArray[np.float64]
    weights: NDArray[np.float64]
    market: MarketState
    rejects: tuple[tuple[float, RejectReason], ...] = ()

    @property
    def n_quotes(self) -> int:
        """Number of usable quotes."""
        return len(self.strikes)

    @property
    def n_rejected(self) -> int:
        """Number of quotes rejected during preparation."""
        return len(self.rejects)

    @property
    def reject_counts(self) -> dict[str, int]:
        """Rejections tallied by reason, ready for the ``CalibrationRun`` JSON column."""
        counts: dict[str, int] = {}
        for _, reason in self.rejects:
            counts[reason.value] = counts.get(reason.value, 0) + 1
        return counts

    @property
    def filtered_ratio(self) -> float:
        """Fraction of the input that was filtered out. Incident rule R007 watches this."""
        total = self.n_quotes + self.n_rejected
        return self.n_rejected / total if total else 0.0


def parity_offset(strike: float, maturity: float, market: MarketState) -> float:
    r""":math:`C - P = S_0 e^{-qT} - K e^{-rT}`, the exact call-put difference."""
    return market.spot * math.exp(-market.dividend * maturity) - strike * market.discount(maturity)


def prepare_slice(
    strikes: NDArray[np.float64],
    mids: NDArray[np.float64],
    maturity: float,
    market: MarketState,
    *,
    quoted_types: NDArray[np.str_] | list[str] | str = "call",
    bids: NDArray[np.float64] | None = None,
    asks: NDArray[np.float64] | None = None,
    max_relative_spread: float = 0.5,
) -> QuoteSlice:
    """Turn raw quotes into implied volatilities and weights, recording every rejection.

    **Everything is converted to the out-of-the-money side**: a put below the forward, a
    call above it. That is the market convention, and Phase 1 measured why it is also the
    right numerical choice — the implied-volatility inversion is conditioned by vega, vega
    collapses for in-the-money options, and a deep in-the-money quote carries a volatility
    that is not resolvable to better than about 1e-6 in float64.

    The conversion is by put-call parity, which is exact, so no information is lost:

    .. math::
        C - P = S_0 e^{-qT} - K e^{-rT}

    Converting rather than reinterpreting matters. Feeding a call price into a put
    inversion produces an implied volatility that is finite, large, and completely wrong —
    it does not raise, it just quietly ruins the fit.

    The bid-ask spread is invariant under parity (the offset is deterministic), but the
    *relative* spread is not, because the out-of-the-money price is smaller. The relative
    spread is therefore computed **after** conversion, which is the honest measure: an
    absolute spread of 0.50 on a deep in-the-money call worth 25 buys no more volatility
    information than the same 0.50 on the 0.20 put it converts to, and the weighting
    should say so.

    Rejections recorded here, per CLAUDE.md invariant 3:

    - non-finite input -> :attr:`RejectReason.MISSING_FIELD`
    - non-positive price, before or after conversion -> :attr:`RejectReason.NEGATIVE_PRICE`
    - crossed book -> :attr:`RejectReason.CROSSED_QUOTE`
    - zero bid -> :attr:`RejectReason.ZERO_BID`
    - relative spread above ``max_relative_spread`` -> :attr:`RejectReason.WIDE_SPREAD`
    - inversion failure -> whatever ``implied_vol`` returned

    Returns
    -------
    QuoteSlice
        The usable quotes, and the reasons the others are missing.
    """
    forward = market.forward(maturity)
    n = len(strikes)
    quoted = [quoted_types] * n if isinstance(quoted_types, str) else [str(t) for t in quoted_types]

    kept_strikes: list[float] = []
    kept_types: list[OptionType] = []
    kept_vols: list[float] = []
    kept_spreads: list[float] = []
    rejects: list[tuple[float, RejectReason]] = []

    for index, (strike, mid) in enumerate(zip(strikes, mids, strict=True)):
        strike_f, mid_f = float(strike), float(mid)

        if not math.isfinite(mid_f) or not math.isfinite(strike_f):
            rejects.append((strike_f, RejectReason.MISSING_FIELD))
            continue
        if mid_f <= 0.0:
            rejects.append((strike_f, RejectReason.NEGATIVE_PRICE))
            continue

        bid = float(bids[index]) if bids is not None else math.nan
        ask = float(asks[index]) if asks is not None else math.nan
        if bids is not None and asks is not None:
            if not (math.isfinite(bid) and math.isfinite(ask)):
                rejects.append((strike_f, RejectReason.MISSING_FIELD))
                continue
            if bid > ask:
                rejects.append((strike_f, RejectReason.CROSSED_QUOTE))
                continue
            if bid <= 0.0:
                rejects.append((strike_f, RejectReason.ZERO_BID))
                continue

        target: OptionType = "put" if strike_f < forward else "call"
        offset = parity_offset(strike_f, maturity, market)

        # Convert the quote to the out-of-the-money side. The offset is deterministic, so
        # it applies identically to the mid and to both sides of the book.
        if quoted[index] != target:
            shift = -offset if target == "put" else offset
            mid_f += shift
            if bids is not None and asks is not None:
                bid += shift
                ask += shift

        if mid_f <= 0.0:
            rejects.append((strike_f, RejectReason.NEGATIVE_PRICE))
            continue

        relative_spread = _DEFAULT_RELATIVE_SPREAD
        if bids is not None and asks is not None:
            relative_spread = (ask - bid) / mid_f
            if relative_spread > max_relative_spread:
                rejects.append((strike_f, RejectReason.WIDE_SPREAD))
                continue
            relative_spread = max(relative_spread, 1e-6)

        result = implied_vol(
            mid_f, market.spot, strike_f, maturity, market.rate, market.dividend, target
        )
        if not result.ok:
            assert result.reject_reason is not None
            rejects.append((strike_f, result.reject_reason))
            continue

        kept_strikes.append(strike_f)
        kept_types.append(target)
        kept_vols.append(result.vol)
        kept_spreads.append(relative_spread)

    weights = _spread_weights(np.asarray(kept_spreads, dtype=np.float64))

    return QuoteSlice(
        maturity=maturity,
        strikes=np.asarray(kept_strikes, dtype=np.float64),
        option_types=tuple(kept_types),
        implied_vols=np.asarray(kept_vols, dtype=np.float64),
        weights=weights,
        market=market,
        rejects=tuple(rejects),
    )


def _spread_weights(relative_spreads: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse relative spread, capped and normalised to average 1.

    Normalising means the objective's scale does not depend on how many quotes survived,
    so RMSE in volatility basis points is comparable across slices with different quote
    counts. Capping stops one suspiciously tight quote from becoming the entire fit.
    """
    if relative_spreads.size == 0:
        return relative_spreads
    raw = 1.0 / np.maximum(relative_spreads, 1e-6)
    capped = np.minimum(raw, _MAX_WEIGHT_RATIO * float(np.median(raw)))
    return capped * (len(capped) / capped.sum())


def model_implied_vols(
    params: HestonParams,
    slice_: QuoteSlice,
    *,
    n_cos: int = cos.N_COS_DEFAULT,
) -> NDArray[np.float64]:
    """Model implied volatilities at the slice's strikes.

    Prices with COS and inverts, on the same out-of-the-money side the observed quotes
    were inverted on. Where the model price falls outside the no-arbitrage bounds — which
    happens at absurd parameter values the global search will visit — the volatility is
    ``NaN``, and :func:`residuals` turns that into a large finite penalty rather than
    letting it poison the optimizer.
    """
    market = slice_.market
    is_put = np.array([t == "put" for t in slice_.option_types])

    prices = np.empty(slice_.n_quotes, dtype=np.float64)
    if is_put.any():
        prices[is_put] = cos.price_put(
            slice_.strikes[is_put], slice_.maturity, params, market, n_cos=n_cos
        )
    if (~is_put).any():
        prices[~is_put] = cos.price_call(
            slice_.strikes[~is_put], slice_.maturity, params, market, n_cos=n_cos
        )

    # Warm-started from the observed volatilities: during a calibration the model is
    # always near the data, so Newton lands in three or four vectorised steps. Anything
    # that does not converge falls back to the same Brent search used everywhere else, so
    # this is a speed-up and not a change of answer.
    return implied_vol_fast(
        prices,
        market.spot,
        slice_.strikes,
        slice_.maturity,
        market.rate,
        market.dividend,
        slice_.option_types,
        initial_guess=slice_.implied_vols,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class TikhonovPrior:
    r"""The regularisation term :math:`\lambda(\Theta-\Theta_0)^\top M (\Theta-\Theta_0)`.

    Attributes
    ----------
    prior
        :math:`\Theta_0`, the parameter vector the fit is pulled towards.
    strength
        :math:`\lambda`. Zero disables the term entirely.
    scales
        The diagonal of :math:`M^{-1/2}`, one per parameter, in the order of
        :data:`PARAM_NAMES`. A deviation of one ``scale`` contributes one unit to the
        penalty, whatever the parameter's units.
    """

    prior: HestonParams
    strength: float = 0.0
    scales: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        # Filled in from the prior when not given, so that the common case needs no
        # thought and the uncommon one stays explicit.
        if self.scales is None:
            object.__setattr__(self, "scales", default_scales(self.prior))

    def residuals(self, params: HestonParams) -> NDArray[np.float64]:
        r"""The five penalty residuals :math:`\sqrt{\lambda}\,(\Theta-\Theta_0)/s`.

        Returned as residuals rather than as a scalar so that ``least_squares`` sees the
        regularisation as five extra rows of the same problem, and the Gauss-Newton
        Jacobian picks it up automatically.
        """
        if self.strength <= 0.0:
            return np.zeros(0, dtype=np.float64)
        difference = params.to_array() - self.prior.to_array()
        assert self.scales is not None  # set in __post_init__
        return math.sqrt(self.strength) * difference / self.scales

    def penalty(self, params: HestonParams) -> float:
        """The scalar penalty, for reporting."""
        residuals = self.residuals(params)
        return float(residuals @ residuals)

    def solution_norm(self, params: HestonParams) -> float:
        r"""The scaled solution norm :math:`\|M^{1/2}(\Theta-\Theta_0)\|`.

        This is the vertical axis of the L-curve, and it deliberately excludes
        :math:`\lambda`: the L-curve plots how far the solution moved against how well it
        fits, as :math:`\lambda` varies.
        """
        difference = params.to_array() - self.prior.to_array()
        assert self.scales is not None  # set in __post_init__
        return float(np.linalg.norm(difference / self.scales))


def default_scales(prior: HestonParams) -> NDArray[np.float64]:
    r"""The diagonal of :math:`M^{-1/2}`: one natural unit per parameter.

    For the four positive parameters the natural unit is the prior value itself, so a
    penalty of 1 corresponds to a 100% relative move. That is the same relative footing
    the log-space Fisher information of :mod:`~voldesk.quant.calibration.identifiability`
    uses, so the regularisation and the conditioning analysis speak about the same
    directions.

    :math:`\rho` is not positive and has no scale of its own, so it takes 0.5 — a third of
    its admissible width, chosen so that a move of 0.5 in correlation counts as comparable
    to a doubling of one of the others. The choice is arbitrary in the same way any choice
    of units is; what matters is that it is stated rather than implicit.
    """
    return np.array(
        [
            max(abs(prior.v0), 1e-4),
            max(abs(prior.kappa), 1e-3),
            max(abs(prior.theta), 1e-4),
            max(abs(prior.sigma), 1e-3),
            0.5,
        ],
        dtype=np.float64,
    )


#: Volatility residual assigned to a quote the model cannot price at all. Large enough to
#: repel the optimizer, finite enough not to break a Jacobian. 5.0 in volatility units is
#: 50000 basis points — far outside anything a real fit produces.
_INFEASIBLE_RESIDUAL: Final[float] = 5.0


def residuals(
    params: HestonParams,
    target: Fittable,
    prior: TikhonovPrior | None = None,
    *,
    n_cos: int = cos.N_COS_DEFAULT,
) -> NDArray[np.float64]:
    r"""The full residual vector: weighted volatility errors, then the Tikhonov rows.

    .. math::
        r_i = \sqrt{w_i}\,\big(\mathrm{iv}^{\text{obs}}_i - \mathrm{iv}^{\text{model}}_i\big)

    so that :math:`\|r\|^2` is exactly :math:`L(\Theta)`.

    Parameter values that produce an unpriceable option get ``_INFEASIBLE_RESIDUAL``
    rather than ``NaN``. A NaN would propagate into the Jacobian and stop the optimizer
    with a message about the linear algebra, several steps away from the actual cause.
    """
    parts = []
    for one in _as_slices(target):
        model_vols = model_implied_vols(params, one, n_cos=n_cos)
        error = one.implied_vols - model_vols
        error = np.where(np.isfinite(error), error, _INFEASIBLE_RESIDUAL)
        parts.append(np.sqrt(one.weights) * error)
    data_residuals = np.concatenate(parts) if parts else np.zeros(0)

    if prior is None:
        return data_residuals
    return np.concatenate([data_residuals, prior.residuals(params)])


def objective_value(
    params: HestonParams,
    target: Fittable,
    prior: TikhonovPrior | None = None,
    *,
    n_cos: int = cos.N_COS_DEFAULT,
) -> float:
    r""":math:`L(\Theta) = \|r\|^2`. The scalar the global search minimises."""
    r = residuals(params, target, prior, n_cos=n_cos)
    return float(r @ r)


def rmse_vol_bps(
    params: HestonParams,
    target: Fittable,
    *,
    n_cos: int = cos.N_COS_DEFAULT,
) -> float:
    r"""Weighted root-mean-square volatility error, in volatility basis points.

    .. math::
        \mathrm{RMSE} = 10^4 \sqrt{\frac{\sum_i w_i\,\Delta\mathrm{iv}_i^2}{\sum_i w_i}}

    The **only** fit-quality number this project reports. CLAUDE.md section 5.5 forbids
    quoting raw objective values because they are not interpretable — they depend on the
    quote count, on the weighting, and on whether regularisation was on. A number in
    volatility basis points can be compared against a bid-ask spread, which is the thing a
    reader actually wants to know. Incident rule R002 fires above 50 of them.
    """
    if target.n_quotes == 0:
        return math.nan
    slices = _as_slices(target)
    model_vols = np.concatenate([model_implied_vols(params, s, n_cos=n_cos) for s in slices])
    error = target.implied_vols - model_vols
    finite = np.isfinite(error)
    if not finite.any():
        return math.inf
    weights = target.weights[finite]
    weighted = float(np.sum(weights * error[finite] ** 2) / np.sum(weights))
    return math.sqrt(weighted) / VOL_BP


def residual_norm(
    params: HestonParams,
    target: Fittable,
    *,
    n_cos: int = cos.N_COS_DEFAULT,
) -> float:
    """The data-only residual norm, the horizontal axis of the L-curve.

    Deliberately excludes the Tikhonov rows: the L-curve is a plot of misfit against
    solution size, and including the penalty in the misfit would make both axes move
    together and destroy the corner.
    """
    r = residuals(params, target, None, n_cos=n_cos)
    return float(np.linalg.norm(r))


def describe_weights(target: Fittable) -> dict[str, float]:
    """Summary of the weighting, for the run record and for figure captions."""
    if target.n_quotes == 0:
        return {"n": 0}
    weights = target.weights
    return {
        "n": float(len(weights)),
        "min": float(weights.min()),
        "median": float(np.median(weights)),
        "max": float(weights.max()),
        "ratio_max_to_min": float(weights.max() / max(weights.min(), 1e-12)),
    }


__all__ = [
    "PARAM_NAMES",
    "VOL_BP",
    "Fittable",
    "QuoteSlice",
    "SurfaceSlices",
    "TikhonovPrior",
    "default_scales",
    "describe_weights",
    "model_implied_vols",
    "objective_value",
    "prepare_slice",
    "residual_norm",
    "residuals",
    "rmse_vol_bps",
]


@dataclasses.dataclass(frozen=True, slots=True)
class SurfaceSlices:
    r"""Several maturity slices fitted with one shared parameter vector.

    Per-slice calibration is the default for Phases 2 and 3 because it makes the
    identifiability analysis clean: one smile, one fit, nothing else contributing
    information. It also turns out to be *under-determined*, and measurably so — see
    ``docs/adr/0007-per-slice-calibration-is-under-determined.md`` for the measurement.

    Joint calibration is the counterpart. The Heston parameters are properties of the
    process, not of a maturity, so one vector must fit every slice at once. The extra
    maturities add exactly the information a single smile lacks: :math:`v_0` governs the
    front of the term structure and :math:`\theta` the back, and with only one slice the
    two are almost interchangeable. With several, they are not.

    This class presents the same interface as :class:`QuoteSlice` so that the optimiser,
    the Fisher information and the L-curve all work on either without knowing which they
    were given.
    """

    slices: tuple[QuoteSlice, ...]

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValueError("a joint calibration needs at least one slice")

    @property
    def n_quotes(self) -> int:
        """Usable quotes across all slices."""
        return sum(s.n_quotes for s in self.slices)

    @property
    def n_rejected(self) -> int:
        """Rejected quotes across all slices."""
        return sum(s.n_rejected for s in self.slices)

    @property
    def rejects(self) -> tuple[tuple[float, RejectReason], ...]:
        """Every rejection, so invariant 3 survives the aggregation."""
        return tuple(reject for s in self.slices for reject in s.rejects)

    @property
    def reject_counts(self) -> dict[str, int]:
        """Rejections tallied by reason across the surface."""
        counts: dict[str, int] = {}
        for s in self.slices:
            for reason, count in s.reject_counts.items():
                counts[reason] = counts.get(reason, 0) + count
        return counts

    @property
    def filtered_ratio(self) -> float:
        """Fraction of the input filtered out. Incident rule R007 watches this."""
        total = self.n_quotes + self.n_rejected
        return self.n_rejected / total if total else 0.0

    @property
    def maturity(self) -> float:
        """Shortest maturity in the set — a label, since a joint fit spans many."""
        return min(s.maturity for s in self.slices)

    @property
    def maturities(self) -> tuple[float, ...]:
        """Every maturity in the set, ascending."""
        return tuple(sorted(s.maturity for s in self.slices))

    @property
    def market(self) -> MarketState:
        """The shared market state."""
        return self.slices[0].market

    @property
    def implied_vols(self) -> NDArray[np.float64]:
        """Observed volatilities, concatenated in slice order."""
        return np.concatenate([s.implied_vols for s in self.slices])

    @property
    def strikes(self) -> NDArray[np.float64]:
        """Strikes, concatenated in slice order."""
        return np.concatenate([s.strikes for s in self.slices])

    @property
    def weights(self) -> NDArray[np.float64]:
        """Weights, concatenated in slice order.

        Each slice's weights already average 1, so no slice dominates by having more
        surviving quotes than another — a front month with 20 usable quotes and a back
        month with 8 contribute in proportion to their quote counts, not in proportion to
        how tightly the front month happens to be quoted.
        """
        return np.concatenate([s.weights for s in self.slices])


#: Anything the calibration machinery can fit: one slice or a whole surface.
Fittable = QuoteSlice | SurfaceSlices


def _as_slices(target: Fittable) -> tuple[QuoteSlice, ...]:
    """Normalise either input to a tuple of slices."""
    return target.slices if isinstance(target, SurfaceSlices) else (target,)


def joint_model_implied_vols(
    params: HestonParams,
    surface: SurfaceSlices,
    *,
    n_cos: int = cos.N_COS_DEFAULT,
) -> NDArray[np.float64]:
    """Model volatilities across every slice, concatenated in slice order."""
    return np.concatenate([model_implied_vols(params, s, n_cos=n_cos) for s in surface.slices])
