r"""The L-curve: choosing the regularisation strength, and checking whether it works.

Tikhonov regularisation trades fit against parameter stability. Too little and the fitted
parameters chase the noise; too much and they are just the prior. The L-curve
(Hansen, 1992) is the standard way to pick the trade-off without knowing the noise level:
plot the residual norm against the solution norm on log-log axes as :math:`\lambda` sweeps,
and the locus has a corner. Left of it, more regularisation costs almost no fit; right of
it, it costs a lot. The corner is the point of maximum curvature.

Reference: Hansen, P.C. (1992), "Analysis of Discrete Ill-Posed Problems by Means of the
L-Curve", *SIAM Review* 34(4), 561-580.

What makes this project's version of the L-curve worth doing
------------------------------------------------------------
The L-curve criterion is a *heuristic*. It has no proof that the corner minimises the
error in the parameters, and there are published counterexamples where it does not. With
real market data one cannot tell, because the true parameters are unknown — which is
exactly the situation that makes the heuristic attractive and unfalsifiable at the same
time.

Here the data is synthetic and the truth is known. So the corner can be located by
Hansen's criterion, the :math:`\lambda` that actually minimises the true parameter error
can be located independently, and **the two can be compared**. CLAUDE.md section 5.7 asks
for exactly this and, importantly, asks for the answer either way:

    Because ground truth is known in synthetic runs, additionally verify whether the
    L-corner coincides with the minimum of the *true* parameter error. That comparison is
    a genuine result — report it either way.

:func:`compare_corner_to_truth` computes both and returns them side by side. Nothing in
this module decides which answer is the good one.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from numpy.typing import NDArray

from voldesk.quant.calibration.objective import (
    QuoteSlice,
    TikhonovPrior,
    residual_norm,
    rmse_vol_bps,
)
from voldesk.quant.calibration.optimizers import (
    CalibrationConfig,
    CalibrationResult,
    calibrate,
    calibrate_local_only,
    relative_parameter_error,
)
from voldesk.quant.model import PARAM_NAMES, HestonParams


@dataclasses.dataclass(frozen=True, slots=True)
class LCurvePoint:
    r"""One :math:`\lambda` on the sweep."""

    strength: float
    params: HestonParams
    residual_norm: float
    solution_norm: float
    rmse_vol_bps: float
    converged: bool

    @property
    def log_residual(self) -> float:
        return math.log10(max(self.residual_norm, 1e-300))

    @property
    def log_solution(self) -> float:
        return math.log10(max(self.solution_norm, 1e-300))


@dataclasses.dataclass(frozen=True, slots=True)
class LCurve:
    r"""A full :math:`\lambda` sweep and the corner detected on it."""

    points: tuple[LCurvePoint, ...]
    prior: HestonParams
    corner_index: int
    curvature: NDArray[np.float64]

    @property
    def strengths(self) -> NDArray[np.float64]:
        return np.array([p.strength for p in self.points])

    @property
    def residual_norms(self) -> NDArray[np.float64]:
        return np.array([p.residual_norm for p in self.points])

    @property
    def solution_norms(self) -> NDArray[np.float64]:
        return np.array([p.solution_norm for p in self.points])

    @property
    def corner(self) -> LCurvePoint:
        """The maximum-curvature point — Hansen's choice of regularisation strength."""
        return self.points[self.corner_index]

    def to_dict(self) -> dict[str, object]:
        return {
            "strengths": self.strengths.tolist(),
            "residual_norms": self.residual_norms.tolist(),
            "solution_norms": self.solution_norms.tolist(),
            "curvature": self.curvature.tolist(),
            "corner_index": self.corner_index,
            "corner_strength": self.corner.strength,
            "corner_params": self.corner.params.to_dict(),
            "prior": self.prior.to_dict(),
        }


def sweep(
    slice_: QuoteSlice,
    prior_params: HestonParams,
    *,
    strengths: NDArray[np.float64] | None = None,
    n_points: int = 25,
    log_min: float = -8.0,
    log_max: float = 2.0,
    config: CalibrationConfig | None = None,
    scales: NDArray[np.float64] | None = None,
) -> LCurve:
    r"""Sweep :math:`\lambda` logarithmically and build the L-curve.

    The sweep runs from strong regularisation down to none, warm-starting each fit from
    the previous solution. Two reasons, and the second is the important one:

    - it is much cheaper than a global search per point;
    - it produces a *continuous* curve. Independent global searches would each land in
      whichever basin they found, and the resulting locus would have jumps that are
      artefacts of the optimiser rather than features of the regularisation. A corner
      detector run on that would find the jump.

    The first point is seeded by a full global calibration at the strongest
    :math:`\lambda`, where the objective is most convex and the fit most reliable.

    Returns
    -------
    LCurve
        Every point, plus the detected corner.
    """
    config = config or CalibrationConfig()
    if strengths is None:
        strengths = np.logspace(log_max, log_min, n_points)
    strengths = np.asarray(strengths, dtype=np.float64)

    points: list[LCurvePoint] = []
    warm_start: HestonParams | None = None

    for strength in strengths:
        prior = TikhonovPrior(prior=prior_params, strength=float(strength), scales=scales)
        result: CalibrationResult
        if warm_start is None:
            result = calibrate(slice_, config, prior)
        else:
            result = calibrate_local_only(slice_, warm_start, config, prior)
        warm_start = result.params

        points.append(
            LCurvePoint(
                strength=float(strength),
                params=result.params,
                # Data-only misfit: the horizontal axis must not contain the penalty, or
                # both axes would move together and the corner would vanish.
                residual_norm=residual_norm(result.params, slice_, n_cos=config.n_cos),
                solution_norm=prior.solution_norm(result.params),
                rmse_vol_bps=rmse_vol_bps(result.params, slice_, n_cos=config.n_cos),
                converged=result.converged,
            )
        )

    curvature = log_log_curvature(
        np.array([p.residual_norm for p in points]),
        np.array([p.solution_norm for p in points]),
    )
    corner_index = int(np.nanargmax(curvature)) if np.any(np.isfinite(curvature)) else 0

    return LCurve(
        points=tuple(points),
        prior=prior_params,
        corner_index=corner_index,
        curvature=curvature,
    )


def log_log_curvature(
    residual_norms: NDArray[np.float64], solution_norms: NDArray[np.float64]
) -> NDArray[np.float64]:
    r"""Curvature of the L-curve in log-log coordinates.

    With :math:`\xi = \log\|r\|` and :math:`\eta = \log\|x\|` parameterised by the sweep
    index, the signed curvature of a plane curve is

    .. math::
        \varkappa = \frac{\xi'\eta'' - \eta'\xi''}
                         {\big(\xi'^2 + \eta'^2\big)^{3/2}}

    Derivatives are taken with ``numpy.gradient``, which is second-order accurate in the
    interior and handles the non-uniform spacing that the log-spaced sweep produces in
    :math:`(\xi, \eta)`.

    The two endpoints are set to ``NaN``. A one-sided second derivative at an endpoint is
    noisy, and the maximum-curvature search would otherwise be drawn to the edge of the
    sweep — reporting a "corner" that is really the end of the range, which is one of the
    classic ways to misuse this method.
    """
    xi = np.log10(np.maximum(residual_norms, 1e-300))
    eta = np.log10(np.maximum(solution_norms, 1e-300))

    d_xi = np.gradient(xi)
    d_eta = np.gradient(eta)
    dd_xi = np.gradient(d_xi)
    dd_eta = np.gradient(d_eta)

    denominator = (d_xi**2 + d_eta**2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.abs(d_xi * dd_eta - d_eta * dd_xi) / denominator
    curvature = np.where(np.isfinite(curvature), curvature, np.nan)

    if len(curvature) > 2:
        curvature[0] = np.nan
        curvature[-1] = np.nan
    return curvature


@dataclasses.dataclass(frozen=True, slots=True)
class CornerComparison:
    r"""Hansen's corner against the :math:`\lambda` that truly minimises parameter error.

    The finding this project is set up to produce. Both numbers are reported whatever they
    say; CLAUDE.md section 11 is explicit that a negative result here is a result.
    """

    corner_strength: float
    corner_index: int
    best_strength: float
    best_index: int
    true_errors: NDArray[np.float64]
    strengths: NDArray[np.float64]
    error_at_corner: float
    error_at_best: float

    @property
    def decades_apart(self) -> float:
        r"""Distance between the two choices of :math:`\lambda`, in decades."""
        return abs(
            math.log10(max(self.corner_strength, 1e-300))
            - math.log10(max(self.best_strength, 1e-300))
        )

    @property
    def excess_error_ratio(self) -> float:
        """How much worse Hansen's choice is, as a multiple of the achievable error."""
        return self.error_at_corner / max(self.error_at_best, 1e-300)

    def verdict(self) -> str:
        """A one-line plain statement of what the comparison found.

        Written to be quotable in the README and the paper without editing, and to say the
        unflattering version just as plainly as the flattering one.
        """
        if self.decades_apart < 0.5:
            return (
                f"The L-curve corner and the true-error minimum coincide to within "
                f"{self.decades_apart:.2f} decades of lambda; Hansen's criterion picks a "
                f"regularisation strength giving {self.excess_error_ratio:.2f}x the "
                f"minimum achievable parameter error."
            )
        return (
            f"The L-curve corner and the true-error minimum are {self.decades_apart:.2f} "
            f"decades apart in lambda. Hansen's criterion selects a solution with "
            f"{self.excess_error_ratio:.2f}x the minimum achievable parameter error. On "
            f"this problem the corner is not a reliable proxy for parameter accuracy."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "corner_strength": self.corner_strength,
            "best_strength": self.best_strength,
            "decades_apart": self.decades_apart,
            "error_at_corner": self.error_at_corner,
            "error_at_best": self.error_at_best,
            "excess_error_ratio": self.excess_error_ratio,
            "strengths": self.strengths.tolist(),
            "true_errors": self.true_errors.tolist(),
            "verdict": self.verdict(),
        }


def parameter_error_metric(fitted: HestonParams, truth: HestonParams) -> float:
    r"""Scalar distance between a fit and the truth, in relative terms.

    The root-mean-square relative error over the four positive parameters plus the
    absolute error in :math:`\rho`. Relative for the positive ones because they span
    orders of magnitude; absolute for :math:`\rho` because a relative error on a
    correlation that can pass through zero is not informative.

    This is the vertical axis of the true-error curve in figure 4, so it needs to be one
    number and it needs to be stated. It is not the only reasonable choice, and a
    different one would move the true-error minimum somewhat — which is itself worth
    knowing when reading the comparison.
    """
    errors = relative_parameter_error(fitted, truth)
    positive = [errors[name] for name in PARAM_NAMES if name != "rho"]
    return float(math.sqrt(np.mean(np.square(positive)) + errors["rho_absolute"] ** 2))


def compare_corner_to_truth(curve: LCurve, truth: HestonParams) -> CornerComparison:
    r"""Locate Hansen's corner and the true-error minimum, and report both.

    Only possible because the data is synthetic. CLAUDE.md invariant 6 keeps ``truth``
    away from the calibrator; it arrives here, on the evaluation side, after every fit on
    the curve is already done.
    """
    strengths = curve.strengths
    errors = np.array([parameter_error_metric(p.params, truth) for p in curve.points])
    best_index = int(np.argmin(errors))

    return CornerComparison(
        corner_strength=curve.corner.strength,
        corner_index=curve.corner_index,
        best_strength=float(strengths[best_index]),
        best_index=best_index,
        true_errors=errors,
        strengths=strengths,
        error_at_corner=float(errors[curve.corner_index]),
        error_at_best=float(errors[best_index]),
    )
