r"""The two-stage calibration pipeline: Differential Evolution, then Levenberg-Marquardt.

CLAUDE.md section 5.6 fixes the order, and the order is the point.

The Heston implied-volatility objective is not convex and is not close to convex. It has a
long, nearly flat valley in the :math:`(\kappa, \sigma)` plane — the degeneracy this whole
project exists to measure — and several local minima scattered around the box. A local
method started from a guess lands wherever the guess pointed it, and the result *looks*
converged: small gradient, plausible parameters, respectable RMSE. Running it from two
different starting points is the only way to notice, and by then it is a mystery rather
than a finding.

So:

1. **Differential Evolution** searches the whole admissible box. It is derivative-free,
   population-based and does not care about the valley, but it converges slowly once it is
   near a minimum — it has no gradient information to exploit.
2. **Levenberg-Marquardt** (``least_squares`` with ``method='trf'``, so that the parameter
   bounds are respected) polishes. It is fast and accurate near a minimum, which is
   exactly the regime stage 1 hands it.

Neither stage alone is adequate. DE alone leaves the last few digits on the table, which
matters for a gate that asks for 1e-3 relative recovery. LM alone finds a minimum, not
*the* minimum.

Everything the run produces is returned in one :class:`CalibrationResult`, including when
it fails. CLAUDE.md invariant 2: *"Every calibration persists a full CalibrationRun row —
no exceptions, including failed runs. Failed runs are the most valuable rows in the
table."* This module never raises for a numerical failure; it returns a record whose
``status`` says what happened.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, least_squares

from voldesk.quant.calibration.objective import (
    Fittable,
    QuoteSlice,
    TikhonovPrior,
    objective_value,
    residual_norm,
    residuals,
    rmse_vol_bps,
)
from voldesk.quant.model import (
    PARAM_NAMES,
    HestonParams,
    bounds_arrays,
    clip_to_bounds,
)
from voldesk.quant.pricing import cos
from voldesk.quant.provenance import git_sha

#: Status values a run can end in. Persisted verbatim on ``CalibrationRun.status``.
STATUS_OK: Final[str] = "ok"
STATUS_NOT_CONVERGED: Final[str] = "not_converged"
STATUS_NO_QUOTES: Final[str] = "no_quotes"
STATUS_FAILED: Final[str] = "failed"


@dataclasses.dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Everything that determines a calibration, so that storing it reproduces the run.

    CLAUDE.md invariant 5: byte-reproducible from the stored config. That is only true if
    the config is complete, which is why the optimizer settings live here rather than as
    call-site keyword arguments.

    Attributes
    ----------
    seed
        Seeds Differential Evolution's population and mutation draws.
    de_popsize
        Population multiplier; the population is ``de_popsize * 5``.
    de_maxiter
        Generation cap for the global stage.
    de_tol
        Relative convergence tolerance for the global stage. Deliberately loose: the
        global stage only has to land in the right basin, and the local stage is far
        better at the last digits. Measured across the three market regimes, tightening
        the global stage from (40 generations, 1e-3) to (200, 1e-8) costs 7.6x the wall
        clock and changes the worst recovery error from 7.8e-7 to 6.5e-7 — that is, it
        buys nothing against a gate that asks for 1e-3.
    de_workers
        Processes for the global stage. **Left at 1 by default.** See
        :func:`calibrate` for why byte-reproducibility and parallelism interact here.
    lm_max_nfev
        Function-evaluation cap for the local stage.
    n_cos
        Cosine terms in the pricer. Part of the config because it changes the answer.
    """

    seed: int = 0
    de_popsize: int = 12
    de_maxiter: int = 40
    de_tol: float = 1e-3
    de_mutation: tuple[float, float] = (0.5, 1.0)
    de_recombination: float = 0.7
    de_workers: int = 1
    lm_max_nfev: int = 2000
    lm_xtol: float = 1e-12
    lm_ftol: float = 1e-12
    n_cos: int = cos.N_COS_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        """As JSON, for the ``CalibrationRun.config`` column."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class StageResult:
    """What one optimisation stage did. Kept so a bad fit can be attributed."""

    name: str
    params: HestonParams
    objective: float
    n_function_evals: int
    duration_seconds: float
    converged: bool
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class CalibrationResult:
    """One calibration, successful or not — the full ``CalibrationRun`` row.

    Every field CLAUDE.md invariant 2 enumerates is here: fitted parameters, seed, git
    SHA, optimizer settings, RMSE, quote counts, Feller ratio, duration, status.
    """

    params: HestonParams
    status: str
    rmse_vol_bps: float
    objective: float
    residual_norm: float
    feller_ratio: float
    n_quotes: int
    n_rejected: int
    reject_counts: dict[str, int]
    filtered_ratio: float
    maturity: float
    seed: int
    git_sha: str
    config: dict[str, Any]
    duration_seconds: float
    stages: tuple[StageResult, ...]
    prior_strength: float
    solution_norm: float
    message: str = ""

    @property
    def converged(self) -> bool:
        """Whether the run reached a minimum the local stage was happy with.

        Incident rule R001 fires when this is false.
        """
        return self.status == STATUS_OK

    @property
    def satisfies_feller(self) -> bool:
        """Rule R003 fires when this is false. It is a signal, never an error."""
        return self.feller_ratio >= 1.0

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for persistence and for experiment output files."""
        record = {
            "status": self.status,
            "rmse_vol_bps": self.rmse_vol_bps,
            "objective": self.objective,
            "residual_norm": self.residual_norm,
            "feller_ratio": self.feller_ratio,
            "n_quotes": self.n_quotes,
            "n_rejected": self.n_rejected,
            "reject_counts": self.reject_counts,
            "filtered_ratio": self.filtered_ratio,
            "maturity": self.maturity,
            "seed": self.seed,
            "git_sha": self.git_sha,
            "config": self.config,
            "duration_seconds": self.duration_seconds,
            "prior_strength": self.prior_strength,
            "solution_norm": self.solution_norm,
            "message": self.message,
            "stages": [
                {
                    "name": s.name,
                    "objective": s.objective,
                    "n_function_evals": s.n_function_evals,
                    "duration_seconds": s.duration_seconds,
                    "converged": s.converged,
                    "message": s.message,
                }
                for s in self.stages
            ],
        }
        record.update({f"param_{name}": getattr(self.params, name) for name in PARAM_NAMES})
        return record


def _empty_result(
    slice_: Fittable,
    config: CalibrationConfig,
    prior: TikhonovPrior | None,
    message: str,
) -> CalibrationResult:
    """A row for a slice with nothing left to fit.

    Still a full row. A slice whose quotes were all rejected is exactly the kind of event
    the operations layer exists to notice, and it cannot notice what was never written.
    """
    fallback = prior.prior if prior is not None else HestonParams(0.04, 1.5, 0.04, 0.3, -0.5)
    return CalibrationResult(
        params=fallback,
        status=STATUS_NO_QUOTES,
        rmse_vol_bps=math.nan,
        objective=math.nan,
        residual_norm=math.nan,
        feller_ratio=fallback.feller_ratio,
        n_quotes=slice_.n_quotes,
        n_rejected=slice_.n_rejected,
        reject_counts=slice_.reject_counts,
        filtered_ratio=slice_.filtered_ratio,
        maturity=slice_.maturity,
        seed=config.seed,
        git_sha=git_sha(),
        config=config.to_dict(),
        duration_seconds=0.0,
        stages=(),
        prior_strength=prior.strength if prior else 0.0,
        solution_norm=math.nan,
        message=message,
    )


def default_initial_guess(slice_: QuoteSlice) -> HestonParams:
    r"""A starting point read off the data rather than hard-coded.

    The at-the-money implied variance is a good estimate of both :math:`v_0` and
    :math:`\theta` for a single slice — for a flat term structure they coincide — so the
    guess uses it for both. The remaining three take mid-range values.

    This matters mostly for the local-only path and for the Tikhonov prior. The global
    stage does not use a starting point at all, which is precisely its advantage.
    """
    if slice_.n_quotes == 0:
        return HestonParams(0.04, 1.5, 0.04, 0.3, -0.5)
    forward = slice_.market.forward(slice_.maturity)
    atm_index = int(np.argmin(np.abs(slice_.strikes - forward)))
    atm_variance = float(np.clip(slice_.implied_vols[atm_index] ** 2, 1e-4, 1.0))
    return HestonParams(
        v0=atm_variance,
        kappa=1.5,
        theta=atm_variance,
        sigma=0.5,
        rho=-0.6,
    )


def calibrate(
    slice_: Fittable,
    config: CalibrationConfig | None = None,
    prior: TikhonovPrior | None = None,
) -> CalibrationResult:
    r"""Fit :math:`\Theta` to one maturity slice. Never raises on a numerical failure.

    Per-slice calibration is the choice recorded in `PROJECT_PLAN.md` for Phases 2 and 3:
    fitting one maturity at a time keeps the identifiability analysis clean and makes the
    :math:`(\kappa, \sigma)` degeneracy easy to visualise, because there is no
    cross-maturity information partially breaking it. Joint calibration is added in
    Phase 3 as the comparison.

    **Reproducibility.** ``de_workers`` defaults to 1. SciPy's Differential Evolution
    draws its randomness in the parent process even when ``updating='deferred'``, so a
    parallel run *should* be reproducible — but "should" is not what CLAUDE.md invariant 5
    asks for, and the serial path costs about a second per slice. Parallelism is available
    for the Phase 3 sweeps, where the outer loop over seeds parallelises anyway and is the
    better place to spend the cores.

    Returns
    -------
    CalibrationResult
        Always. A failure is a record with a ``status``, not an exception — CLAUDE.md
        invariant 2 has no exceptions, and the failed rows are the interesting ones.
    """
    config = config or CalibrationConfig()
    started = time.perf_counter()

    if slice_.n_quotes == 0:
        return _empty_result(slice_, config, prior, "every quote in the slice was rejected")
    if slice_.n_quotes < len(PARAM_NAMES):
        return _empty_result(
            slice_,
            config,
            prior,
            f"{slice_.n_quotes} usable quotes cannot determine {len(PARAM_NAMES)} parameters",
        )

    lower, upper = bounds_arrays()
    stages: list[StageResult] = []

    def scalar_objective(x: NDArray[np.float64]) -> float:
        value = objective_value(HestonParams.from_array(x), slice_, prior, n_cos=config.n_cos)
        # differential_evolution cannot recover from a NaN; a huge finite value tells it
        # the same thing (go elsewhere) without stopping the search.
        return value if math.isfinite(value) else 1e12

    def residual_vector(x: NDArray[np.float64]) -> NDArray[np.float64]:
        r = residuals(HestonParams.from_array(x), slice_, prior, n_cos=config.n_cos)
        return np.where(np.isfinite(r), r, 1e6)

    # --- stage 1: global search ---------------------------------------------------
    stage_started = time.perf_counter()
    try:
        de = differential_evolution(
            scalar_objective,
            bounds=list(zip(lower, upper, strict=True)),
            seed=config.seed,
            popsize=config.de_popsize,
            maxiter=config.de_maxiter,
            tol=config.de_tol,
            mutation=config.de_mutation,
            recombination=config.de_recombination,
            workers=config.de_workers,
            updating="deferred" if config.de_workers != 1 else "immediate",
            polish=False,  # the LM stage is the polish, and it respects the bounds
            init="sobol",
        )
        de_params = clip_to_bounds(HestonParams.from_array(de.x))
        stages.append(
            StageResult(
                name="differential_evolution",
                params=de_params,
                objective=float(de.fun),
                n_function_evals=int(de.nfev),
                duration_seconds=time.perf_counter() - stage_started,
                converged=bool(de.success),
                message=str(de.message),
            )
        )
    except Exception as exc:  # a failed run must still be recorded (invariant 2)
        result = _empty_result(slice_, config, prior, f"global stage raised: {exc!r}")
        return dataclasses.replace(
            result, status=STATUS_FAILED, duration_seconds=time.perf_counter() - started
        )

    # --- stage 2: local polish ----------------------------------------------------
    stage_started = time.perf_counter()
    try:
        lm = least_squares(
            residual_vector,
            x0=de_params.to_array(),
            bounds=(lower, upper),
            method="trf",
            xtol=config.lm_xtol,
            ftol=config.lm_ftol,
            max_nfev=config.lm_max_nfev,
            x_scale=np.array([0.04, 1.5, 0.04, 0.3, 0.5]),  # rough parameter magnitudes
        )
        lm_params = clip_to_bounds(HestonParams.from_array(lm.x))
        lm_objective = float(2.0 * lm.cost)  # least_squares reports 0.5 * ||r||^2
        stages.append(
            StageResult(
                name="levenberg_marquardt",
                params=lm_params,
                objective=lm_objective,
                n_function_evals=int(lm.nfev),
                duration_seconds=time.perf_counter() - stage_started,
                converged=bool(lm.success),
                message=str(lm.message),
            )
        )
    except Exception as exc:  # a failed polish still yields the global result
        lm_params, lm_objective = de_params, float(de.fun)
        stages.append(
            StageResult(
                name="levenberg_marquardt",
                params=de_params,
                objective=lm_objective,
                n_function_evals=0,
                duration_seconds=time.perf_counter() - stage_started,
                converged=False,
                message=f"local stage raised: {exc!r}",
            )
        )

    # The local stage is a polish, not a gamble: if it somehow made things worse, the
    # global result stands. Without this a numerically awkward slice can end up with a
    # answer strictly worse than the one already in hand.
    if lm_objective > stages[0].objective:
        final_params = de_params
        final_objective = stages[0].objective
        note = "local stage did not improve on the global stage; keeping the global result"
    else:
        final_params = lm_params
        final_objective = lm_objective
        note = ""

    # Convergence is judged on the *local* stage. The global stage routinely stops on its
    # generation cap rather than its own tolerance, and reports success=False for doing so
    # — which is normal and says nothing about the fit. What matters is whether the polish
    # reached a point it is satisfied with; that is also what incident rule R001 keys on.
    # The global stage's own flag is kept on its StageResult, where it can be inspected
    # without being mistaken for a verdict on the answer.
    status = STATUS_OK if stages[-1].converged else STATUS_NOT_CONVERGED

    return CalibrationResult(
        params=final_params,
        status=status,
        rmse_vol_bps=rmse_vol_bps(final_params, slice_, n_cos=config.n_cos),
        objective=final_objective,
        residual_norm=residual_norm(final_params, slice_, n_cos=config.n_cos),
        feller_ratio=final_params.feller_ratio,
        n_quotes=slice_.n_quotes,
        n_rejected=slice_.n_rejected,
        reject_counts=slice_.reject_counts,
        filtered_ratio=slice_.filtered_ratio,
        maturity=slice_.maturity,
        seed=config.seed,
        git_sha=git_sha(),
        config=config.to_dict(),
        duration_seconds=time.perf_counter() - started,
        stages=tuple(stages),
        prior_strength=prior.strength if prior else 0.0,
        solution_norm=prior.solution_norm(final_params) if prior else 0.0,
        message="; ".join(m for m in (note, stages[-1].message) if m),
    )


def calibrate_local_only(
    slice_: Fittable,
    start: HestonParams,
    config: CalibrationConfig | None = None,
    prior: TikhonovPrior | None = None,
) -> CalibrationResult:
    """Levenberg-Marquardt from a given start, skipping the global search.

    Used where a good start already exists and the global stage would be wasted work: the
    L-curve sweep, which walks :math:`\\lambda` in small steps and can warm-start each
    point from the previous one, and the profile likelihood, which does the same along a
    fixed parameter.

    It is *not* the default calibration path, and the docstring of :func:`calibrate`
    explains why: a local method on this objective finds a minimum that looks converged
    from the inside whatever it landed in.
    """
    config = config or CalibrationConfig()
    started = time.perf_counter()

    if slice_.n_quotes < len(PARAM_NAMES):
        return _empty_result(slice_, config, prior, "not enough quotes for a local fit")

    lower, upper = bounds_arrays()

    def residual_vector(x: NDArray[np.float64]) -> NDArray[np.float64]:
        r = residuals(HestonParams.from_array(x), slice_, prior, n_cos=config.n_cos)
        return np.where(np.isfinite(r), r, 1e6)

    lm = least_squares(
        residual_vector,
        x0=clip_to_bounds(start).to_array(),
        bounds=(lower, upper),
        method="trf",
        xtol=config.lm_xtol,
        ftol=config.lm_ftol,
        max_nfev=config.lm_max_nfev,
        x_scale=np.array([0.04, 1.5, 0.04, 0.3, 0.5]),
    )
    params = clip_to_bounds(HestonParams.from_array(lm.x))
    stage = StageResult(
        name="levenberg_marquardt",
        params=params,
        objective=float(2.0 * lm.cost),
        n_function_evals=int(lm.nfev),
        duration_seconds=time.perf_counter() - started,
        converged=bool(lm.success),
        message=str(lm.message),
    )

    return CalibrationResult(
        params=params,
        status=STATUS_OK if lm.success else STATUS_NOT_CONVERGED,
        rmse_vol_bps=rmse_vol_bps(params, slice_, n_cos=config.n_cos),
        objective=stage.objective,
        residual_norm=residual_norm(params, slice_, n_cos=config.n_cos),
        feller_ratio=params.feller_ratio,
        n_quotes=slice_.n_quotes,
        n_rejected=slice_.n_rejected,
        reject_counts=slice_.reject_counts,
        filtered_ratio=slice_.filtered_ratio,
        maturity=slice_.maturity,
        seed=config.seed,
        git_sha=git_sha(),
        config=config.to_dict(),
        duration_seconds=time.perf_counter() - started,
        stages=(stage,),
        prior_strength=prior.strength if prior else 0.0,
        solution_norm=prior.solution_norm(params) if prior else 0.0,
        message=stage.message,
    )


def relative_parameter_error(fitted: HestonParams, truth: HestonParams) -> dict[str, float]:
    r"""Per-parameter relative error :math:`|\hat\Theta_i - \Theta_i| / |\Theta_i|`.

    Only meaningful against synthetic ground truth. CLAUDE.md invariant 6 keeps the truth
    away from the calibrator, so this lives on the evaluation side: it is called *after* a
    fit, with the truth the fit never saw.

    :math:`\rho` is reported as an absolute error under the ``rho`` key as well, since a
    relative error on a correlation that can pass through zero is not informative.
    """
    errors = {
        name: abs(getattr(fitted, name) - getattr(truth, name))
        / max(abs(getattr(truth, name)), 1e-12)
        for name in PARAM_NAMES
    }
    errors["rho_absolute"] = abs(fitted.rho - truth.rho)
    errors["max"] = max(errors[name] for name in PARAM_NAMES)
    return errors
