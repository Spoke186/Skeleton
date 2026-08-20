r"""Identifiability: which parameter combinations the data actually determines.

This module is the scientific contribution of the project. Everything else — the pricer,
the operations layer, the dashboards — exists so that the numbers computed here can be
trusted and reproduced.

The question
------------
A calibration returns five numbers and an RMSE. The RMSE says the model fits. It does not
say the five numbers are *determined*. In a sloppy inverse problem some combinations of
parameters are pinned down tightly by the data while others can be varied over orders of
magnitude with almost no effect on the fit — and a point estimate looks identical in both
cases. Distinguishing them requires looking at the curvature of the objective, not its
value.

The Fisher information
----------------------
For Gaussian residuals, the Fisher information matrix is the Gauss-Newton approximation to
the Hessian of the objective (CLAUDE.md section 5.7):

.. math::
    I = J^\top W J

Here the residual vector already carries :math:`\sqrt{w_i}`, so :math:`W` is folded in and
:math:`I = J^\top J` with :math:`J` the Jacobian of the *weighted* residuals. Its
eigenvalues are the curvatures along the corresponding eigenvector directions: a large
eigenvalue is a stiff direction the data constrains, a small one is a sloppy direction it
does not.

Log-parameter space
-------------------
The eigenvalues of :math:`I` in raw parameter space are meaningless to compare across
parameters, because the parameters have different units and differ by orders of magnitude —
:math:`\kappa \in (0, 20]` against :math:`v_0 \in (0, 1]`. Rescaling a parameter rescales
its row and column of :math:`I` and changes the spectrum, so "the smallest eigenvalue"
would be a statement about the units.

CLAUDE.md section 5.7 therefore requires the analysis in log space for the four positive
parameters, where a unit step is a fixed *relative* change and the spectrum is
scale-invariant. The chain rule makes this a column scaling of the Jacobian:

.. math::
    \frac{\partial r}{\partial \log \theta_j}
        = \theta_j \, \frac{\partial r}{\partial \theta_j}

**The choice for** :math:`\rho`, which the specification asks to be documented: it is left
**linear**. Fisher's z-transform :math:`\operatorname{artanh}(\rho)` is the natural
variance-stabilising map for a correlation, and it would be the right choice if
:math:`\rho` were being estimated near :math:`\pm 1`, where the transform's expansion
matters. Calibrated equity-index correlations sit around :math:`-0.7`, comfortably inside
the range where :math:`\operatorname{artanh}` is close to linear (its derivative there is
about 2), and the admissible interval :math:`[-0.999, 0.5]` is bounded away from the
singularity on the side the data occupies. Against that, :math:`\rho` is already
dimensionless and of order one — the very property log space is introduced to give the
others — so transforming it would make the spectrum harder to read, not easier, for a
factor-of-two change in one row. Linear it is, and :func:`fisher_information` takes a flag
so the alternative can be measured rather than argued about.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from voldesk.quant.calibration.objective import QuoteSlice, TikhonovPrior, residuals
from voldesk.quant.calibration.optimizers import CalibrationConfig
from voldesk.quant.model import PARAM_NAMES, POSITIVE_PARAMS, HestonParams, clip_to_bounds
from voldesk.quant.pricing import cos

#: Relative step for the finite-difference Jacobian, in the transformed coordinates.
#: Central differences give an O(h^2) truncation error and an O(eps/h) round-off error, so
#: the optimum sits near eps^(1/3) ~ 6e-6; 1e-4 is a little above that, trading a touch of
#: truncation error for robustness against the pricer's own noise floor, which is the
#: larger effect here. ``tests/test_identifiability.py`` measures the sensitivity.
DEFAULT_FD_STEP: Final[float] = 1e-4


@dataclasses.dataclass(frozen=True, slots=True)
class IdentifiabilityReport:
    r"""The conditioning of one calibration, in transformed coordinates.

    Attributes
    ----------
    eigenvalues
        Eigenvalues of :math:`I`, descending. Figure 2 plots these on a log axis.
    eigenvectors
        Columns are the eigenvectors, in the same order. ``eigenvectors[:, k]`` gives the
        composition of mode ``k`` over :data:`PARAM_NAMES`.
    condition_number
        :math:`\lambda_{\max}/\lambda_{\min}`. The single headline number, though the
        spectrum is the honest version of it.
    fisher_information
        :math:`I` itself, for anyone who wants to do something else with it.
    log_parameters
        Which parameters were analysed in log space.
    """

    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    condition_number: float
    fisher_information: NDArray[np.float64]
    jacobian: NDArray[np.float64]
    params: HestonParams
    log_parameters: tuple[str, ...]
    rho_transform: str

    @property
    def stiffest_direction(self) -> dict[str, float]:
        """Composition of the best-determined mode, by parameter."""
        return self.mode_composition(0)

    @property
    def sloppiest_direction(self) -> dict[str, float]:
        """Composition of the worst-determined mode — the degeneracy, if there is one."""
        return self.mode_composition(len(self.eigenvalues) - 1)

    def mode_composition(self, index: int) -> dict[str, float]:
        """Signed loadings of one eigenvector, by parameter name.

        The sign carries meaning: opposite signs on :math:`\\kappa` and :math:`\\sigma` in
        a sloppy mode say that the data constrains a *ratio* between them, not either one
        separately, which is exactly the classic Heston degeneracy.
        """
        vector = self.eigenvectors[:, index]
        return dict(zip(PARAM_NAMES, (float(v) for v in vector), strict=True))

    def dominant_parameters(self, index: int, threshold: float = 0.3) -> list[str]:
        """Parameters carrying at least ``threshold`` absolute loading in a mode."""
        composition = self.mode_composition(index)
        return [name for name, value in composition.items() if abs(value) >= threshold]

    def to_dict(self) -> dict[str, object]:
        """For the ``CalibrationRun`` JSON column and for experiment output."""
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "condition_number": self.condition_number,
            "log_condition_number": math.log10(max(self.condition_number, 1e-300)),
            "eigenvectors": self.eigenvectors.tolist(),
            "log_parameters": list(self.log_parameters),
            "rho_transform": self.rho_transform,
            "stiffest": self.stiffest_direction,
            "sloppiest": self.sloppiest_direction,
        }


def _transform_scales(params: HestonParams, log_rho: bool) -> NDArray[np.float64]:
    r"""Chain-rule factors converting :math:`\partial/\partial\theta` to the transformed basis.

    For a log-transformed parameter the factor is the parameter value itself. For
    :math:`\rho` left linear it is 1; under Fisher's z-transform it is
    :math:`1 - \rho^2`, the derivative of :math:`\tanh` at the transformed point.
    """
    scales = np.ones(len(PARAM_NAMES), dtype=np.float64)
    for index, name in enumerate(PARAM_NAMES):
        if name in POSITIVE_PARAMS:
            scales[index] = abs(getattr(params, name))
        elif name == "rho" and log_rho:
            scales[index] = 1.0 - params.rho**2
    return scales


def residual_jacobian(
    params: HestonParams,
    slice_: QuoteSlice,
    prior: TikhonovPrior | None = None,
    *,
    step: float = DEFAULT_FD_STEP,
    log_rho: bool = False,
    n_cos: int = cos.N_COS_DEFAULT,
) -> NDArray[np.float64]:
    r"""Central-difference Jacobian of the weighted residuals, in transformed coordinates.

    Shape ``(n_residuals, 5)``. Column :math:`j` is
    :math:`\partial r / \partial z_j` where :math:`z` is the transformed parameter vector.

    Finite differences rather than automatic differentiation, because the residual passes
    through a Brent root-find (the implied-volatility inversion) that has no derivative to
    propagate. The step is taken in the transformed coordinate and mapped back through
    :func:`_transform_scales`, so a step of ``1e-4`` is a relative move of ``1e-4`` for
    every positive parameter regardless of its magnitude — which is the whole reason for
    working in log space.

    Steps that would leave the admissible box are projected back onto it; the resulting
    one-sided difference is slightly less accurate but is preferable to evaluating the
    model at a parameter it is not defined for.
    """
    base = params.to_array()
    scales = _transform_scales(params, log_rho)
    n_residuals = len(residuals(params, slice_, prior, n_cos=n_cos))
    jacobian = np.zeros((n_residuals, len(PARAM_NAMES)), dtype=np.float64)

    for index in range(len(PARAM_NAMES)):
        delta = step * max(scales[index], 1e-12)

        up_values = base.copy()
        up_values[index] += delta
        down_values = base.copy()
        down_values[index] -= delta

        up = clip_to_bounds(HestonParams.from_array(up_values))
        down = clip_to_bounds(HestonParams.from_array(down_values))
        # The realised step after clipping, in the transformed coordinate.
        realised = (up.to_array()[index] - down.to_array()[index]) / max(scales[index], 1e-12)
        if realised <= 0.0:
            continue

        r_up = residuals(up, slice_, prior, n_cos=n_cos)
        r_down = residuals(down, slice_, prior, n_cos=n_cos)
        column = (r_up - r_down) / realised
        jacobian[:, index] = np.where(np.isfinite(column), column, 0.0)

    return jacobian


def fisher_information(
    params: HestonParams,
    slice_: QuoteSlice,
    prior: TikhonovPrior | None = None,
    *,
    step: float = DEFAULT_FD_STEP,
    log_rho: bool = False,
    n_cos: int = cos.N_COS_DEFAULT,
) -> IdentifiabilityReport:
    r"""The Gauss-Newton Fisher information and its spectrum.

    .. math::
        I = J^\top W J

    with :math:`W` already folded into the residuals, so :math:`I = J^\top J`.

    Parameters
    ----------
    log_rho
        Apply Fisher's z-transform to :math:`\rho` instead of leaving it linear. Off by
        default; see the module docstring for the reasoning, and
        ``tests/test_identifiability.py`` for the measurement that the choice moves the
        condition number by a factor of about two and not by an order of magnitude.

    Returns
    -------
    IdentifiabilityReport
        Eigenvalues descending, eigenvectors as columns, condition number.
    """
    jacobian = residual_jacobian(params, slice_, prior, step=step, log_rho=log_rho, n_cos=n_cos)
    information = jacobian.T @ jacobian

    # Symmetric by construction; eigh is both faster and numerically better behaved than
    # the general eigenvalue routine, and guarantees real eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(information)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # eigh can return tiny negative values for a rank-deficient positive-semidefinite
    # matrix. Those are round-off, not curvature; clamping keeps the log axis of figure 2
    # meaningful without pretending the direction is determined.
    eigenvalues = np.maximum(eigenvalues, 0.0)

    largest = float(eigenvalues[0])
    smallest = float(eigenvalues[-1])
    condition = largest / smallest if smallest > 0 else math.inf

    return IdentifiabilityReport(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=condition,
        fisher_information=information,
        jacobian=jacobian,
        params=params,
        log_parameters=POSITIVE_PARAMS,
        rho_transform="fisher_z" if log_rho else "linear",
    )


def finite_difference_hessian(
    params: HestonParams,
    slice_: QuoteSlice,
    prior: TikhonovPrior | None = None,
    *,
    step: float = 1e-3,
    log_rho: bool = False,
    n_cos: int = cos.N_COS_DEFAULT,
) -> NDArray[np.float64]:
    r"""The true Hessian of :math:`L` by second-order central differences, halved.

    The Gauss-Newton form :math:`J^\top J` drops the term
    :math:`\sum_i r_i \nabla^2 r_i`, which vanishes at a perfect fit and is small near a
    good one. This computes the full second derivative so that the approximation can be
    *checked* rather than assumed — the Phase 2 acceptance criterion in `PROJECT_PLAN.md`
    asks for exactly that comparison.

    The factor of one half: :math:`L = \|r\|^2` and
    :math:`\tfrac{1}{2}\nabla^2 L \approx J^\top J`, so the halved Hessian is what is
    comparable to :func:`fisher_information`.

    Expensive — :math:`O(n^2)` objective evaluations — so it is a test and diagnostic tool,
    not part of the production path.
    """
    from voldesk.quant.calibration.objective import objective_value

    n = len(PARAM_NAMES)
    base = params.to_array()
    scales = _transform_scales(params, log_rho)
    steps = step * np.maximum(scales, 1e-12)

    def value_at(offsets: NDArray[np.float64]) -> float:
        candidate = clip_to_bounds(HestonParams.from_array(base + offsets))
        return objective_value(candidate, slice_, prior, n_cos=n_cos)

    zero = np.zeros(n)
    centre = value_at(zero)
    hessian = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = steps[i]
        hessian[i, i] = (value_at(e_i) - 2.0 * centre + value_at(-e_i)) / (
            (steps[i] / scales[i]) ** 2
        )

    for i in range(n):
        for j in range(i + 1, n):
            e_i = np.zeros(n)
            e_i[i] = steps[i]
            e_j = np.zeros(n)
            e_j[j] = steps[j]
            mixed = (
                value_at(e_i + e_j)
                - value_at(e_i - e_j)
                - value_at(-e_i + e_j)
                + value_at(-e_i - e_j)
            ) / (4.0 * (steps[i] / scales[i]) * (steps[j] / scales[j]))
            hessian[i, j] = hessian[j, i] = mixed

    return 0.5 * hessian


@dataclasses.dataclass(frozen=True, slots=True)
class ProfileLikelihood:
    r"""A profile of the objective along one parameter.

    For each value on the grid, the named parameter is **fixed** and every other parameter
    is re-optimised. That is what makes it a profile rather than a slice: it answers "how
    much worse can the fit get if this parameter is forced to that value, given the others
    are free to compensate?" A slice, which holds the others fixed, always looks far more
    curved than the problem really is and would understate the degeneracy badly.

    A flat valley here is the signature of non-identifiability, and `PROJECT_PLAN.md`
    Phase 2 makes finding one for :math:`\kappa` an acceptance criterion: *"if it does not,
    the objective or the scaling is wrong."*
    """

    parameter: str
    grid: NDArray[np.float64]
    objective: NDArray[np.float64]
    best_params: tuple[HestonParams, ...]
    minimum_objective: float

    @property
    def delta_objective(self) -> NDArray[np.float64]:
        """Objective relative to the profile minimum — the shape that matters."""
        return self.objective - self.minimum_objective

    def relative_width(self, tolerance: float = 1.0) -> float:
        r"""Fraction of the swept range within ``tolerance`` of the minimum objective.

        A crude but robust flatness statistic. A well-determined parameter confines the
        objective to a narrow band and returns something small; a degenerate one returns a
        number approaching 1.

        ``tolerance`` is in units of the objective, which is a sum of squared *weighted
        volatility* residuals, so 1.0 corresponds to a change of order one volatility unit
        spread over the slice — enormous, deliberately, so that anything this flags really
        is flat.
        """
        within = self.delta_objective <= tolerance
        return float(np.count_nonzero(within) / len(self.grid))

    def to_dict(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "grid": self.grid.tolist(),
            "objective": self.objective.tolist(),
            "delta_objective": self.delta_objective.tolist(),
            "minimum_objective": self.minimum_objective,
            "relative_width": self.relative_width(),
        }


def profile_likelihood(
    parameter: str,
    centre: HestonParams,
    slice_: QuoteSlice,
    *,
    span: float = 10.0,
    n_points: int = 21,
    prior: TikhonovPrior | None = None,
    config: CalibrationConfig | None = None,
) -> ProfileLikelihood:
    r"""Profile the objective along ``parameter``, re-optimising all the others.

    Parameters
    ----------
    span
        Multiplicative half-width of the sweep for a positive parameter: the grid runs
        from ``centre/span`` to ``centre*span``, logarithmically. For :math:`\rho` the
        sweep is linear across the admissible interval instead.
    n_points
        Grid points. Each one is a bounded local re-optimisation of the other four
        parameters, warm-started from the previous point, so the cost is roughly
        ``n_points`` local fits and not ``n_points`` full calibrations.

    Notes
    -----
    Warm-starting each point from its neighbour is what makes this affordable, and it is
    also what makes the curve smooth: an independent global search at every grid point
    would produce a jagged profile reflecting the search, not the objective.
    """
    if parameter not in PARAM_NAMES:
        raise ValueError(f"unknown parameter {parameter!r}; expected one of {PARAM_NAMES}")

    config = config or CalibrationConfig()
    index = PARAM_NAMES.index(parameter)
    centre_value = getattr(centre, parameter)

    if parameter == "rho":
        grid = np.linspace(max(-0.99, centre_value - 0.4), min(0.45, centre_value + 0.4), n_points)
    else:
        grid = np.exp(
            np.linspace(math.log(centre_value / span), math.log(centre_value * span), n_points)
        )

    objective_values = np.empty(n_points, dtype=np.float64)
    best: list[HestonParams] = []
    warm_start = centre

    for point, value in enumerate(grid):
        fixed = clip_to_bounds(warm_start.replace(**{parameter: float(value)}))
        result = _optimise_with_one_fixed(fixed, index, slice_, prior, config)
        objective_values[point] = result[1]
        best.append(result[0])
        warm_start = result[0]

    return ProfileLikelihood(
        parameter=parameter,
        grid=grid,
        objective=objective_values,
        best_params=tuple(best),
        minimum_objective=float(np.min(objective_values)),
    )


def _optimise_with_one_fixed(
    start: HestonParams,
    fixed_index: int,
    slice_: QuoteSlice,
    prior: TikhonovPrior | None,
    config: CalibrationConfig,
) -> tuple[HestonParams, float]:
    """Re-optimise the four free parameters with one held fixed.

    Implemented by running the ordinary bounded local fit with the fixed parameter pinned
    through a degenerate bound. Reusing the same optimiser as the main pipeline means the
    profile and the fit cannot disagree because they used different machinery.
    """
    from scipy.optimize import least_squares

    from voldesk.quant.calibration.objective import objective_value
    from voldesk.quant.model import bounds_arrays

    lower, upper = bounds_arrays()
    fixed_value = start.to_array()[fixed_index]
    # Pin by collapsing the bound to a point. `trf` handles a zero-width interval by
    # holding the variable, which is exactly the profile's definition.
    lower = lower.copy()
    upper = upper.copy()
    lower[fixed_index] = fixed_value
    upper[fixed_index] = fixed_value

    def residual_vector(x: NDArray[np.float64]) -> NDArray[np.float64]:
        r = residuals(HestonParams.from_array(x), slice_, prior, n_cos=config.n_cos)
        return np.where(np.isfinite(r), r, 1e6)

    try:
        result = least_squares(
            residual_vector,
            x0=np.clip(start.to_array(), lower, upper),
            bounds=(lower, upper),
            method="trf",
            xtol=config.lm_xtol,
            ftol=config.lm_ftol,
            max_nfev=config.lm_max_nfev,
        )
        params = clip_to_bounds(HestonParams.from_array(result.x))
        return params, float(2.0 * result.cost)
    except Exception:  # a profile point that fails is a data point, not a crash
        return start, objective_value(start, slice_, prior, n_cos=config.n_cos)


def objective_grid(
    slice_: QuoteSlice,
    centre: HestonParams,
    parameter_x: str = "kappa",
    parameter_y: str = "sigma",
    *,
    span: float = 6.0,
    n_points: int = 41,
    prior: TikhonovPrior | None = None,
    n_cos: int = cos.N_COS_DEFAULT,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""The objective on a two-parameter grid, everything else held at ``centre``.

    This is the contour data for figure 3. Unlike :func:`profile_likelihood` it does *not*
    re-optimise the other parameters — that is the point of showing both. The contour plot
    displays the raw geometry of the objective surface, and the profile overlaid on it
    shows how much of the apparent curvature disappears once the other parameters are free
    to compensate.

    Returns
    -------
    tuple
        ``(x_grid, y_grid, objective)`` with ``objective`` shaped ``(n_y, n_x)``, ready for
        ``contourf``.
    """
    x_centre = getattr(centre, parameter_x)
    y_centre = getattr(centre, parameter_y)
    x_grid = np.exp(np.linspace(math.log(x_centre / span), math.log(x_centre * span), n_points))
    y_grid = np.exp(np.linspace(math.log(y_centre / span), math.log(y_centre * span), n_points))

    from voldesk.quant.calibration.objective import objective_value

    values = np.empty((n_points, n_points), dtype=np.float64)
    for row, y_value in enumerate(y_grid):
        for column, x_value in enumerate(x_grid):
            candidate = clip_to_bounds(
                centre.replace(**{parameter_x: float(x_value), parameter_y: float(y_value)})
            )
            values[row, column] = objective_value(candidate, slice_, prior, n_cos=n_cos)

    return x_grid, y_grid, values
