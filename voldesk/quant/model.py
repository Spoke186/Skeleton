r"""The Heston model: parameter vector, bounds, and the Feller condition.

Under the risk-neutral measure (CLAUDE.md section 5.1):

.. math::
    dS_t &= (r - q) S_t\,dt + \sqrt{v_t}\, S_t\, dW^S_t \\
    dv_t &= \kappa(\theta - v_t)\,dt + \sigma \sqrt{v_t}\, dW^v_t \\
    d\langle W^S, W^v\rangle_t &= \rho\, dt

Reference: Heston, S. (1993), "A Closed-Form Solution for Options with Stochastic
Volatility with Applications to Bond and Currency Options", *Review of Financial
Studies* 6(2), 327-343, equations (1)-(2).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

#: Calibration bounds, CLAUDE.md section 5.6. Order matches :meth:`HestonParams.to_array`.
PARAM_NAMES: Final[tuple[str, ...]] = ("v0", "kappa", "theta", "sigma", "rho")

PARAM_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "v0": (1e-4, 1.0),
    "kappa": (1e-3, 20.0),
    "theta": (1e-4, 1.0),
    "sigma": (1e-3, 5.0),
    "rho": (-0.999, 0.5),
}

#: Parameters that are strictly positive and are therefore analysed in log space, so that
#: the Fisher information spectrum is scale-invariant across parameters differing by
#: orders of magnitude (CLAUDE.md section 5.7).
POSITIVE_PARAMS: Final[tuple[str, ...]] = ("v0", "kappa", "theta", "sigma")


class ParameterOutOfBoundsError(ValueError):
    """A parameter falls outside the admissible region of CLAUDE.md section 5.6."""


@dataclasses.dataclass(frozen=True, slots=True)
class HestonParams:
    r"""The parameter vector :math:`\Theta = (v_0, \kappa, \theta, \sigma, \rho)`.

    Immutable by construction: a calibration result that could be mutated in place is a
    reproducibility hazard.

    Parameters
    ----------
    v0
        Initial instantaneous variance :math:`v_0`, in variance units, not volatility.
    kappa
        Mean-reversion speed :math:`\kappa` of the variance process.
    theta
        Long-run variance level :math:`\theta`.
    sigma
        Volatility of variance :math:`\sigma`, the "vol of vol".
    rho
        Correlation :math:`\rho` between the spot and variance Brownian motions.
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    # ------------------------------------------------------------------ properties
    @property
    def feller_ratio(self) -> float:
        r""":math:`2\kappa\theta / \sigma^2`.

        The Feller condition :math:`2\kappa\theta \ge \sigma^2`, i.e. ratio >= 1, is what
        keeps the variance process strictly positive; below 1 the origin is attainable.

        CLAUDE.md section 5.1 is explicit that a violation is a **quality signal, not an
        exception**. It is recorded on every run and raises incident R003. It never
        crashes anything: the model remains well defined, the variance process simply
        touches zero, and the pricer handles that.
        """
        return 2.0 * self.kappa * self.theta / (self.sigma**2)

    @property
    def satisfies_feller(self) -> bool:
        r"""Whether :math:`2\kappa\theta \ge \sigma^2` holds."""
        return self.feller_ratio >= 1.0

    # ------------------------------------------------------------------ conversion
    def to_array(self) -> NDArray[np.float64]:
        """As a length-5 array ordered by :data:`PARAM_NAMES`."""
        return np.array([self.v0, self.kappa, self.theta, self.sigma, self.rho], dtype=np.float64)

    @classmethod
    def from_array(
        cls, values: NDArray[np.float64] | list[float] | tuple[float, ...]
    ) -> HestonParams:
        """Inverse of :meth:`to_array`."""
        arr = np.asarray(values, dtype=np.float64)
        if arr.shape != (5,):
            raise ValueError(f"expected 5 parameters ordered {PARAM_NAMES}, got shape {arr.shape}")
        return cls(*(float(x) for x in arr))

    def to_dict(self) -> dict[str, float]:
        """As a plain dict, for the JSON columns on ``CalibrationRun``."""
        return dataclasses.asdict(self)

    def replace(self, **changes: float) -> HestonParams:
        """A copy with some fields changed. Used by finite-difference Jacobians."""
        return dataclasses.replace(self, **changes)

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        """Raise :class:`ParameterOutOfBoundsError` if any parameter is outside its bounds.

        Deliberately *not* called from ``__post_init__``. The optimizer legitimately
        evaluates points on the boundary, and a Fisher-information finite difference
        steps just outside it; raising there would turn a numerical detail into a crash.
        Validation is applied where it means something: at the entry to a calibration and
        when a result is persisted.
        """
        for name in PARAM_NAMES:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ParameterOutOfBoundsError(f"{name} is not finite: {value!r}")
            low, high = PARAM_BOUNDS[name]
            if not (low <= value <= high):
                raise ParameterOutOfBoundsError(
                    f"{name}={value!r} outside admissible bounds [{low}, {high}] "
                    "(CLAUDE.md section 5.6)"
                )

    def is_within_bounds(self) -> bool:
        """Whether :meth:`validate` would pass."""
        try:
            self.validate()
        except ParameterOutOfBoundsError:
            return False
        return True

    def __str__(self) -> str:
        return (
            f"HestonParams(v0={self.v0:.6f}, kappa={self.kappa:.6f}, theta={self.theta:.6f}, "
            f"sigma={self.sigma:.6f}, rho={self.rho:.6f}, feller={self.feller_ratio:.3f})"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class MarketState:
    r"""The deterministic market inputs that sit outside :math:`\Theta`.

    Parameters
    ----------
    spot
        Underlying spot price :math:`S_0`.
    rate
        Continuously compounded risk-free rate :math:`r`.
    dividend
        Continuous dividend yield :math:`q`.
    """

    spot: float
    rate: float = 0.0
    dividend: float = 0.0

    def forward(self, maturity: float) -> float:
        r""":math:`F = S_0 e^{(r-q)T}`."""
        return self.spot * math.exp((self.rate - self.dividend) * maturity)

    def discount(self, maturity: float) -> float:
        r""":math:`e^{-rT}`."""
        return math.exp(-self.rate * maturity)

    def validate(self) -> None:
        """Raise :class:`ValueError` on a non-finite or non-positive market input."""
        if not (self.spot > 0 and math.isfinite(self.spot)):
            raise ValueError(f"spot must be positive and finite, got {self.spot!r}")
        for name in ("rate", "dividend"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} is not finite: {value!r}")


def bounds_arrays() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Lower and upper bound vectors ordered by :data:`PARAM_NAMES`.

    Shaped for ``scipy.optimize.least_squares(bounds=...)`` and for the
    ``differential_evolution`` bounds list.
    """
    lower = np.array([PARAM_BOUNDS[n][0] for n in PARAM_NAMES], dtype=np.float64)
    upper = np.array([PARAM_BOUNDS[n][1] for n in PARAM_NAMES], dtype=np.float64)
    return lower, upper


def clip_to_bounds(params: HestonParams) -> HestonParams:
    """Project a parameter vector onto the admissible box.

    Used only where a projection is the mathematically correct thing to do, for example
    when a finite-difference step would leave the box. It is never used to hide an
    out-of-bounds calibration result: those are persisted as they came out, with a failed
    status. CLAUDE.md invariant 2 makes the failed rows the valuable ones.
    """
    lower, upper = bounds_arrays()
    return HestonParams.from_array(np.clip(params.to_array(), lower, upper))
