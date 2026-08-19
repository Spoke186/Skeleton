"""Shared fixtures and parameter sets for the Phase 1 test suite.

The parameter sets below are the ones every gate is measured on. They are named and
shared so that "the reference case" means the same thing in every test file and in the
numbers quoted in ADRs and in the README.
"""

from __future__ import annotations

import numpy as np
import pytest

from voldesk.quant.model import HestonParams, MarketState

#: A well-behaved calibration-region parameter set. Feller ratio 2.0, so the variance
#: process stays strictly positive and every method is in its comfortable regime.
REFERENCE = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=0.3, rho=-0.7)

#: Feller ratio 0.72 — violated, which is the common case in real calibrations and
#: therefore the case the pricer has to survive.
FELLER_VIOLATING = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=0.5, rho=-0.7)

#: Feller ratio 0.18. Deliberately outside the region any of these methods is accurate in;
#: used to *measure* where accuracy degrades rather than to pretend it does not.
EXTREME = HestonParams(v0=0.04, kappa=1.5, theta=0.06, sigma=1.0, rho=-0.7)

#: The vol-of-vol -> 0 limit, with zero correlation. See ``test_cos.py`` for why rho must
#: be zero for the Black-Scholes limit to be clean.
BS_LIMIT = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=1e-4, rho=0.0)

MATURITIES = (0.1, 0.5, 1.0, 2.0, 5.0)


@pytest.fixture
def market() -> MarketState:
    """Spot 100, 2% rate, 1% dividend yield."""
    return MarketState(spot=100.0, rate=0.02, dividend=0.01)


@pytest.fixture
def strikes() -> np.ndarray:
    """A five-point strike ladder spanning -20% to +25% moneyness."""
    return np.array([80.0, 90.0, 100.0, 110.0, 125.0])


@pytest.fixture
def wide_strikes() -> np.ndarray:
    """A 21-point ladder out to +-35%, the grid the synthetic surfaces use."""
    return np.linspace(65.0, 135.0, 21)


@pytest.fixture
def params() -> HestonParams:
    """The reference parameter set."""
    return REFERENCE
