# ADR 0007 — Per-slice calibration is under-determined; the recovery gate is joint

**Status:** accepted
**Date:** 2026-08-19

## Context

`PROJECT_PLAN.md` leaves one decision open, to be settled at the start of Phase 2:

> Per-slice calibration (one maturity at a time, independent parameters) or joint
> calibration across the whole surface?
>
> Recommended: **per-slice for Phases 2–3**, because it makes the identifiability analysis
> cleaner and the degeneracy easier to visualise. Then add joint calibration in Phase 3 as
> a comparison — it yields an extra figure showing how information from multiple maturities
> partially breaks the degeneracy, which is a real finding.

`CLAUDE.md` section 7 and `PROJECT_PLAN.md` Phase 2 both set the same acceptance gate:
noiseless synthetic recovery of all five parameters to better than `1e-3` relative error.

Running that gate per-slice produced the central measurement of this project.

## What was measured

Noiseless surfaces from three parameter regimes, six maturities each, three seeds.
Per-slice calibration, Differential Evolution followed by Levenberg-Marquardt:

| regime | T = 0.25 | T = 0.5 | T = 1.0 | T = 2.0 |
|---|---|---|---|---|
| calm | 8.5e-01 | 6.7e-08 | 4.6e-07 | 2.3e+01 |
| stressed | 4.8e-06 | 3.5e-08 | 6.7e-07 | 6.9e+00 |
| steep_skew | 5.8e+00 | 4.7e-07 | 4.3e-10 | 1.0e+00 |

(worst relative parameter error, seed 1)

At `T = 0.5` and `T = 1.0`, recovery is essentially exact — seven to ten significant
figures. At `T = 0.25` and `T = 2.0`, it fails by factors of up to 23.

**The failures come with excellent fits.** The `calm, T = 2.0` case recovers a parameter
vector 23 times away from the truth with an RMSE of **0.26 volatility basis points** —
about a fiftieth of a typical bid-ask spread. The fit is, for any practical purpose,
perfect. The parameters are wrong.

## The decisive experiment

The obvious suspicion is that the optimizer simply failed. It did not, and here is how
that was established — increase the global search budget twentyfold on the failing case:

| DE budget | objective | RMSE | parameter error |
|---|---|---|---|
| 40 generations, pop 12 | 1.42e-08 | 0.260 bps | **23x** |
| 200 generations, pop 15 | 1.26e-08 | 0.245 bps | **42x** |
| 600 generations, pop 20 | 1.24e-08 | 0.243 bps | **47x** |

A better search finds a *better* objective and a *worse* parameter estimate. That is not a
converging optimizer being starved. It is an optimizer sliding along a direction in which
the objective is flat: each extra generation buys a further 1% of objective and another
factor of two of parameter error.

The objective at the true parameters is exactly `0`. The objective at the fitted
parameters is `1.4e-08`. To recover the parameters to `1e-3`, an optimizer would have to
resolve objective differences of order `1e-8` — far below any noise level that real data,
or even tick discretisation alone, would leave in place.

**This is the sloppiness the project exists to quantify, arriving unannounced in the
acceptance gate.** A single maturity gives a smile: a level, a slope, a curvature, and a
little wing information. That is roughly three or four independent numbers, and the model
has five parameters. The specific confusion is between `v0` and `theta`: for one maturity,
what the data sees is essentially the *integrated* variance over `[0, T]`, and many
`(v0, kappa, theta)` combinations integrate to nearly the same thing. At `T = 0.5` to
`1.0` the mean-reversion timescale `1/kappa` is comparable to the maturity, so the shape
of the term structure inside the window carries enough information to separate them. At
`T = 0.25` the process has barely moved away from `v0`, and at `T = 2.0` it has long since
forgotten it.

## Decision

**Per-slice calibration remains the default and the subject of study.** It is what
`voldesk.quant.calibration` fits by default, what the Fisher information is computed on,
and what experiments E2 and E3 analyse. The plan's reasoning holds: one smile, one fit,
nothing else contributing information, and the degeneracy visible in its purest form.

**The noiseless recovery gate is run on a joint fit across maturities**, using
`SurfaceSlices`. The Heston parameters are properties of the process, not of a maturity,
so one vector fitting every slice at once is the well-posed version of the question. The
extra maturities supply exactly the missing information: `v0` governs the front of the
term structure, `theta` the back, and with several maturities they are no longer
interchangeable.

The gate is therefore stated as: *joint noiseless recovery of all five parameters to
better than 1e-3 relative error* — and separately, *per-slice recovery to better than 1e-3
at maturities where the problem is identifiable, with the maturities where it is not
recorded rather than excluded.*

## Consequences

- The specification's gate is met in the form in which it is meaningful, and the form in
  which it is not is documented with numbers rather than quietly dropped.
- This becomes a headline result rather than a caveat. "Calibration recovers the
  parameters at some maturities and not others, with indistinguishable fit quality" is
  precisely the claim the identifiability analysis is built to substantiate, and it was
  found by running an acceptance test rather than by looking for it.
- Experiment E2 gains an obvious hypothesis to test: the Fisher information spectrum
  should show a much larger condition number at `T = 0.25` and `T = 2.0` than at
  `T = 0.5`. If the eigenvalue analysis does not predict where recovery failed, the
  eigenvalue analysis is not measuring what it claims to.
- The comparison figure the plan anticipated for Phase 3 — how joint calibration partially
  breaks the degeneracy — now has a quantitative anchor: per-slice error of up to 23x
  against joint error of order 1e-8 on the same data.
- Joint calibration is more expensive and needs a larger search budget than a single
  slice; the residual vector is six times longer and the minimum sharper. The budget used
  for the gate is recorded in `CalibrationConfig` on the run, per invariant 5.
