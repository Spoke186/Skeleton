# ADR 0006 — Carr-Madan: FFT for the method, quadrature for the gate

**Status:** accepted
**Date:** 2026-08-19

## Context

`PROJECT_PLAN.md` Phase 1 asks for a Carr-Madan FFT reference implementation, and
`CLAUDE.md` section 7 sets the gate: *"COS vs Carr-Madan — agreement to < 1e-8."*

Those two requirements are in tension, and the tension is a property of the FFT, not of
either pricer.

Carr-Madan's FFT prices a whole lattice of log-strikes in one transform. That is the
point of the method. But the lattice spacing is fixed by `λ = 2π/(N·η)`, and any strike
that is not on the lattice needs interpolation. Linear interpolation on a lattice of
spacing `λ` carries an error of order `λ² ∂²C/∂k²`. With the default grid
(`N = 4096, η = 0.25`) that works out to about **1e-3** on a spot of 100 — five orders of
magnitude above the gate.

Meeting 1e-8 through the FFT would need a lattice so fine that the transform stops being
an FFT in any useful sense. The gate would then be measuring the grid resolution, not
whether the two pricers agree.

## Decision

Two entry points, each doing the job it is good at.

- **`price_call_fft`** is the method as published: Simpson-weighted integrand, one FFT,
  a whole lattice of prices. It is tested against COS at a tolerance set by its
  interpolation error, and that tolerance is stated in the test as being about the grid.
- **`price_call_quadrature`** evaluates the same `ψ_T(v)` integral by adaptive quadrature
  at the exact strike requested. Not an FFT, slower per strike, no grid error. This is
  what the 1e-8 gate runs against.

Two details in the quadrature were needed to actually reach 1e-8, and both were found by
measurement rather than anticipated:

**Oscillatory weights.** The integrand carries `e^{-ivk}`, which oscillates roughly 150
times over the integration range. A plain adaptive rule stalls near 1e-6 and reports a
round-off warning. Splitting into `Re(ψ)cos(vk) + Im(ψ)sin(vk)` and handing each piece to
QUADPACK's trigonometric-weight routines reaches 1e-13.

**A maturity-dependent cutoff.** The integrand inherits the characteristic function's
Gaussian-like decay, whose width in `v` scales like `1/√T`. A fixed cutoff of 400 looks
perfectly converged at `T = 1` and silently costs 8e-8 at `T = 0.1` — enough to make the
*reference* the thing that fails the gate. The rule is now
`v_max = max(500, 1600/√T)`; at `σ = 0.5, T = 0.1` that reduces the discrepancy against a
converged COS price from 8e-8 to 3e-10.

Both were caught because the quadrature was checked for stability against its own
parameters — sweeping `α` from 0.75 to 3.0 and the cutoff from 100 to 20000 — before being
trusted as a reference. The price must not depend on the damping factor at all; that it
does not, to 1e-10, is what makes it a reference rather than a second opinion.

## Consequences

- The gate measures what it claims to: two independent pricing routes agreeing to between
  4e-14 and 4e-10 across the calibration region.
- The FFT implementation is honest about its accuracy. It is the right tool when a whole
  smile is wanted at once and 1e-3 is fine; it is documented as such rather than presented
  as a high-accuracy reference it cannot be.
- The quadrature costs roughly 100 ms per strike, so it is confined to tests and to
  experiment E4. Nothing in the production path calls it.
- A third, fully independent check exists in the Monte Carlo QE scheme, which shares no
  characteristic function, no transform and no cumulants with either. Agreement across all
  three is the actual evidence that the pricer is right; agreement between COS and
  Carr-Madan alone would only show that the shared characteristic function is used
  consistently.
