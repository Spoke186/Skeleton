# ADR 0005 — COS truncation range and the number of cosine terms

**Status:** accepted
**Date:** 2026-08-19

## Context

The COS method has exactly two accuracy knobs, and they are not independent:

- **`L`**, the truncation width, controls where the density is cut off. Too narrow and
  the tails are discarded; the resulting error is a *floor* that no number of series
  terms can clear.
- **`N_cos`**, the number of cosine terms, controls the series error on whatever interval
  `L` chose. It converges exponentially — until it hits the floor `L` set.

`CLAUDE.md` section 5.3 fixes `L = 12`, gives `N_cos = 256` as a default, and requires
`N_cos` be *"a convergence-tested parameter, not a magic number"*. Fang & Oosterlee (2008)
eq. (49) gives the range in terms of the first, second and fourth cumulants of the
log-return.

Three things came out of actually running that convergence test.

## Decision 1 — the fourth cumulant is computed, not set to zero

Fang & Oosterlee give no closed form for `c4` under Heston. Their own remedy is to set it
to zero and widen `L` to compensate.

Measured against a Carr-Madan quadrature reference at `sigma = 0.5, T = 1`:

| `L` | `N_cos = 256` | `N_cos = 1024` | `N_cos = 4096` |
|---|---|---|---|
| 8  | 4.5e-04 | 4.5e-04 | 4.5e-04 |
| 12 | 2.0e-06 | 2.0e-06 | 2.0e-06 |
| 16 | 9.0e-09 | 9.0e-09 | 9.0e-09 |
| 20 | 4.9e-09 | 4.3e-10 | 4.3e-10 |

The error is **flat in `N_cos`** — the signature of a truncation floor rather than a
series error. At the specified `L = 12`, that floor is 2e-6, which is two orders of
magnitude above the 1e-8 cross-validation gate of `CLAUDE.md` section 7. No number of
cosine terms would have cleared it.

Rather than raise `L` past what the specification says, the missing quantity is computed.
`c4` is obtained by differentiating the cumulant generating function `g(u) = log φ(u)`
numerically at the origin with a five-point stencil — five extra characteristic-function
evaluations. With a real `c4`, `L = 12` gives **4.1e-10** on the same case.

## Decision 2 — `c2` is also computed from the characteristic function

While testing the above, the published closed form for `c2` (Table 11), transcribed as
printed, was found to **disagree with the characteristic function by 0.6%** at the
reference parameters.

Three independent references were used to settle which was wrong:

1. **Finite differences of `log φ`**, refined from `h = 0.4` down to `h = 0.003`,
   converging to `0.052996975`;
2. **A hand derivation** of `Var(X_T)` in the case `ρ = 0, v₀ = θ`, where
   `Var(X_T) = E[I_T] + ¼Var(I_T)` with `I_T = ∫v ds` — agreeing to **1e-12**;
3. **Monte Carlo** on the QE scheme, which shares no code with any of the above:
   `Var(log S_T) = 0.05305 ± 0.00008`, agreeing with the numeric value and excluding the
   closed form (`0.052686`) at roughly four standard errors.

All three agree with each other and disagree with the transcribed table. So all cumulants
are now taken from the characteristic function that is actually used. The published `c1`
*does* agree, to nine significant figures, and is kept as
`first_cumulant_closed_form()` purely so that agreement is asserted in a test — that
control is what makes the `c2` disagreement credible rather than a suspicion about the
differentiation.

The stencil width `h = 0.05` is where the finite-difference truncation error and round-off
meet. Measured spread across `h ∈ {0.1, 0.05, 0.02}`: `c1` 2.5e-6, `c2` 2.4e-7,
`c4` 1.4e-3 relative. `c4` is loosest because a fourth derivative from five points is only
second-order accurate; a fraction of a percent on a truncation *width* is inconsequential.

## Decision 3 — the default `N_cos` is 512, not 256

With the truncation floor removed, the series error becomes the binding constraint, and
256 terms are not enough for the parameter regimes this project runs. On the `steep_skew`
regime (Feller ratio 0.25, `ρ = -0.9`, `σ = 0.6`), which is one of the three named regimes
that experiment E2 compares:

| maturity | `N_cos = 256` | `N_cos = 512` | `N_cos = 1024` |
|---|---|---|---|
| 0.5 | 2.7e-04 | 4.0e-07 | 4.9e-07 |
| 1.0 | 1.3e-03 | 5.6e-06 | 4.4e-06 |

A 1e-3 pricing error would dominate the residuals of any calibration in that regime and
sits four orders of magnitude above the cross-validation gate.

The cost was measured rather than assumed: a 200-option surface takes **8.0 ms** at
`N_cos = 512` against **6.4 ms** at 256, comfortably inside the ~50 ms budget in
`PROJECT_PLAN.md` Phase 1. The extra terms are nearly free because the characteristic
function is evaluated once for a whole maturity slice and shared across all strikes.

This is a departure from the number printed in `CLAUDE.md` section 5.3, and it is exactly
the departure that section asks for by requiring the parameter be convergence-tested.

## Consequences

- The COS engine matches an independent Carr-Madan quadrature to between 4e-14 and 4e-10
  across the calibration region, clearing the 1e-8 gate with four orders of magnitude to
  spare.
- Every pricing call pays five extra characteristic-function evaluations for the
  cumulants. At 8 ms per 200-option surface this is not measurable against the series
  itself.
- Accuracy still degrades outside the calibration region. At `σ = 1.0` (Feller ratio
  0.18) the agreement falls to ~2e-6 at one-month maturity even at `N_cos = 1024`. This is
  asserted as a measurement in
  `tests/test_cross_validation.py::test_agreement_degrades_gracefully_at_extreme_vol_of_vol`
  rather than left to be discovered.
- In the deep wing, where the true price is below 1e-6, both COS and the Carr-Madan
  reference return values that wobble around zero and can go slightly negative. That is a
  property of any truncated Fourier method at the edge of the support. It is five orders
  of magnitude below one tick, so it never becomes a quote, and it is deliberately **not**
  clamped inside the pricer — a clamp would make the wing look better behaved than it is,
  in precisely the region this project's conclusions concern.
