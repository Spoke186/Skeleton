# ADR 0001 — Record architecture decisions

**Status:** accepted
**Date:** 2026-08-19

## Context

`CLAUDE.md` section 10 requires that every numerical choice — truncation range, `N_cos`,
DE population size, regularisation scaling — is written up with its trade-off. Those
choices are not obvious from reading the code: a reader six months from now sees
`L = 12` and cannot tell whether it was reasoned or copied.

The same applies to the places where this project deliberately departs from its own
specification. An undocumented departure is indistinguishable from a mistake.

## Decision

Architecture decisions are recorded as short Markdown files in `docs/adr/`, numbered
sequentially, in the format of Michael Nygard's original ADR proposal: Context,
Decision, Consequences.

An ADR is written when:

- a numerical parameter is chosen and the choice has a trade-off (accuracy vs. cost,
  stability vs. speed);
- the implementation departs from `CLAUDE.md` or `PROJECT_PLAN.md`;
- a mathematical formulation is chosen over an equivalent-looking alternative that is
  numerically worse (the Albrecher characteristic function is the archetype).

ADRs are immutable once accepted. A decision that is later reversed gets a new ADR that
supersedes the old one; the old file stays, marked superseded.

## Consequences

- The `docs/adr/` directory becomes the honest record of why the numbers are what they
  are, and it is the first thing to read when a result looks surprising.
- There is a small tax on every parameter choice. That tax is the point: it discourages
  magic numbers.
