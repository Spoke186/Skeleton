# ADR 0003 — The Makefile wraps Python entrypoints

**Status:** accepted
**Date:** 2026-08-19

## Context

`PROJECT_PLAN.md` Phase 3 requires that every figure regenerates from a single
`make figures`. The production VPS and the CI runner are Ubuntu, where `make` is
available. The development machine is Windows, where it is not.

Writing the build logic inside Makefile recipes would make it unrunnable on the machine
where it is actually developed, and it would put shell quoting between the developer and
the code.

## Decision

Every `Makefile` target is a one-line wrapper over a Python module entrypoint:

```make
figures:
	python -m voldesk.figures.build_all
```

The logic lives in Python. The Makefile is a table of contents.

## Consequences

- `make figures` remains a single command on Linux and in CI, satisfying the Phase 3
  acceptance criterion literally.
- The same work runs on Windows as `python -m voldesk.figures.build_all`, with no
  duplicated logic that could drift.
- Targets are testable: an entrypoint is an importable module with a `main()`, so a test
  can call it directly instead of shelling out.
