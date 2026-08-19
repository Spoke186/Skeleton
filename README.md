# VolDesk

Calibration of the Heston stochastic volatility model to option-implied volatility
surfaces, and characterisation of the **identifiability** of that inverse problem.

The contribution is not that Heston can be calibrated — that is a solved exercise. It is
the measurement of *which parameter combinations option data actually determines*, which
it does not, and how Tikhonov regularisation trades fit against parameter stability —
validated against synthetic ground truth where the true parameters are known.

Build status: **Phase 0 complete.** This README is filled in as the phases land; see
`PROJECT_PLAN.md` for what each phase contains.

## Scope, stated plainly

This is a personal research project. Its primary data is **synthetic and generated
in-repo**, plus exactly one static snapshot of a real QQQ option chain used as a fixture.
Synthetic data is a methodological choice, not a shortcut: with real market data there is
no ground truth, so parameter *recovery* cannot be measured — only fit residual, which is
misleading in an ill-conditioned problem.

Every number that appears here or in `paper/` is produced by a script in this repository
and reproducible from a stored seed.

## Getting started

```bash
python -m pip install -r requirements-dev.txt
cp .env.example .env        # then fill in DATABASE_URL and DJANGO_SECRET_KEY
python scripts/setup_db.py  # verifies the connection and the Job-queue locking primitive
pytest
```

## Documentation

| File | What it holds |
|---|---|
| `CLAUDE.md` | Invariants, mathematical conventions, banned dependencies |
| `PROJECT_PLAN.md` | Phase breakdown with acceptance criteria |
| `ARCHITECTURE.md` | Data and control flow, layering rule |
| `docs/adr/` | Why the numbers and the departures are what they are |
