# Status

Where the build is, what has actually been verified, and what the next session needs to
know. Updated at the end of each phase. `PROJECT_PLAN.md` says what each phase contains;
this file says which of it is done.

Last updated: 2026-08-19, after Phase 2's core modules landed.

---

## Phase state

| Phase | State | Acceptance |
|---|---|---|
| 0 — Foundation | **complete** | ruff clean, mypy clean, pytest green, CI written |
| 1 — Quant core | **complete** | all eight gates of `CLAUDE.md` section 7 pass |
| 2 — Calibration and identifiability | **core written, gates not yet asserted** | see below |
| 3 — Experiments and figures | not started | |
| 4 — Django layer | not started | Postgres connection already verified |
| 5 — VPS deployment | not started | |
| 6 — Operations docs and chaos | not started | |
| 7 — Paper and presentation | not started | |

### Phase 1 — measured results

All eight gates green, 147 tests, 97% coverage on `voldesk/quant` (95.6% on
`voldesk/quant/pricing`, against a 90% CI gate).

| Gate | Measured |
|---|---|
| Black-Scholes limit | 5.2e-08 absolute, 1.0e-07 relative above a 0.1 price floor |
| Put-call parity | < 1e-14 |
| CF continuity at T = 5, 10, 20 | max increment scales linearly with grid spacing |
| COS vs Carr-Madan quadrature | 4e-14 to 4e-10 across the calibration region |
| COS vs MC-QE, 200k antithetic paths | every strike within 1.9 standard errors |
| COS convergence | 1.8e0 → 1.5e-08 from `n_cos` 16 to 256 |
| Noiseless recovery | see Phase 2 — this one moved, deliberately |
| Positivity / no-arbitrage bounds | holds to 1e-10 across the grid |

Benchmark: 200-option COS surface in **8.0 ms** (`PROJECT_PLAN.md` budget: ~50 ms).

### Phase 2 — what is done and what is not

**Written and pushed:** `objective.py`, `optimizers.py`, `identifiability.py`,
`lcurve.py`, `provenance.py`. Lint and type checks clean.

**Not done:** `tests/test_calibration.py` and `tests/test_identifiability.py` do not exist.
The four Phase 2 acceptance criteria have been *measured* but are not yet asserted by the
suite, so Phase 2 is not signed off:

1. noiseless recovery to < 1e-3 — measured, and the result changed the gate (ADR 0007);
2. FIM eigenvalues against a finite-difference Hessian — `finite_difference_hessian()`
   exists, the comparison has not been run;
3. identical seed ⇒ byte-identical output — not yet asserted;
4. profile likelihood for `kappa` showing a flat valley — not yet run.

---

## The finding so far

Recorded in full in `docs/adr/0007-per-slice-calibration-is-under-determined.md`.

Per-slice noiseless recovery is exact at T = 0.5 and T = 1.0 (relative errors of 1e-7 to
1e-10) and fails by up to a factor of **23** at T = 0.25 and T = 2.0 — while producing an
RMSE of **0.26 volatility basis points**, roughly a fiftieth of a bid-ask spread.

Increasing the global search budget twentyfold improves the objective by 13% and makes the
parameter error twice as bad. That is a flat direction seen from the inside, not a starved
optimizer. The objective at the true parameters is exactly 0 and at the fitted point
1.4e-08, so recovering the parameters would require resolving objective differences no
real data could carry.

A single smile does not determine five parameters. The recovery gate therefore runs on a
**joint** fit across maturities (`SurfaceSlices`), and the per-slice result is a headline
measurement rather than a caveat.

This gives experiment E2 a concrete hypothesis to test: the Fisher information spectrum
should show a far larger condition number at T = 0.25 and T = 2.0 than at T = 0.5. If the
eigenvalue analysis does not predict where recovery actually failed, the eigenvalue
analysis is not measuring what it claims to.

---

## Immediate next steps

1. **Finish the joint-fit search budget.** The tuning sweep for `CalibrationConfig` on
   joint fits was still running when the session ended. Known so far: at the per-slice
   default (40 generations, popsize 12) joint recovery reaches 1e-8 for the `stressed`
   regime but only 4e-02 to 2e-01 for `calm` and `steep_skew`, with RMSE of 3 to 16 vol
   bps on noiseless data — that is an unconverged search, not a degeneracy, since the
   objective at the truth is 0. A larger budget is needed for joint fits than for slices.
2. **Write the Phase 2 test suite** and assert the four criteria above.
3. **Phase 3**: experiments E1–E4 and the eight figures as vector PDF.

---

## Environment facts a fresh session needs

Most of these cost time to rediscover.

**Python.** Conda env `Quant`, Python 3.12.13 at
`C:\Users\Esteb\anaconda3\envs\Quant\python.exe`. All dependencies installed.

**Database.** Supabase project `voldesk-dev`, ref `vyzlquqiyohatlsqibck`, PostgreSQL 17.6,
free tier. Credentials are in `.env` (git-ignored; the repo is public). The direct host
`db.<ref>.supabase.co` **does not resolve** from this machine on either address family —
the connection goes through the session-mode pooler on
`aws-0-us-east-1.pooler.supabase.com:5432` with the username form `voldesk_app.<ref>`.
`aws-1` is a different tenant and rejects the login. `scripts/setup_db.py` verifies the
connection and that `SELECT ... FOR UPDATE SKIP LOCKED` hands two concurrent workers two
different rows — already confirmed working. See ADR 0002.

**Real data.** `data/real/qqq_snapshot_2026-08-19.parquet` is committed: 9,315 quotes, 31
expiries, spot 716.08. Figure 6 is unblocked. Do not re-fetch it (`CLAUDE.md` section 4).

**No LaTeX.** Nothing in the `pdflatex`/`xelatex`/`tectonic` family is installed. Phase 7
needs one before "the article compiles clean" can be claimed. Plan: install `tectonic`.

**No `make`.** The `Makefile` targets are one-line wrappers over Python entrypoints so the
same work runs here (ADR 0003).

**Console encoding.** The Windows console is cp1252 and will raise `UnicodeEncodeError` on
any non-ASCII output. Prefix scripts with `PYTHONIOENCODING=utf-8`.

**Background jobs.** Use `python -u`. A 20-minute tuning sweep was lost to stdout
buffering when its timeout killed it before anything flushed.

---

## Working agreements for this project

Decided with the user at the start; they depart from `CLAUDE.md` where noted.

- **Scope**: all seven phases.
- **Git**: commits go **directly to `main`**. This departs from `CLAUDE.md` section 10,
  which asks for one branch and one PR per phase.
- **Postgres**: managed instance in development, native `postgresql-16` in production
  (ADR 0002).

## What cannot be verified from here

Stated so it is never claimed:

1. **Phase 5 on a real VPS** — the nightly timer firing, Grafana rendering live, Sentry
   receiving an exception. The files can be written and every dashboard SQL query can be
   run against the database, which validates the panels; the machine cannot be simulated.
2. **Dashboard screenshots** for the README (Phase 7) — needs a live Grafana.
3. Anything requiring the LaTeX toolchain until it is installed.
