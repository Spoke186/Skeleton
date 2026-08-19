# PROJECT_PLAN.md

Phased execution plan for VolDesk. Read `CLAUDE.md` first — it holds the invariants
and mathematical conventions that constrain every phase below.

**Rule: a phase is not started until the previous phase's acceptance criteria pass.**
Each phase is one branch and one PR.

---

## Phase 0 — Foundation

*Target: half a day*

**Build**
- Repo skeleton per `CLAUDE.md` §6, `pyproject.toml`, pinned `requirements.txt`
- Native Postgres 16 setup script (no container), local `.env` handling
- `ruff` + `mypy` + `pre-commit` config
- GitHub Actions: lint, type-check, `pytest` with coverage report
- `ARCHITECTURE.md` with a Mermaid diagram of the data and control flow
- `docs/adr/0001-record-architecture-decisions.md`

**Acceptance**
- `pytest` runs green on an empty suite; CI passes on a clean checkout
- `ruff check .` and `mypy voldesk/` clean

---

## Phase 1 — Quant core (offline, no Django)

*Target: 3–4 days. This is the load-bearing phase.*

**Build**
- `quant/model.py` — `HestonParams` dataclass, bounds validation, `feller_ratio`
- `quant/charfunc.py` — Albrecher CF exactly as specified in `CLAUDE.md` §5.2
- `quant/pricing/blackscholes.py` — BS price/greeks, Brent IV inversion with
  no-solution handling
- `quant/pricing/cos.py` — COS engine, cumulant-based truncation, put pricing with
  parity recovery for calls, vectorised over strikes
- `quant/pricing/carr_madan.py` — FFT reference implementation
- `quant/pricing/monte_carlo.py` — Andersen QE scheme, `psi_c = 1.5`, martingale
  correction, antithetic variates, returns standard errors
- `quant/synthetic/generator.py` — ground-truth surface generator with the
  microstructure noise model described in `CLAUDE.md` §4
- Full test suite per `CLAUDE.md` §7

**Acceptance — all eight gates in `CLAUDE.md` §7 pass.** Additionally:
- Benchmark recorded: wall-clock to price a 200-option surface via COS. If this is
  above ~50 ms, profile before continuing; the experiments depend on it.
- Coverage on `voldesk/quant/pricing/` above 90%

**Do not proceed if the CF continuity test or the COS-vs-MC test fails.** Those two
failing means the pricer is wrong, and everything downstream would be invalid.

---

## Phase 2 — Calibration and identifiability

*Target: 2 days*

**Build**
- `calibration/objective.py` — spread-weighted IV residuals, Tikhonov term with
  diagonal scaling matrix `M`, RMSE reported in vol basis points
- `calibration/optimizers.py` — Differential Evolution → Levenberg–Marquardt
  pipeline, bounded, seeded, returning a full result record (not just the point
  estimate)
- `calibration/identifiability.py` — Gauss–Newton FIM in log-parameter space,
  eigen-decomposition, condition number, eigenvector composition, profile
  likelihood for `(kappa, sigma)`
- `calibration/lcurve.py` — logarithmic `lambda` sweep, Hansen maximum-curvature
  corner detection

**Acceptance**
- Noiseless synthetic recovery: all five parameters within `< 1e-3` relative error
- FIM eigenvalues verified against finite-difference Hessian on a small case
- Identical seed ⇒ byte-identical calibration output
- Profile likelihood for `kappa` reproduces a visibly flat valley (the expected
  degeneracy) — if it does not, the objective or the scaling is wrong

---

## Phase 3 — Numerical experiments and figures

*Target: 2 days*

**Build**
- `experiments/e1_recovery.py` — sweep over noise level × grid density, N seeds
  each; output bias and variance per parameter
- `experiments/e2_sloppiness.py` — eigenvalue spectrum and eigenvector composition
  across market regimes (calm / stressed / steep-skew parameter sets)
- `experiments/e3_lcurve.py` — regularisation sweep; compare L-corner against the
  true-error minimum, which is knowable here because ground truth exists
- `experiments/e4_cross_validation.py` — COS vs. Carr–Madan vs. MC-QE; convergence
  in `N_cos`, `dt`, `n_paths`
- `figures/` module producing all eight figures listed below as vector PDF

**The eight figures**

| # | Figure |
|---|---|
| 1 | Estimated vs. true parameter, per parameter, with error bars across seeds |
| 2 | FIM eigenvalue spectrum (log axis) + eigenvector composition bars |
| 3 | Objective contours in the `(kappa, sigma)` plane + profile likelihood |
| 4 | L-curve with detected corner and true-error minimum overlaid |
| 5 | Convergence: error vs. `N_cos`, vs. `dt`, vs. `n_paths` |
| 6 | Fitted smile vs. real QQQ snapshot, with residuals in vol bps |
| 7 | 3D implied volatility surface (cover figure) |
| 8 | Feller-violation frequency vs. noise level |

**Acceptance**
- All experiments reproducible end-to-end from stored config + seed
- Every figure regenerates from a single `make figures`
- `plt.savefig` appears nowhere outside `figures/`

---

## Phase 4 — Django: persistence, API, support console

*Target: 2–3 days*

**Build**
- Models per `CLAUDE.md` §8, with migrations
- `Job` queue: `SELECT ... FOR UPDATE SKIP LOCKED`, idempotent handlers,
  exponential backoff, dead-letter after N attempts
- Management commands: `run_worker`, `enqueue_nightly`, `run_experiment`
- Backfill: retro-load Phase 3 experiment results into `CalibrationRun`
- DRF endpoints: surface by date, IV by (strike, maturity), parameter time series,
  run detail
- `quality/` module wired in: arbitrage and quote checks producing
  `QualityCheckResult` and `RejectedQuote` rows
- `incidents/rules.py` — the eight rules R001–R008 with severity and SLA
- Support console: run/date timeline view (input slice → rejections with reasons →
  calibration run → parameters → published surface → incidents) plus the
  "Explain this" plain-language summary action

**Acceptance**
- A deliberately corrupted synthetic slice flows end-to-end: ingest fixture →
  quality checks reject with reasons → calibration degrades → rule fires → incident
  opens with correct severity → visible in the console timeline
- Re-running an identical job is a verified no-op
- Worker survives a killed process mid-job without leaving orphaned locks

---

## Phase 5 — VPS deployment

*Target: 1 day*

**Build**
- `deploy/systemd/` — `voldesk-web`, `voldesk-worker`, `voldesk-nightly` (timer)
- `deploy/caddy/Caddyfile` — reverse proxy, automatic TLS
- `deploy/deploy.sh` — pull, install, migrate, collectstatic, restart
- Grafana: Postgres datasource, two dashboards exported as JSON into
  `deploy/grafana/` — *Calibration Health* and *Support Ops*
- Sentry integration for both web and worker
- `docs/DEPLOYMENT.md` — bare-metal setup from a fresh Ubuntu 24.04 image

**Dashboard panels (all SQL against application tables)**
- Calibration success rate over time; median and p95 RMSE in vol bps
- Parameter time series with drift bands
- Feller violation rate
- Filtered-quote ratio by maturity bucket
- Open incidents by severity; MTTR vs. SLA target
- Surface freshness

**Acceptance**
- Nightly timer fires and produces new rows without manual intervention
- Dashboards render from live VPS data
- A deliberate exception surfaces in Sentry
- Full deploy from a clean VPS reproducible by following `DEPLOYMENT.md` alone

---

## Phase 6 — Operations documentation and chaos

*Target: 1–2 days*

**Build**
- `scripts/chaos.py` — fault injector: corrupted slice, arbitrage-violating
  maturity, pathological parameter region causing non-convergence, forced COS/MC
  divergence, stale surface
- `docs/runbooks/` — at minimum:
  - `calibration-failed-to-converge.md`
  - `feller-condition-violated.md`
  - `cos-mc-reconciliation-mismatch.md`
  - `parameter-drift-alert.md`
  - `surface-stale.md`
- `docs/kb/` — plain-language articles for non-technical readers:
  - `why-did-kappa-change-so-much.md` — the flat-valley degeneracy explained without
    mathematics. This is the flagship article; take it seriously.
  - `what-does-a-surface-under-review-mean.md`
  - `why-are-some-quotes-excluded.md`
- `docs/INCIDENT_RESPONSE.md` — severity matrix, escalation path, SLA table,
  communication templates
- `docs/postmortems/` — two blameless postmortems from real injected faults,
  written from actual timeline data in the database

**Acceptance**
- Each runbook was actually followed to resolve its injected fault, and says so
- Postmortem timelines cite real row IDs and timestamps

---

## Phase 7 — Paper and presentation

*Target: 2 days*

**Build**
- `paper/` LaTeX article: notation table; formal Definition/Lemma/Theorem/Proof
  environments for the CF derivation and the Feller boundary classification; orange
  physical-interpretation boxes; blue deliverable boxes per section; prose strictly
  separated from displayed equations
- Structure: model and CF derivation → Fokker–Planck and boundary classification →
  the inverse problem and its conditioning → numerical methods → results (E1–E4) →
  operational layer → conclusions
- `scripts/report_metrics.py` — queries the live database and emits the measured
  operational metrics
- README: cover figure, problem statement, key findings, reproduction instructions,
  dashboard screenshots, honest scope statement

**Acceptance**
- Every number in the paper and README traces to a script and a seed
- Article compiles clean; figures are the Phase 3 vector PDFs, unmodified
- README states plainly that this is a personal research project using synthetic
  data plus one static real snapshot

---

## Sequencing and cuts

If time runs short, cut in this order: Phase 6 down to two runbooks and one KB
article, then Phase 7 down to README only.

**Phases 1–3 are never cut.** They are the project. Phases 4–6 are what make it an
argument for a Customer Data Engineer role rather than only a numerics exercise.

## Open decision, to settle at the start of Phase 2

Per-slice calibration (one maturity at a time, independent parameters) or joint
calibration across the whole surface?

Recommended: **per-slice for Phases 2–3**, because it makes the identifiability
analysis cleaner and the degeneracy easier to visualise. Then add joint calibration
in Phase 3 as a comparison — it yields an extra figure showing how information from
multiple maturities partially breaks the degeneracy, which is a real finding.