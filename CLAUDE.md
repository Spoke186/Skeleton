# CLAUDE.md

Operating instructions for Claude Code on this repository. Read this file fully
before writing any code. If a request in a session conflicts with this file, say so
and ask before proceeding.

---

## 1. What this project is

**VolDesk** — a research + production platform that calibrates the Heston stochastic
volatility model to option-implied volatility surfaces, and characterises the
*identifiability* of that inverse problem.

The scientific contribution is **not** "we calibrated Heston". That is a solved,
uninteresting exercise. The contribution is:

> Quantifying the sloppiness of the Heston calibration problem — measuring which
> parameter combinations are actually determined by option data, which are not, and
> how Tikhonov regularisation trades fit against parameter stability — validated
> against synthetic ground truth where the true parameters are known.

Everything else (Django app, incident engine, monitoring, runbooks) is the
**production operations layer** wrapped around that numerical core.

## 2. Non-negotiable invariants

These are hard rules. Do not violate them, and do not "simplify" them away.

1. **The pricer must be validated before any Django code is written.** Phase 1's
   test suite must be green first. A wrong pricer makes every downstream result
   elegant garbage.
2. **Every calibration persists a full `CalibrationRun` row** — fitted params, seed,
   git SHA, optimizer settings, RMSE, quote counts, Feller ratio, duration, status.
   No exceptions, including failed runs. Failed runs are the most valuable rows in
   the table.
3. **Every filtered/rejected quote records *why*** — a `reject_reason` enum. Silent
   dropping is forbidden.
4. **All figures are produced by `voldesk/figures/`.** No `plt.savefig` anywhere
   else in the codebase, ever. Figures output vector PDF for LaTeX inclusion.
5. **All randomness is seeded and the seed is persisted.** Every experiment must be
   byte-reproducible from its stored config.
6. **No look-ahead, no fitting on test data.** Synthetic ground truth params are
   never visible to the calibrator — only to the evaluator.

## 3. Stack — and what is explicitly banned

**Use:**
- Python 3.12, NumPy, SciPy, Polars, Numba (only where profiling justifies it)
- Django 5 + Django REST Framework
- PostgreSQL 16 (native install, no container)
- Grafana, reading **directly from Postgres** via SQL panels
- Sentry (free tier) for error tracking
- systemd services + systemd timers for scheduling
- Caddy as reverse proxy (automatic HTTPS)
- pytest, hypothesis, ruff, mypy (non-strict), pre-commit
- matplotlib for figures

**Banned — do not introduce these:**
- **Docker / docker-compose.** The dev machine cannot run it. Native installs only.
- **Celery, Redis, RabbitMQ.** Use the `Job` table + a management command polled by
  a systemd timer. One user, one machine — a broker is unjustified infrastructure.
- **Prometheus.** All metrics already live in Postgres tables. Grafana queries them
  with SQL. This is deliberate: the SQL panels are a project deliverable.
- **Any data-ingestion pipeline, cron scraper, or market data API client running on
  a schedule.** See §4.
- **A JavaScript SPA.** The support console is extended Django admin plus at most
  two custom views with server-rendered templates.
- Heavy frameworks added "for later" — dbt, Airflow, Kafka, MLflow.

## 4. Data policy

**Primary data source is synthetic and generated in-repo.** This is a methodological
choice, not a shortcut: with real market data there is no ground truth, so parameter
*recovery* cannot be measured — only fit residual, which is misleading in an
ill-conditioned problem.

The synthetic generator must produce realistic market microstructure:
- bid/ask spread widening with moneyness and shortening maturity
- tick-size discretisation of quoted prices
- a fraction of stale quotes (priced off a lagged spot)
- zero-bid deep OTM options that must be filtered
- occasional crossed quotes (bid > ask) to exercise the quality layer

**Secondary source:** exactly **one** static snapshot of a real QQQ option chain,
downloaded once by a manual script (`scripts/fetch_snapshot_once.py`), committed to
`data/real/qqq_snapshot_<date>.parquet`. It is a fixture, not a pipeline. Do not
build a scheduler around it. Do not re-fetch it automatically.

## 5. Mathematical conventions

### 5.1 Model

Under the risk-neutral measure:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW^S_t
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW^v_t
    d<W^S, W^v>_t = rho dt

Parameter vector: `Theta = (v0, kappa, theta, sigma, rho)`.

Feller condition: `2 * kappa * theta >= sigma^2`. Store `feller_ratio =
2*kappa*theta / sigma^2` on every run. Violation is a **quality signal, not an
exception** — record it, raise an incident, do not crash.

### 5.2 Characteristic function — CRITICAL

Use the **Albrecher et al. (2007) formulation** ("The Little Heston Trap"), never
Heston's original. The original has a complex-logarithm branch cut that produces
discontinuities for long maturities. This is the single highest-risk bug in the
project.

    d  = sqrt((rho*sigma*i*u - kappa)^2 + sigma^2 * (i*u + u^2))
    g2 = (kappa - rho*sigma*i*u - d) / (kappa - rho*sigma*i*u + d)

    C(u,T) = (kappa - rho*sigma*i*u - d) * T
             - 2 * log((1 - g2 * exp(-d*T)) / (1 - g2))

    D(u,T) = ((kappa - rho*sigma*i*u - d) / sigma^2)
             * (1 - exp(-d*T)) / (1 - g2 * exp(-d*T))

    phi(u,T) = exp(i*u*(log(S0) + (r-q)*T) + C*kappa*theta/sigma^2 + D*v0)

The essential property: `Re(d) >= 0`, so `exp(-d*T)` stays bounded and the
expression is numerically stable. Using `g1 = 1/g2` with `exp(+d*T)` is the trap.

**Mandatory test:** sample `phi` on a dense grid of `u` for `T = 5.0` and assert no
discontinuity — `max |phi(u_{k+1}) - phi(u_k)|` must scale with the grid spacing, not
jump.

### 5.3 Pricing — COS method (Fang & Oosterlee, 2008)

Truncation range from cumulants:

    [a, b] = [c1 - L*sqrt(c2 + sqrt(c4)), c1 + L*sqrt(c2 + sqrt(c4))],  L = 12

Default `N_cos = 256`; must be a convergence-tested parameter, not a magic number.

**Price puts and recover calls by put-call parity.** The COS series for calls is
numerically unstable for large maturities. This is a known pitfall.

Reference implementations for cross-checking (Phase 1 only, not in the hot path):
- Carr–Madan FFT
- Monte Carlo with the **Andersen (2008) QE scheme**, switching rule at
  `psi_c = 1.5`, martingale-corrected drift

### 5.4 Implied volatility

Black–Scholes inversion via Brent on `[1e-6, 5.0]`. Handle no-solution cases
(price outside no-arbitrage bounds) by returning `NaN` and recording a reject
reason — never by clamping silently.

### 5.5 Objective function

Fit in **implied volatility space, not price space.** Price-space fitting
overweights expensive ITM options and fits the wings badly, and the wings are what
users care about.

    L(Theta) = sum_i w_i * (iv_obs_i - iv_model_i(Theta))^2
               + lambda * (Theta - Theta_prior)^T M (Theta - Theta_prior)

with `w_i` inversely proportional to the relative bid-ask spread. `M` is a diagonal
scaling matrix making the parameters dimensionally comparable (they differ by orders
of magnitude — do not regularise raw parameter values).

Report RMSE in **volatility basis points** (1 bp = 0.0001 in vol units). Never
report objective values in raw units; they are not interpretable.

### 5.6 Optimisation

Two stages, always in this order:
1. **Differential Evolution** (SciPy) for global search — escapes the local minima
   that plague this objective
2. **Levenberg–Marquardt** (`scipy.optimize.least_squares`, `method='trf'` with
   bounds) for local polish

Parameter bounds:

    v0    in (1e-4, 1.0]
    kappa in (1e-3, 20.0]
    theta in (1e-4, 1.0]
    sigma in (1e-3, 5.0]
    rho   in [-0.999, 0.5]

### 5.7 Identifiability analysis

Use the Gauss–Newton approximation to the Hessian, which is the Fisher information
matrix for Gaussian residuals:

    I = J^T W J

Compute in **log-parameter space** for the positive parameters (`v0, kappa, theta,
sigma`) so the eigenvalue spectrum is scale-invariant and comparable across
parameters. Handle `rho` separately via Fisher's z-transform or leave it linear —
document the choice.

Report:
- full eigenvalue spectrum on a log axis
- eigenvector composition (which parameter combinations are stiff vs. sloppy)
- condition number
- profile likelihood for `kappa` and `sigma` (the canonical degenerate pair)

**L-curve for regularisation:** sweep `lambda` logarithmically, plot residual norm
vs. solution norm on log-log axes, locate the corner by maximum curvature (Hansen's
criterion). Because ground truth is known in synthetic runs, additionally verify
whether the L-corner coincides with the minimum of the *true* parameter error. That
comparison is a genuine result — report it either way.

## 6. Repository layout

    voldesk/
      quant/
        model.py              # HestonParams dataclass, Feller, validation
        charfunc.py           # Albrecher CF
        pricing/
          cos.py              # COS engine (production path)
          carr_madan.py       # FFT reference
          monte_carlo.py      # QE scheme reference
          blackscholes.py     # BS price + Brent IV inversion
        calibration/
          objective.py        # weighted IV residuals + Tikhonov
          optimizers.py       # DE -> LM pipeline
          identifiability.py  # FIM, eigen-decomposition, profile likelihood
          lcurve.py
        synthetic/
          generator.py        # ground-truth surfaces + microstructure noise
      quality/
        arbitrage.py          # monotonicity, butterfly, calendar, parity, intrinsic
        quotes.py             # crossed, zero-bid, wide-spread, stale
        rules.py              # check -> incident rule mapping
      experiments/
        e1_recovery.py
        e2_sloppiness.py
        e3_lcurve.py
        e4_cross_validation.py
      figures/                # THE ONLY PLACE plt.savefig MAY APPEAR
      apps/                   # Django project
        core/                 # models, Job queue, management commands
        api/                  # DRF endpoints
        support/              # support console views
        incidents/            # rule engine, severity, SLA
      scripts/
        fetch_snapshot_once.py
      docs/
        adr/  runbooks/  kb/  postmortems/
      deploy/
        systemd/  caddy/  grafana/
      tests/
      paper/                  # LaTeX article

## 7. Testing gates

Phase 1 cannot be signed off until all of these pass:

| Test | Assertion |
|---|---|
| BS limit | `sigma=1e-4, v0=theta` reproduces Black–Scholes to `< 1e-6` relative |
| Put-call parity | exact to `< 1e-10` across the whole grid |
| CF continuity | no jump discontinuity in `phi(u)` for `T = 5.0` |
| COS vs MC-QE | agreement within 3 MC standard errors, 200k paths, antithetic |
| COS vs Carr–Madan | agreement to `< 1e-8` |
| COS convergence | error decays with `N_cos`; document the achieved rate |
| Noiseless recovery | all 5 params recovered to `< 1e-3` relative error |
| Positivity | prices satisfy intrinsic-value and no-arbitrage bounds everywhere |

Property-based tests (hypothesis) for arbitrage checks: generate random surfaces,
assert the checker flags known-bad ones and passes known-good ones.

Target coverage on `voldesk/quant/`: **> 90%**. Coverage elsewhere: pragmatic.

## 8. Django layer

Core models (`apps/core/models.py`):

- `Experiment` — named study, config JSON, git SHA
- `CalibrationRun` — see invariant #2; the single most important table
- `QuoteSlice` — a (maturity, strike-grid) block with its provenance
- `RejectedQuote` — quote + `reject_reason` enum
- `SurfacePublication` — the artefact consumers read
- `Job` — `kind`, `payload`, `status`, `attempts`, `locked_at`, `last_error`
- `Incident` — `rule_code`, `severity`, FK to run, `root_cause`, timestamps
- `Ticket` — customer-facing, `severity`, `sla_due_at`, `status`, linked incident
- `QualityCheckResult`

**Job queue pattern:** `SELECT ... FOR UPDATE SKIP LOCKED` to claim work. Idempotent
by construction — re-running a job with the same config must be a no-op or produce
an identical result. Exponential backoff on retry, dead-letter after N attempts.

**Incident rules** (`apps/incidents/rules.py`) — each is a pure function over a
`CalibrationRun` or surface returning `Incident | None`:

| Code | Condition | Severity |
|---|---|---|
| R001 | optimizer did not converge | P2 |
| R002 | RMSE > 50 vol bps | P2 |
| R003 | Feller ratio < 1.0 | P3 |
| R004 | COS vs MC disagreement > tolerance | P1 |
| R005 | parameter drift z-score > 3 vs. rolling window | P3 |
| R006 | static arbitrage in published surface | P1 |
| R007 | filtered-quote ratio > 30% | P3 |
| R008 | surface staleness > SLA | P2 |

SLA targets by severity: P1 1h, P2 4h, P3 24h, P4 72h. Measure actual MTTR against
these — those numbers go in the README.

**Support console:** given a run ID or a date, show the full timeline — input slice,
rejected quotes with reasons, calibration run, resulting parameters, published
surface, and any linked incidents. Plus a "Explain this" action that renders a
plain-language summary suitable for pasting to a non-technical user.

## 9. Deployment (single VPS, no containers)

Ubuntu 24.04, x86-64 (**not ARM** — Numba/llvmlite wheels are less reliable on
aarch64 and debugging that is wasted time).

    postgresql-16      apt, local socket auth
    voldesk-web        systemd unit, gunicorn on 127.0.0.1:8000
    voldesk-worker     systemd unit, polls Job table
    voldesk-nightly    systemd timer -> enqueues the nightly experiment batch
    caddy              systemd, reverse proxy + automatic TLS
    grafana-server     systemd, Postgres datasource, dashboards as JSON in deploy/

Deployment is a shell script: `git pull`, `pip install -r`, `migrate`,
`collectstatic`, `systemctl restart`. No orchestration.

Grafana dashboards are **SQL panels against the application tables** and are
version-controlled as JSON. Two dashboards: *Calibration Health* and *Support Ops*.

## 10. Working style

- **One branch per phase**, PR per phase with a description of what was validated.
- **Do not start a phase before the previous phase's acceptance criteria pass.**
  Ask if unsure whether they do.
- Prefer a correct, readable NumPy implementation first; add Numba only where a
  profile shows it matters, and keep the pure-Python version as a test oracle.
- Type-hint public functions. Docstrings on all `quant/` functions include the
  literature reference and equation number.
- When a numerical choice is made (truncation range, `N_cos`, DE population size,
  regularisation scaling), write an ADR in `docs/adr/` explaining the trade-off.
- Commits: conventional commits. Reference the phase.

## 11. Honesty constraints

This is a portfolio project. It will be presented as such — a personal research
project, in its own CV section, never mixed with professional experience.

- **Never fabricate results.** Every number in the README and the paper is produced
  by a script in this repo and reproducible from a stored seed.
- If an experiment produces a negative or inconclusive result, report it. A finding
  that the L-curve corner does *not* coincide with minimum true error is a real
  result and more interesting than a tidy one.
- No claims of production scale, user counts, or business impact that did not
  happen. Metrics describe this system's own measured behaviour.