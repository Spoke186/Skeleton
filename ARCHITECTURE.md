# Architecture

VolDesk has two halves that meet at exactly one place.

The **numerical core** (`voldesk/quant/`) is a pure library: NumPy and SciPy, no Django,
no database, no I/O beyond what a caller hands it. It can be imported, tested and
profiled with the web stack uninstalled. This is deliberate — `CLAUDE.md` invariant 1
requires the pricer to be validated before any Django code exists, and that is only
enforceable if the pricer cannot reach for a model.

The **operations layer** (`voldesk/apps/`, `voldesk/quality/`, `deploy/`) wraps that core
in persistence, a job queue, quality gates, an incident engine and dashboards. It calls
into the core; the core never calls back.

## Data and control flow

```mermaid
flowchart TB
    subgraph gen["Synthetic generation — voldesk/quant/synthetic/"]
        TRUTH["Ground-truth Theta<br/>(v0, kappa, theta, sigma, rho)"]
        SURF["Arbitrage-free surface<br/>priced by COS"]
        MICRO["Microstructure noise<br/>spreads · tick discretisation<br/>stale quotes · zero bids · crossed"]
        TRUTH --> SURF --> MICRO
    end

    SNAP[("data/real/qqq_snapshot.parquet<br/>one static fixture, fetched manually")]

    subgraph quality["Quality layer — voldesk/quality/"]
        QCHK["quotes.py<br/>crossed · zero-bid · wide · stale"]
        ARB["arbitrage.py<br/>monotonicity · butterfly<br/>calendar · parity · intrinsic"]
        QCHK --> ARB
    end

    REJ[("RejectedQuote<br/>every drop carries a reject_reason")]
    QCHK -. "rejects" .-> REJ
    ARB -. "rejects" .-> REJ

    MICRO --> QCHK
    SNAP --> QCHK

    subgraph calib["Calibration — voldesk/quant/calibration/"]
        OBJ["objective.py<br/>spread-weighted IV residuals<br/>+ Tikhonov, scaled by M"]
        OPT["optimizers.py<br/>Differential Evolution → Levenberg-Marquardt"]
        IDENT["identifiability.py<br/>FIM = J'WJ in log space<br/>eigenspectrum · profile likelihood"]
        LC["lcurve.py<br/>lambda sweep · Hansen corner"]
        OBJ --> OPT --> IDENT
        OPT --> LC
    end

    ARB --> OBJ

    subgraph price["Pricing — voldesk/quant/pricing/"]
        COS["cos.py — production path<br/>puts priced, calls by parity"]
        CM["carr_madan.py — FFT reference"]
        MC["monte_carlo.py — Andersen QE reference"]
        BS["blackscholes.py — Brent IV inversion"]
    end

    CF["charfunc.py<br/>Albrecher (2007) formulation"]
    CF --> COS
    CF --> CM
    COS <-.->|"cross-check<br/>rule R004"| MC
    COS <-.->|"cross-check"| CM
    OBJ --> COS
    COS --> BS

    RUN[("CalibrationRun<br/>params · seed · git SHA · RMSE<br/>Feller ratio · status — failures included")]
    OPT --> RUN
    IDENT --> RUN

    subgraph ops["Operations — voldesk/apps/"]
        RULES["incidents/rules.py<br/>R001-R008, pure functions"]
        INC[("Incident<br/>severity · SLA")]
        PUB[("SurfacePublication")]
        CONSOLE["support/<br/>timeline view · 'Explain this'"]
        RULES --> INC
        INC --> CONSOLE
        PUB --> CONSOLE
        RUN --> CONSOLE
        REJ --> CONSOLE
    end

    RUN --> RULES
    RUN --> PUB
    PUB --> RULES

    subgraph fig["voldesk/figures/ — the only place plt.savefig may appear"]
        F["8 vector PDFs → paper/"]
    end
    IDENT --> F
    LC --> F
    RUN --> F

    GRAF["Grafana<br/>SQL panels straight at the tables"]
    RUN --> GRAF
    INC --> GRAF
    REJ --> GRAF
```

## Job execution

There is no broker. `CLAUDE.md` section 3 rules out Celery, Redis and RabbitMQ as
unjustified infrastructure for one user on one machine. Work is rows in a `Job` table,
claimed with row-level locks.

```mermaid
sequenceDiagram
    participant T as systemd timer<br/>voldesk-nightly
    participant E as enqueue_nightly
    participant DB as PostgreSQL
    participant W as voldesk-worker
    participant Q as quant core

    T->>E: fires on schedule
    E->>DB: INSERT Job (kind, payload, status=pending)
    loop poll
        W->>DB: SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1
        DB-->>W: one job, locked to this worker
        W->>DB: status=running, locked_at=now()
        W->>Q: calibrate(slice, config, seed)
        alt success
            Q-->>W: fitted params + diagnostics
            W->>DB: CalibrationRun row + status=done
        else failure
            Q-->>W: exception
            W->>DB: CalibrationRun row (status=failed) + attempts+1
            Note over W,DB: exponential backoff;<br/>dead-letter after N attempts.<br/>A failed run is still persisted —<br/>invariant 2 has no exceptions.
        end
        W->>DB: run incident rules R001-R008
    end
```

`SKIP LOCKED` is what makes a second worker safe to start without coordination: it takes
the next unlocked row rather than blocking on the one already claimed. A worker killed
mid-job leaves its transaction uncommitted, so the lock dies with the connection and the
row returns to the pool — which is why Phase 4's acceptance criterion is tested by
actually killing the process.

## Layering rule

`voldesk/quant/` may import: numpy, scipy, and itself.
`voldesk/quality/` may import: numpy, polars, and `voldesk.quant`.
`voldesk/apps/` may import: anything, including both of the above.

Nothing in `voldesk/quant/` imports Django. A test enforces this
(`tests/test_layering.py`) — the alternative is that it decays quietly the first time
someone wants a setting.
