# ADR 0002 — Development Postgres is managed; production stays native

**Status:** accepted
**Date:** 2026-08-19

## Context

`CLAUDE.md` section 3 specifies **PostgreSQL 16, native install, no container**, and
section 9 specifies `postgresql-16` installed by apt on the production VPS with local
socket authentication.

The development machine is Windows 11. It has no PostgreSQL installed, no `psql` on
`PATH`, and cannot run Docker (which is banned anyway). Development of Phase 4 — the
`Job` queue built on `SELECT ... FOR UPDATE SKIP LOCKED` — cannot proceed against
SQLite, which has no such construct and no row-level locking at all. Substituting
SQLite would mean the concurrency behaviour that the phase exists to demonstrate is
never actually exercised.

## Decision

Development and testing run against a **managed PostgreSQL instance** reached over the
network via `DATABASE_URL`. Production on the VPS is unchanged: native `postgresql-16`
from apt, local socket, exactly as section 9 specifies.

The application code contains nothing specific to either. It reads `DATABASE_URL` and
talks to PostgreSQL.

## Consequences

- `SELECT ... FOR UPDATE SKIP LOCKED`, advisory locks, and the transactional semantics
  the `Job` queue relies on are exercised against real PostgreSQL, so Phase 4's
  acceptance criteria mean what they claim to mean.
- The managed instance reports engine version 17 rather than 16. `SKIP LOCKED` has been
  available since 9.5 and nothing in this project uses a 17-only feature, so the
  difference is not load-bearing. It is nonetheless a difference between development
  and production, and it is recorded here rather than left to be discovered.
- Development requires network access to the database. Offline work is limited to
  Phases 1–3, which are pure numerics and touch no database at all — this is by design,
  the quant core has no Django dependency.
- CI uses its own PostgreSQL 16 service (see ADR 0004), which matches production's major
  version. So the version skew exists only on the development machine.

## Addendum — connection path

The managed instance's direct host (`db.<ref>.supabase.co`) does not resolve from the
development machine on either address family. The connection therefore goes through the
provider's **session-mode pooler** on port 5432, which resolves over IPv4.

Session mode, not transaction mode, is required: transaction pooling returns a different
backend per statement, which would break `SELECT ... FOR UPDATE` holding a lock across
the claim-and-update sequence of the `Job` queue.

`scripts/setup_db.py` verifies this rather than assuming it — it opens two concurrent
connections and asserts they claim two *different* rows. Measured on this connection:

```
server: PostgreSQL 17.6
SELECT ... FOR UPDATE SKIP LOCKED: two workers, two distinct rows.
```

Production on the VPS connects over a local Unix socket and has no pooler in the path.
