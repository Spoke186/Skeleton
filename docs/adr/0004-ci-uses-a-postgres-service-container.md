# ADR 0004 — CI uses a PostgreSQL service container

**Status:** accepted
**Date:** 2026-08-19

## Context

`CLAUDE.md` section 3 bans Docker and docker-compose: *"The dev machine cannot run it.
Native installs only."* The stated reason is a property of the development machine, and
section 9 reinforces it for production, where deployment is a shell script against
systemd units with no orchestration.

CI needs a PostgreSQL to run the Phase 4 test suite. On GitHub Actions the idiomatic way
to get one is a `services:` block, which is implemented with containers.

## Decision

The CI `test` job declares a `postgres:16` service. The ban is read as it is written —
about the development machine and about production — and not extended to an ephemeral
CI runner that exists for the duration of one job and is then destroyed.

This is stated openly in a comment in `.github/workflows/ci.yml` rather than left for a
reader to notice and read as an inconsistency.

## Consequences

- CI tests run against PostgreSQL **16**, matching the production major version, which
  is a better guarantee than the development environment gives (see ADR 0002).
- No container ever runs on the development machine or on the VPS. The project's
  operational story — systemd units, a shell deploy script, native Postgres — is intact.
- If CI ever moves to a runner without container support, the alternative is an apt
  install of `postgresql-16` in a `before` step. Nothing in the test suite would change;
  it only needs a `DATABASE_URL`.
