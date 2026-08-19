"""Check and prepare the PostgreSQL instance named by ``DATABASE_URL``.

Development runs against a managed PostgreSQL reached over the network; production runs
against a native ``postgresql-16`` on the VPS. See
``docs/adr/0002-development-postgres-is-managed-not-native.md`` and ``docs/DEPLOYMENT.md``.

This script is idempotent and read-mostly. It verifies the three properties the
application actually depends on, rather than assuming them:

1. the connection works at all;
2. the server is PostgreSQL 15 or newer;
3. ``SELECT ... FOR UPDATE SKIP LOCKED`` hands distinct rows to concurrent sessions,
   which is the entire basis of the ``Job`` queue (CLAUDE.md section 8).

Run: ``python scripts/setup_db.py``
"""

from __future__ import annotations

import os
import sys

import psycopg
from dotenv import load_dotenv

PROBE_TABLE = "_voldesk_skiplock_probe"


def _dsn() -> str:
    load_dotenv()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
        raise SystemExit(2)
    return dsn


def check_version(conn: psycopg.Connection) -> None:
    (num,) = conn.execute("SHOW server_version_num").fetchone()  # type: ignore[misc]
    major = int(num) // 10000
    (full,) = conn.execute("SELECT version()").fetchone()  # type: ignore[misc]
    print(f"  server: {full.split(' on ')[0]}")
    if major < 15:
        print(f"  PostgreSQL {major} is too old; 15+ required.")
        raise SystemExit(1)


def check_skip_locked(dsn: str) -> None:
    """Two concurrent sessions must claim two different rows, not block on one."""
    with psycopg.connect(dsn) as setup:
        setup.execute(f"DROP TABLE IF EXISTS {PROBE_TABLE}")
        setup.execute(f"CREATE TABLE {PROBE_TABLE} (id int primary key)")
        setup.execute(f"INSERT INTO {PROBE_TABLE} VALUES (1), (2)")
        setup.commit()

    claim = f"SELECT id FROM {PROBE_TABLE} ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1"
    with psycopg.connect(dsn) as worker_a, psycopg.connect(dsn) as worker_b:
        a = worker_a.execute(claim).fetchone()
        b = worker_b.execute(claim).fetchone()
        worker_a.rollback()
        worker_b.rollback()

    with psycopg.connect(dsn) as teardown:
        teardown.execute(f"DROP TABLE IF EXISTS {PROBE_TABLE}")
        teardown.commit()

    if a == (1,) and b == (2,):
        print("  SELECT ... FOR UPDATE SKIP LOCKED: two workers, two distinct rows.")
        return
    print(f"  SKIP LOCKED did not isolate workers: A got {a}, B got {b}.")
    print("  The Job queue cannot be trusted on this connection — check the pooler mode.")
    raise SystemExit(1)


def main() -> int:
    dsn = _dsn()
    redacted = dsn.split("@")[-1] if "@" in dsn else dsn
    print(f"Connecting to {redacted}")
    try:
        with psycopg.connect(dsn, connect_timeout=15) as conn:
            (user, db) = conn.execute("SELECT current_user, current_database()").fetchone()  # type: ignore[misc]
            print(f"  connected as {user} to {db}")
            check_version(conn)
    except psycopg.OperationalError as exc:
        print(f"  connection failed: {exc}")
        return 1

    check_skip_locked(dsn)
    print("Database is ready. Next: python manage.py migrate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
