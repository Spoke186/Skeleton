"""Provenance stamps: what produced a result, and from what.

CLAUDE.md invariant 2 requires every ``CalibrationRun`` to carry the git SHA alongside the
fitted parameters and the seed, and invariant 5 requires every experiment to be
byte-reproducible from its stored config. Neither is satisfiable if the code version is
not recorded, so recording it is not optional and must not be able to fail.

This module has no dependency beyond the standard library, so the numerical layer can
stamp a result without reaching for the web layer.
"""

from __future__ import annotations

import functools
import pathlib
import subprocess

#: Returned when the code is not running from a git checkout — an installed wheel, a
#: tarball, a notebook. An explicit sentinel rather than an empty string, so that a run
#: with unknown provenance is visible as such in the database instead of looking like a
#: missing field.
UNKNOWN_SHA = "unknown"


@functools.cache
def git_sha(short: bool = False) -> str:
    """The commit the running code came from, or :data:`UNKNOWN_SHA`.

    Never raises. A missing git, a detached worktree, or a non-repository directory all
    produce the sentinel rather than an exception: failing to stamp provenance must not be
    able to fail a calibration, or the invariant that *every* run is persisted (including
    failed ones) would be the first casualty.

    Cached, because it is stamped on every run and cannot change within a process.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN_SHA
    if result.returncode != 0:
        return UNKNOWN_SHA
    return result.stdout.strip() or UNKNOWN_SHA


@functools.cache
def git_is_dirty() -> bool | None:
    """Whether the working tree has uncommitted changes, or ``None`` if unknowable.

    A result produced from a dirty tree is not reproducible from its SHA, and saying so is
    the difference between a reproducibility claim that holds and one that merely looks
    like it does.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=pathlib.Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def provenance() -> dict[str, object]:
    """The full stamp, for the JSON column on a run."""
    return {"git_sha": git_sha(), "git_dirty": git_is_dirty()}
