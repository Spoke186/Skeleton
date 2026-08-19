"""Enforce CLAUDE.md invariant 4: ``plt.savefig`` appears only in ``voldesk/figures/``.

Run as a pre-commit hook and in CI. Exits non-zero on any violation.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALLOWED_PREFIX = ROOT / "voldesk" / "figures"
PATTERN = re.compile(r"\bsavefig\s*\(")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".ruff_cache", "artifacts"}


def main() -> int:
    violations: list[str] = []
    for path in ROOT.rglob("*.py"):
        if SKIP_DIRS & set(path.parts):
            continue
        if path.is_relative_to(ALLOWED_PREFIX) or path == pathlib.Path(__file__):
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if PATTERN.search(line):
                violations.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")

    if violations:
        print("plt.savefig found outside voldesk/figures/ (CLAUDE.md invariant 4):")
        for v in violations:
            print(f"  {v}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
