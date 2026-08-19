"""Structural tests: the layering rule from ARCHITECTURE.md, enforced.

``voldesk/quant/`` is a pure numerical library. It must remain importable with Django
uninstalled — that is what makes CLAUDE.md invariant 1 ("validate the pricer before any
Django code is written") checkable rather than aspirational.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUANT = ROOT / "voldesk" / "quant"

FORBIDDEN_IN_QUANT = ("django", "rest_framework", "voldesk.apps", "psycopg")


def _imported_modules(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("path", sorted(QUANT.rglob("*.py")), ids=lambda p: p.name)
def test_quant_does_not_import_the_web_layer(path: pathlib.Path) -> None:
    imported = _imported_modules(path)
    offending = {
        name
        for name in imported
        for forbidden in FORBIDDEN_IN_QUANT
        if name == forbidden or name.startswith(forbidden + ".")
    }
    assert not offending, (
        f"{path.relative_to(ROOT)} imports {sorted(offending)}. "
        "voldesk/quant/ must stay a pure numerical library — see ARCHITECTURE.md."
    )


def test_savefig_is_confined_to_the_figures_package() -> None:
    """CLAUDE.md invariant 4, checked in the test suite as well as in pre-commit."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_savefig.py")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
