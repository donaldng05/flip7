"""Tests that the rules engine stays free of RL framework imports."""

import ast
from pathlib import Path

import flip7.core

FORBIDDEN_MODULES = {"numpy", "gymnasium", "pettingzoo"}


def test_core_sources_do_not_import_rl_libraries() -> None:
    core_dir = Path(flip7.core.__file__).resolve().parent
    imported: set[str] = set()
    for path in core_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(
                    alias.name.split(".", maxsplit=1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module.split(".", maxsplit=1)[0])

    assert imported.isdisjoint(FORBIDDEN_MODULES)
