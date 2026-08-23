"""Smoke tests for the installed package structure."""

import importlib

import flip7


def test_package_exposes_expected_version() -> None:
    assert flip7.__version__ == "0.1.0"


def test_expected_subpackages_import() -> None:
    package_names = (
        "flip7.agents",
        "flip7.core",
        "flip7.envs",
        "flip7.evaluation",
        "flip7.training",
    )

    assert all(importlib.import_module(name) for name in package_names)
