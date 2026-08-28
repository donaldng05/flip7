"""Tests for experiment configuration loading."""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from flip7.config.loader import load_config


def test_load_config_reads_yaml_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\nplayers: 4\n", encoding="utf-8")

    assert load_config(path) == {"seed": 7, "players": 4}


def test_defaults_config_uses_baseline_player_count() -> None:
    config = load_config(Path("configs/defaults.yaml"))

    assert config["players"] == 3
    assert config["env"] == {
        "observation": "competitive",
        "reward": "sparse_win",
        "learner_id": 0,
    }


def test_phase6_config_declares_the_six_condition_matrix() -> None:
    config = load_config(Path("configs/phase6.yaml"))
    conditions = cast(list[object], config["conditions"])

    assert isinstance(conditions, list)
    typed_conditions = [
        cast(Mapping[str, object], condition) for condition in conditions
    ]
    assert [condition["name"] for condition in typed_conditions] == [
        "basic_fixed",
        "basic_random",
        "competitive_fixed",
        "competitive_random",
        "deck_aware_fixed",
        "deck_aware_random",
    ]


def test_phase7_config_declares_control_and_population_conditions() -> None:
    config = load_config(Path("configs/phase7.yaml"))
    conditions = cast(list[object], config["conditions"])
    typed_conditions = [
        cast(Mapping[str, object], condition) for condition in conditions
    ]

    assert [condition["name"] for condition in typed_conditions] == [
        "baseline_control",
        "league_mixed",
        "latest_only",
    ]


def test_phase7_stability_config_declares_balanced_conditions_and_fallback() -> None:
    config = load_config(Path("configs/phase7-follow-up-stability.yaml"))
    screening = cast(Mapping[str, object], config["screening"])
    conditions = [
        cast(Mapping[str, object], condition)
        for condition in cast(list[object], screening["conditions"])
    ]

    assert [condition["name"] for condition in conditions] == [
        "balanced_control",
        "balanced_latest_only",
        "balanced_temporal",
        "balanced_response_diverse",
    ]
    assert config["fallback"] == {
        "name": "seat_aware_response_diverse",
        "observation": "seat_aware",
        "enabled": True,
    }


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        load_config(path)
