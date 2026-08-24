"""Tests for experiment configuration loading."""

from pathlib import Path

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


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.yaml")


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="top-level mapping"):
        load_config(path)
