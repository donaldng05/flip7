"""Unit tests for flip7.experiment.io."""

import json
from pathlib import Path
from typing import cast

from flip7.experiment.io import sha256_file, write_json, write_stage_summary


def test_write_json_and_sha256(tmp_path: Path) -> None:
    target = tmp_path / "sub" / "data.json"
    data = {"b": 2, "a": 1}
    write_json(target, data)
    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert content == '{\n  "a": 1,\n  "b": 2\n}\n'

    h1 = sha256_file(target)
    assert isinstance(h1, str)
    assert len(h1) == 64


def test_write_stage_summary(tmp_path: Path) -> None:
    out = tmp_path / "out"
    write_stage_summary(out, "screening", {"experiment": "test_exp", "score": 10})
    screening_path = out / "screening-summary.json"
    summary_path = out / "summary.json"
    assert screening_path.is_file()
    assert summary_path.is_file()

    summary_data = cast(
        dict[str, object], json.loads(summary_path.read_text(encoding="utf-8"))
    )
    assert summary_data["experiment"] == "test_exp"
    assert "screening" in cast(list[object], summary_data["stages"])

    # Write another stage
    write_stage_summary(out, "confirmation", {"experiment": "test_exp", "score": 20})
    summary_data2 = cast(
        dict[str, object], json.loads(summary_path.read_text(encoding="utf-8"))
    )
    stages = cast(list[object], summary_data2["stages"])
    assert "screening" in stages
    assert "confirmation" in stages
