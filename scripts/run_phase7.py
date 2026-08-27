"""Run the reproducible Phase 7 league and tournament experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import cast

from flip7 import __version__
from flip7.agents import PPOAgent
from flip7.config.loader import load_config
from flip7.envs import ObservationFamily
from flip7.evaluation import (
    heldout_factories,
    run_rotated_matchups,
    run_round_robin,
    summarize_rotated_results,
    write_tournament,
)
from flip7.evaluation.phase7 import TournamentParticipant
from flip7.training import (
    LeagueConfig,
    LeaguePPOTrainer,
    PolicySnapshot,
    PPOConfig,
    PPOTrainer,
    baseline_factories,
    write_history,
    write_population,
)

MANIFEST_PATH_KEYS = (
    "checkpoint",
    "training_history",
    "population",
    "evaluation",
    "tournament",
    "manifest",
)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    values = _list(value, name)
    if not all(isinstance(item, str) for item in values):
        raise ValueError(f"{name} must contain only strings")
    return tuple(cast(str, item) for item in values)


def _integers(value: object, name: str) -> tuple[int, ...]:
    values = _list(value, name)
    if not all(isinstance(item, int) for item in values):
        raise ValueError(f"{name} must contain only integers")
    return tuple(cast(int, item) for item in values)


def _matchups(value: object, name: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(_list(value, name)):
        pair = _strings(item, f"{name}[{index}]")
        if len(pair) != 2:
            raise ValueError(f"each {name} entry must contain two opponents")
        pairs.append((pair[0], pair[1]))
    if not pairs:
        raise ValueError(f"{name} must not be empty")
    return tuple(pairs)


def _config_for_condition(
    root: Mapping[str, object], condition: Mapping[str, object], seed: int
) -> PPOConfig:
    environment = _mapping(root["env"], "env")
    training = _mapping(root["training"], "training")
    config_values: dict[str, object] = dict(training)
    config_values.update(
        {
            "seed": seed,
            "player_count": int(root["players"]),
            "learner_id": int(environment["learner_id"]),
            "learner_seat_mode": str(environment["learner_seat_mode"]),
            "observation": str(environment["observation"]),
            "reward": str(environment["reward"]),
        }
    )
    config_values.pop("algorithm", None)
    if "trainer" not in condition:
        raise ValueError("condition is missing trainer")
    return PPOConfig(**config_values)


def _league_config(
    root: Mapping[str, object], condition: Mapping[str, object]
) -> LeagueConfig:
    population = _mapping(condition["population"], "condition.population")
    opponents = _mapping(root["opponents"], "opponents")
    values: dict[str, object] = dict(population)
    values["baseline_names"] = _strings(opponents["training"], "opponents.training")
    return LeagueConfig(**values)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_manifest(manifest: Mapping[str, object]) -> None:
    """Reject an incomplete manifest before reporting a run as complete."""
    for key in MANIFEST_PATH_KEYS:
        value = manifest.get(key)
        if not isinstance(value, str) or not Path(value).is_file():
            raise ValueError(f"artifact manifest is missing a file for {key}")


def _rating(result: Mapping[str, object], name: str) -> float | None:
    elo = _mapping(result["elo"], "tournament.elo")
    rows = _list(elo["ratings"], "tournament.elo.ratings")
    for row_value in rows:
        row = _mapping(row_value, "tournament.elo.rating")
        if row["name"] == name:
            return float(row["rating"])
    return None


def _participant_for_snapshot(snapshot: PolicySnapshot) -> TournamentParticipant:
    return TournamentParticipant(
        snapshot.policy_id,
        lambda path=snapshot.path: PPOAgent.from_checkpoint(path, deterministic=True),
        snapshot.observation,
    )


def _aggregate_rotated_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Pool rotated summaries before calculating the cross-seed seat spread."""
    if not summaries:
        raise ValueError("cannot aggregate empty rotated summaries")
    total_games = sum(int(summary["games"]) for summary in summaries)
    pooled_win_share = (
        sum(
            float(summary["win_share"]) * int(summary["games"]) for summary in summaries
        )
        / total_games
    )
    per_seat: dict[str, float] = {}
    for seat in range(3):
        seat_key = str(seat)
        per_seat[seat_key] = (
            sum(
                float(
                    _mapping(
                        _mapping(summary["per_seat"], "rotated per-seat summary")[
                            seat_key
                        ],
                        "rotated seat summary",
                    )["win_share"]
                )
                * int(summary["games"])
                for summary in summaries
            )
            / total_games
        )
    return {
        "games": total_games,
        "win_share": pooled_win_share,
        "seat_spread": max(per_seat.values()) - min(per_seat.values()),
        "per_seat_win_share": per_seat,
    }


def _run_seed(
    root: Mapping[str, object],
    condition: Mapping[str, object],
    seed: int,
    output_root: Path,
) -> dict[str, object]:
    condition_name = str(condition["name"])
    run_dir = output_root / condition_name / f"seed-{seed}"
    checkpoint = run_dir / "checkpoint.pt"
    history_path = run_dir / "training.json"
    population_path = run_dir / "population.json"
    evaluation_path = run_dir / "evaluation.json"
    tournament_path = run_dir / "tournament.json"
    manifest_path = run_dir / "manifest.json"

    config = _config_for_condition(root, condition, seed)
    trainer_name = str(condition["trainer"])
    if trainer_name == "ppo":
        trainer: PPOTrainer | LeaguePPOTrainer = PPOTrainer(
            config,
            opponent_names=_strings(
                _mapping(root["opponents"], "opponents")["training"],
                "opponents.training",
            ),
        )
    elif trainer_name == "league":
        trainer = LeaguePPOTrainer(
            config, league_config=_league_config(root, condition)
        )
    else:
        raise ValueError(f"unsupported Phase 7 trainer: {trainer_name}")

    history = trainer.train(checkpoint)
    write_history(history_path, history)
    if isinstance(trainer, LeaguePPOTrainer):
        write_population(population_path, trainer.league)
        snapshots = trainer.league.snapshots
    else:
        _write_json(
            population_path,
            {
                "config": {"mode": "baseline_control"},
                "observation": config.observation,
                "source_seed": seed,
                "snapshots": [],
                "opponent_exposure": {},
            },
        )
        snapshots = ()

    evaluation = _mapping(root["evaluation"], "evaluation")
    games_per_seat = int(evaluation["games_per_seat"])
    seed_bases = _integers(evaluation["seed_bases"], "evaluation.seed_bases")
    matchups = _matchups(evaluation["matchups"], "evaluation.matchups")
    heldout_matchup_values = evaluation.get("heldout_matchups")
    if heldout_matchup_values is None:
        raise ValueError("evaluation.heldout_matchups is required")
    heldout_matchup_values = _matchups(
        heldout_matchup_values, "evaluation.heldout_matchups"
    )
    heldout_seed_values = _integers(
        evaluation.get("heldout_seed_bases", []),
        "evaluation.heldout_seed_bases",
    )
    if len(heldout_seed_values) != len(heldout_matchup_values):
        raise ValueError("one held-out evaluation seed base is required per matchup")
    observation = ObservationFamily(config.observation)

    def policy() -> PPOAgent:
        return PPOAgent.from_checkpoint(checkpoint, deterministic=True)

    baseline_roster = baseline_factories()
    rotated_results = run_rotated_matchups(
        policy,
        baseline_roster,
        games=games_per_seat,
        seed_bases=seed_bases,
        observation=observation,
        matchups=matchups,
    )
    heldout_roster = baseline_roster | heldout_factories()
    heldout_results = run_rotated_matchups(
        policy,
        heldout_roster,
        games=games_per_seat,
        seed_bases=heldout_seed_values,
        observation=observation,
        matchups=heldout_matchup_values,
    )

    tournament_values = _mapping(root["tournament"], "tournament")
    participants = [
        TournamentParticipant(name, factory, ObservationFamily.DECK_AWARE)
        for name, factory in baseline_roster.items()
    ]
    participants.append(TournamentParticipant("final", policy, observation))
    participants.extend(_participant_for_snapshot(snapshot) for snapshot in snapshots)
    tournament = run_round_robin(
        participants,
        games=int(tournament_values["games_per_lineup"]),
        seed=int(tournament_values["seed"]) + seed,
        initial_rating=float(tournament_values["initial_rating"]),
        k_factor=float(tournament_values["k_factor"]),
    )
    write_tournament(tournament_path, tournament)

    baseline_summary = summarize_rotated_results(rotated_results)
    heldout_summary = summarize_rotated_results(heldout_results)
    final_rating = _rating(tournament.as_dict(), "final")
    warmup_rating = (
        _rating(tournament.as_dict(), snapshots[0].policy_id) if snapshots else None
    )
    metadata: dict[str, object] = {
        "experiment": str(root["experiment"]),
        "condition": condition_name,
        "trainer": trainer_name,
        "seed": seed,
        "package_version": __version__,
        "config": asdict(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "training_history": str(history_path),
        "population": str(population_path),
        "evaluation": str(evaluation_path),
        "tournament": str(tournament_path),
        "manifest": str(manifest_path),
        "observation": config.observation,
        "learner_seat_mode": config.learner_seat_mode,
        "baseline_matchups": [list(pair) for pair in matchups],
        "heldout_matchups": [list(pair) for pair in heldout_matchup_values],
        "heldout_seed_bases": list(heldout_seed_values),
        "summary": {
            "baseline_rotated": baseline_summary,
            "heldout_rotated": heldout_summary,
            "tournament_games": tournament.games,
            "final_rating": final_rating,
            "warmup_rating": warmup_rating,
            "snapshot_count": len(snapshots),
        },
    }
    evaluation_payload: dict[str, object] = dict(metadata)
    evaluation_payload["matchups"] = [
        result.as_dict() for result in rotated_results + heldout_results
    ]
    evaluation_payload["summary"] = {
        "baseline_rotated": baseline_summary,
        "heldout_rotated": heldout_summary,
    }
    _write_json(evaluation_path, evaluation_payload)
    _write_json(manifest_path, metadata)
    validate_artifact_manifest(metadata)
    return {
        "seed": seed,
        "trainer": trainer_name,
        "baseline": baseline_summary,
        "heldout": heldout_summary,
        "final_rating": final_rating,
        "warmup_rating": warmup_rating,
        "snapshot_count": len(snapshots),
        "artifacts": {
            "checkpoint": str(checkpoint),
            "evaluation": str(evaluation_path),
            "tournament": str(tournament_path),
            "manifest": str(manifest_path),
        },
    }


def _run_condition(
    root: Mapping[str, object],
    condition: Mapping[str, object],
    seeds: Sequence[int],
    output_root: Path,
) -> dict[str, object]:
    seed_summaries = [_run_seed(root, condition, seed, output_root) for seed in seeds]
    return _aggregate_condition(condition, seeds, seed_summaries)


def _aggregate_condition(
    condition: Mapping[str, object],
    seeds: Sequence[int],
    seed_summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline_summaries = [
        cast(Mapping[str, object], summary["baseline"]) for summary in seed_summaries
    ]
    heldout_summaries = [
        cast(Mapping[str, object], summary["heldout"]) for summary in seed_summaries
    ]
    baseline_aggregate = _aggregate_rotated_summaries(baseline_summaries)
    heldout_aggregate = _aggregate_rotated_summaries(heldout_summaries)
    final_ratings = [
        float(summary["final_rating"])
        for summary in seed_summaries
        if summary["final_rating"] is not None
    ]
    warmup_deltas = [
        float(summary["final_rating"]) - float(summary["warmup_rating"])
        for summary in seed_summaries
        if summary["final_rating"] is not None and summary["warmup_rating"] is not None
    ]
    return {
        "trainer": str(condition["trainer"]),
        "seeds": list(seeds),
        "seed_summaries": seed_summaries,
        "aggregate": {
            "baseline_rotated": {
                **baseline_aggregate,
            },
            "heldout_rotated": {
                **heldout_aggregate,
            },
            "final_rating_median": median(final_ratings) if final_ratings else None,
            "warmup_delta_median": median(warmup_deltas) if warmup_deltas else None,
            "warmup_delta_min": min(warmup_deltas) if warmup_deltas else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/phase7.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/phase7"))
    args = parser.parse_args()

    root = _mapping(load_config(args.config), "configuration")
    seeds = _integers(root["seeds"], "seeds")
    conditions = _list(root["conditions"], "conditions")
    condition_results: dict[str, object] = {}
    for index, condition_value in enumerate(conditions):
        condition = _mapping(condition_value, f"conditions[{index}]")
        if "name" not in condition:
            raise ValueError(f"conditions[{index}] is missing name")
        condition_results[str(condition["name"])] = _run_condition(
            root, condition, seeds, args.output_root
        )

    reference = _mapping(root["phase6_reference"], "phase6_reference")
    reference_win_share = float(reference["pooled_win_share"])
    reference_spread = float(reference["seat_spread"])
    control = _mapping(condition_results["baseline_control"], "baseline_control")
    league = _mapping(condition_results["league_mixed"], "league_mixed")
    latest = _mapping(condition_results["latest_only"], "latest_only")
    control_baseline = _mapping(control["aggregate"], "baseline aggregate")[
        "baseline_rotated"
    ]
    league_aggregate = _mapping(league["aggregate"], "league aggregate")
    latest_aggregate = _mapping(latest["aggregate"], "latest aggregate")
    control_rotated = _mapping(control_baseline, "control rotated")
    league_rotated = _mapping(league_aggregate["baseline_rotated"], "league rotated")
    league_heldout = _mapping(league_aggregate["heldout_rotated"], "league heldout")
    latest_heldout = _mapping(latest_aggregate["heldout_rotated"], "latest heldout")
    league_warmup_delta_min = league_aggregate["warmup_delta_min"]
    gates = {
        "control_reproduces_phase6": (
            abs(float(control_rotated["win_share"]) - reference_win_share) <= 0.01
            and abs(float(control_rotated["seat_spread"]) - reference_spread) <= 0.01
        ),
        "league_robustness": (
            float(league_rotated["win_share"]) >= 0.604
            and float(league_rotated["seat_spread"]) <= 0.05
        ),
        "population_protection": (
            float(league_heldout["win_share"])
            >= float(latest_heldout["win_share"]) - 0.05
            and float(league_heldout["win_share"])
            >= float(latest_heldout["win_share"]) + 0.03
        ),
        "tournament_adaptation": (
            league_warmup_delta_min is not None
            and float(league_warmup_delta_min) >= 25.0
            and float(league_aggregate["final_rating_median"]) >= 1500.0
        ),
    }
    summary = {
        "experiment": str(root["experiment"]),
        "reference": {
            "phase6_pooled_win_share": reference_win_share,
            "phase6_seat_spread": reference_spread,
        },
        "conditions": condition_results,
        "gates": gates,
    }
    _write_json(args.output_root / "summary.json", summary)
    print(f"wrote summary: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
