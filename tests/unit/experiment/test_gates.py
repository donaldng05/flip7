"""Unit tests for flip7.experiment.gates."""

from flip7.experiment.gates import (
    GateConfig,
    evaluate_phase7_gates,
    is_candidate_robust,
)


def test_gate_config_defaults():
    gates = GateConfig.from_config({})
    assert gates.min_baseline_win_share == 0.604
    assert gates.max_seat_spread == 0.05
    assert gates.min_heldout_difference == 0.03
    assert gates.max_seed_degradation == -0.02
    assert gates.min_final_rating == 1500.0
    assert gates.min_final_minus_warmup_elo == 25.0


def test_gate_config_from_yaml_mapping():
    root = {
        "gates": {
            "min_baseline_win_share": 0.65,
            "max_seat_spread": 0.04,
            "min_warmup_gain": 30.0,
        }
    }
    gates = GateConfig.from_config(root)
    assert gates.min_baseline_win_share == 0.65
    assert gates.max_seat_spread == 0.04
    assert gates.min_final_minus_warmup_elo == 30.0


def test_evaluate_phase7_gates():
    gates = GateConfig(
        min_baseline_win_share=0.60,
        max_seat_spread=0.05,
        min_heldout_difference=0.03,
        min_final_rating=1500.0,
        min_final_minus_warmup_elo=25.0,
    )
    control_rotated = {"win_share": 0.65, "seat_spread": 0.02}
    league_rotated = {"win_share": 0.62, "seat_spread": 0.04}
    league_heldout = {"win_share": 0.70}
    latest_heldout = {"win_share": 0.65}
    league_aggregate = {"final_rating_median": 1550.0}

    result = evaluate_phase7_gates(
        control_rotated=control_rotated,
        league_rotated=league_rotated,
        league_heldout=league_heldout,
        latest_heldout=latest_heldout,
        league_warmup_delta_min=30.0,
        league_aggregate=league_aggregate,
        reference_win_share=0.65,
        reference_spread=0.02,
        gates=gates,
    )
    assert result["control_reproduces_phase6"] is True
    assert result["league_robustness"] is True
    assert result["population_protection"] is True
    assert result["tournament_adaptation"] is True


def test_is_candidate_robust():
    gates = GateConfig(min_baseline_win_share=0.60, max_seat_spread=0.05)
    row_pass = {"baseline": {"win_share": 0.65, "seat_spread": 0.03}}
    row_fail_win = {"baseline": {"win_share": 0.55, "seat_spread": 0.03}}
    row_fail_spread = {"baseline": {"win_share": 0.65, "seat_spread": 0.06}}
    assert is_candidate_robust(row_pass, gates) is True
    assert is_candidate_robust(row_fail_win, gates) is False
    assert is_candidate_robust(row_fail_spread, gates) is False
