# Baseline Timing Reference

Status: frozen smoke reference for the `phase-7-resolve` cleanup decision.
This note records wall-clock cost on the current branch before any
refactoring or language-change work. It is not a scientific result and does
not authorize Phase 8.

## Provenance

- Branch: `phase-7-resolve`
- Commit: `80fb731 fix(typing): resolve pyright diagnostics test type casts`
- Config: `configs/phase7-resolve.yaml`
- Date: 2026-09-22 (UTC)
- Device: CPU, Windows, `uv 0.12.1`
- Output root: `artifacts/baseline-timing-smoke` (ignored, not committed)

## Smoke command

```powershell
uv run python scripts/run_phase7_follow_up_stability.py `
  --config configs/phase7-resolve.yaml `
  --stage screening `
  --smoke `
  --skip-full-population-tournament `
  --output-root artifacts/baseline-timing-smoke
```

Smoke budget: 1 seed (`7`), 5 conditions
(`balanced_control`, `balanced_latest_only`, `balanced_temporal`,
`balanced_response_diverse`, plus auto `seat_aware_response_diverse`
fallback), 2 updates, 1 game per seat, focused tournament only.

## Frozen result

- Wall clock: `WALL_SECONDS=61.1` (67s including harness startup)
- Summary: `artifacts/baseline-timing-smoke/screening-summary.json`
- Fallback: required (`seat_aware_response_diverse`), expected on 2-update
  noise; see `artifacts/baseline-timing-smoke/screening/fallback-decision.json`

## Micro-benchmarks (same machine, CPU)

| Piece | Time |
| --- | --- |
| Pure baseline game (`risk`/`threshold`/`random`) | 8ms/game, ~131 games/s |
| Learned-policy game (`basic` PPO + 2 baselines) | 17ms/game, ~58 games/s |
| PPO rollout, 1026 seat-balanced steps | 2.3s (~2ms/learner-step) |
| PPO update, 4 epochs | 0.1s |
| Trainer init | 0.8s |
| State-bank build, 2048 states | 0.2s |

Method: `flip7.evaluation.run_game` over 20 seeded games;
`PPOTrainer` with `basic`, `separate` + seat-conditioned critic;
`flip7.evaluation.build_state_bank(states=2048)`.

## Extrapolation to full screening

Per seed per condition, approximate:

- Training: 100 updates x ~2.4s = ~240s
- Rotated eval (900 baseline + 600 held-out games) = ~20-25s
- Focused tournament (up to 35 lineups x 6 perms x 20 games) = ~50s
  serial, ~15s with 8 workers
- Full tournament if enabled (~11.6k games) = ~175s serial
- Diversity (46 archived x 2048 states, batched) + response signatures
  (46 x 9 games) = ~15-35s
- Paired held-out = ~18s

Total: ~5-7 min per run x 12 runs (4 conditions x 3 seeds) = ~60-85 min
for screening, before fallback conditions and 5-seed confirmation at
200 games per seat.

## Interpretation

Hours come from run fan-out and tournament combinatorics, not per-game
engine speed. A compiled engine alone would cut only the 8-17ms/game
portion (~20-30% of wall-clock). The fast loop is therefore:

1. Smoke-by-default (~61s reference above).
2. `--skip-full-population-tournament` for screening.
3. Parallelize across seeds/conditions (today only tournament games
   parallelize).
4. Full 100-update / 100-200 games-per-seat budgets only for
   confirmation.

## Reproduction

Re-run the smoke command above on `phase-7-resolve` at the recorded
commit. Expect ~1 min wall-clock and 5 conditions in
`screening-summary.json`. Do not compare smoke win shares across
machines; only wall-clock and run completion are frozen here.

## Fast-loop guardrail (frozen reference)

Seat-robustness regression signal for cleanup/speed work, on `main`
post-dedup. Same optimizer, matchups, and seed bases as
`configs/phase6.yaml`; reduced to `basic_random` x seeds `[7, 17]` x
10 updates x 20 games/seat (360 games).

```powershell
uv run python scripts/run_phase6.py `
  --config artifacts/guardrail/guardrail.yaml `
  --output-root artifacts/guardrail/run1
```

Frozen result (`WALL_SECONDS=52.6`, 2026-09-22, CPU/Windows):

| Seed | Win share | 95% CI | Spread | Seat 0/1/2 | Final | Bust |
| ---: | ---: | --- | ---: | --- | ---: | ---: |
| 7 | 57.5% | [50.3, 64.7] | 6.7pp | 53.3/59.2/60.0% | 198.4 | 15.9% |
| 17 | 69.7% | [63.0, 76.4] | 10.8pp | 63.3/74.2/71.7% | 201.2 | 13.8% |
| pooled | 63.6% | — | 8.3pp | — | — | — |

Sanity: per-seed spreads sit inside the historical Phase 6 per-seed
band (~8.5-13.5pp); final scores and bust rates match the 50-update
`basic_random` profile (~200 final, ~14% bust). Win shares bracket the
50-update 65.4% pooled result at 1/5 the training budget with wide
CIs (60 games per seat-block), as expected.

Regression rule: future fast-loop work must reproduce these per-seed
numbers within CI overlap. Investigate if either seed's win share
falls outside its frozen CI or spread grows by more than 5pp. Never
gate on pooled spread alone (see `phase-7-resolution.md`).
