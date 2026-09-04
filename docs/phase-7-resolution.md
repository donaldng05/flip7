# Phase 7 scientific resolution

Status: implementation complete; empirical resolution pending. No Phase 8
promotion decision has been made.

This work is isolated on the `phase-7-resolve` branch and preserves the
previous Phase 7 artifacts as immutable references.

## Pre-registered question

The Phase 7 screening showed positive adaptation but failed the five-point
seat-spread gate. The primary hypothesis is that the seat-balanced trainer
concatenated independently reset seat blocks before computing GAE. The value
bootstrap and GAE trace could therefore cross from one learner seat's rollout
into another seat's unrelated rollout.

The first change is a correctness fix only: each independently collected seat
block is represented as a separate GAE segment with its own bootstrap value.
The only learning intervention is the existing training-only `potential_win`
reward, evaluated with sparse-win metrics.

## Fixed protocol

- The stronger Phase 7 PPO recipe, environment, opponents, schedules, and
  thresholds are unchanged.
- Screening uses seeds `7, 17, 27`, 100 updates, and 100 games per learner
  seat.
- Confirmation uses seeds `7, 17, 27, 37, 47`, 100 updates, and 200 games per
  learner seat.
- Only update 100 is eligible for promotion.
- The full-population tournament is non-gating and may be skipped for CPU
  runtime.
- MAPPO is explicitly deferred from this resolution.

## Scientific decision rule

- **PASS:** the corrected potential-win candidate passes every required
  confirmation gate for every confirmation seed.
- **FAIL:** the evaluation protocol is validated, but the corrected recipe or
  potential-win intervention fails a required gate.
- **INCONCLUSIVE:** the known-good Phase 6 reference cannot be reproduced or
  the evaluation/training provenance is incomplete.

## Commands

The diagnostic command audits existing artifacts without retraining:

```text
python scripts/diagnose_phase7_stability.py
```

Corrected sparse-reward screening uses a new artifact root:

```text
python scripts/run_phase7_follow_up_stability.py `
  --config configs/phase7-resolve.yaml `
  --output-root artifacts/phase7-resolve/sparse-corrected `
  --skip-full-population-tournament
```

The one intervention is run only after protocol validation:

```text
python scripts/run_phase7_follow_up_stability.py `
  --config configs/phase7-resolve.yaml `
  --training-reward potential_win `
  --output-root artifacts/phase7-resolve/potential-corrected `
  --skip-full-population-tournament
```

Confirmation is permitted only when potential-win screening passes and uses
the same potential-win artifact root with `--stage confirmation`.

## Results

The one-game smoke audit completed and exercised the real Phase 6 and Phase 7
manifest layouts, but it is intentionally not evidence for a scientific
decision. The pre-registered 1,000-game, two-batch audit was started against
the immutable Phase 6 reference, current balanced control, and current
seat-aware fallback. It remained CPU-bound for roughly 90 minutes and was
stopped before its final JSON write; no corrected training or reward
intervention was started. A future run must complete the audit before any
screening or confirmation result can be classified as PASS or FAIL.

Existing failed results are not overwritten.
