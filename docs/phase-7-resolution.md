# Phase 7 scientific resolution

Status: Phase 7 remains inconclusive; no Phase 8 promotion decision has been
made.

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

## Calibration rule

The original Phase 6 pooled `basic_random` spread is retired as a seat-
robustness reference. It combined policies from different training seeds and
could hide seed-specific seat instability. Historical Phase 6 artifacts remain
immutable, but their `seed_summaries` are now the source of truth.

Calibration uses the training seed as the independent unit. For every seed it
reports the three seat win shares, the derived seat spread, pooled win share,
and 95% uncertainty intervals. Fresh evaluation is stratified by matchup and
learner seat; when raw game outcomes are available, intervals use a seeded
game-level bootstrap. Legacy summary-only artifacts use a conservative normal
approximation and are labeled accordingly.

The 5-point spread remains the long-term robustness target, not a protocol
validity test. A new candidate is compared with the calibrated reference per
seed: lower-spread confidence intervals are `better`, higher-spread intervals
are `worse`, and overlapping intervals are `compatible`. A weak reference is
reported as `reference-not-robust / candidate-comparison-only`; it is never
reported as `protocol-invalid` or used to claim a recipe failure.

Run the artifact-only recalibration with:

```powershell
uv run python scripts/calibrate_phase6_reference.py --reevaluate-reference
```

This writes `artifacts/phase7-resolve/calibration-v2.json` and does not train
or modify any Phase 6 or Phase 7 checkpoint.

Omit `--reevaluate-reference` to reanalyze the existing fresh audit using its
summary-level uncertainty when a completed audit is already available.

## Scientific decision rule

- **PASS:** the corrected potential-win candidate passes every required
  confirmation gate for every confirmation seed.
- **FAIL:** the evaluation protocol is validated, but the corrected recipe or
  potential-win intervention fails a required gate.
- **INCONCLUSIVE:** the reference needs recalibration, is not robust under
  per-seed evaluation, or the evaluation/training provenance is incomplete.

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

The required order is calibration-v2, corrected sparse-reward control
interpretation, potential-win screening, confirmation, and only then any MAPPO
pilot. Phase 8 remains locked until a final candidate passes the per-seed
strength, relative seat-stability, diversity, and tournament gates.

## Results

The completed fresh audit reproduced the Phase 6 `basic_random` win share
within 0.35 percentage points, but it exposed the pooled spread problem. The
historical pooled spread was 2.72 points, while the fresh pooled spread was
5.44 points and the per-training-seed fresh spreads were approximately 9--10
points. The historical per-seed spreads were already approximately 8.5--13.5
points.

`artifacts/phase7-resolve/calibration-v2.json` records the corrected
interpretation: the reference reproduces pooled strength, is not reliably
seat-robust under the 5-point target, and both current controls are compatible
with the reference on a per-seed basis. Therefore Phase 7 remains
**INCONCLUSIVE**; this does not establish that segmented GAE caused or failed
to cause the original result. No corrected reward intervention, MAPPO pilot,
or Phase 8 promotion is authorized by this artifact.

The historical pooled result is not overwritten. Existing failed results are
not overwritten either; the calibration-v2 artifact is a separate report.
