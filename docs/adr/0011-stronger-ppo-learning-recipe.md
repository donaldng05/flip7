# ADR 0011: Stronger PPO learning recipe for Phase 7 completion

- Status: Accepted
- Date: 2026-09-02

## Context

Phase 6 `basic_random` demonstrated that the existing masked PPO pipeline can
learn a strong seat-robust policy. Phase 7 league follow-up experiments then
showed that snapshot retention alone did not create a useful population: the
novelty condition improved held-out win share by only 2.17 points, and seat
spread remained seed-sensitive. Training histories also showed entropy falling
from approximately 0.68 to 0.24 during the 50-update recipe while value loss
remained noisy.

The next intervention must improve credit assignment and optimizer stability
without changing the immutable Phase 6/7 controls or evaluating a favorable
intermediate checkpoint.

## Decision

Add an opt-in PPO recipe while retaining the existing shared actor-critic as the
default compatibility path. The new recipe uses:

- independent actor and critic feature extractors;
- an actor-independent learner-seat one-hot input to the value function;
- orthogonal initialization;
- scheduled learning rate and entropy coefficient;
- value-function clipping and target-KL diagnostics;
- equal transition quotas for all learner seats; and
- a 100-update final-checkpoint contract.

Add `potential_win` as a training-only reward mode. It preserves the terminal
sparse-win reward and adds a discounted delta of a bounded public score-
differential potential. Evaluation remains `sparse_win`; held-out opponents and
variants are never used by the training reward or recipe selection.

## Consequences

The value function can model seat-dependent returns without changing the basic
actor input, and scheduled regularization can be evaluated against the observed
entropy collapse. Potential shaping provides a controlled credit-assignment
ablation without redefining the primary objective. New checkpoints record their
network and critic-context dimensions so incompatible restoration fails
explicitly, while existing Phase 5–7 checkpoints remain readable.

This ADR does not promote a recipe or advance Phase 8. Promotion requires the
predeclared per-seed strength, seat-spread, held-out, behavioral-diversity, and
tournament gates in the Phase 7 stability documentation.
