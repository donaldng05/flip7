# Architecture Decision Records

This directory records durable technology, architecture, reproducibility, and
deployment decisions for the Flip 7 research platform.

Each ADR uses a short Nygard layout:

1. **Status** — accepted, superseded, or deprecated.
2. **Context** — the problem and constraints at the time of the decision.
3. **Decision** — what we chose.
4. **Consequences** — what becomes easier, harder, or out of scope.

ADRs do not replace the rules specification or the environment specification.
They explain why a boundary exists so later phases do not silently reopen it.

Phase 5 training decisions are recorded in
[`0007-custom-pytorch-ppo.md`](0007-custom-pytorch-ppo.md).
Phase 7 league self-play and tournament decisions are recorded in
[`0008-population-self-play-and-tournament-evaluation.md`](0008-population-self-play-and-tournament-evaluation.md).
Phase 7 follow-up diversity and MAPPO criteria are recorded in
[`0009-phase7-follow-up-diversity-and-mappo.md`](0009-phase7-follow-up-diversity-and-mappo.md).
Phase 7 stability and diversity-benefit evaluation is recorded in
[`0010-phase7-stability-and-diversity-benefit.md`](0010-phase7-stability-and-diversity-benefit.md).
