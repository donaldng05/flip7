# Architecture

## Foundation boundaries

```text
flip7.core
    |
    +--> flip7.envs
    |        |
    |        +--> flip7.agents
    |        +--> flip7.training
    |        +--> flip7.evaluation
    |
    +--> flip7.config
```

The core engine will own authoritative game state and deterministic mechanics.
Environment adapters expose that state through RL interfaces. Agents consume
observations and return actions; training coordinates policy updates; evaluation
runs controlled comparisons. Configuration is an input boundary shared by
experiments, not a place for hidden defaults.

The dependency direction is intentionally one-way. Core does not import agents,
training, evaluation, or external RL frameworks. This keeps rule correctness
testable without installing a GPU stack and makes later adapters replaceable.

## Quality boundary

The same checks run locally and in CI:

```text
Ruff format -> Ruff lint -> Pyright -> Pytest + coverage -> package build
```

CI validates pull requests and pushes. Tagged releases build validated package
artifacts and attach them to a GitHub release; production hosting is not part of
the foundation.
