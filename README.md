# Flip 7 RL Agent

A research software project for building and evaluating reinforcement-learning
agents that play Flip 7 against heuristic, algorithmic, and learned opponents.

## Status

Phases 0–2 are in place: the engineering foundation, the authoritative rules
specification, and a deterministic Flip 7 engine. Phase 3 exposes that engine as
a PettingZoo AEC environment with a Gymnasium vs-opponents wrapper. Baseline
agents, training, and evaluation remain later milestones.

## Documentation

- [Project overview](docs/project-overview.md)
- [Requirements and scope](docs/requirements-and-scope.md)
- [Project plan](docs/project-plan.md)
- [Game rules specification](docs/game-rules-specification.md)
- [Environment specification](docs/environment-specification.md)
- [Architecture decision records](docs/adr/README.md)
- [Engineering standards](docs/engineering-standards.md)

## Development

The project uses Python, `uv`, and a `src/` package layout. Install `uv`, then
run:

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The complete local quality gate is available through `scripts/check.ps1` on
Windows or `scripts/check.sh` on Linux/macOS. Docker is an optional alternative
for a reproducible environment; see [development environment](docs/development-environment.md).

## Architecture

The package is intentionally split into replaceable boundaries:

- `flip7.core`: deterministic game mechanics and state
- `flip7.envs`: PettingZoo and Gymnasium adapters
- `flip7.agents`: heuristic and learned policies
- `flip7.training`: training orchestration
- `flip7.evaluation`: benchmarks and statistical evaluation
- `flip7.config`: configuration and experiment inputs

`flip7.core` does not import NumPy, Gymnasium, or PettingZoo. Those libraries
are used only by the environment adapters.

## License

This project is licensed under the [MIT License](LICENSE).
