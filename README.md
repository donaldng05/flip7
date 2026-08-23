# Flip 7 RL Agent

A research software project for building and evaluating reinforcement-learning
agents that play Flip 7 against heuristic, algorithmic, and learned opponents.

## Status

The repository is currently establishing its engineering foundation. Game rules,
the simulator, RL environment, agents, training, and evaluation will be added
in later milestones.

## Documentation

- [Project overview](docs/project-overview.md)
- [Requirements and scope](docs/requirements-and-scope.md)
- [Project plan](docs/project-plan.md)
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
- `flip7.envs`: RL environment adapters
- `flip7.agents`: heuristic and learned policies
- `flip7.training`: training orchestration
- `flip7.evaluation`: benchmarks and statistical evaluation
- `flip7.config`: configuration and experiment inputs

The initial dependency set remains small. RL frameworks, distributed execution,
experiment tracking, and model serving are added only when their roadmap phase
requires them.

## License

This project is licensed under the [MIT License](LICENSE).
