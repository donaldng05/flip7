# Engineering Standards

## Scope of this milestone

The repository foundation is implemented before the game-rules specification.
This milestone establishes repeatable development, quality gates, CI, release
artifacts, and architecture boundaries; it does not define or implement Flip 7
behavior.

## Tooling

- Python 3.12 or newer is the supported runtime.
- `uv.lock` is committed and is the source of truth for resolved dependencies.
- Ruff owns formatting and linting.
- Pyright checks package typing.
- Pytest owns automated tests and coverage.
- Pre-commit runs fast repository hygiene checks before commits.
- GitHub Actions runs the same quality gates in a clean Linux environment.

## Design boundaries

`flip7.core` owns deterministic game state and mechanics. `flip7.envs` adapts
the core to external RL interfaces. Agents consume environment observations;
training orchestrates policies; evaluation measures them. These layers must not
reach through one another's implementation details.

## Reproducibility

Every future simulation or experiment that uses randomness must accept an
explicit seed or RNG object. Tests must use fixed seeds where randomness is
relevant. Training and evaluation use separate recorded seeds and configurations.
Generated checkpoints, logs, metrics, and datasets are not source files and
remain outside version control unless intentionally curated.

Configuration files are version-controlled experiment inputs. Each run should
record the exact configuration and seed alongside its generated artifacts.

## Testing

Game rules require unit tests for individual mechanics and integration tests for
complete rounds and games. Environment tests must verify legal actions,
observations, rewards, and seeded transitions. Evaluation tests must verify
metric calculations with small deterministic fixtures.

## Dependency policy

Prefer the standard library for stable core behavior. Add NumPy when simulation
needs numerical arrays, PettingZoo/Gymnasium when the environment adapter is
implemented, PyTorch when learning begins, and Ray/Optuna/tracking/serving
libraries only when measured project needs justify them.
