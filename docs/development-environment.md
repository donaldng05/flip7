# Development Environment

## Native setup

Install Python 3.12 or newer and `uv`, then run:

```powershell
uv sync
uv run pre-commit install
```

Run the checks with `scripts/check.ps1` on Windows or
`bash scripts/check.sh` on Linux/macOS.

## Docker setup

Docker is optional and provides an isolated environment matching CI:

```powershell
docker compose build
docker compose run --rm dev
```

The source tree is mounted into the container, while the dependency cache is
kept in a named volume. The container does not include credentials, datasets,
checkpoints, or experiment logs.

The native `uv` workflow remains the primary development workflow because it is
faster for iterative work and uses the same lockfile.
