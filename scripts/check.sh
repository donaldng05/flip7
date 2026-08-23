#!/usr/bin/env bash
set -euo pipefail

uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest --cov=flip7 --cov-report=term-missing --cov-report=xml
