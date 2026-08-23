FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN useradd --create-home --shell /bin/bash app
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

WORKDIR /workspace
COPY pyproject.toml uv.lock README.md LICENSE ./

COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs
RUN uv sync --locked
RUN chown -R app:app /workspace

USER app
CMD ["bash", "scripts/check.sh"]
