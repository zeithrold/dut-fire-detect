FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

ADD . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen
