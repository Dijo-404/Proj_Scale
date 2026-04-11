ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
ARG BUILD_MODE=standalone

FROM ${BASE_IMAGE} AS builder

WORKDIR /app/env

ENV UV_PROJECT_ENVIRONMENT=/app/env/.venv

COPY pyproject.toml uv.lock README.md openenv.yaml requirements.txt ./
COPY __init__.py client.py models.py tasks.py graders.py ./
COPY inference.py inference_config.py inference_prompts.py inference_runner.py inference_strategies.py ./
COPY config ./config
COPY server ./server

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev --no-install-project

FROM ${BASE_IMAGE} AS runtime

ARG BUILD_MODE=standalone
ENV BUILD_MODE=${BUILD_MODE}

WORKDIR /app/env

COPY --from=builder /app/env /app/env

ENV PATH=/app/env/.venv/bin:${PATH}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "server"]
