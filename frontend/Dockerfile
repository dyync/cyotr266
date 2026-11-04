# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False GRADIO_TEMP_DIR=/tmp/gradio

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project

COPY favicon.ico ./
COPY webui ./webui
RUN useradd -u 1000 -m app
USER app

EXPOSE 7860
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 7860), 3)" || exit 1
CMD ["python", "-m", "webui.app"]
