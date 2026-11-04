# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY orchestrator ./orchestrator

# Runs as root because it needs docker.sock; it is the only container with it
# and it is not reachable from outside the internal network (except /api via Caddy).
EXPOSE 7861
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7861/health', timeout=4)"
CMD ["uvicorn", "orchestrator.main:app", "--host", "0.0.0.0", "--port", "7861", "--proxy-headers", "--forwarded-allow-ips", "*"]
