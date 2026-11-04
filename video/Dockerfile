# syntax=docker/dockerfile:1.7
# torch wheels bundle the CUDA 13 runtime, so a slim base image is enough.
FROM python:3.12-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-install-project

COPY media ./media
RUN useradd -u 1000 -m app
USER app

EXPOSE 1376
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:1376/health', timeout=4)"
CMD ["uvicorn", "media.main:app", "--host", "0.0.0.0", "--port", "1376"]
