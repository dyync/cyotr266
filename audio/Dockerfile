# syntax=docker/dockerfile:1.7
# CTranslate2 4.8 is built against CUDA 12.x + cuDNN 9 -> use the matching
# NVIDIA runtime image (the old python:slim image had no cuBLAS/cuDNN, and the
# unpinned torch dependency pulled CUDA 13 libraries that CTranslate2 can't use).
FROM nvidia/cuda:12.9.2-cudnn-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 HF_HOME=/models/hf

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && uv sync --locked --no-install-project --python 3.12

COPY asr ./asr
RUN useradd -u 1000 -m app 2>/dev/null || usermod -l app $(id -nu 1000)
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "asr.main:app", "--host", "0.0.0.0", "--port", "8000"]
