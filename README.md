# AI Studio — multi-user LLM / image / video / speech on one RTX 4090

Gradio 6 web app for several simultaneous users on a single 24 GB GPU:

| Tab   | Model (default)                                  | Runs in            |
|-------|--------------------------------------------------|--------------------|
| Chat  | Qwen3.5-9B (FP8) via official **vLLM 0.33** server, streaming, optional thinking mode | `vllm` |
| Image | **Z-Image-Turbo** (6B, 9 steps)                  | `video` (media)    |
| Video | **Wan 2.2 TI2V-5B**, text→video and image→video, 24 fps | `video` (media) |
| VIP   | OpenCV art filters (Öl, Aquarell, Bleistift, Pointillismus), CPU | `video` (media) |
| Audio | **faster-whisper large-v3-turbo**, transcription + translation, SRT export | `audio` |
| Admin | GPU/disk stats, container logs, switch the LLM to any HF model | `backend` |

## Architecture

```
Internet ─443─> caddy (HTTPS) ─> frontend  (Gradio 6, login, no GPU, no docker.sock)
                     │ /api, /images/api-*            │ internal network only
                     └──────────────> backend = orchestrator (FastAPI)
                                      ├─ GPU scheduler (Redis job queue + chat lease)
                                      ├─ /v1/chat/completions streaming proxy
                                      └─ only container with docker.sock
          vllm (sleep mode) · video (media) · audio (ASR) · redis
```

**One GPU, many users — "swap + queue":**

* The **LLM is the home state**. All chats go through the orchestrator and vLLM batches them (continuous batching), so 10+ people can chat at once.
* **Image / video jobs go into one FIFO queue.** For a job, the scheduler stops new chats from starting and lets running chats finish (up to 60 s). It then puts vLLM to sleep, which moves the weights to CPU RAM and frees ~19 GB in seconds, and loads the media model. The UI shows the queue position and progress.
* Jobs of the same type run back to back, so the model is loaded only once. If chats are waiting, the scheduler hands the GPU back to the LLM after 4 jobs or 2 minutes, and chats get at least 20 s before the next batch starts.
* Whisper (~2 GB) stays loaded the whole time, so transcription never waits for the queue.
* Limits: each user can have 2 jobs queued (only 1 of them video) and 30 chat messages per minute. Change these in `.env`.

VRAM budget (24 GB): vLLM 0.78 × 24 ≈ 18.7 GB (weights ~10 GB + KV cache) + Whisper ~2 GB. When vLLM is asleep, Z-Image (~13 GB peak with offload), or Wan 2.2 5B (~20 GB peak with offload) get the GPU instead.

## Requirements

* Ubuntu 22.04 or 24.04, RTX 4090 (24 GB), **NVIDIA driver ≥ 580** (the containers use CUDA 13 / 12.9)
* **≥ 32 GB RAM**: vLLM sleep mode keeps the LLM weights in RAM while media jobs run
* ~120 GB disk for models (`/models`) and space for outputs (`./data`)
* A DNS name pointing at the VM, for automatic HTTPS. Without one, use `DOMAIN=":80"` on a trusted LAN.

## Setup

```bash
# 1. host: driver 580, Docker, NVIDIA Container Toolkit, firewall (reboot if the driver changed)
sudo bash scripts/setup-host.sh

# 2. config
cp .env.example .env           # set DOMAIN, PUBLIC_BASE_URL, HF_TOKEN
sed -i "s/^INTERNAL_TOKEN=.*/INTERNAL_TOKEN=$(openssl rand -hex 32)/; s/^REDIS_PASSWORD=.*/REDIS_PASSWORD=$(openssl rand -hex 32)/" .env
bash scripts/add_user.sh admin --admin     # creates config/users.toml
bash scripts/add_user.sh alice

# 3. models (optional but recommended: ~45 GB, otherwise downloaded on first use)
bash scripts/prefetch_models.sh

# 4. start
docker compose up -d --build --wait
```

Open `https://$DOMAIN` and log in.

### Migrating from the old stack

The old compose file published Redis, the Docker-control backend, vLLM and every service port to the internet. Stop it **before** starting the new one; the container names are reused.

```bash
docker rm -f container_redis container_backend container_audio container_video   container_frontend container_tr container_vllm_xoo container_vllm_oai 2>/dev/null
docker network rm sys_net 2>/dev/null
```

The old code is still in the git history (`git show bed08b0:legacy/NOTE.md`). The old `/models` content is not reused; the new HF cache lives in `/models/hf`.

## Operating

| Task | Command |
|------|---------|
| Logs | `docker compose logs -f backend` (scheduler decisions), `… vllm`, `… video` |
| Queue / GPU state | Admin tab, or `docker compose exec backend python -c "import urllib.request;print(urllib.request.urlopen('http://localhost:7861/health').read())"` |
| Add/replace a user | `bash scripts/add_user.sh <name> [--admin]` (hot-reloaded, no restart) |
| Remove a user | delete their `[users.<name>]` block in `config/users.toml` |
| Switch the LLM | Admin tab → search HF → **Deploy model**. If the new model fails to start, the previous one is restored. The compose default is used again after `docker compose up` recreates `vllm`. |
| End-to-end test | see below |
| Update images | bump the pinned versions in `docker-compose.yml` / `*/pyproject.toml`, then run `uv lock` and `docker compose up -d --build` |

**Smoke test** (chat, image, transcription of `audio/nk.mp3`, all VIP styles, video, `/api`, admin auth):

```bash
docker compose run --rm --no-deps -e ORCH_URL=http://backend:7861 \
  -v ./scripts:/scripts:ro -v ./video/dragon.png:/assets/dragon.png:ro -v ./audio/nk.mp3:/assets/nk.mp3:ro \
  backend python /scripts/smoke_test.py
# concurrency: add  -e LOAD=1   (10 parallel chats + 3 image jobs + 2 transcriptions)
# skip the slow video job: -e SKIP=video
```

Unit tests for the scheduler (no GPU needed): `cd backend && uv run pytest`.

### Legacy `/api` endpoint (breaking change)

`POST https://$DOMAIN/api` with `{"method": "vllm" | "image", "prompt": "…"}` still works and returns the same shape as before: a string answer, or an image URL under `/images/…`. It now **requires** `Authorization: Bearer <key>`, with keys set in `API_KEYS` in `.env`. If `API_KEYS` is empty, the endpoint is disabled.

## What changed in 2.0 and why

**Why image generation and audio were broken** (all fixed):

* **Audio:** a fresh install pulls PyAV 19, which removed `av.open(metadata_errors=…)`. faster-whisper 1.2.1 still passes that argument, so every transcription failed. PyAV is now pinned to 18.1.0.
* **Audio:** the container used `runtime: nvidia`, which needs a manually configured runtime on newer toolkits. The base image was `python:slim` (no cuBLAS/cuDNN for CTranslate2), and an unpinned `torch` pulled in CUDA 13 libraries that CTranslate2 can't use. It now uses an `nvidia/cuda:12.9.2-cudnn-runtime` base, the model is loaded once at startup, and if CUDA is unavailable it falls back to CPU.
* **Audio:** the model dropdown sat in a hidden accordion with no value, and the audio file was passed as a `/tmp` path shared between containers. Uploads now go to the service directly as multipart.
* **Image:** every image request *restarted* the old TRELLIS container, which immediately loaded TRELLIS back into VRAM, so SDXL ran out of memory. On top of that, `HF_HOME` wasn't on `/models`, so models were downloaded again after each of these restarts; filenames came from the prompt, so long or unicode prompts crashed; and a LoRA scale was passed with no LoRA loaded.

**Multi-user and security:**

* All per-user state now lives in Gradio session state; the old code used module globals shared by everyone.
* All backend calls are async, with timeouts.
* Login (bcrypt) with an admin role, and HTTPS through Caddy.
* Only ports 80 and 443 are open. Before, Redis without a password and a Docker-control API were public, which amounted to remote root.
* Removed a route that served the frontend container's entire filesystem (`/files`, `/f/…`).
* Only the orchestrator gets `docker.sock`, and it may only control containers of this compose project.
* Non-root app containers, pinned image tags and `uv.lock` lockfiles, health checks, log rotation.

## Configuration reference

See `.env.example`. Useful knobs in `backend/orchestrator/config.py` (all settable via env on the `backend` service):
`DRAIN_TIMEOUT`, `CHAT_WAIT_TIMEOUT`, `BATCH_MAX_JOBS`, `BATCH_MAX_SECONDS`, `LLM_MIN_WINDOW`, `MAX_VIDEO_JOBS_PER_USER`.

## Notes / known limitations

* TRELLIS (3D) was removed in 2.0.
* While a long video job (several minutes) holds the GPU, new chats wait. That is the trade-off of swap + queue on one GPU; the chat UI tells users why.
* Running jobs can't be cancelled, only queued ones.
