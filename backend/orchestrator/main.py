"""Orchestrator API: the only service that touches the GPU schedule and docker.sock.

Reachable only on the internal docker network, except ``/api`` and ``/images``
which Caddy exposes for the (bearer-key protected) legacy external API.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from huggingface_hub import HfApi
from pydantic import BaseModel, Field
from redis.asyncio import Redis

from . import docker_ops
from .backends import GpuBackends, ensure_dir, safe_user, under
from .config import get_settings
from .scheduler import GpuBusy, LimitExceeded, Scheduler
from .stats import StatsSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("orchestrator")
S = get_settings()

MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][\w.-]{0,95}/[\w.-]{1,95}$")
EXTRA_ARGS_RE = re.compile(r"^[\w\s.,:=/\-{}\"']*$")


@dataclass
class Caller:
    user: str
    admin: bool


class AppState:
    redis: Redis
    http: httpx.AsyncClient
    backends: GpuBackends
    scheduler: Scheduler
    stats: StatsSampler


st = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    st.redis = Redis.from_url(S.redis_url, decode_responses=True)
    st.http = httpx.AsyncClient(timeout=httpx.Timeout(30, read=None), limits=httpx.Limits(max_connections=200))
    st.backends = GpuBackends(st.http, S)
    st.scheduler = Scheduler(st.redis, st.backends, S)
    st.stats = StatsSampler()
    for d in ("uploads", "outputs"):
        ensure_dir(Path(S.data_dir) / d)
    st.scheduler.start()
    st.stats.start()
    yield
    await st.scheduler.stop()
    await st.stats.stop()
    await st.http.aclose()
    await st.redis.aclose()


app = FastAPI(title="orchestrator", lifespan=lifespan, docs_url=None, redoc_url=None)


# ---------------------------------------------------------------- auth/limits
def caller(request: Request) -> Caller:
    token = request.headers.get("x-internal-token") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token.encode(), S.internal_token.encode()):
        raise HTTPException(401, "unauthorized")
    user = safe_user(request.headers.get("x-user", "anonymous"))
    return Caller(user=user, admin=request.headers.get("x-user-role") == "admin")


def admin(c: Caller = Depends(caller)) -> Caller:
    if not c.admin:
        raise HTTPException(403, "admin only")
    return c


async def rate_limit(user: str, bucket: str, per_minute: int) -> None:
    key = f"rl:{bucket}:{user}:{int(time.time() // 60)}"
    n = await st.redis.incr(key)
    if n == 1:
        await st.redis.expire(key, 70)
    if n > per_minute:
        raise HTTPException(429, f"Rate limit: max {per_minute} {bucket} requests per minute")


@app.get("/health")
async def health():
    return {"ok": True, "holder": st.scheduler.holder}


# ----------------------------------------------------------------------- chat
async def _chat_upstream(body: dict[str, Any]):
    """Proxy one chat completion to vLLM while holding a shared GPU lease."""
    body["model"] = "chat"
    if body.get("max_tokens"):
        body["max_tokens"] = min(int(body["max_tokens"]), S.chat_max_tokens)
    if body.get("max_completion_tokens"):
        body["max_completion_tokens"] = min(int(body["max_completion_tokens"]), S.chat_max_tokens)

    lease = st.scheduler.chat_lease()
    try:
        await lease.__aenter__()
    except GpuBusy as e:
        raise HTTPException(503, str(e)) from e

    url = f"{S.vllm_url}/v1/chat/completions"
    if not body.get("stream"):
        try:
            r = await st.http.post(url, json=body, timeout=600)
            return JSONResponse(r.json(), status_code=r.status_code)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"vLLM error: {e}") from e
        finally:
            await lease.__aexit__(None, None, None)

    try:
        req = st.http.build_request("POST", url, json=body, timeout=httpx.Timeout(30, read=600))
        resp = await st.http.send(req, stream=True)
    except BaseException as e:
        await lease.__aexit__(None, None, None)
        if isinstance(e, httpx.HTTPError):
            raise HTTPException(502, f"vLLM error: {e}") from e
        raise

    if resp.status_code != 200:
        content = await resp.aread()
        await resp.aclose()
        await lease.__aexit__(None, None, None)
        return JSONResponse({"error": content.decode(errors="replace")[:2000]}, status_code=resp.status_code)

    async def relay():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()
            await lease.__aexit__(None, None, None)

    return StreamingResponse(relay(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, c: Caller = Depends(caller)):
    await rate_limit(c.user, "chat", S.chat_rpm)
    return await _chat_upstream(await request.json())


@app.get("/v1/models")
async def models(_: Caller = Depends(caller)):
    try:
        r = await st.http.get(f"{S.vllm_url}/v1/models", timeout=10)
        return JSONResponse(r.json(), status_code=r.status_code)
    except httpx.HTTPError as e:
        raise HTTPException(503, f"vLLM unavailable: {e}") from e


# ----------------------------------------------------------------------- jobs
class ImageParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field("", max_length=2000)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(9, ge=1, le=50)
    guidance: float = Field(0.0, ge=0, le=20)
    seed: int = Field(-1, ge=-1)


class VideoParams(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field("", max_length=2000)
    image_path: str | None = None
    resolution: str = Field("480p", pattern="^(480p|720p)$")
    num_frames: int = Field(81, ge=17, le=121)
    steps: int = Field(30, ge=4, le=60)
    guidance: float = Field(5.0, ge=0, le=20)
    fps: int = Field(24, ge=8, le=30)
    seed: int = Field(-1, ge=-1)


class JobRequest(BaseModel):
    type: str = Field(pattern="^(image|video)$")
    params: dict[str, Any]


PARAM_MODELS = {"image": ImageParams, "video": VideoParams}
UPLOADS = Path(S.data_dir) / "uploads"


@app.post("/jobs")
async def submit_job(req: JobRequest, c: Caller = Depends(caller)):
    params = PARAM_MODELS[req.type](**req.params).model_dump()
    for key in ("image_path",):
        if params.get(key):
            try:
                params[key] = str(under(params[key], UPLOADS))
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
    try:
        return await st.scheduler.submit(c.user, req.type, params)
    except LimitExceeded as e:
        raise HTTPException(429, str(e)) from e


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, c: Caller = Depends(caller)):
    job = await st.scheduler.get_job(job_id)
    if not job or (job["user"] != c.user and not c.admin):
        raise HTTPException(404, "job not found")
    job["gpu"] = await st.scheduler.state()
    return job


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, c: Caller = Depends(caller)):
    if not await st.scheduler.cancel(job_id, c.user, c.admin):
        raise HTTPException(409, "job is not queued (or not yours)")
    return {"cancelled": job_id}


@app.get("/queue")
async def queue_state(_: Caller = Depends(caller)):
    return await st.scheduler.state()


# --------------------------------------------------------- ASR / VIP (no lease)
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
    response_format: str = Form("verbose_json"),
    task: str = Form("transcribe"),
    c: Caller = Depends(caller),
):
    await rate_limit(c.user, "transcribe", 10)
    data = {"response_format": response_format, "task": task}
    if language:
        data["language"] = language
    files = {"file": (file.filename or "audio", file.file, file.content_type or "application/octet-stream")}
    try:
        r = await st.http.post(f"{S.asr_url}/v1/audio/transcriptions", data=data, files=files, timeout=1800)
    except httpx.HTTPError as e:
        raise HTTPException(503, f"ASR service unavailable: {e}") from e
    if response_format in ("text", "srt", "vtt"):
        return JSONResponse({"text": r.text}, status_code=r.status_code)
    return JSONResponse(r.json(), status_code=r.status_code)


class VipRequest(BaseModel):
    style: str = Field(pattern="^(oil|aquarell|bleistift|point)$")
    image_path: str
    params: dict[str, Any] = {}


@app.post("/vip")
async def vip(req: VipRequest, c: Caller = Depends(caller)):
    await rate_limit(c.user, "vip", 60)
    try:
        path = str(under(req.image_path, UPLOADS))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    r = await st.http.post(
        f"{S.media_url}/vip/{req.style}",
        json={"image_path": path, "user": c.user, "params": req.params},
        timeout=300,
    )
    return JSONResponse(r.json(), status_code=r.status_code)


# ---------------------------------------------------------------------- admin
@app.get("/admin/stats")
async def admin_stats(_: Caller = Depends(admin)):
    return {**st.stats.latest, "scheduler": await st.scheduler.state(), "llm": st.backends.read_llm_config()}


@app.get("/admin/containers")
async def admin_containers(_: Caller = Depends(admin)):
    return await docker_ops.list_containers()


@app.get("/admin/containers/{name}/logs")
async def admin_logs(name: str, tail: int = 300, _: Caller = Depends(admin)):
    try:
        return {"logs": await docker_ops.logs(name, min(tail, 5000))}
    except (docker_ops.NotFound, PermissionError) as e:
        raise HTTPException(404, str(e)) from e


@app.post("/admin/containers/{name}/{action}")
async def admin_container_action(name: str, action: str, _: Caller = Depends(admin)):
    if name == S.vllm_container:
        raise HTTPException(409, "vLLM is managed by the scheduler (use model deploy instead)")
    fn = {"start": docker_ops.start, "stop": docker_ops.stop, "restart": docker_ops.restart}.get(action)
    if fn is None:
        raise HTTPException(400, "action must be start|stop|restart")
    try:
        await fn(name)
    except (docker_ops.NotFound, PermissionError) as e:
        raise HTTPException(404, str(e)) from e
    return {"ok": True}


@app.get("/admin/hf/search")
async def hf_search(q: str, limit: int = 30, _: Caller = Depends(admin)):
    def _search():
        return [
            {"id": m.id, "downloads": m.downloads, "likes": m.likes, "pipeline_tag": m.pipeline_tag}
            for m in HfApi().list_models(search=q, pipeline_tag="text-generation", sort="downloads", limit=min(limit, 100))
        ]

    try:
        return await asyncio.to_thread(_search)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Hugging Face search failed: {e}") from e


@app.get("/admin/hf/info")
async def hf_info(model_id: str, _: Caller = Depends(admin)):
    if not MODEL_ID_RE.match(model_id):
        raise HTTPException(400, "invalid model id")

    def _info():
        info = HfApi().model_info(model_id, files_metadata=True)
        weight_bytes = sum(
            (f.size or 0) for f in (info.siblings or []) if f.rfilename.endswith((".safetensors", ".bin")) and "/" not in f.rfilename.strip("/")
        )
        cfg = info.config or {}
        params = info.safetensors.total if info.safetensors else None
        est_gb = round(weight_bytes / 1024**3 * 1.15, 1) if weight_bytes else None
        return {
            "id": info.id,
            "pipeline_tag": info.pipeline_tag,
            "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"),
            "gated": info.gated,
            "private": info.private,
            "license": (info.card_data or {}).get("license") if info.card_data else None,
            "params_b": round(params / 1e9, 2) if params else None,
            "dtypes": list(info.safetensors.parameters.keys()) if info.safetensors else None,
            "weights_gb": round(weight_bytes / 1024**3, 2) if weight_bytes else None,
            "est_vram_gb": est_gb,
            "fits_4090": (est_gb is not None and est_gb < 24 * 0.78 - 2),
            "downloads": info.downloads,
        }

    try:
        return await asyncio.to_thread(_info)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"model info failed: {e}") from e


class DeployRequest(BaseModel):
    model: str
    max_model_len: int = Field(16384, ge=512, le=262144)
    gpu_memory_utilization: float = Field(0.78, ge=0.3, le=0.9)
    extra_args: str = Field("", max_length=500)


@app.get("/admin/llm")
async def llm_config(_: Caller = Depends(admin)):
    return st.backends.read_llm_config()


@app.post("/admin/llm/deploy")
async def llm_deploy(req: DeployRequest, c: Caller = Depends(admin)):
    if not MODEL_ID_RE.match(req.model):
        raise HTTPException(400, "invalid model id")
    if not EXTRA_ARGS_RE.match(req.extra_args):
        raise HTTPException(400, "extra_args contains forbidden characters")
    return await st.scheduler.submit(c.user, "deploy", req.model_dump())


# ------------------------------------------------- legacy public /api endpoint
def _api_user(request: Request) -> str:
    key = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if not key or not any(secrets.compare_digest(key.encode(), k.encode()) for k in S.api_key_set):
        raise HTTPException(401, "missing or invalid API key")
    return "api-" + hashlib.sha256(key.encode()).hexdigest()[:8]


@app.post("/api")
async def legacy_api(request: Request):
    """Backwards compatible with the old {"method": "vllm"|"image", "prompt": ...} API."""
    user = _api_user(request)
    data = await request.json()
    method, prompt = data.get("method"), str(data.get("prompt", ""))[:4000]
    if not prompt:
        raise HTTPException(400, "prompt required")

    if method == "vllm":
        await rate_limit(user, "chat", S.chat_rpm)
        resp = await _chat_upstream(
            {
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(data.get("max_tokens", 1024)),
                "temperature": 0.7,
                "top_p": 0.8,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        parsed = json.loads(resp.body)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, parsed)
        return parsed["choices"][0]["message"]["content"]

    if method == "image":
        try:
            job = await st.scheduler.submit(user, "image", ImageParams(prompt=prompt).model_dump())
        except LimitExceeded as e:
            raise HTTPException(429, str(e)) from e
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            job = await st.scheduler.get_job(job["id"])
            if job["status"] == "done":
                name = Path(job["result"]["path"]).name
                return f"{S.public_base_url.rstrip('/')}/images/{user}/{name}"
            if job["status"] in ("failed", "cancelled"):
                raise HTTPException(500, job.get("error", "image generation failed"))
            await asyncio.sleep(1)
        raise HTTPException(504, "image generation timed out")

    raise HTTPException(400, "method must be 'vllm' or 'image'")


@app.get("/images/{user}/{name}")
async def legacy_image(user: str, name: str):
    if not user.startswith("api-") or not re.fullmatch(r"[0-9a-f]{32}\.png", name):
        raise HTTPException(404)
    path = Path(S.data_dir) / "outputs" / safe_user(user) / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")
