"""Media service: image + video generation (GPU) and VIP filters (CPU).

Internal only. GPU scheduling decisions are made by the orchestrator; the lock
here is a second line of defence so two GPU jobs can never overlap.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import vip
from .pipelines import PipelineManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("media")

DATA = Path(os.getenv("DATA_DIR", "/data"))
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"

app = FastAPI(title="media", docs_url=None, redoc_url=None)
manager = PipelineManager()
gpu_lock = threading.Lock()
cpu_limiter = anyio.CapacityLimiter(int(os.getenv("VIP_CONCURRENCY", "2")))
progress: dict[str, float] = {}


def _out(user: str, suffix: str) -> Path:
    safe = "".join(c for c in user if c.isalnum() or c in "-_.")[:64] or "anon"
    d = OUTPUTS / safe
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{uuid.uuid4().hex}{suffix}"


def _check_upload(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path).resolve()
    if not p.is_relative_to(UPLOADS.resolve()) or not p.is_file():
        raise HTTPException(400, "image_path must be an existing file under /data/uploads")
    return str(p)


def _run_gpu(fn, *args):
    if not gpu_lock.acquire(timeout=5):
        raise HTTPException(409, "GPU busy in media service")
    try:
        return fn(*args)
    finally:
        gpu_lock.release()


@app.get("/health")
async def health():
    return {"ok": True, "loaded": manager.kind}


class LoadReq(BaseModel):
    kind: str = Field(pattern="^(image|video)$")


@app.post("/load")
async def load(req: LoadReq):
    await anyio.to_thread.run_sync(_run_gpu, manager.load, req.kind)
    return {"loaded": manager.kind}


@app.post("/unload")
async def unload():
    await anyio.to_thread.run_sync(_run_gpu, manager.unload)
    return {"loaded": None}


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id not in progress:
        raise HTTPException(404)
    return {"progress": progress[job_id]}


class GenReq(BaseModel):
    model_config = {"extra": "allow"}
    job_id: str
    user: str = "anon"
    prompt: str


async def _generate(kind: str, req: GenReq, suffix: str) -> dict[str, Any]:
    params = req.model_dump()
    if kind == "video":
        params["image_path"] = _check_upload(params.get("image_path"))
    out = _out(req.user, suffix)
    progress[req.job_id] = 0.0

    def cb(p: float) -> None:
        progress[req.job_id] = round(p, 3)

    fn = manager.generate_image if kind == "image" else manager.generate_video
    try:
        return await anyio.to_thread.run_sync(_run_gpu, fn, params, out, cb)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.exception("%s generation failed", kind)
        raise HTTPException(500, f"{kind} generation failed: {e}") from e
    finally:
        asyncio.get_running_loop().call_later(300, progress.pop, req.job_id, None)


@app.post("/generate/image")
async def generate_image(req: GenReq):
    return await _generate("image", req, ".png")


@app.post("/generate/video")
async def generate_video(req: GenReq):
    return await _generate("video", req, ".mp4")


class VipReq(BaseModel):
    image_path: str
    user: str = "anon"
    params: dict[str, Any] = {}


@app.post("/vip/{style}")
async def vip_filter(style: str, req: VipReq):
    fn = vip.STYLES.get(style)
    if fn is None:
        raise HTTPException(404, f"unknown style {style}")
    path = _check_upload(req.image_path)
    out = _out(req.user, ".png")
    try:
        result = await anyio.to_thread.run_sync(lambda: fn(path, out, **req.params), limiter=cpu_limiter)
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e)) from e
    return {"path": result}
