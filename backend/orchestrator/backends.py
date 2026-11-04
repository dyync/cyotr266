"""Concrete GPU backends used by the scheduler (vLLM, media service)."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Any

import httpx

from . import docker_ops
from .config import Settings

log = logging.getLogger("backends")

LLM_ENV_KEYS = ("LLM_MODEL", "LLM_MAX_MODEL_LEN", "LLM_GPU_UTIL", "LLM_EXTRA_ARGS")


def safe_user(user: str) -> str:
    return "".join(c for c in user if c.isalnum() or c in "-_.")[:64] or "anon"


APP_UID = int(os.getenv("APP_UID", "1000"))


def own(path: Path) -> Path:
    """Files/dirs created by this (root) container must stay writable for the uid-1000 services."""
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        os.chown(path, APP_UID, APP_UID)
    return path


def ensure_dir(path: Path) -> Path:
    if not path.is_dir():
        path.mkdir(parents=True, exist_ok=True)
        own(path)
    return path


def under(path: str | Path, root: str | Path) -> Path:
    """Resolve *path* and make sure it stays inside *root*."""
    p = Path(path).resolve()
    if not p.is_relative_to(Path(root).resolve()):
        raise ValueError(f"path {path} outside {root}")
    return p


class GpuBackends:
    def __init__(self, http: httpx.AsyncClient, settings: Settings):
        self.http = http
        self.s = settings

    # ------------------------------------------------------------------ vLLM
    async def _llm_get(self, path: str, timeout: float = 10) -> httpx.Response:
        return await self.http.get(f"{self.s.vllm_url}{path}", timeout=timeout)

    async def llm_healthy(self) -> bool:
        try:
            return (await self._llm_get("/health", 5)).status_code == 200
        except httpx.HTTPError:
            return False

    async def llm_is_sleeping(self) -> bool:
        r = await self._llm_get("/is_sleeping")
        r.raise_for_status()
        return bool(r.json().get("is_sleeping"))

    async def _wait_llm(self, timeout: float, want_awake: bool = True) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if await self.llm_healthy() and (not want_awake or not await self.llm_is_sleeping()):
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        raise TimeoutError("vLLM did not become ready in time")

    async def llm_sleep(self, abort: bool) -> None:
        await self._wait_llm(self.s.llm_wake_timeout, want_awake=False)
        if await self.llm_is_sleeping():
            return
        mode = "abort" if abort else "wait"
        r = await self.http.post(f"{self.s.vllm_url}/sleep", params={"level": 1, "mode": mode}, timeout=600)
        r.raise_for_status()
        log.info("vLLM sleeping (mode=%s)", mode)

    async def llm_wake(self) -> None:
        await self._wait_llm(self.s.llm_wake_timeout, want_awake=False)
        if await self.llm_is_sleeping():
            r = await self.http.post(f"{self.s.vllm_url}/wake_up", timeout=600)
            r.raise_for_status()
        await self._wait_llm(self.s.llm_wake_timeout)
        log.info("vLLM awake")

    async def llm_ensure_awake(self) -> None:
        await self.llm_wake()

    # ----------------------------------------------------------- media (GPU)
    async def media_load(self, kind: str) -> None:
        r = await self.http.post(f"{self.s.media_url}/load", json={"kind": kind}, timeout=1800)
        r.raise_for_status()

    async def media_unload(self) -> None:
        try:
            r = await self.http.post(f"{self.s.media_url}/unload", timeout=120)
            r.raise_for_status()
        except httpx.ConnectError:
            log.warning("media service unreachable during unload (treated as unloaded)")

    async def _media_generate(self, kind: str, job: dict[str, Any], on_progress) -> dict[str, Any]:
        payload = {**job["params"], "job_id": job["id"], "user": safe_user(job["user"])}

        async def poll() -> None:
            while True:
                await asyncio.sleep(1.5)
                try:
                    r = await self.http.get(f"{self.s.media_url}/progress/{job['id']}", timeout=5)
                    if r.status_code == 200:
                        await on_progress(r.json().get("progress", 0))
                except httpx.HTTPError:
                    pass

        poller = asyncio.create_task(poll())
        try:
            r = await self.http.post(f"{self.s.media_url}/generate/{kind}", json=payload, timeout=None)
        finally:
            poller.cancel()
        if r.status_code != 200:
            raise RuntimeError(_error_text(r))
        return r.json()

    # ---------------------------------------------------------------- deploy
    def read_llm_config(self) -> dict[str, str]:
        cfg: dict[str, str] = {}
        p = Path(self.s.llm_config_path)
        if p.is_file():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    try:
                        cfg[k.strip()] = shlex.split(v)[0] if v.strip() else ""
                    except ValueError:
                        cfg[k.strip()] = v.strip()
        return cfg

    def _write_llm_config(self, cfg: dict[str, str]) -> None:
        p = Path(self.s.llm_config_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text("".join(f"{k}={shlex.quote(cfg[k])}\n" for k in LLM_ENV_KEYS if cfg.get(k)))
        os.replace(tmp, p)

    async def _deploy(self, job: dict[str, Any], on_progress) -> dict[str, Any]:
        params = job["params"]
        old = self.read_llm_config()
        new = {
            "LLM_MODEL": params["model"],
            "LLM_MAX_MODEL_LEN": str(int(params.get("max_model_len", 16384))),
            "LLM_GPU_UTIL": str(float(params.get("gpu_memory_utilization", 0.78))),
            "LLM_EXTRA_ARGS": params.get("extra_args", ""),
        }
        self._write_llm_config(new)
        await on_progress(0.1)
        await docker_ops.restart(self.s.vllm_container)
        try:
            await self._wait_llm(self.s.llm_deploy_timeout)
        except TimeoutError:
            log.error("deploy of %s failed, restoring previous config", new["LLM_MODEL"])
            logs = await docker_ops.logs(self.s.vllm_container, tail=40)
            if old:
                self._write_llm_config(old)
            else:
                Path(self.s.llm_config_path).unlink(missing_ok=True)
            await docker_ops.restart(self.s.vllm_container)
            await self._wait_llm(self.s.llm_deploy_timeout)
            raise RuntimeError(f"vLLM failed to start {new['LLM_MODEL']}; previous model restored.\n{logs[-3000:]}")
        return {"model": new["LLM_MODEL"], "config": new}

    # -------------------------------------------------------------- dispatch
    async def run_job(self, job: dict[str, Any], on_progress) -> dict[str, Any]:
        kind = job["type"]
        if kind in ("image", "video"):
            return await self._media_generate(kind, job, on_progress)
        if kind == "deploy":
            return await self._deploy(job, on_progress)
        raise ValueError(f"unknown job type {kind}")


def _error_text(r: httpx.Response) -> str:
    try:
        body = r.json()
        return str(body.get("detail") or body)[:2000]
    except ValueError:
        return f"HTTP {r.status_code}: {r.text[:2000]}"
