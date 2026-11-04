"""Docker SDK helpers, restricted to containers of this compose project."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import docker
from docker.errors import NotFound

from .config import get_settings

__all__ = ["NotFound", "list_containers", "start", "stop", "restart", "logs", "status"]


@lru_cache
def _client() -> docker.DockerClient:
    return docker.from_env(timeout=120)


def _get(name: str):
    c = _client().containers.get(name)
    project = c.labels.get("com.docker.compose.project")
    if project != get_settings().compose_project:
        raise PermissionError(f"container {name} is not part of project {get_settings().compose_project}")
    return c


def _list() -> list[dict[str, Any]]:
    project = get_settings().compose_project
    out = []
    for c in _client().containers.list(all=True, filters={"label": f"com.docker.compose.project={project}"}):
        state = c.attrs.get("State", {})
        out.append(
            {
                "name": c.name,
                "service": c.labels.get("com.docker.compose.service", ""),
                "image": (c.image.tags or [c.attrs.get("Config", {}).get("Image", "")])[0] if c.image else "",
                "status": c.status,
                "health": (state.get("Health") or {}).get("Status", ""),
                "started": state.get("StartedAt", ""),
            }
        )
    return sorted(out, key=lambda x: x["service"])


async def list_containers() -> list[dict[str, Any]]:
    return await asyncio.to_thread(_list)


async def start(name: str) -> None:
    await asyncio.to_thread(lambda: _get(name).start())


async def stop(name: str, timeout: int = 30) -> None:
    def _stop():
        c = _get(name)
        if c.status in ("running", "restarting", "paused"):
            c.stop(timeout=timeout)

    await asyncio.to_thread(_stop)


async def restart(name: str, timeout: int = 30) -> None:
    await asyncio.to_thread(lambda: _get(name).restart(timeout=timeout))


async def status(name: str) -> str:
    def _status():
        c = _get(name)
        c.reload()
        return c.status

    return await asyncio.to_thread(_status)


async def logs(name: str, tail: int = 300) -> str:
    def _logs():
        return _get(name).logs(tail=tail, timestamps=True).decode("utf-8", errors="replace")

    return await asyncio.to_thread(_logs)
