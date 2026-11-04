"""Shared frontend plumbing: config, users/auth, orchestrator client, job watching.

Everything that is per-user lives in gr.State / component values; nothing here
is mutable per-session module state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import bcrypt
import gradio as gr
import httpx
from openai import AsyncOpenAI

log = logging.getLogger("webui")

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:7861")
INTERNAL_TOKEN = os.environ["INTERNAL_TOKEN"]
USERS_FILE = Path(os.getenv("USERS_FILE", "/config/users.toml"))
DATA = Path(os.getenv("DATA_DIR", "/data"))
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
DEFAULTS = json.loads(Path(os.getenv("DEFAULTS_FILE", "/app/utils/defaults.json")).read_text(encoding="utf-8"))["frontend"]


# ----------------------------------------------------------------- users/auth
@dataclass(frozen=True)
class User:
    name: str
    admin: bool


_DUMMY_HASH = bcrypt.hashpw(b"dummy", bcrypt.gensalt(12))


class UserStore:
    """users.toml: [users.<name>] password_hash = "$2b$..." / admin = true. Hot-reloaded."""

    def __init__(self, path: Path):
        self.path = path
        self._mtime = 0.0
        self._users: dict[str, dict[str, Any]] = {}

    def _reload(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._users = {}
            return
        if mtime != self._mtime:
            self._users = tomllib.loads(self.path.read_text(encoding="utf-8")).get("users", {})
            self._mtime = mtime
            log.info("loaded %d user(s) from %s", len(self._users), self.path)

    def check(self, username: str, password: str) -> bool:
        self._reload()
        entry = self._users.get(username)
        if not entry or not entry.get("password_hash"):
            bcrypt.checkpw(b"x", _DUMMY_HASH)  # same cost for unknown users
            return False
        try:
            return bcrypt.checkpw(password.encode(), entry["password_hash"].encode())
        except ValueError:
            return False

    def is_admin(self, username: str | None) -> bool:
        self._reload()
        return bool(username and self._users.get(username, {}).get("admin"))


USERS = UserStore(USERS_FILE)


def current_user(request: gr.Request | None) -> User:
    name = getattr(request, "username", None) if request else None
    if not name:
        raise gr.Error("Not logged in")
    return User(name=name, admin=USERS.is_admin(name))


# ---------------------------------------------------------- orchestrator API
class Backend:
    def __init__(self) -> None:
        self.http = httpx.AsyncClient(
            base_url=BACKEND_URL,
            headers={"X-Internal-Token": INTERNAL_TOKEN},
            timeout=httpx.Timeout(30, read=1800),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
        self.openai = AsyncOpenAI(base_url=f"{BACKEND_URL}/v1", api_key=INTERNAL_TOKEN, timeout=900, max_retries=0)

    @staticmethod
    def headers(user: User) -> dict[str, str]:
        return {"X-User": user.name, "X-User-Role": "admin" if user.admin else "user"}

    async def call(self, user: User, method: str, path: str, **kw) -> Any:
        try:
            r = await self.http.request(method, path, headers=self.headers(user), **kw)
        except httpx.HTTPError as e:
            raise gr.Error(f"Backend unreachable: {e}") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail") or r.json().get("error") or r.text
            except ValueError:
                detail = r.text
            raise gr.Error(str(detail)[:500])
        return r.json()

    async def submit(self, user: User, job_type: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self.call(user, "POST", "/jobs", json={"type": job_type, "params": params})


BACKEND = Backend()


def stage_upload(path: str | None, user: User) -> str | None:
    """Copy a Gradio temp upload into the shared /data/uploads volume."""
    if not path:
        return None
    d = UPLOADS / user.name
    d.mkdir(parents=True, exist_ok=True)
    dst = d / f"{uuid.uuid4().hex}{Path(path).suffix.lower()[:10]}"
    shutil.copyfile(path, dst)
    return str(dst)


HOLDER_LABEL = {
    "llm": "the language model",
    "image": "image generation",
    "video": "video generation",
    "maintenance": "a model switch",
    "unknown": "start-up",
}


def describe(job: dict[str, Any]) -> str:
    s = job["status"]
    gpu = job.get("gpu", {})
    if s == "queued":
        if not job.get("position"):
            return "⏳ Starting… (loading model onto the GPU)"
        holder = gpu.get("holder")
        busy = f" · GPU currently busy with {HOLDER_LABEL.get(holder, holder)}" if holder else ""
        return f"⏳ Queued — position **{job['position']}**{busy}"
    if s == "running":
        pct = int(round(job.get("progress", 0) * 100))
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        return f"⚙️ Running `{bar}` {pct}%"
    if s == "done":
        r = job.get("result") or {}
        extra = f" · {r['seconds']}s" if r.get("seconds") else ""
        extra += f" · seed {r['seed']}" if r.get("seed") is not None else ""
        return f"✅ Done{extra}"
    if s == "cancelled":
        return "🚫 Cancelled"
    return f"❌ Failed: {job.get('error', 'unknown error')}"


async def watch_job(user: User, job: dict[str, Any], poll: float = 1.0) -> AsyncIterator[dict[str, Any]]:
    """Yield job snapshots until it reaches a terminal state."""
    while True:
        job = await BACKEND.call(user, "GET", f"/jobs/{job['id']}")
        yield job
        if job["status"] in ("done", "failed", "cancelled"):
            return
        await asyncio.sleep(poll)


async def cancel_job(job_id: str | None, request: gr.Request) -> str:
    user = current_user(request)
    if not job_id:
        return "Nothing to cancel."
    await BACKEND.call(user, "DELETE", f"/jobs/{job_id}")
    return "🚫 Cancelled"
