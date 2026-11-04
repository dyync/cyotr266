"""Single-GPU scheduler: one "holder" owns the GPU at a time.

* ``llm`` is the home state. Chat requests take a *shared* lease so vLLM keeps
  batching across users.
* Media jobs (image / video / trellis / deploy) run from a FIFO queue in Redis.
  The worker takes the GPU *exclusively*: new chats wait (writer preference),
  running chats are drained (bounded by ``drain_timeout``), then vLLM is put to
  sleep and the target model is loaded.
* Consecutive jobs of the same type run back-to-back to amortise swaps, but if
  chats are waiting the worker yields back to the LLM for ``llm_min_window``
  seconds after ``batch_max_jobs`` / ``batch_max_seconds``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Protocol

from redis.asyncio import Redis

from .config import Settings

log = logging.getLogger("scheduler")

Q_PENDING = "gpu:queue"
Q_RUNNING = "gpu:running"
HOME = "llm"
MAINTENANCE = "maintenance"

JOB_TYPES = {"image", "video", "trellis", "deploy"}
# which holder a job type needs
HOLDER_FOR = {"image": "image", "video": "video", "trellis": "trellis", "deploy": MAINTENANCE}
ACTIVE = {"queued", "running"}


class GpuBusy(Exception):
    """Raised when a chat could not get the GPU within chat_wait_timeout."""


class LimitExceeded(Exception):
    pass


class Backends(Protocol):
    async def llm_sleep(self, abort: bool) -> None: ...
    async def llm_wake(self) -> None: ...
    async def llm_ensure_awake(self) -> None: ...
    async def media_load(self, kind: str) -> None: ...
    async def media_unload(self) -> None: ...
    async def trellis_start(self) -> None: ...
    async def trellis_stop(self) -> None: ...
    async def run_job(self, job: dict[str, Any], on_progress) -> dict[str, Any]: ...


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def user_key(user: str) -> str:
    return f"user_jobs:{user}"


class Scheduler:
    def __init__(self, redis: Redis, backends: Backends, settings: Settings):
        self.r = redis
        self.b = backends
        self.s = settings
        self.holder = "unknown"
        self.holder_since = time.monotonic()
        self.current_job: str | None = None
        self._cond = asyncio.Condition()
        self._readers = 0
        self._chat_waiters = 0
        self._exclusive = True  # chats wait until _init_gpu() finished
        self._llm_since = 0.0
        self._job_event = asyncio.Event()
        self._submit_lock = asyncio.Lock()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ chat
    @asynccontextmanager
    async def chat_lease(self):
        """Shared lease for one chat request (held for the whole stream)."""
        async with self._cond:
            self._chat_waiters += 1
            try:
                async with asyncio.timeout(self.s.chat_wait_timeout):
                    await self._cond.wait_for(lambda: not self._exclusive)
            except TimeoutError as e:
                raise GpuBusy(f"GPU busy with '{self.holder}' job") from e
            finally:
                self._chat_waiters -= 1
            self._readers += 1
        try:
            yield
        finally:
            async with self._cond:
                self._readers -= 1
                self._cond.notify_all()

    async def _enter_exclusive(self) -> bool:
        """Block new chats and wait for running ones. Returns True if drained."""
        async with self._cond:
            self._exclusive = True
            try:
                async with asyncio.timeout(self.s.drain_timeout):
                    await self._cond.wait_for(lambda: self._readers == 0)
                return True
            except TimeoutError:
                log.warning("drain timeout: %d chat(s) will be aborted", self._readers)
                return False

    async def _leave_exclusive(self) -> None:
        async with self._cond:
            self._exclusive = False
            self._cond.notify_all()

    # ------------------------------------------------------------- swapping
    async def _switch(self, target: str) -> None:
        if self.holder == target:
            return
        log.info("GPU swap %s -> %s", self.holder, target)
        prev = self.holder

        if prev == HOME:
            drained = await self._enter_exclusive()
            if target != MAINTENANCE:
                await self.b.llm_sleep(abort=not drained)
        elif prev == "unknown":
            await self._enter_exclusive()

        # release the previous non-LLM holder
        if prev in ("image", "video", "unknown") and target not in ("image", "video"):
            await self.b.media_unload()
        if prev in ("trellis", "unknown"):
            await self.b.trellis_stop()

        # acquire the target
        if target == HOME:
            if prev == "unknown":
                await self.b.llm_ensure_awake()
            elif prev != MAINTENANCE:
                await self.b.llm_wake()
            self._set_holder(HOME)
            self._llm_since = time.monotonic()
            await self._leave_exclusive()
            return
        if prev == "unknown" and target != MAINTENANCE:
            await self.b.llm_sleep(abort=True)
        if target in ("image", "video"):
            await self.b.media_load(target)
        elif target == "trellis":
            await self.b.trellis_start()
        self._set_holder(target)

    def _set_holder(self, holder: str) -> None:
        self.holder = holder
        self.holder_since = time.monotonic()

    async def _give_chats_a_window(self) -> None:
        """After returning home, let waiting chats run for a while."""
        start = time.monotonic()
        await asyncio.sleep(0)  # let woken chats grab their lease
        while time.monotonic() - start < self.s.llm_min_window:
            if self._readers == 0 and self._chat_waiters == 0:
                return
            await asyncio.sleep(0.5)

    # ----------------------------------------------------------------- jobs
    async def submit(self, user: str, job_type: str, params: dict[str, Any]) -> dict[str, Any]:
        if job_type not in JOB_TYPES:
            raise ValueError(f"unknown job type {job_type}")
        async with self._submit_lock:
            active = await self._active_jobs(user)
            if job_type != "deploy":
                if len(active) >= self.s.max_jobs_per_user:
                    raise LimitExceeded(
                        f"You already have {len(active)} job(s) queued/running "
                        f"(limit {self.s.max_jobs_per_user})."
                    )
                if job_type == "video" and sum(1 for t in active.values() if t == "video") >= self.s.max_video_jobs_per_user:
                    raise LimitExceeded("Only one video job per user at a time.")
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "type": job_type,
                "user": user,
                "status": "queued",
                "params": json.dumps(params),
                "progress": "0",
                "created": str(time.time()),
            }
            pipe = self.r.pipeline()
            pipe.hset(job_key(job_id), mapping=job)
            pipe.expire(job_key(job_id), self.s.job_ttl)
            pipe.sadd(user_key(user), job_id)
            pipe.expire(user_key(user), self.s.job_ttl)
            pipe.rpush(Q_PENDING, job_id)
            await pipe.execute()
        self._job_event.set()
        return await self.get_job(job_id)  # type: ignore[return-value]

    async def _active_jobs(self, user: str) -> dict[str, str]:
        """job_id -> type of the user's queued/running jobs (self-healing set)."""
        ids = await self.r.smembers(user_key(user))
        active: dict[str, str] = {}
        for jid in ids:
            status, jtype = await self.r.hmget(job_key(jid), "status", "type")
            if status in ACTIVE:
                active[jid] = jtype
            else:
                await self.r.srem(user_key(user), jid)
        return active

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        raw = await self.r.hgetall(job_key(job_id))
        if not raw:
            return None
        job: dict[str, Any] = dict(raw)
        job["params"] = json.loads(job.get("params") or "{}")
        job["result"] = json.loads(job["result"]) if job.get("result") else None
        job["progress"] = float(job.get("progress") or 0)
        if job["status"] == "queued":
            pos = await self.r.lpos(Q_PENDING, job_id)
            job["position"] = (pos + 1) if pos is not None else 0  # 0 = being picked up
        else:
            job["position"] = 0
        return job

    async def cancel(self, job_id: str, user: str, is_admin: bool = False) -> bool:
        job = await self.get_job(job_id)
        if not job or (job["user"] != user and not is_admin):
            return False
        if job["status"] != "queued":
            return False
        removed = await self.r.lrem(Q_PENDING, 0, job_id)
        if removed:
            await self.r.hset(job_key(job_id), mapping={"status": "cancelled", "finished": str(time.time())})
            await self.r.srem(user_key(job["user"]), job_id)
        return bool(removed)

    async def _update(self, job_id: str, **fields: Any) -> None:
        await self.r.hset(job_key(job_id), mapping={k: str(v) for k, v in fields.items()})

    async def _run(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        if not job or job["status"] != "queued":
            await self.r.lrem(Q_RUNNING, 0, job_id)
            return
        self.current_job = job_id
        await self._update(job_id, status="running", started=time.time())

        async def on_progress(p: float) -> None:
            await self._update(job_id, progress=round(max(0.0, min(1.0, p)), 3))

        try:
            result = await self.b.run_job(job, on_progress)
            await self.r.hset(
                job_key(job_id),
                mapping={"status": "done", "progress": "1", "result": json.dumps(result), "finished": str(time.time())},
            )
        except Exception as e:  # noqa: BLE001 - report every failure to the user
            log.exception("job %s (%s) failed", job_id, job["type"])
            await self._update(job_id, status="failed", error=str(e)[:2000], finished=time.time())
        finally:
            self.current_job = None
            await self.r.lrem(Q_RUNNING, 0, job_id)
            await self.r.srem(user_key(job["user"]), job_id)

    async def _peek_type(self) -> str | None:
        nxt = await self.r.lindex(Q_PENDING, 0)
        if nxt is None:
            return None
        return await self.r.hget(job_key(nxt), "type")

    # --------------------------------------------------------------- worker
    async def _recover(self) -> None:
        """Jobs that were running when the orchestrator died are failed."""
        while (job_id := await self.r.lpop(Q_RUNNING)) is not None:
            await self._update(job_id, status="failed", error="interrupted by orchestrator restart", finished=time.time())

    async def _init_gpu(self) -> None:
        await self._recover()
        while True:
            try:
                await self._switch(HOME)
                return
            except Exception:  # noqa: BLE001
                log.exception("GPU init failed, retrying in 10s")
                self.holder = "unknown"
                await asyncio.sleep(10)

    async def run_forever(self) -> None:
        await self._init_gpu()
        batch_count, batch_start = 0, 0.0
        while True:
            try:
                job_id = await self.r.lmove(Q_PENDING, Q_RUNNING, "LEFT", "RIGHT")
                if job_id is None:
                    if self.holder != HOME:
                        await self._switch(HOME)
                    self._job_event.clear()
                    try:
                        async with asyncio.timeout(2):
                            await self._job_event.wait()
                    except TimeoutError:
                        pass
                    continue

                jtype = await self.r.hget(job_key(job_id), "type")
                target = HOLDER_FOR.get(jtype or "", None)
                if target is None:
                    await self.r.lrem(Q_RUNNING, 0, job_id)
                    continue
                if self.holder != target:
                    await self._switch(target)
                    batch_count, batch_start = 0, time.monotonic()

                await self._run(job_id)
                batch_count += 1
                if target == MAINTENANCE:
                    # deploy restarted vLLM: it is awake again
                    self._set_holder(HOME)
                    self._llm_since = time.monotonic()
                    await self._leave_exclusive()
                    continue

                next_type = await self._peek_type()
                batch_exhausted = (
                    batch_count >= self.s.batch_max_jobs
                    or time.monotonic() - batch_start >= self.s.batch_max_seconds
                )
                chats_waiting = self._chat_waiters > 0
                if next_type is None or (chats_waiting and (next_type != jtype or batch_exhausted)):
                    await self._switch(HOME)
                    batch_count = 0
                    if next_type is not None:
                        await self._give_chats_a_window()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - keep the worker alive
                log.exception("scheduler loop error; resetting GPU state")
                self.holder = "unknown"
                await asyncio.sleep(5)
                await self._init_gpu()

    def start(self) -> None:
        self._task = asyncio.create_task(self.run_forever(), name="gpu-scheduler")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def state(self) -> dict[str, Any]:
        return {
            "holder": self.holder,
            "holder_seconds": round(time.monotonic() - self.holder_since, 1),
            "current_job": self.current_job,
            "queued": await self.r.llen(Q_PENDING),
            "chats_active": self._readers,
            "chats_waiting": self._chat_waiters,
        }
