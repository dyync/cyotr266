"""Scheduler tests against fake Redis and fake GPU backends (no GPU needed)."""

from __future__ import annotations

import asyncio
import os

import fakeredis
import pytest

os.environ.setdefault("INTERNAL_TOKEN", "test")

from orchestrator.config import Settings  # noqa: E402
from orchestrator.scheduler import GpuBusy, LimitExceeded, Scheduler  # noqa: E402


class FakeBackends:
    def __init__(self, job_seconds: float = 0.05):
        self.events: list[str] = []
        self.job_seconds = job_seconds
        self.fail_types: set[str] = set()
        self.llm_awake = True
        self.media_loaded: str | None = None
        self.gpu_users: set[str] = set()  # what is resident on the GPU right now
        self.max_concurrent_gpu = 0

    def _track(self):
        self.max_concurrent_gpu = max(self.max_concurrent_gpu, len(self.gpu_users))

    async def llm_sleep(self, abort: bool) -> None:
        self.events.append(f"llm_sleep(abort={abort})")
        self.llm_awake = False
        self.gpu_users.discard("llm")

    async def llm_wake(self) -> None:
        self.events.append("llm_wake")
        self.llm_awake = True
        self.gpu_users.add("llm")
        self._track()

    async def llm_ensure_awake(self) -> None:
        await self.llm_wake()

    async def media_load(self, kind: str) -> None:
        self.events.append(f"media_load({kind})")
        self.gpu_users.discard("image")
        self.gpu_users.discard("video")
        self.media_loaded = kind
        self.gpu_users.add(kind)
        self._track()

    async def media_unload(self) -> None:
        self.events.append("media_unload")
        self.media_loaded = None
        self.gpu_users.discard("image")
        self.gpu_users.discard("video")

    async def run_job(self, job, on_progress):
        t = job["type"]
        assert t == "deploy" or t in self.gpu_users, f"{t} ran without its model loaded: {self.gpu_users}"
        assert "llm" not in self.gpu_users or t == "deploy", "LLM still resident during media job"
        self.events.append(f"run({t}:{job['params'].get('n')})")
        await on_progress(0.5)
        await asyncio.sleep(self.job_seconds)
        if t in self.fail_types:
            raise RuntimeError("boom")
        return {"path": f"/data/outputs/x/{job['id']}.png"}


def make(**overrides) -> tuple[Scheduler, FakeBackends]:
    cfg = dict(
        internal_token="t",
        drain_timeout=1,
        chat_wait_timeout=5,
        llm_min_window=0.3,
        batch_max_jobs=2,
        batch_max_seconds=60,
        max_jobs_per_user=3,
    )
    settings = Settings(**{**cfg, **overrides})
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    backends = FakeBackends()
    return Scheduler(redis, backends, settings), backends


async def wait_status(s: Scheduler, job_id: str, statuses=("done", "failed"), timeout=5.0):
    async with asyncio.timeout(timeout):
        while True:
            job = await s.get_job(job_id)
            if job["status"] in statuses:
                return job
            await asyncio.sleep(0.02)


async def wait_holder(s: Scheduler, holder: str, timeout=5.0):
    async with asyncio.timeout(timeout):
        while s.holder != holder:
            await asyncio.sleep(0.02)


@pytest.fixture
async def sched():
    s, b = make()
    s.start()
    await wait_holder(s, "llm")
    yield s, b
    await s.stop()


async def test_init_returns_home_and_cleans_gpu(sched):
    s, b = sched
    assert s.holder == "llm"
    assert "media_unload" in b.events and "llm_wake" in b.events


async def test_image_job_swaps_and_returns_home(sched):
    s, b = sched
    b.events.clear()
    job = await s.submit("alice", "image", {"n": 1})
    done = await wait_status(s, job["id"])
    assert done["status"] == "done" and done["result"]["path"].endswith(".png")
    await wait_holder(s, "llm")
    assert b.events == ["llm_sleep(abort=False)", "media_load(image)", "run(image:1)", "media_unload", "llm_wake"]
    assert b.max_concurrent_gpu == 1  # never two models resident at once


async def test_chat_waits_while_media_job_runs(sched):
    s, b = sched
    b.job_seconds = 0.5
    job = await s.submit("alice", "video", {"n": 1})
    await wait_holder(s, "video")
    t0 = asyncio.get_running_loop().time()
    async with s.chat_lease():
        assert s.holder == "llm"  # chat only proceeds once LLM is home
    assert asyncio.get_running_loop().time() - t0 > 0.2
    assert (await s.get_job(job["id"]))["status"] == "done"


async def test_running_chat_is_drained_before_sleep(sched):
    s, b = sched
    b.events.clear()
    order: list[str] = []

    async def chat():
        async with s.chat_lease():
            await asyncio.sleep(0.3)
            order.append("chat_done")

    task = asyncio.create_task(chat())
    await asyncio.sleep(0.05)
    job = await s.submit("bob", "image", {"n": 1})
    await wait_status(s, job["id"])
    await task
    sleep_idx = b.events.index("llm_sleep(abort=False)")
    assert order == ["chat_done"] and sleep_idx == 0


async def test_drain_timeout_aborts(sched):
    s, b = sched
    b.events.clear()
    release = asyncio.Event()

    async def long_chat():
        async with s.chat_lease():
            await release.wait()

    task = asyncio.create_task(long_chat())
    await asyncio.sleep(0.05)
    job = await s.submit("bob", "image", {"n": 1})
    await asyncio.sleep(1.3)  # drain_timeout=1
    assert "llm_sleep(abort=True)" in b.events
    release.set()
    await task
    await wait_status(s, job["id"])


async def test_same_type_jobs_are_batched(sched):
    s, b = sched
    b.events.clear()
    jobs = [await s.submit(u, "image", {"n": i}) for i, u in enumerate(["a", "b", "c"])]
    for j in jobs:
        await wait_status(s, j["id"])
    await wait_holder(s, "llm")
    assert b.events.count("media_load(image)") == 1
    assert b.events.count("llm_wake") == 1


async def test_batch_yields_to_waiting_chats():
    s, b = make()
    b.job_seconds = 0.2
    s.start()
    await wait_holder(s, "llm")
    b.events.clear()
    jobs = [await s.submit(f"u{i}", "image", {"n": i}) for i in range(4)]
    await wait_holder(s, "image")
    async with s.chat_lease():  # waits until the worker yields after batch_max_jobs=2
        served_after = [e for e in b.events if e.startswith("run")]
    assert len(served_after) == 2
    for j in jobs:
        await wait_status(s, j["id"], timeout=10)
    await s.stop()


async def test_direct_switch_between_media_types_without_chats(sched):
    s, b = sched
    b.events.clear()
    j1 = await s.submit("a", "image", {"n": 1})
    j2 = await s.submit("b", "video", {"n": 2})
    j3 = await s.submit("c", "image", {"n": 3})
    for j in (j1, j2, j3):
        await wait_status(s, j["id"])
    await wait_holder(s, "llm")
    assert b.events.count("llm_wake") == 1  # no detour through the LLM
    loads = [e for e in b.events if e.startswith("media_load")]
    assert loads == ["media_load(image)", "media_load(video)", "media_load(image)"]
    assert b.max_concurrent_gpu == 1


async def test_failed_job_reports_error_and_frees_slot(sched):
    s, b = sched
    b.fail_types.add("image")
    job = await s.submit("alice", "image", {"n": 1})
    failed = await wait_status(s, job["id"])
    assert failed["status"] == "failed" and "boom" in failed["error"]
    assert await s._active_jobs("alice") == {}
    await wait_holder(s, "llm")


async def test_per_user_limits():
    s, _ = make()  # worker not started: jobs stay queued
    for i in range(3):
        await s.submit("alice", "image", {"n": i})
    with pytest.raises(LimitExceeded):
        await s.submit("alice", "image", {"n": 9})
    await s.submit("bob", "video", {"n": 1})
    with pytest.raises(LimitExceeded):
        await s.submit("bob", "video", {"n": 2})


async def test_queue_position_and_cancel():
    s, _ = make()
    a = await s.submit("alice", "image", {"n": 1})
    b_ = await s.submit("bob", "image", {"n": 2})
    assert (await s.get_job(a["id"]))["position"] == 1
    assert (await s.get_job(b_["id"]))["position"] == 2
    assert not await s.cancel(a["id"], "bob")  # not yours
    assert await s.cancel(a["id"], "alice")
    assert (await s.get_job(a["id"]))["status"] == "cancelled"
    assert (await s.get_job(b_["id"]))["position"] == 1


async def test_deploy_returns_home_without_wake(sched):
    s, b = sched
    b.events.clear()
    job = await s.submit("admin", "deploy", {"model": "x/y"})
    await wait_status(s, job["id"])
    await wait_holder(s, "llm")
    assert "llm_sleep(abort=False)" not in b.events and "llm_wake" not in b.events
    async with s.chat_lease():
        pass


async def test_crash_recovery_fails_interrupted_jobs():
    s, b = make()
    job = await s.submit("alice", "image", {"n": 1})
    await s.r.lmove("gpu:queue", "gpu:running", "LEFT", "RIGHT")
    await s.r.hset(f"job:{job['id']}", "status", "running")
    s.start()
    failed = await wait_status(s, job["id"])
    assert failed["status"] == "failed" and "restart" in failed["error"]
    await s.stop()


async def test_chat_timeout_raises_gpubusy():
    s, b = make(chat_wait_timeout=0.2)
    # never started -> GPU stays exclusive (init not done)
    with pytest.raises(GpuBusy):
        async with s.chat_lease():
            pass
