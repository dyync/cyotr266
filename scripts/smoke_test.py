r"""End-to-end smoke / load test against the orchestrator API.

Runs in a throw-away backend container (httpx + the shared /data volume):

    docker compose run --rm --no-deps -e ORCH_URL=http://backend:7861 \
        -v ./scripts:/scripts:ro -v ./video/dragon.png:/assets/dragon.png:ro \
        -v ./audio/nk.mp3:/assets/nk.mp3:ro backend python /scripts/smoke_test.py

    add  -e LOAD=1  for the concurrency test, -e SKIP=video,trellis to skip slow parts.

Env: ORCH_URL, INTERNAL_TOKEN + API_KEYS (taken from the container env), DATA_DIR,
     ASSETS_DIR (default /assets), SKIP, LOAD.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import httpx

ORCH = os.getenv("ORCH_URL", "http://127.0.0.1:7861")
TOKEN = os.environ["INTERNAL_TOKEN"]
API_KEY = (os.getenv("API_KEYS", "").split(",") or [""])[0].strip()
DATA = Path(os.getenv("DATA_DIR", "/data"))
ASSETS = Path(os.getenv("ASSETS_DIR", "/assets"))
SKIP = set(filter(None, os.getenv("SKIP", "").split(",")))
RESULTS: list[tuple[str, bool, str]] = []


def hdr(user: str = "smoke", admin: bool = False) -> dict[str, str]:
    return {"X-Internal-Token": TOKEN, "X-User": user, "X-User-Role": "admin" if admin else "user"}


def record(name: str, ok: bool, info: str = "") -> None:
    RESULTS.append((name, ok, info))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {info}", flush=True)


def stage(name: str) -> str:
    src = ASSETS / name
    dst_dir = DATA / "uploads" / "smoke"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / name
    shutil.copyfile(src, dst)
    return str(dst)


async def chat(c: httpx.AsyncClient, user: str, prompt: str, thinking: bool = False) -> tuple[str, float]:
    t0 = time.monotonic()
    first = None
    text = ""
    body = {
        "model": "chat",
        "stream": True,
        "max_tokens": 128,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    async with c.stream("POST", f"{ORCH}/v1/chat/completions", json=body, headers=hdr(user), timeout=900) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {(await r.aread())[:300]!r}")
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"] if json.loads(line[6:]).get("choices") else {}
            if delta.get("content"):
                first = first or time.monotonic() - t0
                text += delta["content"]
    return text, first or -1


async def run_job(c: httpx.AsyncClient, jtype: str, params: dict, user: str = "smoke", timeout: float = 1800) -> dict:
    r = await c.post(f"{ORCH}/jobs", json={"type": jtype, "params": params}, headers=hdr(user))
    r.raise_for_status()
    job = r.json()
    deadline = time.monotonic() + timeout
    positions = []
    while time.monotonic() < deadline:
        job = (await c.get(f"{ORCH}/jobs/{job['id']}", headers=hdr(user))).json()
        if job["status"] == "queued":
            positions.append(job["position"])
        if job["status"] in ("done", "failed", "cancelled"):
            job["_positions"] = positions
            return job
        await asyncio.sleep(1)
    raise TimeoutError(f"{jtype} job timed out")


async def smoke(c: httpx.AsyncClient) -> None:
    r = await c.get(f"{ORCH}/health")
    record("orchestrator health", r.status_code == 200, r.text)

    try:
        text, ttft = await chat(c, "smoke", "Say hello in German, one word.")
        record("chat streaming", bool(text.strip()), f"ttft={ttft:.2f}s reply={text.strip()[:60]!r}")
    except Exception as e:  # noqa: BLE001
        record("chat streaming", False, str(e))

    job = await run_job(c, "image", {"prompt": "a red fox in snow, photo", "width": 1024, "height": 1024, "steps": 9})
    ok = job["status"] == "done" and Path(job["result"]["path"]).is_file()
    record("image job", ok, json.dumps(job.get("result") or job.get("error"))[:200])

    with open(stage("nk.mp3"), "rb") as f:
        r = await c.post(f"{ORCH}/transcribe", headers=hdr(), files={"file": ("nk.mp3", f)}, data={"response_format": "verbose_json"}, timeout=1800)
    body = r.json() if r.status_code == 200 else {}
    record("transcription nk.mp3", r.status_code == 200 and len(body.get("text", "")) > 50, f"lang={body.get('language')} t={body.get('processing_seconds')}s")

    img = stage("dragon.png")
    for style, params in {
        "oil": {"pinsel": 10, "radius": 25},
        "aquarell": {"pinsel": 170, "radius": 0.62},
        "bleistift": {"buntstift": "Buntstift", "pinsel": 20, "radius": 0.18, "schatten": 0.05},
        "point": {"colors": 20},
    }.items():
        r = await c.post(f"{ORCH}/vip", headers=hdr(), json={"style": style, "image_path": img, "params": params})
        record(f"vip {style}", r.status_code == 200 and Path(r.json()["path"]).is_file(), r.text[:120])

    if "video" not in SKIP:
        job = await run_job(c, "video", {"prompt": "waves rolling onto a beach at sunset", "resolution": "480p", "num_frames": 49, "steps": 20})
        record("video job (480p, 2s)", job["status"] == "done", json.dumps(job.get("result") or job.get("error"))[:200])
    if "trellis" not in SKIP:
        job = await run_job(c, "trellis", {"image_path": img})
        record("trellis job", job["status"] == "done", json.dumps(job.get("result") or job.get("error"))[:200])

    r = await c.post(f"{ORCH}/api", json={"method": "vllm", "prompt": "hi"})
    record("/api rejects missing key", r.status_code == 401, str(r.status_code))
    if API_KEY:
        r = await c.post(f"{ORCH}/api", json={"method": "vllm", "prompt": "Say OK"}, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=600)
        record("/api vllm with key", r.status_code == 200 and bool(r.json()), r.text[:80])
        r = await c.post(f"{ORCH}/api", json={"method": "image", "prompt": "a lighthouse"}, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=1200)
        url = r.json() if r.status_code == 200 else ""
        ok = r.status_code == 200 and "/images/api-" in url
        if ok:
            img_r = await c.get(f"{ORCH}/images/{url.split('/images/')[1]}")
            ok = img_r.status_code == 200 and img_r.headers["content-type"] == "image/png"
        record("/api image with key + fetch", ok, url)

    r = await c.get(f"{ORCH}/admin/stats", headers=hdr())
    record("admin endpoints reject non-admin", r.status_code == 403, str(r.status_code))
    r = await c.get(f"{ORCH}/admin/stats", headers=hdr(admin=True))
    record("admin stats", r.status_code == 200, json.dumps(r.json().get("scheduler"))[:200])


async def load(c: httpx.AsyncClient) -> None:
    """3 users: 10 parallel chats, 3 image jobs, 2 transcriptions at the same time."""
    users = ["u1", "u2", "u3"]
    mp3 = stage("nk.mp3")

    async def transcribe(u: str):
        with open(mp3, "rb") as f:
            r = await c.post(f"{ORCH}/transcribe", headers=hdr(u), files={"file": ("nk.mp3", f)}, data={"response_format": "json"}, timeout=1800)
        return r.status_code

    t0 = time.monotonic()
    chats = [chat(c, users[i % 3], f"Count to {i + 3}.") for i in range(10)]
    images = [run_job(c, "image", {"prompt": f"test image {i}", "steps": 9}, user=users[i]) for i in range(3)]
    asr = [transcribe(users[i]) for i in range(2)]
    res = await asyncio.gather(*chats, *images, *asr, return_exceptions=True)
    chat_res, img_res, asr_res = res[:10], res[10:13], res[13:]
    ok_chats = [r for r in chat_res if isinstance(r, tuple) and r[0]]
    record("load: 10 parallel chats", len(ok_chats) == 10, f"ok={len(ok_chats)} ttft max={max((r[1] for r in ok_chats), default=-1):.1f}s errors={[str(r)[:80] for r in chat_res if not isinstance(r, tuple)]}")
    ok_img = [r for r in img_res if isinstance(r, dict) and r["status"] == "done"]
    record("load: 3 image jobs", len(ok_img) == 3, f"queue positions seen={[r.get('_positions', [])[:3] for r in ok_img]}")
    record("load: 2 transcriptions", all(r == 200 for r in asr_res), str(asr_res))
    record("load: wall time", True, f"{time.monotonic() - t0:.1f}s")


async def main() -> int:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60, read=1800)) as c:
        if os.getenv("LOAD"):
            await load(c)
        else:
            await smoke(c)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__" or __name__ == "<stdin>":
    sys.exit(asyncio.run(main()))
