"""Admin tab (only rendered usable for users with admin = true)."""

from __future__ import annotations

import gradio as gr
import pandas as pd

from .core import BACKEND, DEFAULTS, current_user, describe, watch_job


def _admin(request: gr.Request):
    user = current_user(request)
    if not user.admin:
        raise gr.Error("Admins only")
    return user


async def refresh_stats(request: gr.Request):
    user = _admin(request)
    s = await BACKEND.call(user, "GET", "/admin/stats")
    sch = s["scheduler"]
    gpu_df = pd.DataFrame(s.get("gpu", []))
    disk_df = pd.DataFrame(s.get("disk", []))
    llm = s.get("llm") or {}
    md = (
        f"**GPU holder:** `{sch['holder']}` for {sch['holder_seconds']}s · **queued jobs:** {sch['queued']} · "
        f"**active chats:** {sch['chats_active']} · **waiting chats:** {sch['chats_waiting']}  \n"
        f"**LLM:** `{llm.get('LLM_MODEL', 'compose default')}`"
    )
    return md, gpu_df, disk_df


async def refresh_containers(request: gr.Request):
    user = _admin(request)
    rows = await BACKEND.call(user, "GET", "/admin/containers")
    names = [r["name"] for r in rows]
    return pd.DataFrame(rows), gr.update(choices=names, value=names[0] if names else None)


async def container_logs(name: str, request: gr.Request):
    user = _admin(request)
    if not name:
        return ""
    res = await BACKEND.call(user, "GET", f"/admin/containers/{name}/logs", params={"tail": 400})
    return "\n".join(reversed(res["logs"].splitlines()))


def container_action(action: str):
    async def fn(name: str, request: gr.Request):
        user = _admin(request)
        await BACKEND.call(user, "POST", f"/admin/containers/{name}/{action}")
        gr.Info(f"{action} {name}: ok")

    return fn


async def hf_search(q: str, request: gr.Request):
    user = _admin(request)
    if len(q.strip()) < 2:
        raise gr.Error("Type at least 2 characters")
    rows = await BACKEND.call(user, "GET", "/admin/hf/search", params={"q": q.strip(), "limit": 50})
    ids = [r["id"] for r in rows]
    return gr.update(choices=ids, value=ids[0] if ids else None, label=f"{len(ids)} models")


async def hf_info(model_id: str, request: gr.Request):
    user = _admin(request)
    if not model_id:
        return None, ""
    info = await BACKEND.call(user, "GET", "/admin/hf/info", params={"model_id": model_id})
    verdict = (
        "✅ should fit next to the ASR model" if info.get("fits_4090")
        else "⚠️ probably too large for 24 GB at the configured utilisation (try an FP8/AWQ variant)"
        if info.get("est_vram_gb") else "❔ size unknown"
    )
    return info, f"Estimated weights in VRAM: **{info.get('est_vram_gb')} GB** — {verdict}"


async def deploy(model_id: str, max_len: int, util: float, extra: str, request: gr.Request):
    user = _admin(request)
    if not model_id:
        raise gr.Error("Select a model first")
    job = await BACKEND.call(
        user,
        "POST",
        "/admin/llm/deploy",
        json={"model": model_id, "max_model_len": int(max_len), "gpu_memory_utilization": float(util), "extra_args": extra},
    )
    yield f"Deploy job queued ({job['id'][:8]}) — chats pause while vLLM restarts with `{model_id}`."
    async for job in watch_job(user, job, poll=3):
        yield describe(job) + (f"\n\n```\n{job['error'][-1500:]}\n```" if job["status"] == "failed" else "")


def build() -> gr.Timer:
    gr.Markdown("### System")
    stats_md = gr.Markdown()
    with gr.Row():
        gpu_df = gr.Dataframe(label="GPU", interactive=False)
        disk_df = gr.Dataframe(label="Disk", interactive=False)
    timer = gr.Timer(5, active=False)  # activated on load for admins only
    timer.tick(refresh_stats, None, [stats_md, gpu_df, disk_df], show_progress="hidden", concurrency_limit=None)

    with gr.Accordion("Containers", open=False):
        cont_df = gr.Dataframe(interactive=False)
        with gr.Row():
            cont_name = gr.Dropdown(label="Container", choices=[])
            btn_refresh = gr.Button("Refresh")
            btn_logs = gr.Button("Logs")
            btn_restart = gr.Button("Restart", variant="stop")
            btn_stop = gr.Button("Stop", variant="stop")
            btn_start = gr.Button("Start")
        logs = gr.Code(label="Logs (newest first)", language=None, lines=20)
    btn_refresh.click(refresh_containers, None, [cont_df, cont_name])
    btn_logs.click(container_logs, cont_name, logs)
    btn_restart.click(container_action("restart"), cont_name, None)
    btn_stop.click(container_action("stop"), cont_name, None)
    btn_start.click(container_action("start"), cont_name, None)

    with gr.Accordion("Language model", open=True):
        gr.Markdown(
            "Switching the model restarts vLLM: running chats are drained, then everyone waits until the new model is "
            "loaded (first download can take a while). If the new model fails to start, the previous one is restored."
        )
        with gr.Row():
            suggested = gr.Dropdown(DEFAULTS["suggested_llms"], label="Suggested models", allow_custom_value=True)
            query = gr.Textbox(label="Search Hugging Face (text-generation)", placeholder="e.g. Qwen3.5 FP8")
            btn_search = gr.Button("Search")
        found = gr.Dropdown(label="Results", choices=[], allow_custom_value=True)
        info_md = gr.Markdown()
        info_json = gr.JSON(label="Model info")
        with gr.Row():
            max_len = gr.Slider(2048, 65536, value=16384, step=1024, label="max_model_len")
            util = gr.Slider(0.4, 0.9, value=0.78, step=0.01, label="gpu_memory_utilization")
            extra = gr.Textbox(label="Extra vLLM args", placeholder="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder")
        btn_deploy = gr.Button("Deploy model", variant="primary")
        deploy_status = gr.Markdown()
    suggested.change(lambda m: gr.update(value=m), suggested, found)
    btn_search.click(hf_search, query, found)
    query.submit(hf_search, query, found)
    found.change(hf_info, found, [info_json, info_md])
    btn_deploy.click(deploy, [found, max_len, util, extra], deploy_status)
    return timer
