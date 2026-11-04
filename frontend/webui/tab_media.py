"""Image, Video, 3D, VIP and Audio tabs. GPU work goes through the job queue."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from .core import BACKEND, DEFAULTS, cancel_job, current_user, describe, stage_upload, watch_job

NO_CHANGE = gr.skip()


# ---------------------------------------------------------------------- image
async def run_image(prompt: str, size: str, steps: int, seed: float, gallery: list | None, request: gr.Request):
    user = current_user(request)
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    w, h = DEFAULTS["image"]["sizes"][size]
    job = await BACKEND.submit(user, "image", {"prompt": prompt, "width": w, "height": h, "steps": int(steps), "seed": int(seed)})
    yield job["id"], describe({**job, "gpu": {}}), NO_CHANGE, NO_CHANGE
    async for job in watch_job(user, job):
        if job["status"] == "done":
            path = job["result"]["path"]
            yield None, describe(job), path, [path, *(gallery or [])][:24]
        else:
            yield job["id"], describe(job), NO_CHANGE, NO_CHANGE


def build_image() -> None:
    d = DEFAULTS["image"]
    job_id = gr.State(None)
    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(d["prompt"], label="Prompt", lines=4, max_length=4000)
            with gr.Row():
                size = gr.Dropdown(list(d["sizes"]), value=next(iter(d["sizes"])), label="Size")
                steps = gr.Slider(4, 20, value=d["steps"], step=1, label="Steps")
                seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
            with gr.Row():
                go = gr.Button("Generate image", variant="primary")
                stop = gr.Button("Cancel", variant="secondary")
            status = gr.Markdown()
        with gr.Column(scale=3):
            out = gr.Image(label="Result", type="filepath", interactive=False, height=520)
    gallery = gr.Gallery(label="Your images (this session)", columns=6, height="auto", interactive=False)
    ev = dict(fn=run_image, inputs=[prompt, size, steps, seed, gallery], outputs=[job_id, status, out, gallery], concurrency_limit=None)
    go.click(**ev)
    prompt.submit(**ev)
    stop.click(cancel_job, job_id, status)


# ---------------------------------------------------------------------- video
async def run_video(mode: str, prompt: str, image: str | None, resolution: str, seconds: int, steps: int, seed: float, request: gr.Request):
    user = current_user(request)
    if not prompt.strip():
        raise gr.Error("Please enter a prompt (also used for image→video).")
    if mode == "Image → Video" and not image:
        raise gr.Error("Please upload a start image.")
    frames = int(seconds) * 24 + 1
    params = {
        "prompt": prompt,
        "resolution": resolution,
        "num_frames": min(frames, 121),
        "steps": int(steps),
        "seed": int(seed),
        "fps": 24,
        "image_path": stage_upload(image, user) if mode == "Image → Video" else None,
    }
    job = await BACKEND.submit(user, "video", params)
    yield job["id"], describe({**job, "gpu": {}}), NO_CHANGE
    async for job in watch_job(user, job, poll=2):
        if job["status"] == "done":
            yield None, describe(job), job["result"]["path"]
        else:
            yield job["id"], describe(job), NO_CHANGE


def build_video() -> None:
    d = DEFAULTS["video"]
    job_id = gr.State(None)
    with gr.Row():
        with gr.Column(scale=2):
            mode = gr.Radio(["Text → Video", "Image → Video"], value="Text → Video", show_label=False)
            prompt = gr.Textbox(d["prompt"], label="Prompt", lines=4, max_length=4000)
            image = gr.Image(label="Start image", type="filepath", visible=False, height=260)
            with gr.Row():
                resolution = gr.Radio(["480p", "720p"], value=d["resolution"], label="Resolution")
                seconds = gr.Slider(2, 5, value=d["seconds"], step=1, label="Length (s, 24 fps)")
            with gr.Accordion("Advanced", open=False):
                steps = gr.Slider(10, 50, value=d["steps"], step=1, label="Steps")
                seed = gr.Number(-1, label="Seed (-1 = random)", precision=0)
            gr.Markdown(d["note"])
            with gr.Row():
                go = gr.Button("Generate video", variant="primary")
                stop = gr.Button("Cancel")
            status = gr.Markdown()
        with gr.Column(scale=3):
            out = gr.Video(label="Result", interactive=False, height=520)
    mode.change(lambda m: gr.update(visible=m == "Image → Video"), mode, image)
    go.click(run_video, [mode, prompt, image, resolution, seconds, steps, seed], [job_id, status, out], concurrency_limit=None)
    stop.click(cancel_job, job_id, status)


# ------------------------------------------------------------------------- 3D
async def run_trellis(image: str | None, request: gr.Request):
    user = current_user(request)
    if not image:
        raise gr.Error("Please upload an image.")
    job = await BACKEND.submit(user, "trellis", {"image_path": stage_upload(image, user)})
    yield job["id"], describe({**job, "gpu": {}}), NO_CHANGE, NO_CHANGE, NO_CHANGE
    async for job in watch_job(user, job, poll=2):
        if job["status"] != "done":
            yield job["id"], describe(job), NO_CHANGE, NO_CHANGE, NO_CHANGE
            continue
        files = list((job["result"].get("files") or {}).values()) or [job["result"]["path"]]
        video = next((f for f in files if f.endswith((".mp4", ".webm"))), None)
        model = next((f for f in files if f.endswith((".glb", ".gltf", ".obj", ".ply"))), None)
        yield (
            None,
            describe(job),
            gr.update(value=video, visible=video is not None),
            gr.update(value=model, visible=model is not None),
            files,
        )


def build_trellis() -> None:
    job_id = gr.State(None)
    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(label="Input image", type="filepath", height=320)
            with gr.Row():
                go = gr.Button("Image → 3D", variant="primary")
                stop = gr.Button("Cancel")
            status = gr.Markdown()
        with gr.Column(scale=3):
            video = gr.Video(label="Preview", interactive=False, visible=False)
            model = gr.Model3D(label="3D model", visible=False)
            files = gr.File(label="Downloads", file_count="multiple", interactive=False)
    go.click(run_trellis, image, [job_id, status, video, model, files], concurrency_limit=None)
    stop.click(cancel_job, job_id, status)


# ------------------------------------------------------------------------ VIP
VIP_STYLES = DEFAULTS["vip_styles"]  # label -> backend style


async def run_vip(staged: str | None, style: str, oil_p, oil_r, aq_p, aq_r, bl_color, bl_p, bl_r, bl_s, pt_c, request: gr.Request):
    if not staged:
        return NO_CHANGE
    user = current_user(request)
    key = VIP_STYLES[style]
    params = {
        "oil": {"pinsel": oil_p, "radius": oil_r},
        "aquarell": {"pinsel": aq_p, "radius": aq_r},
        "bleistift": {"buntstift": bl_color, "pinsel": bl_p, "radius": bl_r, "schatten": bl_s},
        "point": {"colors": pt_c},
    }[key]
    res = await BACKEND.call(user, "POST", "/vip", json={"style": key, "image_path": staged, "params": params})
    return res["path"]


def stage_vip(image: str | None, request: gr.Request) -> str | None:
    return stage_upload(image, current_user(request)) if image else None


def build_vip() -> None:
    staged = gr.State(None)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload", type="filepath", height=360)
            style = gr.Radio(list(VIP_STYLES), value=next(iter(VIP_STYLES)), label="Stil")
            with gr.Group(visible=True) as g_oil:
                oil_p = gr.Slider(1, 50, value=10, step=1, label="Pinsel", info="Stärke")
                oil_r = gr.Slider(1, 100, value=25, step=1, label="Radius", info="Ähnliche Pixel")
            with gr.Group(visible=False) as g_aq:
                aq_p = gr.Slider(1, 200, value=170, step=1, label="Pinsel", info="Stärke")
                aq_r = gr.Slider(0.01, 1, value=0.62, step=0.01, label="Radius")
            with gr.Group(visible=False) as g_bl:
                bl_color = gr.Radio(["Bleistift", "Buntstift"], value="Bleistift", label="Farbe")
                bl_p = gr.Slider(1, 50, value=20, step=1, label="Pinsel", info="Stärke")
                bl_r = gr.Slider(0.01, 1, value=0.18, step=0.01, label="Radius")
                bl_s = gr.Slider(0.01, 0.1, value=0.05, step=0.005, label="Schatten")
            with gr.Group(visible=False) as g_pt:
                pt_c = gr.Slider(2, 64, value=20, step=1, label="Farben")
        with gr.Column():
            out = gr.Image(label="Ergebnis", type="filepath", interactive=False, height=520)

    groups = [g_oil, g_aq, g_bl, g_pt]
    inputs = [staged, style, oil_p, oil_r, aq_p, aq_r, bl_color, bl_p, bl_r, bl_s, pt_c]
    style.change(lambda s: [gr.update(visible=VIP_STYLES[s] == k) for k in ("oil", "aquarell", "bleistift", "point")], style, groups)
    image.change(stage_vip, image, staged).then(run_vip, inputs, out)
    for comp in [style, oil_p, oil_r, aq_p, aq_r, bl_color, bl_p, bl_r, bl_s, pt_c]:
        (comp.release if isinstance(comp, gr.Slider) else comp.change)(run_vip, inputs, out, trigger_mode="always_last")


# ---------------------------------------------------------------------- audio
async def run_audio(audio: str | None, language: str, task: str, request: gr.Request):
    user = current_user(request)
    if not audio:
        raise gr.Error("Please upload or record audio.")
    with open(audio, "rb") as f:
        data = {"language": "" if language == "auto" else language, "task": "translate" if task.startswith("Translate") else "transcribe", "response_format": "verbose_json"}
        res = await BACKEND.call(user, "POST", "/transcribe", data=data, files={"file": (Path(audio).name, f)})
    lines = []
    for s in res.get("segments", []):
        lines.append(f"[{_mmss(s['start'])} → {_mmss(s['end'])}] {s['text']}")
    srt = Path(audio).with_suffix(".srt")
    srt.write_text(_srt(res.get("segments", [])), encoding="utf-8")
    info = (
        f"**Language:** {res.get('language')} ({res.get('language_probability', 0):.0%}) · "
        f"**Duration:** {_mmss(res.get('duration', 0))} · **Processing:** {res.get('processing_seconds')}s"
    )
    return res.get("text", ""), "\n".join(lines), info, str(srt)


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def _srt(segments: list[dict]) -> str:
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    return "\n".join(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{s['text']}\n" for i, s in enumerate(segments, 1))


def build_audio() -> None:
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="Audio", sources=["upload", "microphone"], type="filepath")
            with gr.Row():
                language = gr.Dropdown(DEFAULTS["audio"]["languages"], value="auto", label="Language")
                task = gr.Radio(["Transcribe", "Translate to English"], value="Transcribe", label="Task")
            go = gr.Button("Audio → Text", variant="primary")
            info = gr.Markdown()
        with gr.Column():
            text = gr.Textbox(label="Transcript", lines=10, buttons=["copy"])
            with gr.Accordion("Timestamps", open=False):
                segments = gr.Textbox(show_label=False, lines=10, buttons=["copy"])
            srt = gr.File(label="Subtitles (.srt)", interactive=False)
    go.click(run_audio, [audio, language, task], [text, segments, info, srt], concurrency_limit=8)
