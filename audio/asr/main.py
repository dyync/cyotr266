"""Speech-to-text service with an OpenAI-compatible /v1/audio/transcriptions endpoint.

faster-whisper (CTranslate2) large-v3-turbo, loaded once at startup and kept
resident (~1.5-2 GB VRAM in int8_float16). Falls back to CPU int8 if CUDA is
not available so the service never crash-loops.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from faster_whisper import BatchedInferencePipeline, WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("asr")

MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")
DEVICE = os.getenv("ASR_DEVICE", "cuda")
COMPUTE = os.getenv("ASR_COMPUTE_TYPE", "int8_float16")
BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE", "8"))
MAX_UPLOAD_MB = int(os.getenv("ASR_MAX_UPLOAD_MB", "200"))
limiter = anyio.CapacityLimiter(int(os.getenv("ASR_CONCURRENCY", "2")))

state: dict = {}


def _load() -> None:
    try:
        model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
        state["device"] = f"{DEVICE}/{COMPUTE}"
    except Exception as e:  # noqa: BLE001 - e.g. no GPU / missing CUDA libs
        log.error("could not load %s on %s (%s); falling back to CPU int8", MODEL, DEVICE, e)
        model = WhisperModel(MODEL, device="cpu", compute_type="int8")
        state["device"] = "cpu/int8"
    state["model"] = model
    state["batched"] = BatchedInferencePipeline(model=model)
    log.info("loaded whisper %s on %s", MODEL, state["device"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    await anyio.to_thread.run_sync(_load)
    yield


app = FastAPI(title="asr", lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/health")
async def health():
    if "model" not in state:
        raise HTTPException(503, "loading")
    return {"ok": True, "model": MODEL, "device": state["device"]}


def _ts(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _transcribe(path: str, language: str | None, task: str, vad: bool) -> dict:
    t0 = time.time()
    segments, info = state["batched"].transcribe(
        path,
        language=language or None,
        task=task,
        batch_size=BATCH_SIZE,
        vad_filter=vad,
    )
    segs = [{"id": i, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for i, s in enumerate(segments)]
    return {
        "task": task,
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 2),
        "text": " ".join(s["text"] for s in segs).strip(),
        "segments": segs,
        "processing_seconds": round(time.time() - t0, 2),
    }


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(MODEL),  # accepted for OpenAI compatibility, ignored
    language: str = Form(""),
    response_format: str = Form("json"),
    task: str = Form("transcribe"),
    vad_filter: bool = Form(True),
):
    if "model" not in state:
        raise HTTPException(503, "model still loading")
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task must be transcribe|translate")
    suffix = os.path.splitext(file.filename or "")[1][:10] or ".bin"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        size = 0
        with os.fdopen(fd, "wb") as tmp:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
                tmp.write(chunk)
        try:
            result = await anyio.to_thread.run_sync(_transcribe, tmp_path, language, task, vad_filter, limiter=limiter)
        except Exception as e:  # noqa: BLE001 - e.g. undecodable audio
            log.exception("transcription failed")
            raise HTTPException(400, f"transcription failed: {e}") from e
    finally:
        os.unlink(tmp_path)

    if response_format == "text":
        return PlainTextResponse(result["text"])
    if response_format in ("srt", "vtt"):
        sep = "," if response_format == "srt" else "."
        lines = ["WEBVTT\n"] if response_format == "vtt" else []
        for i, s in enumerate(result["segments"], 1):
            lines.append(f"{i}\n{_ts(s['start'], sep)} --> {_ts(s['end'], sep)}\n{s['text']}\n")
        return PlainTextResponse("\n".join(lines))
    if response_format == "verbose_json":
        return JSONResponse(result)
    return JSONResponse({"text": result["text"]})
