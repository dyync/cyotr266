"""Chat tab: streaming chat through the orchestrator's OpenAI-compatible proxy."""

from __future__ import annotations

import time
from typing import Any

import gradio as gr
import openai

from .core import BACKEND, DEFAULTS, HOLDER_LABEL, current_user


def _text(content: Any) -> str:
    """Gradio 6 normalises message content to a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "") if content.get("type", "text") == "text" else ""
    if isinstance(content, list):
        return "\n".join(t for t in (_text(c) for c in content) if t)
    return ""


def _extra(delta: Any, *names: str) -> str:
    for n in names:
        v = getattr(delta, n, None)
        if v is None and getattr(delta, "model_extra", None):
            v = delta.model_extra.get(n)
        if v:
            return v
    return ""


async def chat_fn(message: str, history: list[dict], system_prompt: str, thinking: bool, temperature: float, top_p: float, max_tokens: int, request: gr.Request):
    user = current_user(request)
    msgs: list[dict[str, str]] = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt.strip()})
    for m in history:
        if (m.get("metadata") or {}).get("title"):
            continue  # skip "thinking" bubbles and status notes
        text = _text(m.get("content"))
        if text:
            msgs.append({"role": m["role"], "content": text})
    msgs.append({"role": "user", "content": message})

    state = await BACKEND.call(user, "GET", "/queue")
    if state["holder"] != "llm":
        yield {
            "role": "assistant",
            "content": f"The GPU is busy with {HOLDER_LABEL.get(state['holder'], state['holder'])} "
            f"({state['queued']} job(s) queued). Your answer starts as soon as the language model is back…",
            "metadata": {"title": "⏳ Waiting for GPU", "status": "pending"},
        }

    try:
        stream = await BACKEND.openai.chat.completions.create(
            model="chat",
            messages=msgs,
            stream=True,
            temperature=temperature,
            top_p=top_p,
            max_tokens=int(max_tokens),
            extra_headers=BACKEND.headers(user),
            extra_body={
                "top_k": 20,
                "presence_penalty": 1.5,
                "chat_template_kwargs": {"enable_thinking": bool(thinking)},
            },
        )
        reasoning, answer, t0, t_think = "", "", time.monotonic(), None
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            r = _extra(delta, "reasoning", "reasoning_content")
            if r:
                reasoning += r
            if delta.content:
                if t_think is None:
                    t_think = time.monotonic() - t0
                answer += delta.content
            out: list[dict] = []
            if reasoning:
                out.append(
                    {
                        "role": "assistant",
                        "content": reasoning,
                        "metadata": {
                            "title": "💭 Thinking",
                            "status": "done" if answer else "pending",
                            **({"duration": round(t_think, 1)} if t_think else {}),
                        },
                    }
                )
            out.append({"role": "assistant", "content": answer})
            yield out
    except openai.APIStatusError as e:
        detail = e.body.get("detail") if isinstance(e.body, dict) else None
        yield f"⚠️ {detail or e.message}"
    except openai.APIConnectionError as e:
        yield f"⚠️ Language model unreachable ({e.__class__.__name__}). Please retry in a moment."


def build() -> None:
    c = DEFAULTS["chat"]
    gr.ChatInterface(
        chat_fn,
        chatbot=gr.Chatbot(
            height="65vh",
            show_label=False,
            buttons=["copy", "copy_all"],
            placeholder=c["welcome"],
            feedback_options=None,
        ),
        textbox=gr.Textbox(placeholder=c["placeholder"], autofocus=True, show_label=False, submit_btn=True),
        additional_inputs=[
            gr.Textbox(c["system_prompt"], label="System prompt", lines=2),
            gr.Checkbox(False, label="Thinking mode (slower, better at hard questions)"),
            gr.Slider(0.0, 1.5, value=c["temperature"], step=0.05, label="temperature"),
            gr.Slider(0.05, 1.0, value=c["top_p"], step=0.05, label="top_p"),
            gr.Slider(64, 8192, value=c["max_tokens"], step=64, label="max_tokens"),
        ],
        additional_inputs_accordion=gr.Accordion("Settings", open=False),
        editable=True,
        save_history=True,
        concurrency_limit=128,
        api_visibility="private",
        fill_height=True,
    )
