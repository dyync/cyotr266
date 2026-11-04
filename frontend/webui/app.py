"""Gradio 6 multi-user frontend. Run: python -m webui.app"""

from __future__ import annotations

import logging
import os

import gradio as gr

from . import tab_admin, tab_chat, tab_media
from .core import OUTPUTS, USERS, current_user

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CSS = """
footer {visibility: hidden}
#topbar {align-items: center}
"""


def on_load(request: gr.Request):
    user = current_user(request)
    return (
        f"Signed in as **{user.name}**" + (" · admin" if user.admin else ""),
        gr.update(visible=user.admin),
        gr.Timer(active=user.admin),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="AI Studio", fill_height=True) as demo:
        with gr.Row(elem_id="topbar"):
            gr.Markdown("## AI Studio")
            whoami = gr.Markdown()
            gr.Button("Logout", link="/logout", size="sm", scale=0)
        with gr.Tabs():
            with gr.Tab("Chat"):
                tab_chat.build()
            with gr.Tab("Image"):
                tab_media.build_image()
            with gr.Tab("Video"):
                tab_media.build_video()
            with gr.Tab("VIP"):
                tab_media.build_vip()
            with gr.Tab("Audio"):
                tab_media.build_audio()
            with gr.Tab("Admin", visible=False) as admin_tab:
                timer = tab_admin.build()
        demo.load(on_load, None, [whoami, admin_tab, timer])
    return demo


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    demo = build_app()
    demo.queue(default_concurrency_limit=32, api_open=False).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("FRONTEND_PORT", "7860")),
        auth=USERS.check,
        auth_message="Please sign in.",
        allowed_paths=[str(OUTPUTS)],
        footer_links=[],
        max_file_size="200mb",
        ssr_mode=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
        favicon_path=os.getenv("FAVICON", "/app/favicon.ico"),
    )


if __name__ == "__main__":
    main()
