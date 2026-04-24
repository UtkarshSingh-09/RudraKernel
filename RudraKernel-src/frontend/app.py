"""Step 24 Gradio storytelling app."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from frontend.components.arms_race import render_arms_race_text
from frontend.components.before_after import render_before_after_text
from frontend.components.war_room import render_war_room_text


ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "data" / "demo_episodes" / "step24_demo_episode.jsonl"
CSS_PATH = ROOT / "assets" / "css" / "storytelling.css"


def load_demo_episode_text() -> str:
    lines = []
    for raw in DEMO_PATH.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        lines.append(f"[{row['step']}] {row['speaker']}: {row['message']}")
    return "\n".join(lines)


def build_app() -> gr.Blocks:
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    with gr.Blocks(css=css, title="RudraKernel War Room") as demo:
        gr.Markdown("## RudraKernel: Money Shot Demo", elem_classes=["rk-title"])

        with gr.Tabs():
            with gr.Tab("War Room"):
                gr.Markdown(render_war_room_text(), elem_classes=["rk-note"])
                out = gr.Textbox(label="Demo Episode Playback", lines=6)
                gr.Button("Play Demo").click(fn=load_demo_episode_text, inputs=None, outputs=out)

            with gr.Tab("Before-After"):
                gr.Markdown(render_before_after_text(), elem_classes=["rk-note"])

            with gr.Tab("Arms Race"):
                gr.Markdown(render_arms_race_text(), elem_classes=["rk-note"])

    demo.rudra_tabs = ["War Room", "Before-After", "Arms Race"]
    return demo


if __name__ == "__main__":
    build_app().launch()
