"""Step 24 Gradio storytelling app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from frontend.components.arms_race import render_arms_race_text
from frontend.components.before_after import render_before_after_text
from frontend.components.war_room import (
    get_live_scoreboard,
    get_training_curve,
    render_score_summary,
    render_war_room_text,
)

ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "data" / "demo_episodes" / "step24_demo_episode.jsonl"
CSS_PATH = ROOT / "assets" / "css" / "storytelling.css"
PLOTS_PATH = ROOT.parent / "docs" / "plots"


def _load_demo_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in DEMO_PATH.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def load_demo_episode_text() -> str:
    lines = []
    for row in _load_demo_rows():
        lines.append(f"[{row['step']}] {row['speaker']}: {row['message']}")
    return "\n".join(lines)


def render_turn_state(turn_index: int) -> tuple[str, str]:
    rows = _load_demo_rows()
    idx = max(0, min(turn_index, len(rows) - 1))
    row = rows[idx]
    event = f"### Current Event\n`[{row['step']}] {row['speaker']}`: {row['message']}"
    return event, render_score_summary(idx)


def run_learning_simulation() -> tuple[str, list[dict[str, Any]]]:
    curve = get_training_curve()
    final = curve[-1]
    summary = (
        "### Learning Run Complete\n"
        f"- Episodes: **{len(curve)}**\n"
        f"- Final mean reward: **{final['mean_reward']:.2f}**\n"
        f"- Final win rate: **{final['win_rate'] * 100:.1f}%**"
    )
    return summary, curve


def build_app() -> gr.Blocks:
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    with gr.Blocks(title="RudraKernel War Room") as demo:
        if css:
            gr.HTML(f"<style>{css}</style>")
        gr.Markdown("## RudraKernel: Agent Learning & Score Simulator", elem_classes=["rk-title"])
        gr.Markdown(
            "Interactive incident playback with learning metrics, reward trends, and "
            "before/after generalization evidence.",
            elem_classes=["rk-note"],
        )

        with gr.Tabs():
            with gr.Tab("War Room"):
                gr.Markdown(render_war_room_text(), elem_classes=["rk-note"])
                with gr.Row():
                    with gr.Column(scale=3):
                        out = gr.Textbox(label="Demo Episode Playback", lines=8)
                        play_btn = gr.Button("Play Demo Timeline")
                        turn_slider = gr.Slider(
                            minimum=0,
                            maximum=max(len(_load_demo_rows()) - 1, 0),
                            step=1,
                            value=0,
                            label="Inspect Decision Step",
                        )
                        current_event = gr.Markdown()
                    with gr.Column(scale=2):
                        score_summary = gr.Markdown(render_score_summary(0))
                        gr.Dataframe(
                            value=get_live_scoreboard(),
                            headers=["agent", "trust", "risk", "alignment"],
                            label="Live Trust/Risk Board",
                            interactive=False,
                        )
                with gr.Row():
                    learning_summary = gr.Markdown("### Learning Run Pending")
                    simulate_btn = gr.Button("Run 12-Episode Learning Simulation")
                learning_curve = gr.Dataframe(
                    value=[],
                    headers=["episode", "mean_reward", "win_rate"],
                    label="Learning Curve Snapshot",
                    interactive=False,
                )

                play_btn.click(fn=load_demo_episode_text, inputs=None, outputs=out)
                turn_slider.change(
                    fn=render_turn_state,
                    inputs=turn_slider,
                    outputs=[current_event, score_summary],
                )
                simulate_btn.click(
                    fn=run_learning_simulation,
                    inputs=None,
                    outputs=[learning_summary, learning_curve],
                )

            with gr.Tab("Before-After"):
                gr.Markdown(render_before_after_text(), elem_classes=["rk-note"])
                with gr.Row():
                    gr.Image(
                        value=str(PLOTS_PATH / "generalization_gap.png"),
                        label="Generalization Gap",
                        interactive=False,
                    )
                    gr.Image(
                        value=str(PLOTS_PATH / "ablation_comparison.png"),
                        label="Ablation Comparison",
                        interactive=False,
                    )

            with gr.Tab("Arms Race"):
                gr.Markdown(render_arms_race_text(), elem_classes=["rk-note"])
                with gr.Row():
                    gr.Image(
                        value=str(PLOTS_PATH / "arms_race_curve.png"),
                        label="Arms Race Curve",
                        interactive=False,
                    )
                    gr.Image(
                        value=str(PLOTS_PATH / "reward_components.png"),
                        label="Reward Components",
                        interactive=False,
                    )

    demo.rudra_tabs = ["War Room", "Before-After", "Arms Race"]
    return demo


if __name__ == "__main__":
    build_app().launch()
