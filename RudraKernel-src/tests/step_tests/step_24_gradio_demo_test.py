"""Gate tests for Step 24 — Gradio money-shot frontend."""

from __future__ import annotations

import pytest

try:
    import gradio  # noqa: F401

    _GRADIO_AVAILABLE = True
except Exception:
    _GRADIO_AVAILABLE = False

if _GRADIO_AVAILABLE:
    from frontend.app import build_app, load_demo_episode_text


@pytest.mark.skipif(not _GRADIO_AVAILABLE, reason="gradio not installed")
def test_gradio_app_boots_and_has_three_tabs() -> None:
    app = build_app()
    assert app is not None
    assert getattr(app, "rudra_tabs", []) == ["War Room", "Before-After", "Arms Race"]


@pytest.mark.skipif(not _GRADIO_AVAILABLE, reason="gradio not installed")
def test_demo_episode_playback_text_is_nonempty() -> None:
    text = load_demo_episode_text()
    assert "Agent4" in text
    assert "YOU" in text
    assert len(text.splitlines()) >= 3
