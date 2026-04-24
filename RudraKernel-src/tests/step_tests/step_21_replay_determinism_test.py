"""Gate tests for Step 21 — Determinism + Replay."""

from __future__ import annotations

from pathlib import Path

from siege_env.replay.logger import ReplayLogger
from siege_env.replay.player import replay_file
from siege_env.server.siege_environment import SIEGEEnvironment


def test_replay_logger_round_trip() -> None:
    path = Path("/tmp/siege_step21_round_trip.jsonl")
    if path.exists():
        path.unlink()
    logger = ReplayLogger(path)
    logger.append({"step": 1, "tool": "diagnose"})
    logger.append({"step": 2, "tool": "challenge"})
    events = logger.read_all()
    assert len(events) == 2


def test_replay_player_reads_logged_events() -> None:
    path = Path("/tmp/siege_step21_player.jsonl")
    if path.exists():
        path.unlink()
    logger = ReplayLogger(path)
    logger.append({"step": 1, "tool": "diagnose"})
    assert len(replay_file(path)) == 1


def test_environment_step_info_exposes_replay_path() -> None:
    env = SIEGEEnvironment(seed=101)
    env.reset()
    _, _, _, info = env.step({"tool_name": "diagnose", "arguments": {"root_cause": "x", "confidence": 0.5, "evidence": ["e"]}})
    assert "replay_log_path" in info
