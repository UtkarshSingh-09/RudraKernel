from __future__ import annotations

from typing import Any


def render_war_room_text() -> str:
    return """
### Live War Room
Track incident pressure, trust collapse, and coalition response in one view.

Key streams:
- Incident timeline playback
- Agent trust and risk score
- Learning signal over episodes
""".strip()


def get_live_scoreboard() -> list[dict[str, Any]]:
    return [
        {"agent": "Agent1", "trust": 0.76, "risk": 0.34, "alignment": "stable"},
        {"agent": "Agent2", "trust": 0.68, "risk": 0.42, "alignment": "watch"},
        {"agent": "Agent3", "trust": 0.59, "risk": 0.49, "alignment": "volatile"},
        {"agent": "Agent4", "trust": 0.54, "risk": 0.73, "alignment": "critical"},
    ]


def get_training_curve() -> list[dict[str, float]]:
    episodes = list(range(1, 13))
    mean_reward = [-0.35, -0.22, -0.14, -0.07, 0.01, 0.08, 0.15, 0.21, 0.29, 0.33, 0.36, 0.41]
    win_rate = [0.28, 0.31, 0.36, 0.42, 0.48, 0.53, 0.59, 0.64, 0.69, 0.72, 0.75, 0.79]
    return [
        {"episode": float(ep), "mean_reward": reward, "win_rate": win}
        for ep, reward, win in zip(episodes, mean_reward, win_rate, strict=True)
    ]


def render_score_summary(step: int) -> str:
    trajectory = [
        {"label": "Initial suspicion", "team_score": 47, "threat_score": 62, "trust_agent4": 0.82},
        {"label": "Counter-evidence applied", "team_score": 58, "threat_score": 54, "trust_agent4": 0.67},
        {"label": "Coalition correction", "team_score": 71, "threat_score": 44, "trust_agent4": 0.54},
    ]
    idx = max(0, min(step, len(trajectory) - 1))
    point = trajectory[idx]
    return (
        f"### Step {idx}: {point['label']}\n"
        f"- Team learning score: **{point['team_score']} / 100**\n"
        f"- Threat score: **{point['threat_score']} / 100**\n"
        f"- Trust(Agent4): **{point['trust_agent4']:.2f}**"
    )
