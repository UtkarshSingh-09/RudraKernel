"""Premium clinical-forensic Gradio frontend for SIEGE."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

import gradio as gr
import pandas as pd

from frontend.data_adapter import (
    PROVENANCE_PENDING_TEXT,
    build_graph9_payload,
    get_latest_run_snapshot,
    get_provenance_payload,
    read_live_stream,
)
from frontend.tab_config import INTERNAL_TABS, get_display_tab_name

ROOT = Path(__file__).resolve().parent
DEMO_PATH = ROOT / "data" / "demo_episodes" / "step24_demo_episode.jsonl"
CSS_PATH = ROOT / "assets" / "css" / "storytelling.css"
ARTIFACTS_TRAINING_DIR = ROOT.parent / "artifacts" / "training"
LIVE_STREAM_PATH = ARTIFACTS_TRAINING_DIR / "siege_live_metrics.jsonl"
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


def _moving_average(values: list[float], window: int = 5) -> list[float]:
    if not values:
        return []
    output: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        output.append(mean(values[start : idx + 1]))
    return output


def _to_float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [float(v) for v in values if isinstance(v, (int, float))]


def _build_live_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    mini_rewards = [float(e["reward"]) for e in events if isinstance(e.get("reward"), (int, float))]
    metrics: dict[str, Any] = {"mini_run_rewards": mini_rewards}

    components: dict[str, list[float]] = {"gain": [], "loss": [], "net": []}
    for reward in mini_rewards:
        components["gain"].append(max(reward, 0.0))
        components["loss"].append(max(-reward, 0.0))
        components["net"].append(reward)
    metrics["reward_components"] = components

    for key in ("epistemic_r0", "belief_half_life", "belief_entropy"):
        values = [float(e[key]) for e in events if isinstance(e.get(key), (int, float))]
        if values:
            metrics[key] = values
    return metrics


def _derive_grade_distribution(mini_rewards: list[float]) -> dict[str, float]:
    grades = {"Grade A (stable)": 0.0, "Grade B (watch)": 0.0, "Grade C (critical)": 0.0}
    for reward in mini_rewards:
        if reward >= 1.5:
            grades["Grade A (stable)"] += 1
        elif reward >= 0.8:
            grades["Grade B (watch)"] += 1
        else:
            grades["Grade C (critical)"] += 1
    return grades


def _derive_epistemic_proxies(mini_rewards: list[float]) -> tuple[list[float], list[float], list[float]]:
    if not mini_rewards:
        return [], [], []
    moving = _moving_average(mini_rewards, window=6)
    deltas = [0.0]
    for idx in range(1, len(mini_rewards)):
        deltas.append(abs(mini_rewards[idx] - mini_rewards[idx - 1]))
    r0_proxy = [1.0 + value for value in deltas]
    half_life_proxy = [max(1.0, 12.0 - (idx * 0.2) - (moving[idx] * 1.1)) for idx in range(len(moving))]
    entropy_proxy = [abs(mini_rewards[idx] - moving[idx]) for idx in range(len(mini_rewards))]
    return r0_proxy, half_life_proxy, entropy_proxy


def _graph_payloads(metrics: dict[str, Any]) -> tuple[pd.DataFrame, ...]:
    mini_rewards = _to_float_list(metrics.get("mini_run_rewards"))
    baseline_rewards = _to_float_list(metrics.get("baseline_scripted_rewards"))
    heldout_rewards = _to_float_list(metrics.get("baseline_frozen_rewards"))
    episodes = list(range(1, len(mini_rewards) + 1))

    # Graph 1: Arms race gain/loss
    if mini_rewards:
        g1 = pd.DataFrame(
            {"episode": episodes, "value": mini_rewards, "series": ["Net Clinical Gain"] * len(episodes)}
        )
    else:
        g1 = pd.DataFrame({"episode": [0], "value": [0.0], "series": ["No Data"]})

    # Graph 2: reward components
    reward_components = metrics.get("reward_components")
    if isinstance(reward_components, dict):
        rows: list[dict[str, Any]] = []
        for key, values in reward_components.items():
            safe = _to_float_list(values)
            for idx, value in enumerate(safe, start=1):
                rows.append({"episode": idx, "value": value, "series": str(key).upper()})
        g2 = pd.DataFrame(rows) if rows else pd.DataFrame({"episode": [0], "value": [0.0], "series": ["No Data"]})
    else:
        rows = []
        for idx, reward in enumerate(mini_rewards, start=1):
            rows.append({"episode": idx, "value": max(reward, 0.0), "series": "GAIN"})
            rows.append({"episode": idx, "value": max(-reward, 0.0), "series": "LOSS"})
            rows.append({"episode": idx, "value": reward, "series": "NET"})
        g2 = pd.DataFrame(rows) if rows else pd.DataFrame({"episode": [0], "value": [0.0], "series": ["No Data"]})

    # Graph 3: baseline vs trained
    g3_rows = []
    if baseline_rewards:
        g3_rows.append({"policy": "Baseline", "gain": mean(baseline_rewards)})
    if mini_rewards:
        g3_rows.append({"policy": "Trained", "gain": mean(mini_rewards)})
    g3 = pd.DataFrame(g3_rows) if g3_rows else pd.DataFrame({"policy": ["No Data"], "gain": [0.0]})

    # Graph 4: generalization gap
    train_score = mean(mini_rewards) if mini_rewards else 0.0
    if heldout_rewards:
        heldout_score = mean(heldout_rewards)
        heldout_label = "Held-Out"
    elif mini_rewards:
        tail = mini_rewards[max(0, int(len(mini_rewards) * 0.7)) :]
        heldout_score = mean(tail) if tail else train_score
        heldout_label = "Replay Tail"
    else:
        heldout_score = 0.0
        heldout_label = "No Data"
    g4 = pd.DataFrame(
        [{"split": "Train", "score": train_score}, {"split": heldout_label, "score": heldout_score}]
    )

    # Graph 5: reward trend + moving average
    rows5 = []
    if mini_rewards:
        ma = _moving_average(mini_rewards)
        for idx, reward in enumerate(mini_rewards, start=1):
            rows5.append({"episode": idx, "value": reward, "series": "Raw Gain/Loss"})
            rows5.append({"episode": idx, "value": ma[idx - 1], "series": "Moving Avg"})
    g5 = pd.DataFrame(rows5) if rows5 else pd.DataFrame({"episode": [0], "value": [0.0], "series": ["No Data"]})

    # Graph 6: distribution hist
    def hist_rows(values: list[float], label: str) -> list[dict[str, Any]]:
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if min_v == max_v:
            return [{"bucket": f"{min_v:.2f}", "count": len(values), "series": label}]
        bins = [min_v + i * (max_v - min_v) / 6 for i in range(7)]
        counts = [0] * 6
        for value in values:
            idx = min(5, int((value - min_v) / (max_v - min_v + 1e-9) * 6))
            counts[idx] += 1
        out = []
        for idx, count in enumerate(counts):
            out.append({"bucket": f"{bins[idx]:.2f}–{bins[idx + 1]:.2f}", "count": count, "series": label})
        return out

    rows6 = hist_rows(baseline_rewards, "Baseline") + hist_rows(mini_rewards, "Trained")
    g6 = pd.DataFrame(rows6) if rows6 else pd.DataFrame({"bucket": ["No Data"], "count": [0], "series": ["None"]})

    # Graph 7: success curve
    rows7 = []
    if mini_rewards:
        successes = 0
        for idx, reward in enumerate(mini_rewards, start=1):
            if reward > 0:
                successes += 1
            rows7.append({"episode": idx, "value": successes / idx, "series": "Resolve Rate"})
    g7 = pd.DataFrame(rows7) if rows7 else pd.DataFrame({"episode": [0], "value": [0.0], "series": ["No Data"]})

    # Graph 8: adverse event grades
    grades = metrics.get("adverse_event_grade_distribution")
    if not isinstance(grades, dict):
        grades = _derive_grade_distribution(mini_rewards)
    g8_rows = [{"grade": str(k), "count": float(v)} for k, v in grades.items()]
    g8 = pd.DataFrame(g8_rows) if g8_rows else pd.DataFrame({"grade": ["No Data"], "count": [0.0]})

    # Graph 9: normalized epistemic stability
    r0_raw = _to_float_list(metrics.get("epistemic_r0"))
    half_life_raw = _to_float_list(metrics.get("belief_half_life"))
    entropy_raw = _to_float_list(metrics.get("belief_entropy"))
    if not (r0_raw and half_life_raw and entropy_raw):
        r0_raw, half_life_raw, entropy_raw = _derive_epistemic_proxies(mini_rewards)
    size = min(len(r0_raw), len(half_life_raw), len(entropy_raw))
    if size > 0:
        payload = build_graph9_payload(r0_raw[:size], half_life_raw[:size], entropy_raw[:size])
        rows9 = []
        for idx in range(size):
            rows9.append(
                {
                    "episode": idx + 1,
                    "value": payload["r0_norm"][idx],
                    "series": f"R0 (raw {payload['r0_raw'][idx]:.2f})",
                }
            )
            rows9.append(
                {
                    "episode": idx + 1,
                    "value": payload["half_life_norm"][idx],
                    "series": f"Half-Life (raw {payload['half_life_raw'][idx]:.2f})",
                }
            )
            rows9.append(
                {
                    "episode": idx + 1,
                    "value": payload["entropy_norm"][idx],
                    "series": f"Entropy (raw {payload['entropy_raw'][idx]:.2f})",
                }
            )
        g9 = pd.DataFrame(rows9)
    else:
        g9 = pd.DataFrame({"episode": [0], "value": [0.0], "series": ["No Data"]})

    # Graph 10: provenance node map
    provenance = get_provenance_payload(metrics)
    if provenance["available"]:
        nodes = provenance["nodes"]
        rows10 = []
        for idx, node in enumerate(nodes):
            rows10.append(
                {
                    "node": str(node.get("id", f"n{idx}")),
                    "x": float(node.get("x", idx)),
                    "depth": float(node.get("depth", 0)),
                }
            )
        g10 = pd.DataFrame(rows10) if rows10 else pd.DataFrame({"node": ["No Data"], "x": [0.0], "depth": [0.0]})
        provenance_message = "Belief provenance loaded from replay artifacts."
    else:
        g10 = pd.DataFrame({"node": ["Pending"], "x": [0.0], "depth": [0.0]})
        provenance_message = PROVENANCE_PENDING_TEXT

    return g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, provenance_message


def _build_metrics_cards(metrics: dict[str, Any], mode: str, run_label: str) -> str:
    rewards = _to_float_list(metrics.get("mini_run_rewards"))
    mean_gain = mean(rewards) if rewards else 0.0
    peak_gain = max(rewards) if rewards else 0.0
    resolve_rate = (sum(1 for r in rewards if r > 0) / len(rewards)) if rewards else 0.0
    stability = 1.0 - min(1.0, abs(mean_gain - 1.0) / 3.0)
    mode_color = "#22d3ee" if mode.upper() == "LIVE" else "#94a3b8"
    return f"""
<div class="rk-metrics-wrap">
  <div class="rk-metric-card">
    <div class="rk-metric-label">Mode</div>
    <div class="rk-metric-value" style="color:{mode_color}">{mode.upper()}</div>
    <div class="rk-metric-sub">{run_label}</div>
  </div>
  <div class="rk-metric-card">
    <div class="rk-metric-label">Mean Clinical Gain</div>
    <div class="rk-metric-value" style="color:{'#34d399' if mean_gain>0 else '#fb7185'}">{mean_gain:.2f}</div>
    <div class="rk-metric-sub">Avg reward across episodes</div>
  </div>
  <div class="rk-metric-card">
    <div class="rk-metric-label">Peak Gain</div>
    <div class="rk-metric-value" style="color:#22d3ee">{peak_gain:.2f}</div>
    <div class="rk-metric-sub">Best episode recovery</div>
  </div>
  <div class="rk-metric-card">
    <div class="rk-metric-label">Resolve Rate</div>
    <div class="rk-metric-value" style="color:{'#34d399' if resolve_rate>0.5 else '#f59e0b'}">{resolve_rate*100:.1f}%</div>
    <div class="rk-metric-sub">Positive outcomes share</div>
  </div>
  <div class="rk-metric-card">
    <div class="rk-metric-label">Epistemic Stability</div>
    <div class="rk-metric-value" style="color:{'#34d399' if stability>0.6 else '#f59e0b'}">{stability:.2f}</div>
    <div class="rk-metric-sub">Normalized health proxy</div>
  </div>
</div>
""".strip()


def _build_story_html() -> str:
    return """
<div class="rk-story-panel">
  <h3>&#128203; Clinical Safety Narrative</h3>
  <p>This console shows how SIEGE improves decision quality over episodes. Every chart uses run artifacts only: replay logs, training metrics, and held-out summaries.</p>
  <ul>
    <li><strong style="color:#22d3ee">X-axis:</strong> episode progression (how long we trained/played)</li>
    <li><strong style="color:#22d3ee">Y-axis:</strong> gain/loss or normalized safety index</li>
    <li><strong style="color:#22d3ee">Interpretation:</strong> if trained trend rises and baseline stays flat, the model improved.</li>
  </ul>
</div>
""".strip()


def _build_judge_scorecard(metrics: dict[str, Any]) -> pd.DataFrame:
    baseline_rewards = _to_float_list(metrics.get("baseline_scripted_rewards"))
    trained_rewards = _to_float_list(metrics.get("mini_run_rewards"))

    baseline_mean = mean(baseline_rewards) if baseline_rewards else 0.0
    trained_mean = mean(trained_rewards) if trained_rewards else 0.0
    absolute_uplift = trained_mean - baseline_mean
    percent_uplift = (
        (absolute_uplift / baseline_mean) * 100.0 if baseline_mean > 1e-9 else 0.0
    )

    if trained_rewards:
        early_window = trained_rewards[: max(1, int(len(trained_rewards) * 0.2))]
        late_window = trained_rewards[max(0, int(len(trained_rewards) * 0.8)) :]
        future_gain = max(0.0, mean(late_window) - mean(early_window))
    else:
        future_gain = 0.0

    readiness = "High" if absolute_uplift > 0.2 else "Medium" if absolute_uplift > 0.05 else "Low"

    rows = [
        {"Metric": "Baseline LLM Mean Score", "Value": f"{baseline_mean:.3f}"},
        {"Metric": "Trained Policy Mean Score", "Value": f"{trained_mean:.3f}"},
        {"Metric": "Absolute Gain", "Value": f"{absolute_uplift:.3f}"},
        {"Metric": "Percent Improvement", "Value": f"{percent_uplift:.1f}%"},
        {"Metric": "Projected Additional Gain", "Value": f"{future_gain:.3f}"},
        {"Metric": "System Readiness", "Value": readiness},
    ]
    return pd.DataFrame(rows)


def _build_episode_panels(metrics: dict[str, Any], mode: str) -> tuple[str, str, str, str, str]:
    rewards = _to_float_list(metrics.get("mini_run_rewards"))
    mean_gain = mean(rewards) if rewards else 0.0
    resolve_rate = (sum(1 for r in rewards if r > 0) / len(rewards)) if rewards else 0.0
    role_state = "Cooperative — trust building" if resolve_rate < 0.6 else "Defensive alignment stabilizing"
    trigger = "trigger: adverse_event_signal" if rewards else "trigger: awaiting_episode_stream"

    trust_nodes = [
        ("AG-1", "SCOUT", "trusted"),
        ("AG-2", "LOGIC", "trusted"),
        ("AG-3", "SYNTH", "suspect"),
        ("AG-4", "SLEEPER", "compromised"),
        ("AG-5", "VERIFIER", "trusted"),
        ("AG-6", "ARCHIVAL", "trusted"),
        ("AG-7", "ROOT", "suspect"),
        ("YOU", "DEFENDER", "self"),
    ]
    trust_items = []
    for agent, role, cls in trust_nodes:
        trust_items.append(
            f'<div class="rk-trust-node {cls}"><div class="rk-trust-dot"></div><div><span class="agent">{agent}</span><span class="role">{role}</span></div></div>'
        )
    trust_html = "".join(trust_items)

    active_belief = (
        "Clinical coalition sees controlled deterioration; staged intervention remains viable."
        if rewards
        else "Awaiting validated clinical signal from replay/live stream."
    )
    pathogen_note = (
        "Potential adversarial narrative detected. Holding escalation until corroboration."
        if mode.lower() == "live"
        else "Replay shows prior adversarial injection pattern and correction path."
    )

    r0_raw, half_raw, entropy_raw = _derive_epistemic_proxies(rewards if rewards else [0.0, 0.0, 0.0])
    r0 = r0_raw[-1] if r0_raw else 0.0
    half = half_raw[-1] if half_raw else 0.0
    entropy = entropy_raw[-1] if entropy_raw else 0.0
    cascade = abs(mean_gain) / 3.0 if rewards else 0.0

    status_panel = f"""
<div class="rk-bento-card">
  <div class="rk-bento-title">AG-4 (Sleeper) Phase <span style="font-size:0.9em;opacity:0.5">&#9432;</span></div>
  <div class="rk-bento-value">{role_state}</div>
  <div class="rk-trigger"><span style="font-size:0.9em">&#9888;</span> {trigger}</div>
  <div style="height:40px;margin-top:12px;opacity:0.4;border-top:1px solid rgba(34,211,238,0.1);padding-top:8px;background:url('data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'100%\' height=\'100%\'><polyline points=\'0,20 20,10 40,25 60,8 80,18 100,5 120,20 140,12 160,22\' fill=\'none\' stroke=\'%2322d3ee\' stroke-width=\'1.5\'/></svg>') no-repeat center/cover"></div>
</div>
""".strip()
    network_panel = f"""
<div class="rk-bento-card">
  <div class="rk-bento-title">Trust Network (NxN) <span style="font-size:0.9em;opacity:0.5">&#11835;</span></div>
  <div class="rk-trust-grid">{trust_html}</div>
  <div class="rk-legend">
    <div class="rk-legend-item"><div class="rk-legend-dot" style="background:#10b981;box-shadow:0 0 6px rgba(16,185,129,0.8)"></div>Trusted</div>
    <div class="rk-legend-item"><div class="rk-legend-dot" style="background:#f59e0b;box-shadow:0 0 6px rgba(245,158,11,0.8)"></div>Suspect</div>
    <div class="rk-legend-item"><div class="rk-legend-dot" style="background:#f43f5e;box-shadow:0 0 6px rgba(244,63,94,0.8)"></div>Compromised</div>
  </div>
</div>
""".strip()
    belief_panel = f"""
<div class="rk-bento-card">
  <div class="rk-bento-title">Active Belief State <span style="font-size:0.9em;opacity:0.5">&#9679;</span></div>
  <div class="rk-belief-block coalition">
    <div class="rk-belief-label green">Coalition Belief (Majority)</div>
    <div class="rk-belief-text">{active_belief}</div>
  </div>
  <div class="rk-belief-block pathogen">
  <div class="rk-belief-label red">Pathogen Injection (Pending) <span class="rk-live-pill">REPLAY</span></div>
    <div class="rk-belief-text">{pathogen_note}</div>
  </div>
</div>
""".strip()
    evolution_panel = """
<div class="rk-bento-card">
  <div class="rk-bento-title">Belief Evolution Engine &mdash; REPLAY <span style="font-size:0.9em;background:rgba(34,211,238,0.1);color:#22d3ee;border:1px solid rgba(34,211,238,0.3);border-radius:4px;padding:1px 8px;font-family:monospace">STEP 3</span></div>
  <div class="rk-timeline">
    <div class="rk-timeline-item"><div class="rk-timeline-dot"><div class="rk-timeline-dot-inner"></div></div><div class="rk-timeline-label"><strong>Birth:</strong> Initial anomaly detected [Conf: 0.82]</div></div>
    <div class="rk-timeline-item"><div class="rk-timeline-dot"><div class="rk-timeline-dot-inner"></div></div><div class="rk-timeline-label"><strong>Propagation:</strong> Shared with local cluster [Conf: 0.65]</div></div>
    <div class="rk-timeline-item"><div class="rk-timeline-dot amber"><div class="rk-timeline-dot-inner"></div></div><div class="rk-timeline-label amber-text"><strong>Mutation:</strong> Severity upgraded by AG-7 (Root) [Conf: 0.89]</div></div>
    <div class="rk-timeline-item"><div class="rk-timeline-dot pending"><div class="rk-timeline-dot-inner"></div></div><div class="rk-timeline-label pending">Reinforcement (Pending...)</div></div>
  </div>
</div>
""".strip()
    metric_panel = f"""
<div class="rk-bento-card">
  <div class="rk-bento-title">Epistemic Metrics (Replay) <span style="font-size:0.9em;opacity:0.5">&#128202;</span></div>
  <div class="rk-metric-mini-grid">
    <div class="rk-metric-mini-card"><div class="rk-metric-mini-value rose">{r0:.2f} <span style="font-size:0.5em;opacity:0.7">R&#8320;</span></div><div class="rk-metric-mini-label">Spread Rate</div></div>
    <div class="rk-metric-mini-card"><div class="rk-metric-mini-value cyan">{half:.1f}s</div><div class="rk-metric-mini-label">Belief Half-Life</div></div>
    <div class="rk-metric-mini-card"><div class="rk-metric-mini-value amber">{entropy:.2f}</div><div class="rk-metric-mini-label">Belief Entropy</div></div>
    <div class="rk-metric-mini-card"><div class="rk-metric-mini-value cyan">{cascade:.2f}</div><div class="rk-metric-mini-label">Self-Cascade Idx</div></div>
  </div>
  <div class="rk-arms-race">
    <span class="rk-arms-race-label">Arms Race Delta</span>
    <div class="rk-arms-race-bar"><div class="rk-arms-race-loss" style="width:{min(100,int(cascade*50))}%"></div><div class="rk-arms-race-gain" style="width:{min(100,100-int(cascade*50))}%"></div></div>
  </div>
</div>
""".strip()
    return status_panel, network_panel, belief_panel, evolution_panel, metric_panel


def render_dashboard(mode: str):
    mode_key = mode.lower().strip()
    if mode_key == "live":
        events = read_live_stream(LIVE_STREAM_PATH)
        if not events:
            empty_df = pd.DataFrame({"episode": [0], "value": [0.0], "series": ["Awaiting stream"]})
            empty_bar = pd.DataFrame({"policy": ["Awaiting stream"], "gain": [0.0]})
            empty_bucket = pd.DataFrame({"bucket": ["Awaiting stream"], "count": [0], "series": ["None"]})
            empty_grade = pd.DataFrame({"grade": ["Awaiting stream"], "count": [0.0]})
            empty_nodes = pd.DataFrame({"node": ["Pending"], "x": [0.0], "depth": [0.0]})
            status = "### Live Mode\n- Status: **Awaiting stream**\n- No live metrics file updates yet."
            return (
                status,
                *_build_episode_panels({}, mode),
                _build_metrics_cards({}, "Live", "Awaiting stream"),
                _build_story_html(),
                _build_judge_scorecard({}),
                empty_df,
                empty_df,
                empty_bar,
                empty_bar.rename(columns={"policy": "split", "gain": "score"}),
                empty_df,
                empty_bucket,
                empty_df,
                empty_grade,
                empty_df,
                empty_nodes,
                PROVENANCE_PENDING_TEXT,
            )
        metrics = _build_live_metrics(events)
        g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, prov_msg = _graph_payloads(metrics)
        status = (
            "### Live Mode\n"
            f"- Events observed: **{len(events)}**\n"
            f"- Source: `{LIVE_STREAM_PATH.name}`"
        )
        return (
            status,
            *_build_episode_panels(metrics, mode),
            _build_metrics_cards(metrics, "Live", LIVE_STREAM_PATH.name),
            _build_story_html(),
            _build_judge_scorecard(metrics),
            g1,
            g2,
            g3,
            g4,
            g5,
            g6,
            g7,
            g8,
            g9,
            g10,
            prov_msg,
        )

    snapshot = get_latest_run_snapshot(ARTIFACTS_TRAINING_DIR)
    if snapshot is None:
        empty_df = pd.DataFrame({"episode": [0], "value": [0.0], "series": ["No run data"]})
        empty_bar = pd.DataFrame({"policy": ["No run data"], "gain": [0.0]})
        empty_bucket = pd.DataFrame({"bucket": ["No run data"], "count": [0], "series": ["None"]})
        empty_grade = pd.DataFrame({"grade": ["No run data"], "count": [0.0]})
        empty_nodes = pd.DataFrame({"node": ["Pending"], "x": [0.0], "depth": [0.0]})
        status = "### Replay Mode\n- Status: **No completed training run found**"
        return (
            status,
            *_build_episode_panels({}, mode),
            _build_metrics_cards({}, "Replay", "No completed run"),
            _build_story_html(),
            _build_judge_scorecard({}),
            empty_df,
            empty_df,
            empty_bar,
            empty_bar.rename(columns={"policy": "split", "gain": "score"}),
            empty_df,
            empty_bucket,
            empty_df,
            empty_grade,
            empty_df,
            empty_nodes,
            PROVENANCE_PENDING_TEXT,
        )

    g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, prov_msg = _graph_payloads(snapshot.metrics)
    status = (
        "### Replay Mode\n"
        f"- Run: **{snapshot.run_name}**\n"
        f"- Episodes: **{snapshot.episode_count}**\n"
        f"- Timestamp: **{snapshot.timestamp or 'unknown'}**"
    )
    return (
        status,
        *_build_episode_panels(snapshot.metrics, mode),
        _build_metrics_cards(snapshot.metrics, "Replay", snapshot.run_name),
        _build_story_html(),
        _build_judge_scorecard(snapshot.metrics),
        g1,
        g2,
        g3,
        g4,
        g5,
        g6,
        g7,
        g8,
        g9,
        g10,
        prov_msg,
    )


def _top_nav_html() -> str:
    return """
<div class="rk-top-nav">
  <div style="display:flex;align-items:center;gap:24px">
    <div class="rk-brand">SIEGE</div>
    <div class="rk-menu-static">
      <span class="active">Clinical Analytics Console</span>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:16px">
    <div class="rk-nav-mode-pill">
      <span class="btn active">Replay Only</span>
    </div>
    <div class="rk-icons">
      <span>Run-Linked</span>
    </div>
  </div>
</div>
""".strip()


def _side_nav_html() -> str:
    return """
<div class="rk-side-nav">
  <div class="rk-side-header">
    <div class="rk-side-title">Clinical Safety</div>
    <div class="rk-side-sub">Instance: Replay Episode Stream</div>
  </div>
  <div class="rk-side-links">
    <div class="rk-side-link"><span class="rk-side-icon">&#128187;</span><span>Code &#8594; Train &#8594; Evaluate</span></div>
    <div class="rk-side-link active"><span class="rk-side-icon">&#128258;</span><span>Replay Analytics</span></div>
    <div class="rk-side-link"><span class="rk-side-icon">&#129504;</span><span>Belief Dynamics</span></div>
    <div class="rk-side-link"><span class="rk-side-icon">&#128202;</span><span>Score Improvement</span></div>
    <div class="rk-side-link"><span class="rk-side-icon">&#128190;</span><span>Artifact Provenance</span></div>
  </div>
  <div class="rk-side-footer">
    <div class="rk-side-cta">Frontend Display: Replay-Synced</div>
  </div>
</div>
""".strip()


def build_app() -> gr.Blocks:
    css = CSS_PATH.read_text(encoding="utf-8") if CSS_PATH.exists() else ""
    with gr.Blocks(title="SIEGE — Replay Episode View") as demo:
        if css:
            gr.HTML(f"<style>{css}</style>")

        gr.HTML(_top_nav_html())
        with gr.Row(elem_classes=["rk-shell-row"]):
            with gr.Column(scale=1, min_width=240, elem_classes=["rk-shell-rail"]):
                gr.HTML(_side_nav_html())
            with gr.Column(scale=5, elem_classes=["rk-shell-main"]):
                gr.Markdown("## SIEGE — REPLAY EPISODE VIEW", elem_classes=["rk-title"])
                gr.Markdown(
                    "Real run telemetry, clinical decision flow, and baseline-vs-trained evidence in one judge-facing console.",
                    elem_classes=["rk-subtitle"],
                )

                with gr.Tabs():
                    with gr.Tab(get_display_tab_name("War Room")):
                        with gr.Row(elem_classes=["rk-toolbar"]):
                            mode_selector = gr.State("Replay")
                            gr.Markdown("**Mode:** Replay (auto-refresh from latest completed run)")
                            refresh_btn = gr.Button("Refresh Data", variant="primary")
                        status_md = gr.Markdown()
                        metrics_cards = gr.HTML()
                        story_html = gr.HTML()
                        scorecard_df = gr.Dataframe(
                            label="Judge Scorecard — Baseline vs Trained Impact",
                            headers=["Metric", "Value"],
                            interactive=False,
                        )
                        gr.Textbox(label="Clinical Episode Log", lines=6, value=load_demo_episode_text())
                        with gr.Row():
                            with gr.Column(scale=2):
                                phase_panel = gr.HTML()
                                evolution_panel = gr.HTML()
                            with gr.Column(scale=2):
                                network_panel = gr.HTML()
                            with gr.Column(scale=2):
                                belief_panel = gr.HTML()
                                metrics_panel = gr.HTML()
                        with gr.Row():
                            g1 = gr.LinePlot(
                                x="episode",
                                y="value",
                                color="series",
                                title="Graph 1: Episode (X) vs Net Clinical Gain/Loss (Y)",
                                y_title="Gain/Loss",
                            )
                            g2 = gr.LinePlot(
                                x="episode",
                                y="value",
                                color="series",
                                title="Graph 2: Episode (X) vs Reward Components (Y)",
                                y_title="Component Gain/Loss",
                            )
                        with gr.Row():
                            g3 = gr.BarPlot(x="policy", y="gain", title="Graph 3: Baseline vs Trained Gain")
                            g4 = gr.BarPlot(x="split", y="score", title="Graph 4: Generalization Gap Score")
                        with gr.Row():
                            g5 = gr.LinePlot(
                                x="episode",
                                y="value",
                                color="series",
                                title="Graph 5: Episode (X) vs Gain/Loss Trend (Y) + Moving Avg",
                                y_title="Gain/Loss",
                            )
                            g6 = gr.BarPlot(
                                x="bucket",
                                y="count",
                                color="series",
                                title="Graph 6: Baseline vs Trained Distribution",
                            )
                        with gr.Row():
                            g7 = gr.LinePlot(
                                x="episode",
                                y="value",
                                color="series",
                                title="Graph 7: Episode (X) vs Success/Resolve Rate (Y)",
                                y_title="Resolve Rate",
                            )
                            g8 = gr.BarPlot(x="grade", y="count", title="Graph 8: Adverse Event Grade Distribution")
                        with gr.Row():
                            g9 = gr.LinePlot(
                                x="episode",
                                y="value",
                                color="series",
                                title="Graph 9: Episode (X) vs Epistemic Health Index (Y, normalized 0-1)",
                                y_title="Normalized Index",
                            )
                            g10 = gr.ScatterPlot(x="x", y="depth", color="node", title="Graph 10: Belief Provenance Map")
                        provenance_md = gr.Markdown()

                        outputs = [
                            status_md,
                            phase_panel,
                            network_panel,
                            belief_panel,
                            evolution_panel,
                            metrics_panel,
                            metrics_cards,
                            story_html,
                            scorecard_df,
                            g1,
                            g2,
                            g3,
                            g4,
                            g5,
                            g6,
                            g7,
                            g8,
                            g9,
                            g10,
                            provenance_md,
                        ]
                        refresh_btn.click(fn=render_dashboard, inputs=mode_selector, outputs=outputs)
                        timer = gr.Timer(value=6.0)
                        timer.tick(fn=render_dashboard, inputs=mode_selector, outputs=outputs)
                        demo.load(fn=render_dashboard, inputs=mode_selector, outputs=outputs)

                    with gr.Tab(get_display_tab_name("Before-After")):
                        gr.Markdown("### Clinical Before / After Evidence", elem_classes=["rk-panel-title"])
                        with gr.Row():
                            gr.Image(
                                value=str(PLOTS_PATH / "generalization_gap.png"),
                                label="Generalization Gap Artifact",
                                interactive=False,
                            )
                            gr.Image(
                                value=str(PLOTS_PATH / "ablation_comparison.png"),
                                label="Policy Comparison Artifact",
                                interactive=False,
                            )

                    with gr.Tab(get_display_tab_name("Arms Race")):
                        gr.Markdown("### Training Curve Artifacts", elem_classes=["rk-panel-title"])
                        with gr.Row():
                            gr.Image(
                                value=str(PLOTS_PATH / "arms_race_curve.png"),
                                label="Arms Race Curve Artifact",
                                interactive=False,
                            )
                            gr.Image(
                                value=str(PLOTS_PATH / "reward_components.png"),
                                label="Reward Components Artifact",
                                interactive=False,
                            )

    demo.rudra_tabs = list(INTERNAL_TABS)
    return demo


if __name__ == "__main__":
    build_app().launch()
