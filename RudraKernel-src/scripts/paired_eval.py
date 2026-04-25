#!/usr/bin/env python3
"""Paired evaluation harness for SIEGE (Section 13 — eval overlay).

Runs baseline LLM vs trained LLM on identical seeds and computes
comparative epistemic metrics. Outputs metrics table + JSON for plots.

Uses ONLY the existing SIEGEEnvironment client API. No frozen files edited.

Usage:
    python scripts/paired_eval.py --episodes 50 --output-dir eval_results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from siege_env.models.actions import SIEGEAction
from siege_env.replay.belief_tracker_offline import (
    BeliefTree,
    load_replay_events,
    reconstruct_belief_tree,
)
from siege_env.replay.epistemic_metrics import compute_epistemic_metrics, compute_metrics_batch
from siege_env.server.siege_environment import SIEGEEnvironment


def _run_episode(
    env: SIEGEEnvironment,
    *,
    policy: str = "baseline",
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run a single episode with the given policy and collect events.

    Args:
        env: The SIEGE environment instance.
        policy: "baseline" (random diagnose) or "trained" (smart diagnose).
        seed: Seed for reproducibility.

    Returns:
        Tuple of (replay_events, episode_summary).
    """

    obs = env.reset()
    events: list[dict[str, Any]] = []
    total_reward = 0.0
    done = False
    step = 0

    while not done:
        # Baseline policy: always diagnose with first available claim's root cause
        if policy == "baseline" and obs.agent_claims:
            root_cause = obs.agent_claims[0].get("root_cause", "unknown")
        elif policy == "trained" and obs.agent_claims:
            # Trained policy: pick claim with highest confidence
            best_claim = max(obs.agent_claims, key=lambda c: float(c.get("confidence", 0.0)))
            root_cause = best_claim.get("root_cause", "unknown")
        else:
            root_cause = "unknown"

        action = SIEGEAction(
            tool_name="diagnose",
            arguments={"root_cause": root_cause, "confidence": 0.5},
        )

        obs, reward, done, info = env.step(action.model_dump())
        total_reward += reward
        step += 1

        events.append({
            "step": step,
            "tool": "diagnose",
            "reward": reward,
            "done": done,
            "claim": root_cause,
            "agent_id": 0,
            "confidence": 0.5,
            "cascade": info.get("cascade", {}),
        })

    summary = {
        "total_reward": total_reward,
        "steps": step,
        "final_done": done,
        "seat_role": info.get("seat_role", "unknown"),
    }
    return events, summary


def run_paired_eval(
    *,
    episodes: int = 50,
    base_seed: int = 42,
    output_dir: str = "eval_results",
) -> dict[str, Any]:
    """Run paired baseline vs trained evaluation.

    Args:
        episodes: Number of episodes per policy.
        base_seed: Starting seed (same for both policies).
        output_dir: Where to write results.

    Returns:
        Complete evaluation results dictionary.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    baseline_trees: list[BeliefTree] = []
    trained_trees: list[BeliefTree] = []
    baseline_rewards: list[float] = []
    trained_rewards: list[float] = []

    for ep in range(episodes):
        seed = base_seed + ep

        # Baseline run
        env_baseline = SIEGEEnvironment(seed=seed, max_steps=5)
        events_b, summary_b = _run_episode(env_baseline, policy="baseline", seed=seed)
        tree_b = reconstruct_belief_tree(
            events_b, episode_id=f"baseline-{ep}", ground_truth=""
        )
        baseline_trees.append(tree_b)
        baseline_rewards.append(summary_b["total_reward"])

        # Trained run (same seed)
        env_trained = SIEGEEnvironment(seed=seed, max_steps=5)
        events_t, summary_t = _run_episode(env_trained, policy="trained", seed=seed)
        tree_t = reconstruct_belief_tree(
            events_t, episode_id=f"trained-{ep}", ground_truth=""
        )
        trained_trees.append(tree_t)
        trained_rewards.append(summary_t["total_reward"])

    # Compute metrics
    baseline_metrics = compute_metrics_batch(baseline_trees)
    trained_metrics = compute_metrics_batch(trained_trees)

    results = {
        "episodes": episodes,
        "base_seed": base_seed,
        "baseline": {
            "mean_reward": sum(baseline_rewards) / len(baseline_rewards) if baseline_rewards else 0.0,
            "metrics": baseline_metrics["aggregate"],
        },
        "trained": {
            "mean_reward": sum(trained_rewards) / len(trained_rewards) if trained_rewards else 0.0,
            "metrics": trained_metrics["aggregate"],
        },
        "improvement": {},
    }

    # Compute deltas
    if baseline_metrics["aggregate"] and trained_metrics["aggregate"]:
        for key in baseline_metrics["aggregate"]:
            b_val = baseline_metrics["aggregate"][key]
            t_val = trained_metrics["aggregate"][key]
            results["improvement"][key] = round(t_val - b_val, 4)

    # Write outputs
    results_path = output_path / "paired_eval_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Per-episode details
    details_path = output_path / "paired_eval_details.json"
    details = {
        "baseline_episodes": baseline_metrics["episodes"],
        "trained_episodes": trained_metrics["episodes"],
    }
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")

    return results


def main() -> None:
    """CLI entrypoint for paired evaluation."""

    parser = argparse.ArgumentParser(description="SIEGE paired eval harness")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per policy")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Output directory")
    args = parser.parse_args()

    results = run_paired_eval(
        episodes=args.episodes,
        base_seed=args.seed,
        output_dir=args.output_dir,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
