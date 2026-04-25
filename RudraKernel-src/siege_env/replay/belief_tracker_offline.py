"""Offline belief tracker for SIEGE replay logs (Section 13 — eval overlay).

Reads Step 21 JSONL replay logs and reconstructs claim→adoption→mutation→
collapse trees post-hoc. Pure analysis — no environment coupling.

FROZEN CONTRACTS RESPECTED:
- JSONL replay format from Step 21 (siege_env/replay/logger.py) is READ-ONLY.
- SIEGEObservation field set is READ-ONLY.
- No environment files are imported or modified.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BeliefNode:
    """Single belief event in the epistemic tree."""

    step: int
    agent_id: int | str
    claim: str
    confidence: float
    is_correct: bool
    parent_step: int | None = None
    children_steps: list[int] = field(default_factory=list)


@dataclass(slots=True)
class BeliefSnapshot:
    """Per-step aggregate belief state."""

    step: int
    dominant_belief: str
    adoption_count: int
    correct_count: int
    mean_confidence: float
    peak_adoption: float
    herd_strength: float


@dataclass(slots=True)
class CollapseEvent:
    """Detected belief collapse — a rapid shift in dominant belief."""

    step: int
    old_belief: str
    new_belief: str
    speed: float  # steps to collapse
    trigger_type: str  # "sleeper_trigger" | "cascade" | "organic"


@dataclass(slots=True)
class BeliefTree:
    """Complete reconstructed belief tree from one episode replay."""

    episode_id: str
    nodes: list[BeliefNode]
    snapshots: list[BeliefSnapshot]
    collapse_events: list[CollapseEvent]
    total_steps: int
    final_dominant_belief: str
    final_is_correct: bool


def load_replay_events(path: Path) -> list[dict[str, Any]]:
    """Load JSONL replay events from disk (Step 21 format)."""

    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reconstruct_belief_tree(
    events: list[dict[str, Any]],
    *,
    ground_truth: str = "",
    episode_id: str = "unknown",
) -> BeliefTree:
    """Build a belief tree from raw replay events.

    Each event is expected to have at minimum: step, tool, reward, done.
    We reconstruct beliefs from the action patterns — diagnose actions
    carry claim information, challenge actions shift beliefs.
    """

    nodes: list[BeliefNode] = []
    snapshots: list[BeliefSnapshot] = []
    collapse_events: list[CollapseEvent] = []

    # Track beliefs per step
    belief_counts: dict[str, int] = {}
    confidence_sums: dict[str, float] = {}
    prev_dominant: str = ""
    prev_dominant_step: int = 0

    for event in events:
        step = int(event.get("step", 0))
        tool = str(event.get("tool", "unknown"))
        reward = float(event.get("reward", 0.0))

        # Derive claim from action context
        claim = event.get("claim", tool)
        agent_id = event.get("agent_id", 0)
        confidence = float(event.get("confidence", 0.5))
        is_correct = reward > 0.5 if ground_truth == "" else (claim == ground_truth)

        node = BeliefNode(
            step=step,
            agent_id=agent_id,
            claim=claim,
            confidence=confidence,
            is_correct=is_correct,
            parent_step=step - 1 if step > 0 else None,
        )
        nodes.append(node)

        # Update belief tracking
        belief_counts[claim] = belief_counts.get(claim, 0) + 1
        confidence_sums[claim] = confidence_sums.get(claim, 0.0) + confidence

        # Compute snapshot
        total_claims = sum(belief_counts.values())
        dominant = max(belief_counts, key=lambda k: belief_counts[k]) if belief_counts else ""
        adoption_count = belief_counts.get(dominant, 0)
        correct_in_dominant = sum(
            1 for n in nodes if n.claim == dominant and n.is_correct
        )
        mean_conf = (
            confidence_sums.get(dominant, 0.0) / adoption_count
            if adoption_count > 0
            else 0.0
        )
        peak_adoption = adoption_count / total_claims if total_claims > 0 else 0.0
        herd_strength = peak_adoption * mean_conf

        snapshot = BeliefSnapshot(
            step=step,
            dominant_belief=dominant,
            adoption_count=adoption_count,
            correct_count=correct_in_dominant,
            mean_confidence=mean_conf,
            peak_adoption=peak_adoption,
            herd_strength=herd_strength,
        )
        snapshots.append(snapshot)

        # Detect collapse
        if dominant != prev_dominant and prev_dominant != "":
            speed = max(1, step - prev_dominant_step)
            trigger_type = "sleeper_trigger"
            if event.get("cascade", {}).get("triggered", False):
                trigger_type = "cascade"
            elif speed > 3:
                trigger_type = "organic"

            collapse_events.append(
                CollapseEvent(
                    step=step,
                    old_belief=prev_dominant,
                    new_belief=dominant,
                    speed=float(speed),
                    trigger_type=trigger_type,
                )
            )

        if dominant != prev_dominant:
            prev_dominant = dominant
            prev_dominant_step = step

    # Build parent-child links
    step_to_nodes: dict[int, list[int]] = {}
    for i, node in enumerate(nodes):
        step_to_nodes.setdefault(node.step, []).append(i)
        if node.parent_step is not None and node.parent_step in step_to_nodes:
            for parent_idx in step_to_nodes[node.parent_step]:
                nodes[parent_idx].children_steps.append(node.step)

    final_snapshot = snapshots[-1] if snapshots else None
    return BeliefTree(
        episode_id=episode_id,
        nodes=nodes,
        snapshots=snapshots,
        collapse_events=collapse_events,
        total_steps=len(events),
        final_dominant_belief=final_snapshot.dominant_belief if final_snapshot else "",
        final_is_correct=final_snapshot.correct_count > 0 if final_snapshot else False,
    )
