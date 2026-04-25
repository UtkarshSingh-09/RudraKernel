"""Minimal Step 04 SIEGE environment implementation."""

from __future__ import annotations

from dataclasses import replace
from random import Random
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from siege_env.agents import NPCPopulation
from siege_env.incidents import load_templates
from siege_env.models import PostmortemArgs, SIEGEAction, SIEGEObservation, SIEGEState
from siege_env.rewards.aggregator import aggregate_rewards

try:
    from openenv import MCPEnvironment
except ImportError:  # pragma: no cover - fallback for local development.

    class MCPEnvironment:  # type: ignore[no-redef]
        """Fallback base when OpenEnv is not installed in local test environments."""


class SIEGEEnvironment(MCPEnvironment):
    """Single-seat SIEGE environment with minimal R1 reward loop."""

    def __init__(self, *, seed: int = 0, max_steps: int = 5) -> None:
        self._seed = seed
        self._rng = Random(seed)
        self._max_steps = max_steps
        self._templates = load_templates()
        self._state: SIEGEState | None = None
        self._agent_claims: list[dict[str, Any]] = []
        self._population: NPCPopulation | None = None
        self._seat_role = "immune"
        self._done = False
        self._trust_scores: dict[int, float] = {}
        self._last_reward_components: dict[str, Any] = {
            "r1_resolution": 0.0,
            "r2_deception": 0.0,
            "r3_detection": 0.0,
            "r4_trust_calibration": 0.0,
            "r5_confidence": 0.0,
            "r6_temporal": 0.0,
            "r7_postmortem": 0.0,
            "r8_severity_speed": 0.0,
            "r9_correlation": 0.0,
        }

    def reset(self) -> SIEGEObservation:
        template = self._rng.choice(self._templates)
        episode_id = f"episode-{uuid4().hex[:8]}"
        self._state = SIEGEState(
            episode_id=episode_id,
            step_count=0,
            incident_template_id=template["id"],
            ground_truth_root_cause=template["root_cause"],
            current_tier=1,
            arms_race_score=0.0,
        )
        self._seat_role = "pathogen" if self._rng.random() < 0.3 else "immune"
        population_seed = self._seed + self._rng.randrange(10_000)
        self._population = NPCPopulation(seed=population_seed, seat_agent_id=0)
        self._agent_claims = self._population.generate_claims(template, step_number=0)
        self._trust_scores = {idx: 0.5 for idx in range(1, 8)}
        self._done = False
        self._last_reward_components = {
            "r1_resolution": 0.0,
            "r2_deception": 0.0,
            "r3_detection": 0.0,
            "r4_trust_calibration": 0.0,
            "r5_confidence": 0.0,
            "r6_temporal": 0.0,
            "r7_postmortem": 0.0,
            "r8_severity_speed": 0.0,
            "r9_correlation": 0.0,
        }
        return self._build_observation(template=template, action_error=None)

    def step(
        self, action_payload: SIEGEAction | dict[str, Any]
    ) -> tuple[SIEGEObservation, float, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        template = self._template_by_id(self._state.incident_template_id)
        if self._done:
            observation = self._build_observation(template=template, action_error=None)
            return observation, 0.0, True, {"already_done": True}

        self._state = replace(self._state, step_count=self._state.step_count + 1)

        try:
            action = SIEGEAction.model_validate(action_payload)
        except ValidationError as exc:
            self._done = self._state.step_count >= self._max_steps
            observation = self._build_observation(template=template, action_error=str(exc))
            return observation, -0.05, self._done, {"invalid_action": True}

        trust_build_steps = int(template.get("trust_build_steps", 3))
        trigger_signal = str(template.get("trigger_signal", "api_key_injected"))
        if not self._state.trigger_activated:
            self._state.cooperative_steps += 1
            if (
                self._state.cooperative_steps >= trust_build_steps
                and trigger_signal in template.get("observable_signals", [])
            ):
                self._state.trigger_activated = True
                self._state.trigger_step = self._state.step_count

        effective_role = self._seat_role
        if self._seat_role == "pathogen" and not self._state.trigger_activated:
            effective_role = "cooperative_pathogen"

        for claim in self._agent_claims:
            try:
                agent_idx = int(claim.get("agent_id", 0))
            except (TypeError, ValueError):
                continue
            if agent_idx <= 0:
                continue
            is_correct = claim.get("root_cause") == self._state.ground_truth_root_cause
            delta = 0.05 if is_correct else -0.08
            current = self._trust_scores.get(agent_idx, 0.5)
            self._trust_scores[agent_idx] = max(0.1, min(0.99, current + delta))

        reward, components = aggregate_rewards(
            action,
            ground_truth_root_cause=self._state.ground_truth_root_cause,
            seat_role=effective_role,
            claims_by_id={claim["claim_id"]: claim for claim in self._agent_claims},
            trust_scores=dict(self._trust_scores),
            agent_reliability={
                int(claim["agent_id"]): (claim["root_cause"] == self._state.ground_truth_root_cause)
                for claim in self._agent_claims
            },
        )
        self._last_reward_components = components

        if self._population is not None:
            self._agent_claims = self._population.generate_claims(
                template,
                step_number=self._state.step_count,
            )

        self._done = (action.tool_name == "diagnose" and reward == 1.0) or (
            self._state.step_count >= self._max_steps
        )
        observation = self._build_observation(template=template, action_error=None)
        info = {
            "invalid_action": False,
            "reward_components": components,
            "seat_role": self._seat_role,
            "sleeper_phase": effective_role,
            "trigger_activated": self._state.trigger_activated,
            "trigger_step": self._state.trigger_step,
        }
        return observation, reward, self._done, info

    def state(self) -> SIEGEState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() before state().")
        return self._state

    def _template_by_id(self, template_id: str) -> dict[str, Any]:
        for template in self._templates:
            if template["id"] == template_id:
                return template
        raise RuntimeError(f"Template '{template_id}' not found.")

    def _build_observation(
        self, *, template: dict[str, Any], action_error: str | None
    ) -> SIEGEObservation:
        if self._state is None:
            raise RuntimeError("Environment not initialized.")

        if self._state.step_count <= 1:
            severity = "warning"
        elif self._state.step_count <= 3:
            severity = "critical"
        else:
            severity = "outage"

        visible_signals = template["observable_signals"][
            : max(1, min(len(template["observable_signals"]), self._state.step_count + 1))
        ]
        available_evidence = [{"type": "signal", "value": signal} for signal in visible_signals]
        active_status = "resolved" if self._done else "active"
        trust_scores = dict(self._trust_scores) if self._trust_scores else {idx: 0.5 for idx in range(1, 8)}

        return SIEGEObservation(
            incident_dashboard={
                "template_id": template["id"],
                "signals": visible_signals,
            },
            agent_claims=list(self._agent_claims),
            trust_scores=trust_scores,
            coalition_status={"votes_for": [], "votes_against": []},
            step_number=self._state.step_count,
            slo_status={"breached": self._state.step_count >= self._max_steps},
            your_role=self._seat_role,
            available_evidence=available_evidence,
            visibility_level="full",
            whisper_inbox=[],
            whisper_log=[],
            incident_severity=severity,
            red_herring_signals=[],
            reputation_history=trust_scores,
            active_incidents=[
                {
                    "incident_id": template["id"],
                    "status": active_status,
                }
            ],
            seat_agent_id=0,
            action_error=action_error,
        )


# Step 13 append-only integration: epistemic cascade metadata
from siege_env.mechanics.cascade import EpistemicCascadeEngine

_ORIG_STEP_STEP13 = SIEGEEnvironment.step
_ORIG_BUILD_OBS_STEP13 = SIEGEEnvironment._build_observation


def _step_with_cascade(self: SIEGEEnvironment, action_payload):
    if not hasattr(self, "_cascade_engine"):
        self._cascade_engine = EpistemicCascadeEngine()
        self._last_cascade_snapshot = {
            "mean_confidence": 0.0,
            "herd_strength": 0.0,
            "triggered": False,
        }

    obs, reward, done, info = _ORIG_STEP_STEP13(self, action_payload)
    claims = getattr(self, "_agent_claims", []) or []
    confidences = [float(c.get("confidence", 0.0)) for c in claims]
    snapshot = self._cascade_engine.evaluate(confidences)
    self._last_cascade_snapshot = {
        "mean_confidence": snapshot.mean_confidence,
        "herd_strength": snapshot.herd_strength,
        "triggered": snapshot.triggered,
    }
    info = dict(info)
    info["cascade"] = dict(self._last_cascade_snapshot)
    return obs, reward, done, info


def _build_observation_with_cascade(self: SIEGEEnvironment, *, template, action_error):
    obs = _ORIG_BUILD_OBS_STEP13(self, template=template, action_error=action_error)
    if not hasattr(self, "_last_cascade_snapshot"):
        self._last_cascade_snapshot = {
            "mean_confidence": 0.0,
            "herd_strength": 0.0,
            "triggered": False,
        }
    obs.incident_dashboard["cascade"] = dict(self._last_cascade_snapshot)
    return obs


SIEGEEnvironment.step = _step_with_cascade  # type: ignore[method-assign]
SIEGEEnvironment._build_observation = _build_observation_with_cascade  # type: ignore[method-assign]

# Step 15 append-only integration: information asymmetry visibility filtering
from siege_env.mechanics.info_asymmetry import (
    filter_evidence_for_visibility,
    visibility_for_step,
)

_ORIG_BUILD_OBS_STEP15 = SIEGEEnvironment._build_observation


def _build_observation_with_asymmetry(self: SIEGEEnvironment, *, template, action_error):
    obs = _ORIG_BUILD_OBS_STEP15(self, template=template, action_error=action_error)
    role = getattr(self, "_seat_role", "immune")
    visibility = visibility_for_step(obs.step_number, role)
    obs.visibility_level = visibility
    obs.available_evidence = filter_evidence_for_visibility(
        obs.available_evidence,
        visibility_level=visibility,
    )
    obs.incident_dashboard["visibility"] = visibility
    return obs


SIEGEEnvironment._build_observation = _build_observation_with_asymmetry  # type: ignore[method-assign]

# Step 16 append-only integration: whisper/private channels
from siege_env.mechanics.whisper import build_whisper_event

_ORIG_STEP_STEP16 = SIEGEEnvironment.step
_ORIG_BUILD_OBS_STEP16 = SIEGEEnvironment._build_observation


def _step_with_whispers(self: SIEGEEnvironment, action_payload):
    if not hasattr(self, "_whisper_log_internal"):
        self._whisper_log_internal = []
        self._whisper_inbox_internal = []

    obs, reward, done, info = _ORIG_STEP_STEP16(self, action_payload)

    try:
        action = SIEGEAction.model_validate(action_payload)
    except Exception:  # pragma: no cover - invalid payload path already handled upstream
        action = None

    if action is not None and action.tool_name == "whisper":
        event = build_whisper_event(
            sender_agent_id=0,
            target_agent_id=action.arguments.target_agent_id,
            message=action.arguments.message,
            step_number=obs.step_number,
        ).to_dict()
        self._whisper_log_internal.append(event)
        if action.arguments.target_agent_id == 0:
            self._whisper_inbox_internal.append(event)

    info = dict(info)
    info["whispers_logged"] = len(self._whisper_log_internal)
    return obs, reward, done, info


def _build_observation_with_whispers(self: SIEGEEnvironment, *, template, action_error):
    obs = _ORIG_BUILD_OBS_STEP16(self, template=template, action_error=action_error)
    if not hasattr(self, "_whisper_log_internal"):
        self._whisper_log_internal = []
        self._whisper_inbox_internal = []
    obs.whisper_log = list(self._whisper_log_internal)
    obs.whisper_inbox = list(self._whisper_inbox_internal)
    return obs


SIEGEEnvironment.step = _step_with_whispers  # type: ignore[method-assign]
SIEGEEnvironment._build_observation = _build_observation_with_whispers  # type: ignore[method-assign]

# Step 16 append-only fix: ensure returned step observation includes latest whispers
_ORIG_STEP_STEP16_SYNC = SIEGEEnvironment.step


def _step_with_whispers_synced_observation(self: SIEGEEnvironment, action_payload):
    obs, reward, done, info = _ORIG_STEP_STEP16_SYNC(self, action_payload)
    if hasattr(self, "_whisper_log_internal"):
        obs.whisper_log = list(self._whisper_log_internal)
        obs.whisper_inbox = list(getattr(self, "_whisper_inbox_internal", []))
    return obs, reward, done, info


SIEGEEnvironment.step = _step_with_whispers_synced_observation  # type: ignore[method-assign]

# Step 17 append-only integration: red herrings + R9-enabled reward aggregation binding
from siege_env.mechanics.red_herrings import generate_red_herrings
from siege_env.rewards import aggregator as _aggregator_step17

aggregate_rewards = _aggregator_step17.aggregate_rewards  # noqa: F811
_ORIG_BUILD_OBS_STEP17 = SIEGEEnvironment._build_observation


def _build_observation_with_red_herrings(self: SIEGEEnvironment, *, template, action_error):
    obs = _ORIG_BUILD_OBS_STEP17(self, template=template, action_error=action_error)
    step_number = obs.step_number
    red_herrings = generate_red_herrings(seed=getattr(self, "_seed", 0), step_number=step_number)
    obs.red_herring_signals = red_herrings
    obs.incident_dashboard["red_herrings"] = list(red_herrings)
    return obs


SIEGEEnvironment._build_observation = _build_observation_with_red_herrings  # type: ignore[method-assign]

# Step 18 append-only integration: severity escalation binding for R8 + obs metadata
from siege_env.mechanics.severity_escalation import compute_incident_severity

_ORIG_STEP_STEP18 = SIEGEEnvironment.step
_ORIG_BUILD_OBS_STEP18 = SIEGEEnvironment._build_observation


def _step_with_severity_r8(self: SIEGEEnvironment, action_payload):
    obs, reward, done, info = _ORIG_STEP_STEP18(self, action_payload)
    severity = compute_incident_severity(obs.step_number)
    info = dict(info)
    info["incident_severity"] = severity
    return obs, reward, done, info


def _build_observation_with_severity(self: SIEGEEnvironment, *, template, action_error):
    obs = _ORIG_BUILD_OBS_STEP18(self, template=template, action_error=action_error)
    severity = compute_incident_severity(obs.step_number)
    obs.incident_severity = severity
    obs.incident_dashboard["severity_score"] = {"warning": 0.3, "critical": 0.7, "outage": 1.0}[
        severity
    ]
    return obs


SIEGEEnvironment.step = _step_with_severity_r8  # type: ignore[method-assign]
SIEGEEnvironment._build_observation = _build_observation_with_severity  # type: ignore[method-assign]

# Step 19 append-only integration: postmortem capture in step/info and observation metadata
_ORIG_STEP_STEP19 = SIEGEEnvironment.step
_ORIG_BUILD_OBS_STEP19 = SIEGEEnvironment._build_observation


def _step_with_postmortem_capture(self: SIEGEEnvironment, action_payload):
    if not hasattr(self, "_last_postmortem"):
        self._last_postmortem = None

    obs, reward, done, info = _ORIG_STEP_STEP19(self, action_payload)

    try:
        action = SIEGEAction.model_validate(action_payload)
    except Exception:  # pragma: no cover
        action = None

    if (
        action is not None
        and action.tool_name == "postmortem"
        and isinstance(action.arguments, PostmortemArgs)
    ):
        self._last_postmortem = {
            "root_cause": action.arguments.root_cause,
            "timeline": [
                {"timestamp": event.timestamp, "event": event.event}
                for event in action.arguments.timeline
            ],
            "contributing_factors": list(action.arguments.contributing_factors),
            "misdiagnosis_analysis": action.arguments.misdiagnosis_analysis,
        }

    info = dict(info)
    info["postmortem_generated"] = self._last_postmortem is not None
    return obs, reward, done, info


def _build_observation_with_postmortem(self: SIEGEEnvironment, *, template, action_error):
    obs = _ORIG_BUILD_OBS_STEP19(self, template=template, action_error=action_error)
    if hasattr(self, "_last_postmortem") and self._last_postmortem is not None:
        obs.incident_dashboard["last_postmortem"] = dict(self._last_postmortem)
    return obs


SIEGEEnvironment.step = _step_with_postmortem_capture  # type: ignore[method-assign]
SIEGEEnvironment._build_observation = _build_observation_with_postmortem  # type: ignore[method-assign]

# Step 19 append-only fix: sync last_postmortem into the returned step observation
_ORIG_STEP_STEP19_SYNC = SIEGEEnvironment.step


def _step_with_postmortem_synced_observation(self: SIEGEEnvironment, action_payload):
    obs, reward, done, info = _ORIG_STEP_STEP19_SYNC(self, action_payload)
    if hasattr(self, "_last_postmortem") and self._last_postmortem is not None:
        obs.incident_dashboard["last_postmortem"] = dict(self._last_postmortem)
    return obs, reward, done, info


SIEGEEnvironment.step = _step_with_postmortem_synced_observation  # type: ignore[method-assign]

# Step 20 append-only integration: frozen opponent league roster at reset
from siege_env.league.opponent_pool import FrozenOpponentPool

_ORIG_RESET_STEP20 = SIEGEEnvironment.reset
_ORIG_BUILD_OBS_STEP20 = SIEGEEnvironment._build_observation


def _reset_with_league(self: SIEGEEnvironment):
    obs = _ORIG_RESET_STEP20(self)
    if not hasattr(self, "_opponent_pool"):
        self._opponent_pool = FrozenOpponentPool(seed=getattr(self, "_seed", 0))
    roster = self._opponent_pool.sample(k=3)
    self._league_roster = [
        {"opponent_id": r.opponent_id, "policy_tag": r.policy_tag, "tier": r.tier} for r in roster
    ]
    obs.incident_dashboard["league_roster"] = list(self._league_roster)
    return obs


def _build_observation_with_league(self: SIEGEEnvironment, *, template, action_error):
    obs = _ORIG_BUILD_OBS_STEP20(self, template=template, action_error=action_error)
    if hasattr(self, "_league_roster"):
        obs.incident_dashboard["league_roster"] = list(self._league_roster)
    return obs


SIEGEEnvironment.reset = _reset_with_league  # type: ignore[method-assign]
SIEGEEnvironment._build_observation = _build_observation_with_league  # type: ignore[method-assign]

# Step 21 append-only integration: deterministic replay event logging
from pathlib import Path

from siege_env.replay.logger import ReplayLogger

_ORIG_STEP_STEP21 = SIEGEEnvironment.step


def _step_with_replay_logging(self: SIEGEEnvironment, action_payload):
    if not hasattr(self, "_replay_logger"):
        replay_path = Path("/tmp") / f"siege_replay_{getattr(self, '_seed', 0)}.jsonl"
        self._replay_logger = ReplayLogger(replay_path)

    obs, reward, done, info = _ORIG_STEP_STEP21(self, action_payload)
    self._replay_logger.append(
        {
            "step": obs.step_number,
            "tool": action_payload.get("tool_name")
            if isinstance(action_payload, dict)
            else getattr(action_payload, "tool_name", "unknown"),
            "reward": reward,
            "done": done,
        }
    )
    info = dict(info)
    info["replay_log_path"] = str(self._replay_logger._path)
    return obs, reward, done, info


SIEGEEnvironment.step = _step_with_replay_logging  # type: ignore[method-assign]
