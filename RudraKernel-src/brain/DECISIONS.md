# DECISIONS

## ADR-000: Bootstrap conventions
Status: Accepted

Context:
- Step 00 requires repository skeleton, quality gates, and context tooling.

Decision:
- Establish brain files and tooling first, then implement environment logic in step-gated order.

Consequences:
- Faster onboarding and stricter execution tracking.

## ADR-001: Strict Pydantic actions with dataclass transport models
Status: Accepted

Context:
- Step 02 introduces six MCP-style tools whose payloads must reject malformed inputs cleanly.
- Later environment steps need JSON-serializable observations and state snapshots without over-coupling runtime state to Pydantic models.

Decision:
- Use Pydantic v2 models for all tool argument schemas and the top-level `SIEGEAction`.
- Keep `SIEGEObservation` and `SIEGEState` as validated standard-library dataclasses with explicit `to_dict`/`from_dict` and `to_json`/`from_json` helpers.

Consequences:
- Tool payloads get strict validation, helpful error surfaces, and JSON schema export for future MCP registration.
- Observation and state objects stay lightweight, explicit, and easy to serialize into brain snapshots and replay logs.

## ADR-002: Seed incidents are fixed canonical templates with deterministic variants
Status: Accepted

Context:
- Step 03 requires five real post-mortem seeds that later steps can expand without destabilizing tests.
- Replay and debugging workflows benefit from deterministic variant generation.

Decision:
- Keep a strict minimal template schema in `templates.json` with only required keys.
- Validate templates at load time and generate variants deterministically by rotating list fields and suffixing IDs.

Consequences:
- Step 03 remains stable, predictable, and easy to test.
- Step 14 can expand from this contract without breaking early-phase behavior.

## ADR-003: Step 04 uses MCPEnvironment-compatible minimal loop with graceful invalid actions
Status: Accepted

Context:
- Step 04 needs a working `reset/step/state` loop with R1 reward only.
- Local test environments may not always have OpenEnv installed.

Decision:
- Implement `SIEGEEnvironment(MCPEnvironment)` with a local fallback base if `openenv` import is unavailable.
- Keep invalid actions non-fatal: return `action_error` observation + reward `-0.05` and continue until terminal criteria.

Consequences:
- Step 04 remains aligned with MCP-first architecture and is testable in constrained local environments.
- Later steps can replace the fallback path without changing environment behavior contracts.

## ADR-004: Scripted NPC population uses deterministic role-conditioned policies
Status: Accepted

Context:
- Step 05 requires fast scripted NPCs with deterministic claim generation and role compliance tests.
- Phase A prioritizes reproducibility over realism.

Decision:
- Introduce `ScriptedNPCAgent` roles (`lead`, `verifier`, `contrarian`) with fixed confidence bands.
- Add `NPCPopulation` orchestrator that builds 7 NPC agents (for a single seat setup) and emits deterministic claims from a seed.

Consequences:
- Claim generation is stable for test gates and replay/debug workflows.
- Later steps can swap/augment policies without changing population contract.

## ADR-005: Trust is modeled with per-observer Bayesian updates plus weighted coalition ratification
Status: Accepted

Context:
- Step 06 requires an `N x N` trust matrix and weighted coalition voting gate.
- The implementation needs deterministic behavior suitable for strict gate tests.

Decision:
- Add `BayesianTrustNetwork` with diagonal self-trust fixed at `1.0` and posterior updates from claim correctness signals.
- Add `CoalitionVoting` that computes weighted support/oppose tallies and ratifies only when support beats oppose and clears a threshold.

Consequences:
- Trust evolution and coalition decisions are now explicit primitives for Steps 07+.
- The voting contract is stable and testable (including ties, abstentions, and default weights).

## ADR-006: Seat role is sampled per episode (70/30) and R2/R3 are role-gated
Status: Accepted

Context:
- Step 07 introduces pathogen role assignment and reward components for deception/detection.
- Step 04/05 contracts must remain stable while adding adversarial dynamics.

Decision:
- Sample seat role on reset with probability `P(pathogen)=0.3`, `P(immune)=0.7`.
- Add `R2` (pathogen deception reward for incorrect diagnose) and `R3` (immune detection reward for correctly challenging incorrect claims) as separate components in the aggregator.

Consequences:
- Role-split behavior is now testable at environment level.
- Later reward steps can build on a clean, composable reward-component interface.

## ADR-007: Trust calibration uses Brier scoring and exploit-counter tests for R1-R4
Status: Accepted

Context:
- Step 08 introduces R4 trust calibration and requires reward-hacking resistance tests for R1-R4.
- Existing reward aggregation must remain bounded and composable.

Decision:
- Implement `R4` as `1 - mean_brier(trust_scores, actual_reliability)` with clamping to `[0, 1]`.
- Extend aggregator inputs with `trust_scores` and `agent_reliability` and include `r4_trust_calibration` component.
- Add explicit exploit-counter gate tests for trivial reward hacks across R1-R4.

Consequences:
- Trust calibration now has mathematically grounded scoring and deterministic tests.
- Reward surface is harder to game with naive policies before adding later reward components.
