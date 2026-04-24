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
