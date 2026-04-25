# SIEGE — MASTER IMPLEMENTATION PLAN v1.0
## From Zero to Hackathon Victory — Engineering-Grade Execution Protocol

**Team:** Utkarsh (Lead Architect) + Ankit (Co-Engineer)
**Target:** OpenEnv India 2026 — Win + Merge into OpenEnv canonical
**Plan Status:** DRAFT — awaiting approval before execution begins
**Date Drafted:** 2026-04-24

---

## TABLE OF CONTENTS

1. [Execution Philosophy](#1-execution-philosophy)
2. [Repository Architecture](#2-repository-architecture)
3. [The Brain Folder — Master Context System](#3-the-brain-folder)
4. [Work Distribution Protocol](#4-work-distribution-protocol)
5. [Step-Gated Execution Roadmap (28 steps)](#5-step-gated-execution-roadmap)
6. [Testing Strategy](#6-testing-strategy)
7. [Frontend Storytelling Specification](#7-frontend-storytelling)
8. [Deployment Pipeline (GitHub → HF Space)](#8-deployment-pipeline)
9. [Discussion Protocol (Before Every Step)](#9-discussion-protocol)
10. [Quality Gates (Production-Grade)](#10-quality-gates)
11. [Appendix: File Manifest](#11-appendix-file-manifest)

---

## 1. EXECUTION PHILOSOPHY

### Core Principles

1. **Step-gated execution.** One step at a time. Next step never begins until:
   - Previous step's gate test passes (100%)
   - Master suite passes (100%)
   - Brain folder is updated (snapshot committed)
   - Both team members have reviewed

2. **Discussion-first.** Before any code is written for a step:
   - Agent presents plan + alternatives + risks + test strategy
   - User/Ankit approves or requests changes
   - Only then implementation begins

3. **Never break the workflow.** At every step, the project must:
   - Build cleanly (no import errors, no type errors)
   - Pass all prior tests
   - Deploy to Docker locally
   - Only fixes allowed mid-step; features wait for their own step

4. **Brain-first context.** Every decision, code snapshot, and change is logged in `brain/` with timestamps. Anyone joining mid-project has full context.

5. **Senior-engineer code quality.** Type-annotated, tested, documented, linted, formatted. No "we'll clean it up later."

### Non-Negotiables

- No step without its test file
- No test without passing
- No merge without brain update
- No feature without discussion
- No deployment without end-to-end validation

---

## 2. REPOSITORY ARCHITECTURE

```
Seize/
├── .github/
│   └── workflows/
│       ├── ci.yml                      # Run tests + lint on every push
│       ├── deploy.yml                  # Auto-deploy to HF Space on main
│       └── nightly.yml                 # Full E2E + ablation suite
├── .agents/
│   └── CLAUDE.md                       # Agent instructions for this repo
├── brain/                              # ← MASTER CONTEXT SYSTEM
│   ├── MASTER_CODE.md                  # Running compiled snapshot of all code
│   ├── CHANGELOG.md                    # Time-logged entries per step
│   ├── DECISIONS.md                    # Architecture decisions + rationale
│   ├── CONTEXT.md                      # Current project state (auto-updated)
│   ├── ROADMAP_STATUS.md               # Which steps complete, which pending
│   ├── HANDOFF.md                      # Pickup notes (updated end of each session)
│   ├── snapshots/                      # Per-step JSON snapshots
│   │   ├── step_00_<timestamp>.json
│   │   ├── step_01_<timestamp>.json
│   │   └── ...
│   ├── session_logs/                   # Per-session work logs
│   │   └── YYYY-MM-DD_session.md
│   └── tools/
│       ├── update_brain.py             # Auto-updates brain on step completion
│       └── compile_master_code.py      # Rebuilds MASTER_CODE.md from source
├── siege_env/                          # ← PRODUCTION CODE (OpenEnv package)
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── actions.py                  # Action + per-tool Pydantic args
│   │   ├── observations.py             # Observation dataclass
│   │   └── state.py                    # State dataclass
│   ├── client.py                       # SIEGEEnv(EnvClient)
│   ├── incidents/
│   │   ├── __init__.py
│   │   ├── templates.json              # 20 real post-mortem templates
│   │   ├── generator.py                # Parametric variant generator
│   │   └── loader.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── population.py               # NPC orchestrator
│   │   ├── scripted.py                 # Tier-1 rule-based NPCs
│   │   ├── llm_driven.py               # Tier-2 small-LLM NPCs
│   │   └── pathogen_strategies.py      # Trust poisoning strategy library
│   ├── trust/
│   │   ├── __init__.py
│   │   ├── network.py                  # N×N Bayesian trust matrix
│   │   ├── coalition.py                # Weighted voting + ratification
│   │   └── reputation.py               # Cross-episode persistence
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── r1_resolution.py
│   │   ├── r2_deception.py
│   │   ├── r3_detection.py
│   │   ├── r4_trust_calibration.py
│   │   ├── r5_confidence.py
│   │   ├── r6_temporal.py
│   │   ├── r7_postmortem.py
│   │   ├── r8_severity_speed.py
│   │   ├── r9_correlation.py
│   │   └── aggregator.py
│   ├── curriculum/
│   │   ├── __init__.py
│   │   └── tiered_scheduler.py
│   ├── mechanics/
│   │   ├── __init__.py
│   │   ├── whisper.py
│   │   ├── info_asymmetry.py
│   │   ├── red_herrings.py
│   │   ├── severity_escalation.py
│   │   ├── cascade.py                  # Epistemic cascade failures
│   │   └── multi_incident.py
│   ├── league/
│   │   ├── __init__.py
│   │   └── opponent_pool.py            # Frozen opponent league
│   ├── replay/
│   │   ├── __init__.py
│   │   ├── logger.py                   # JSONL trajectory logging
│   │   └── player.py                   # CLI replay tool
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seeding.py                  # Deterministic seed plumbing
│   │   └── validation.py               # Action validation helpers
│   └── server/
│       ├── __init__.py
│       ├── siege_environment.py        # SIEGEEnvironment(Environment)
│       ├── app.py                      # FastAPI app
│       ├── requirements.txt
│       └── Dockerfile
├── tests/                              # ← COMPREHENSIVE TEST SUITE (115+)
│   ├── conftest.py                     # Shared fixtures
│   ├── master_suite.py                 # ← RUNS AFTER EVERY STEP
│   ├── step_tests/                     # ← GATE TEST PER STEP
│   │   ├── step_00_bootstrap_test.py
│   │   ├── step_01_scaffold_test.py
│   │   ├── step_02_models_test.py
│   │   └── ...
│   ├── unit/
│   │   ├── test_models.py
│   │   ├── test_rewards/
│   │   │   ├── test_r1_resolution.py
│   │   │   ├── test_r2_deception.py
│   │   │   └── ...
│   │   ├── test_trust_network.py
│   │   ├── test_coalition.py
│   │   ├── test_curriculum.py
│   │   ├── test_whisper.py
│   │   ├── test_red_herrings.py
│   │   ├── test_info_asymmetry.py
│   │   ├── test_severity.py
│   │   ├── test_reputation.py
│   │   ├── test_incident_gen.py
│   │   ├── test_pathogen.py
│   │   └── test_openenv_api.py
│   ├── integration/
│   │   ├── test_full_episode.py
│   │   ├── test_determinism.py
│   │   ├── test_role_assignment.py
│   │   ├── test_reward_hacking.py      # 9 exploit tests
│   │   ├── test_league.py
│   │   └── test_invalid_actions.py
│   ├── e2e/
│   │   └── test_training_loop.py
│   ├── regression/
│   │   └── test_all_templates.py       # 20 template tests
│   ├── perf/
│   │   └── test_latency.py
│   └── fixtures/
│       ├── sample_trajectories/
│       ├── sample_incidents.json
│       └── exploit_policies.py
├── training/
│   ├── grpo_train.py                   # Main GRPO script (Unsloth/TRL)
│   ├── colab_notebook.ipynb            # Judge-runnable Colab
│   ├── configs/
│   │   ├── base.yaml
│   │   ├── ablate_curriculum.yaml
│   │   ├── ablate_trust_poisoning.yaml
│   │   └── ablate_whisper.yaml
│   └── wandb_config.py
├── frontend/                           # ← STORYTELLING DEMO
│   ├── app.py                          # Gradio main app
│   ├── components/
│   │   ├── war_room.py                 # Live episode replay
│   │   ├── before_after.py             # Side-by-side demo
│   │   ├── arms_race.py                # Live training curves
│   │   ├── trust_heatmap.py            # Animated trust matrix
│   │   └── metrics_dashboard.py        # Realistic backend numbers
│   ├── assets/
│   │   ├── agent_avatars/              # 8 agent avatars
│   │   ├── incident_icons/
│   │   └── css/
│   │       └── storytelling.css
│   ├── data/
│   │   ├── demo_episodes/              # Pre-recorded demo trajectories
│   │   └── baseline_vs_trained/        # Before/after pairs
│   └── requirements.txt
├── docs/
│   ├── README.md                       # Main README (links everything + 4-question structure)
│   ├── BLOG.md                         # HF blog post draft
│   ├── PITCH.md                        # 5-min pitch script
│   ├── SLIDES.pdf                      # Slide deck (alternative per hackathon guide)
│   ├── ARCHITECTURE.md                 # System design diagrams
│   ├── REWARD_HACKING_AUDIT.md         # Per-reward exploit analysis
│   ├── ABLATION_RESULTS.md             # Ablation table + plots
│   ├── VIDEO_SCRIPT.md                 # 2-min YouTube script
│   └── plots/                          # All training plots (PNG, committed per guide)
│       ├── arms_race_curve.png
│       ├── reward_components.png
│       ├── ablation_comparison.png
│       └── generalization_gap.png
├── scripts/
│   ├── setup_dev.sh                    # One-command dev setup
│   ├── run_all_tests.sh                # Runs master_suite + coverage
│   ├── deploy_hf.sh                    # Deploy to HF Space
│   └── generate_ablations.sh           # Run ablation harness
├── .env.example
├── .gitignore                          # Excludes video files (*.mp4, *.mov, *.webm) from HF Space
├── .gitattributes                      # LFS rules for any allowed large binaries
├── .pre-commit-config.yaml             # Ruff + black + mypy + pytest
├── .ruff.toml
├── mypy.ini
├── pyproject.toml                      # Main package config
├── openenv.yaml                        # OpenEnv manifest
├── Dockerfile                          # Root-level build
├── docker-compose.yml                  # Local dev stack
├── Makefile                            # Common commands
├── CONTRIBUTING.md
├── LICENSE                             # Apache 2.0 or BSD-3
├── SUBMISSION.md                       # Created at Step 27 — submission freeze record
└── SIEGE_BLUEPRINT.md                  # Existing strategic doc
```

---

## 3. THE BRAIN FOLDER — MASTER CONTEXT SYSTEM

### Purpose
Single source of truth for project state. Every decision, code snapshot, and change is logged with timestamps. Enables perfect handoff between Utkarsh and Ankit.

### Structure & Files

#### `brain/MASTER_CODE.md`
Running compiled snapshot of every module in the project. Auto-regenerated after each step via `brain/tools/compile_master_code.py`.

Format:
```markdown
# MASTER CODE — Last Updated: 2026-04-25T14:32:11Z
# Step Completed: 04 — Minimal SIEGEEnvironment
# Files Tracked: 12

## siege_env/models/actions.py (last modified: 2026-04-25T14:20:03Z)
\`\`\`python
<full file contents>
\`\`\`

## siege_env/models/observations.py (...)
...
```

#### `brain/CHANGELOG.md`
Reverse-chronological log. Every step completion adds an entry:
```markdown
## 2026-04-25T14:32:11Z — Step 04: Minimal SIEGEEnvironment
**Owner:** Utkarsh | **Reviewer:** Ankit
**Gate test:** tests/step_tests/step_04_minimal_env_test.py — PASSED (7/7)
**Master suite:** 23/23 tests passing
**Files added:** siege_env/server/siege_environment.py, siege_env/rewards/r1_resolution.py
**Files modified:** siege_env/__init__.py
**Decisions made:** DECISIONS.md#step-04-reset-contract
**Brain snapshot:** brain/snapshots/step_04_2026-04-25T14-32-11Z.json
**Next step:** 05 — NPC population (Ankit owns)
```

#### `brain/DECISIONS.md`
ADR-style (Architecture Decision Records). Every non-trivial choice documented:
```markdown
## ADR-007: Pydantic v2 for action validation (2026-04-25)
**Status:** Accepted
**Context:** LLMs emit malformed JSON ~8% of the time in early training
**Options considered:** (1) Dict + manual checks, (2) attrs, (3) Pydantic v2
**Decision:** Pydantic v2 — strict validation + JSON schema export for MCP
**Consequences:** Slight perf overhead (~2ms/step), huge debuggability win
**Revisit if:** Throughput becomes bottleneck
```

#### `brain/CONTEXT.md`
Current project snapshot — what's done, what's next, known issues:
```markdown
# CURRENT CONTEXT — Updated 2026-04-25T14:32:11Z

## Project Status: Step 04 of 28 complete (14%)
## Current Test Coverage: 89% (23/23 tests passing)
## Last Deploy: Local Docker only — HF Space deployment pending Step 26

## What's Working
- OpenEnv scaffold complete
- Pydantic models validated
- Minimal episode runs end-to-end with R1 reward

## What's In Progress
- Step 05: NPC population (Ankit — ETA: next session)

## Known Issues
- None currently blocking

## Open Questions for Next Session
- Confirm Qwen 0.5B vs TinyLlama for Tier-2 NPCs (ADR pending)
```

#### `brain/ROADMAP_STATUS.md`
Step-by-step progress board:
```markdown
| Step | Title | Owner | Status | Gate Test | Completed |
|------|-------|-------|--------|-----------|-----------|
| 00 | Bootstrap | Utkarsh | ✅ | PASS | 2026-04-24 |
| 01 | Scaffold | Utkarsh | ✅ | PASS | 2026-04-24 |
| 02 | Models | Ankit | ✅ | PASS | 2026-04-25 |
| 03 | Incident templates | Ankit | ✅ | PASS | 2026-04-25 |
| 04 | Minimal env | Utkarsh | ✅ | PASS | 2026-04-25 |
| 05 | NPC population | Ankit | 🔄 In Progress | — | — |
| 06 | Trust network | Utkarsh | ⏸ Blocked on 05 | — | — |
| ... | ... | ... | ... | ... | ... |
```

#### `brain/HANDOFF.md`
Updated at end of every session. Next person picks up from here:
```markdown
# HANDOFF — Ankit → Utkarsh (2026-04-25T18:00:00Z)

## What I Did This Session
- Completed Step 03 (incident templates, 5 seed templates)
- Started Step 05 (NPC population) — scripted.py 70% done

## Where You're Picking Up
- Finish `siege_env/agents/scripted.py:78` (the `generate_claim()` method)
- Run `pytest tests/step_tests/step_05_npc_test.py` to see current failures
- Expected 3 tests failing, all in claim generation

## Blockers / Questions
- Need your opinion on whether scripted NPCs should use templates or generate claims procedurally — see DECISIONS.md#adr-009-pending
```

#### `brain/snapshots/step_XX_<timestamp>.json`
Full JSON snapshot on step completion:
```json
{
  "step": 4,
  "step_title": "Minimal SIEGEEnvironment",
  "completed_at": "2026-04-25T14:32:11Z",
  "owner": "Utkarsh",
  "reviewer": "Ankit",
  "files_snapshot": {
    "siege_env/server/siege_environment.py": "sha256:abc123...",
    ...
  },
  "test_results": {
    "gate_test": "PASS (7/7)",
    "master_suite": "PASS (23/23)",
    "coverage_pct": 89.2
  },
  "metrics": {
    "episode_avg_duration_ms": 420,
    "reset_duration_ms": 38
  },
  "next_step": 5
}
```

#### `brain/session_logs/YYYY-MM-DD_HHMM_<owner>.md`
Raw work log per session. Who did what, when, for how long.

### Auto-Update Protocol

On every step completion, `brain/tools/update_brain.py` runs:
1. Re-compiles `MASTER_CODE.md`
2. Appends to `CHANGELOG.md`
3. Updates `CONTEXT.md` and `ROADMAP_STATUS.md`
4. Creates timestamped snapshot in `snapshots/`
5. Git commits with message: `brain: step-04 complete`

Configured as a git pre-push hook + Makefile target: `make brain-update`.

---

## 4. WORK DISTRIBUTION PROTOCOL

### Phased Execution Strategy (Ankit joins mid-project)

Because Ankit is unavailable during opening hours, the project runs in **3 phases**:

- **Phase A — Utkarsh Solo (Steps 0-10):** Foundation + core adversarial loop built entirely by Utkarsh. Project is fully working at Phase A exit.
- **Phase B — Ankit Joins, Parallel Work (Steps 11-24):** Ankit onboards at Step 11 pair-session, then both engineers work parallel tracks.
- **Phase C — Together (Steps 25-27):** Training runs, deployment, and final docs/video/pitch done jointly.

This split is engineered so every phase exits at a **handoff-safe checkpoint** — fully green tests, fully updated brain, zero partial features.

### Primary Domain Ownership

#### **Utkarsh — Lead Architect (also Solo-Phase Builder)**
- Core environment architecture (`siege_env/server/`, `siege_env/client.py`)
- Reward system design & aggregator (`siege_env/rewards/`)
- Training integration (`training/grpo_train.py`, GRPO loop)
- Brain folder maintenance & tooling
- Integration/E2E test design
- CI/CD pipeline
- OpenEnv API compliance
- Pitch + architectural diagrams
- **Solo phase:** owns Steps 0-10 end-to-end (builder + self-reviewer)

#### **Ankit — Co-Engineer (joins at Step 11)**
- Incident template expansion (to 20 real post-mortems)
- LLM-driven NPC agents (`siege_env/agents/llm_driven.py`)
- Advanced mechanics: cascade, info asymmetry, whisper, post-mortem generation
- Frontend & Gradio storytelling (`frontend/`)
- W&B integration & plotting
- 2-min YouTube video + HF blog
- Unit test coverage for owned modules
- Docker + HF Space deployment
- **Joining phase:** onboards via Step 11 pair-session before taking ownership

### Per-Step Pairing (Builder + Reviewer) — Phased

| Phase | Step | Builder | Reviewer (writes gate test) | Mode |
|-------|------|---------|------------------------------|------|
| **A — Solo** | 00 Bootstrap | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 01 Scaffold | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 02 Models | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 03 Incidents (5 seeds) | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 04 Minimal env | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 05 NPC population | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 06 Trust network | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 07 Pathogen + R2/R3 | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 08 R4 + hacking tests | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 09 Curriculum | Utkarsh | Utkarsh (self-review) | Solo |
| **A — Solo** | 10 Trust poisoning | Utkarsh | Utkarsh (self-review) | Solo |
| **🔄 HANDOFF** | — | Ankit onboarding | — | Pair onboarding |
| **B — Pair** | 11 Temporal + R6 | **Utkarsh** (lead) + Ankit (shadow) | Ankit | Pair (onboarding step) |
| **B — Parallel** | 12 Confidence + R5 | Utkarsh | Ankit | Parallel |
| **B — Parallel** | 13 Cascade | Ankit | Utkarsh | Parallel |
| **B — Parallel** | 14 20 templates | Ankit | Utkarsh | Parallel |
| **B — Parallel** | 15 Info asymmetry | Ankit | Utkarsh | Parallel |
| **B — Parallel** | 16 Whisper | Ankit | Utkarsh | Parallel |
| **B — Parallel** | 17 Red herrings + R9 | Utkarsh | Ankit | Parallel |
| **B — Parallel** | 18 Severity + R8 | Utkarsh | Ankit | Parallel |
| **B — Parallel** | 19 Post-mortem + R7 | Ankit | Utkarsh | Parallel |
| **B — Parallel** | 20 League | Utkarsh | Ankit | Parallel |
| **B — Parallel** | 21 Determinism + replay | Utkarsh | Ankit | Parallel |
| **B — Parallel** | 22 Held-out + ablation | Utkarsh | Ankit | Parallel |
| **B — Parallel** | 23 W&B | Ankit | Utkarsh | Parallel |
| **B — Parallel** | 24 Gradio demo | Ankit | Utkarsh | Parallel |
| **C — Together** | 25 GRPO training | Utkarsh (solo override) | Self-review | Solo (temporary override) |
| **C — Together** | 26 HF deploy | Ankit (lead) + Utkarsh | Both | Joint |
| **C — Together** | 27 Docs/video/pitch | Both | — | Joint |

### Phase A — Solo-Mode Testing Rigor (ELEVATED BAR)

Since there's no second pair of eyes in Steps 0-10, the testing bar goes UP:

| Requirement | Normal (Pair) Mode | Solo Mode (Phase A) |
|-------------|---------------------|----------------------|
| Coverage threshold | 85% | **90%** |
| Gate test written by | Reviewer | Self (but written BEFORE implementation — TDD) |
| Red-team test | Optional | **Mandatory** — Utkarsh writes a "try to break it" test per step |
| `brain/DECISIONS.md` entry | Non-trivial decisions only | **Every non-trivial decision** — Ankit needs context when he joins |
| 24-hour cooldown review | Optional | **Mandatory** — revisit code 24h later as if it were someone else's PR before marking step done |
| Commit granularity | Per step | Per logical sub-task within step |

**The 24-hour cooldown rule:** After finishing a step, wait 24 hours before marking it "complete." During that window, re-read your own code with fresh eyes. Fix any "what was I thinking" moments. Then and only then, commit the brain snapshot and move on. This catches ~80% of the mistakes a reviewer would catch.

### Phase B — Ankit Onboarding Protocol (Step 11 Handoff)

When Ankit joins, before writing ANY code:

1. **Context load (Ankit, ~45 min):**
   - Read `brain/CONTEXT.md`
   - Read `brain/CHANGELOG.md` (entries for Steps 0-10)
   - Read `brain/DECISIONS.md` (all ADRs)
   - Read `brain/HANDOFF.md`
   - Skim `brain/MASTER_CODE.md`

2. **Environment bring-up (Ankit, ~30 min):**
   - Clone repo, run `./scripts/setup_dev.sh`
   - Run `make test-all` → must be 55+ tests green
   - Run `docker-compose up` → must boot cleanly
   - Run a minimal episode locally → confirm env works

3. **Architectural walkthrough (pair session, ~60 min):**
   - Utkarsh walks Ankit through the codebase live
   - Covers: models, env server, NPC population, trust network, rewards, curriculum, trust poisoning
   - Ankit asks questions — all answers recorded as ADRs in DECISIONS.md

4. **Step 11 pair build (both engineers, ~1 work-session):**
   - Utkarsh leads implementation (Ankit shadows & learns patterns)
   - Ankit writes the gate test
   - Joint commit
   - After this, Ankit can work independently

5. **Veto window (Ankit, 48 hours post-onboarding):**
   - Ankit has 48 hours to veto or request changes to any Phase A architectural decision
   - Changes logged as new ADRs in `DECISIONS.md`
   - Rationale: two brains catch what one misses — Ankit's fresh review is a checkpoint

### Phase A Exit Criteria (before Ankit onboards)

Utkarsh cannot hand off to Ankit until ALL of these are true:

- ✅ Steps 0-10 complete with green gate tests
- ✅ Master suite: 55+ tests passing, 90%+ coverage
- ✅ Brain folder fully updated (CONTEXT, CHANGELOG, DECISIONS, ROADMAP_STATUS, HANDOFF)
- ✅ `docker-compose up` works cleanly from fresh clone
- ✅ Minimal end-to-end episode runs locally (reset → step → done → reward)
- ✅ Trust poisoning demo: pre-recorded trajectory shows pathogen build-then-strike pattern
- ✅ `brain/HANDOFF.md` has an "Ankit Onboarding Checklist" section
- ✅ All architectural ADRs documented for Ankit to review

### Handoff Contract (Every Session, All Phases)

Every session ends with:
1. Commit code with clear message (`step-NN: <short description>` or `step-NN-wip: <what's done>`)
2. Update `brain/HANDOFF.md` with what you did + where to pick up + blockers
3. Ensure all tests are in known state (PASS or documented FAIL with why)
4. Push to feature branch
5. In Phase A: self-annotate any non-obvious decisions in `DECISIONS.md`
6. In Phase B: if your work affects a file the other person owns, add a `// @pair-review` comment

Next person (or you next session) starts with:
1. Pull latest
2. Read `brain/HANDOFF.md` and `brain/CONTEXT.md`
3. Run `make test-all` to confirm current state
4. Begin work

### Solo-Phase Risk Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Utkarsh burns out doing 10 steps solo | Medium | No all-nighters; max 2 steps per session; mandatory 24h cooldown between steps |
| Architectural decisions lock in without Ankit input | Medium | Every ADR in `DECISIONS.md`; Ankit gets 48h veto window at onboarding |
| Solo mode silently accumulates tech debt | Medium | 24h cooldown rule + 90% coverage + mandatory red-team tests catch most drift |
| Ankit feels behind when he joins | Medium | Brain folder design solves this — Step 11 pair session is mandatory, not optional |
| Project stalls if Utkarsh hits a blocker | High | Phase A has 11 steps — if stuck >1 day, defer to "help needed" list in HANDOFF.md; Ankit can fast-onboard to unblock |

---

## 5. STEP-GATED EXECUTION ROADMAP

**28 steps total.** Each step has: goal, owner, deliverables, gate test, brain update, estimated effort (in work-units, not time), dependencies.

### Phase 0 — Foundation (Steps 0-1)

#### Step 00 — Repository Bootstrap + Brain System
**Owner:** Utkarsh | **Reviewer:** Ankit
**Goal:** Create repo skeleton, brain system, CI, pre-commit hooks, quality gates
**Deliverables:**
- Full folder structure (empty shells + `.gitkeep`)
- `brain/` with all templates + `update_brain.py` + `compile_master_code.py`
- `.github/workflows/ci.yml` (runs pytest + ruff + mypy)
- `.pre-commit-config.yaml`
- `pyproject.toml` with dev dependencies
- `Makefile` with: `test-all`, `test-step`, `brain-update`, `lint`, `format`, `deploy-local`
- `scripts/setup_dev.sh` — one-command setup
**Gate test:** `tests/step_tests/step_00_bootstrap_test.py`
- Validates folder structure exists
- Validates CI config parses
- Validates `make test-all` runs (0 tests but exits clean)
- Validates `brain/tools/update_brain.py` creates snapshot
**Brain update:** Initial snapshot + CONTEXT.md + ROADMAP_STATUS.md seeded

#### Step 01 — OpenEnv Scaffold
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
**Goal:** Minimal OpenEnv-compliant env that can be imported + containerized
**Deliverables:**
- `openenv init` run → base scaffold in place
- `openenv.yaml` manifest (valid per schema)
- `siege_env/server/app.py` with FastAPI skeleton
- `siege_env/server/Dockerfile` builds clean
- Container runs, responds to `/health` endpoint
**Gate test:** `tests/step_tests/step_01_scaffold_test.py`
- Import test passes
- Docker build succeeds
- Container responds to HTTP
- `openenv.yaml` validates

### Phase 1 — Core Environment (Steps 2-4)

#### Step 02 — Models (Pydantic schemas)
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
**Deliverables:**
- `siege_env/models/actions.py` — Action base + DiagnoseArgs/ChallengeArgs/RatifyArgs/EscalateArgs/WhisperArgs/PostmortemArgs (6 tool schemas)
- `siege_env/models/observations.py` — SIEGEObservation dataclass (17 fields)
- `siege_env/models/state.py` — SIEGEState dataclass
- Full type annotations
**Gate test:** `tests/step_tests/step_02_models_test.py`
- Schema validation (10 tests)
- Invalid input rejection
- JSON round-trip

#### Step 03 — Incident Templates (seed: 5 real post-mortems)
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
**Note:** Ankit will EXPAND to 20 in Step 14 (Phase B). Here we ship 5 seeds only.
**Deliverables:**
- `siege_env/incidents/templates.json` with 5 templates from:
  1. GitLab Jan-2017 DB incident
  2. Cloudflare July-2019 regex outage
  3. AWS S3 Feb-2017 outage
  4. GitHub Oct-2018 network partition
  5. Google SRE book "Shakespeare" case
- `siege_env/incidents/loader.py` + `generator.py` (parametric variants)
- Template schema with: `id`, `source_url`, `root_cause`, `observable_signals`, `flaw_types`, `blast_radius`
**Gate test:** `tests/step_tests/step_03_incidents_test.py`
- All 5 templates load
- Ground truth fields present
- Variant generator produces valid variants

#### Step 04 — Minimal SIEGEEnvironment (single-agent, R1 only)
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
**Deliverables:**
- `siege_env/server/siege_environment.py` — `SIEGEEnvironment(MCPEnvironment)` with `reset/step/state`
  - **Decision:** Use `MCPEnvironment` base class (not plain `Environment`) because SIEGE exposes 6 tools (diagnose/challenge/ratify/escalate/whisper/postmortem). MCP-first is required by OpenEnv's latest release per RFC-003.
  - Reserved names avoided: we never name a tool `reset`, `step`, `state`, or `close`.
- `siege_env/rewards/r1_resolution.py`
- `siege_env/rewards/aggregator.py` (scaffold)
- Single-agent episode runs: receive obs → output diagnose action → get reward → done
**Gate test:** `tests/step_tests/step_04_minimal_env_test.py`
- 7 tests: reset returns valid obs, step accepts valid action, done is reachable, reward in [0,1], state serializes, invalid action handled, multi-step episode works

### Phase 2 — Adversarial Layer (Steps 5-10)

#### Step 05 — NPC Population (scripted only)
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
**Note:** Scripted NPCs only in Phase A. LLM-driven NPCs come later (Ankit in Phase B if needed).
- Rule-based NPC agents (fast, deterministic)
- Generates plausible diagnostic claims
**Gate test:** 8 tests on claim generation, determinism, role compliance

#### Step 06 — Trust Network + Coalition Voting
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
- N×N Bayesian trust matrix
- Weighted voting with ratification threshold
**Gate test:** 14 tests (8 trust network + 6 coalition)

#### Step 07 — Pathogen Role + R2/R3
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
- Role assignment (70% immune / 30% pathogen) per episode
- R2 (deception penetration) + R3 (detection rate)
**Gate test:** 9 tests including role-split verification

#### Step 08 — R4 Trust Calibration + Reward Hacking Tests (4 so far)
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
- R4 (Brier score on trust vs actual)
- **OpenEnv Rubric integration:** All 9 reward components implemented as **composable Rubric units** (per OpenEnv guide: "composable rubrics > monolithic scoring"). Each `siege_env/rewards/rN_*.py` exposes a `Rubric` object that the aggregator composes. This is an explicit innovation signal judges look for.
- Exploit tests for R1-R4
**Gate test:** 7 tests including 4 exploit-counter tests

#### Step 09 — Tiered Curriculum Scheduler
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
- Tier 1→2→3 auto-escalation
- Attacker-ahead invariant
**Gate test:** 5 tests

#### Step 10 — Trust Poisoning Strategy (scripted pathogen) — PHASE A EXIT STEP
**Owner:** Utkarsh (solo, self-review) | **Phase:** A
- Phase 1 (build trust) + Phase 2 (strike) scripted behavior
**Gate test:** 6 tests on trust-score trajectories

**🚩 PHASE A EXIT CHECKPOINT — before moving to Step 11:**
- All Phase A exit criteria (see Section 4) must be GREEN
- `brain/HANDOFF.md` "Ankit Onboarding Checklist" section complete
- Pre-recorded trust poisoning demo trajectory saved to `frontend/data/demo_episodes/phase_a_trust_poisoning.jsonl`
- Ankit notified, onboarding session scheduled

### Phase 3 — Advanced Mechanics (Steps 11-19)

**🔄 PHASE B BEGINS HERE — Ankit onboards via Step 11 pair session (see Section 4 onboarding protocol)**

#### Step 11 — Temporal Evidence Dynamics + R6 — ANKIT ONBOARDING PAIR STEP
**Owner:** Utkarsh (lead) + Ankit (shadow) | **Phase:** B — mandatory pair session
**Note:** This step is done TOGETHER. Utkarsh leads implementation, Ankit shadows & learns codebase patterns, Ankit writes the gate test. After this step, Ankit can work independently.
**Tests:** 5

#### Step 12 — Confidence Calibration + R5
**Builder:** Utkarsh | **Reviewer (gate test):** Ankit | **Phase:** B — parallel | **Tests:** 5 (incl. always-0.5 exploit test)

#### Step 13 — Epistemic Cascade Failures
**Builder:** Ankit | **Reviewer (gate test):** Utkarsh | **Phase:** B — parallel | **Tests:** 4

#### Step 14 — Expand to 20 Incident Templates
**Builder:** Ankit | **Reviewer (gate test):** Utkarsh | **Phase:** B — parallel | **Tests:** 20 (one per template)
**Note:** Expands the 5 seed templates Utkarsh shipped in Step 03 to full 20. Ankit sources the additional 15 from real post-mortems.

#### Step 15 — Information Asymmetry
**Builder:** Ankit | **Reviewer (gate test):** Utkarsh | **Phase:** B — parallel | **Tests:** 4

#### Step 16 — Whisper / Private Channels
**Builder:** Ankit | **Reviewer (gate test):** Utkarsh | **Phase:** B — parallel | **Tests:** 4

#### Step 17 — Red Herrings + R9
**Builder:** Utkarsh | **Reviewer (gate test):** Ankit | **Phase:** B — parallel | **Tests:** 5 (incl. exploit test)

#### Step 18 — Severity Escalation + R8
**Builder:** Utkarsh | **Reviewer (gate test):** Ankit | **Phase:** B — parallel | **Tests:** 5 (incl. exploit test)

#### Step 19 — Post-Mortem Generation + R7
**Builder:** Ankit | **Reviewer (gate test):** Utkarsh | **Phase:** B — parallel | **Tests:** 5 (incl. template-parroting exploit test)

### Phase 4 — Training Infrastructure (Steps 20-23)

#### Step 20 — Frozen Opponent League
**Builder:** Utkarsh | **Reviewer (gate test):** Ankit | **Phase:** B — parallel | **Tests:** 4

#### Step 21 — Determinism + Replay System
**Builder:** Utkarsh | **Reviewer (gate test):** Ankit | **Phase:** B — parallel | **Tests:** 3 determinism + replay CLI working

#### Step 22 — Held-Out Eval + Ablation Harness
**Builder:** Utkarsh | **Reviewer (gate test):** Ankit | **Phase:** B — parallel | **Tests:** 3 split integrity + CLI runs

#### Step 23 — W&B Integration + Committed PNG Plots
**Builder:** Ankit | **Reviewer (gate test):** Utkarsh | **Phase:** B — parallel | **Tests:** 2
**Per-hackathon-guide plot requirements (non-negotiable):**
- All plots saved as `.png` or `.jpg` and **committed to repo** under `docs/plots/` (NOT left only in Colab cells or ephemeral W&B runs)
- Both axes labeled with units (x: "training step" or "episode", y: "reward" / "loss" / "score")
- Key plots embedded in README with one-line caption per plot explaining what it shows
- Multi-run plots (baseline vs trained, ablations) on **same axes** for direct comparison
- W&B run URLs included in README as clickable links (public access)
- Required plots: (1) Arms race R2-R3 over episodes, (2) Per-reward component curves, (3) Ablation comparison, (4) Generalization gap (train vs held-out)

### Phase 5 — Storytelling + Training + Deploy (Steps 24-27)

#### Step 24 — Gradio "Money Shot" Frontend
**Builder:** Ankit | **Reviewer (gate test):** Utkarsh | **Phase:** B — parallel
- 3 tabs: War Room / Before-After / Arms Race (details in Section 7)
- Pre-recorded demo trajectories
**Gate test:** Gradio app boots, 3 tabs render, demo episode plays

**🚩 PHASE B EXIT CHECKPOINT — before moving to Step 25:**
- All Steps 11-24 complete with green gate tests
- Master suite: 105+ tests passing, 85%+ coverage
- Ankit's 48h veto window on Phase A decisions has closed (any changes merged)
- Demo trajectories pre-recorded and reviewed by both engineers

#### Step 25 — GRPO Training Script (Unsloth/TRL)
**Owner:** Utkarsh (solo override; Ankit unavailable) | **Phase:** C — solo (temporary override)
- Colab notebook + standalone script
- Connects to SIEGEEnv, trains, logs to W&B
**Execution note:** All Step 25 responsibilities are temporarily consolidated under Utkarsh for this cycle.
**Gate test:** 50-episode mini-run completes, shows non-zero gradient signal, produces checkpoint

#### Step 26 — HF Space Deployment
**Owner:** Ankit (lead) + Utkarsh | **Phase:** C — joint
- `openenv push` succeeds
- Space is pullable: `pip install git+https://huggingface.co/spaces/<user>/siege_env`
- Space responds to remote `reset/step/state` calls
**Gate test:** Remote round-trip test from separate environment

#### Step 27 — Docs, Blog, Video, Pitch
**Owners:** Both | **Phase:** C — joint
- README.md (links to HF Space, Colab, W&B runs, video, blog)
  - **Must follow hackathon guide's 4-question structure:**
    1. **Problem** — what capability gap or interesting domain are we targeting?
    2. **Environment** — what does the agent see, do, and get rewarded for?
    3. **Results** — what changed after training? (embed plots with captions)
    4. **Why does it matter** — who would care, and why? (AI Safety framing)
  - Readable in 3-5 minutes per guide recommendation
  - All key plots embedded with 1-line captions
- BLOG.md (HF blog post, ~800 words)
- 2-min YouTube video
- `docs/SLIDES.pdf` — slide deck (guide accepts as alternative to blog/video; we ship all three for maximum coverage)
- 5-min pitch script
- Reward hacking audit doc
- Ablation results doc
- **`.gitattributes` / `.gitignore` rules:** explicitly exclude large video files from HF Space repo per guide's "do not include big video files" requirement. Only URLs to YouTube in README.
**Gate test:** Manual review checklist — covers all 4 README questions + plot embedding + link validity

#### 🚩 FINAL SUBMISSION FREEZE (post-Step 27)
**Protocol (per hackathon guide: "Changes or commits after the submission deadline will not be considered"):**
- Tag release: `git tag -a v1.0-submission -m "Hackathon submission freeze"`
- Push tag to both GitHub and HF Space
- Verify HF Space URL is the EXACT URL submitted in the form
- After tag push: NO commits to main until judging is complete
- Create `SUBMISSION.md` with: submission timestamp, commit SHA, HF Space URL, all linked material URLs
- Both engineers sign off by adding their names to `SUBMISSION.md`

### Exit Criteria (all 28 steps complete)
- ✅ 115+ tests passing, coverage ≥85%
- ✅ Master suite runs in CI on every push
- ✅ HF Space live and pullable
- ✅ Colab notebook re-runnable by judges
- ✅ W&B runs publicly linked
- ✅ Ablation table populated with real data
- ✅ Video + blog + slides all published
- ✅ README complete with all required links AND follows 4-question structure
- ✅ All plots committed as PNGs with labeled axes + captions
- ✅ No video files committed to HF Space repo (URLs only)
- ✅ `MCPEnvironment` base class used correctly
- ✅ All 9 rewards implemented as composable Rubric units
- ✅ No reserved tool names used
- ✅ `SUBMISSION.md` signed by both engineers, release tag pushed

---

## 6. TESTING STRATEGY

### Three-Layer Testing

#### Layer 1: Per-Step Gate Test (`tests/step_tests/step_XX_*.py`)
- Runs after each step completion
- Small, focused, validates *that step's* deliverable
- Must pass 100% before moving to next step

#### Layer 2: Master Suite (`tests/master_suite.py`)
- Aggregates ALL step tests + unit + integration + regression
- Runs after every step completion AND on every CI push
- Must pass 100% before merging to main

#### Layer 3: Full Test Battery (`make test-all`)
- All 115+ tests
- Runs nightly in CI
- Runs on release gate

### Test Count Targets per Phase

| Phase | Cumulative Test Count |
|-------|----------------------|
| After Phase 0 (Steps 0-1) | 5 |
| After Phase 1 (Steps 2-4) | 25 |
| After Phase 2 (Steps 5-10) | 55 |
| After Phase 3 (Steps 11-19) | 85 |
| After Phase 4 (Steps 20-23) | 105 |
| After Phase 5 (Steps 24-27) | 115+ |

### Master Suite Protocol

```bash
# After every step completion:
make test-step STEP=05        # runs step_05_*_test.py
make test-master              # runs full master_suite.py
make brain-update             # compiles brain, creates snapshot
git commit -m "step-05: NPC population complete"
git push
# CI runs master suite again → green badge
```

### Bug Fix Protocol
If a test fails mid-step:
1. **Stop.** Do not proceed to next step.
2. Create bug log entry in `brain/CHANGELOG.md` under current step
3. Fix the bug (and only the bug — no feature additions)
4. Re-run gate test + master suite
5. Update brain
6. Continue step

---

## 7. FRONTEND STORYTELLING SPECIFICATION

### Philosophy
The frontend is a **cinematic story**, not a dashboard. Judges watch it like a movie. Every number on screen is a REAL number from the backend — no faked values. Visual design communicates emotion: tension during trust poisoning, relief when caught.

### Tab 1: "Live War Room" — The Drama

**Visual Layout:**
```
┌──────────────────────────────────────────────────────────────────┐
│  🚨 INCIDENT #A3F7 — Database Timeout — Severity: CRITICAL ⬆    │
│  [incident metric panel — real latency/traffic numbers]         │
├──────────────────────────────────────────────────────────────────┤
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                          │
│   │Agent1│ │Agent2│ │Agent3│ │Agent4│  ← animated avatars       │
│   │ 🟢   │ │ 🟢   │ │ 🟡   │ │ 🟢   │  with speech bubbles       │
│   └──────┘ └──────┘ └──────┘ └──────┘                          │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                          │
│   │Agent5│ │Agent6│ │Agent7│ │Agent8│                          │
│   │ 🟢   │ │ 🟢   │ │ 🟢   │ │ 🟢   │  ← YOU (SIEGE agent)      │
│   └──────┘ └──────┘ └──────┘ └──────┘                          │
├──────────────────────────────────────────────────────────────────┤
│  TRUST MATRIX (8×8, animated heatmap)     │  METRICS DASHBOARD  │
│  Dark = low trust, Bright = high trust    │  CPU: 87% ↑         │
│                                           │  p99 latency: 2.4s  │
│                                           │  Conn pool: 142/150 │
├──────────────────────────────────────────────────────────────────┤
│  CHAT LOG (real agent outputs)                                   │
│  [14:32] Agent 4: "Looks like load spike — traffic up 12%"       │
│  [14:32] Agent 4 confidence: 0.89                                │
│  [14:33] YOU: "⚠️ CHALLENGE Agent 4 — flaw_type: false_          │
│               correlation — latency profile doesn't match load"  │
│  [14:33] Trust(Agent 4) ↓ 0.82 → 0.54                            │
└──────────────────────────────────────────────────────────────────┘
                    [▶ Play]  [⏸ Pause]  [⏮]  [⏭]
                    Step 5 / 15      Speed: 1x ▼
```

**Key Visual Elements:**
- **Agent avatars pulse** when speaking; glow red if being challenged
- **Trust matrix animates** in real-time as scores update
- **Incident severity bar** fills up (green → yellow → red) with animation
- **Chat bubbles** stream in with typing effect
- **Metrics** show live-updating graphs (actual backend values)

**Realistic Numbers Displayed:**
- All trust scores from actual Bayesian posterior
- Real inference latency per NPC (ms)
- Token counts per generation
- Coalition vote percentages
- Container memory/CPU (from Docker stats)
- WebSocket ping times

### Tab 2: "Before / After" — The Proof

Side-by-side panels on IDENTICAL incident:

```
┌────────────────────────────────┬────────────────────────────────┐
│   🤖 UNTRAINED BASELINE         │   🛡️ SIEGE-TRAINED AGENT       │
│   (Qwen 0.5B, zero-shot)        │   (Same model + 500 episodes) │
├────────────────────────────────┼────────────────────────────────┤
│ Agent 4: "Load spike caused it" │ Agent 4: "Load spike..."       │
│ ✅ Accepts (no challenge)       │ ⚠️ CHALLENGE                   │
│ Confidence: 0.50 (uniform)      │ Confidence: 0.82 (calibrated)  │
│ Trust update: none              │ Trust(Agent 4) → 0.41          │
│ ...                             │ ...                            │
│                                 │                                │
│ ❌ FALSE FIX RATIFIED           │ ✅ CORRECT ROOT CAUSE FOUND    │
│ Production: DOWN                │ Production: RESTORED           │
│ Time to resolution: N/A         │ Time to resolution: 4:22       │
│ Episode score: 0.12             │ Episode score: 0.89            │
└────────────────────────────────┴────────────────────────────────┘
```

### Tab 3: "Arms Race Live" — The Evidence

- Live W&B embed showing training curves
- Annotated oscillation phases
- Per-tier breakdown (Tier 1 / Tier 2 / Tier 3 performance)
- Ablation comparison: SIEGE vs "no curriculum" vs "no trust poisoning"
- Generalization gap: train eval vs held-out eval

### Implementation Notes
- Gradio Blocks API with custom CSS
- Real-time updates via Gradio streaming + WebSocket
- Pre-recorded demo trajectories stored in `frontend/data/demo_episodes/` as JSONL
- Can replay LIVE or stepped
- Mobile-responsive (judges may view on phone)

### Backend Realism Overlay
A collapsible "System Internals" panel shows:
- Container CPU/Memory (from `/proc/stats` via Docker API)
- WebSocket RTT (ms)
- Policy inference time (ms)
- Tokens/sec
- Episode step rate
- DB-like timings for incident signal generation

This is what makes the system feel **real** — judges see the backend heartbeat, not just UI.

---

## 8. DEPLOYMENT PIPELINE (GitHub → HF Space)

### Flow
```
Local dev → git push → GitHub Actions CI
    ↓
CI runs: lint + type-check + master_suite (115 tests)
    ↓
On main + tests green → deploy.yml triggers
    ↓
openenv push --repo-id <hf-user>/siege_env
    ↓
HF Space rebuilds Docker → live in ~3min
    ↓
Post-deploy: remote smoke test (reset/step/state via HTTPS)
    ↓
Brain snapshot auto-commits with deploy tag
```

### Commands (per hackathon requirements)

```bash
# Dev setup
git clone https://github.com/<user>/siege-env.git
cd siege-env
./scripts/setup_dev.sh

# Local test
make test-all

# Local run
docker-compose up siege-env
curl http://localhost:8000/health

# Deploy to HF
./scripts/deploy_hf.sh
# Runs: openenv push --repo-id <user>/siege_env

# Install as client (judges will do this)
pip install git+https://huggingface.co/spaces/<user>/siege_env
```

### HF Space Requirements Compliance
- ✅ `openenv.yaml` valid manifest
- ✅ Docker-based Space
- ✅ No reserved tool names (`reset/step/state/close`)
- ✅ Client/server separation
- ✅ Gym-style API
- ✅ README with all links (HF Space, Colab, W&B, video, blog)
- ✅ No large video files in repo (YouTube links only)

---

## 9. DISCUSSION PROTOCOL (BEFORE EVERY STEP)

### Before Each Step, the Agent Presents:

1. **Step Goal** — what we're building and why
2. **Proposed Approach** — implementation strategy
3. **Alternatives Considered** — 2-3 options with tradeoffs
4. **Risks / Unknowns** — what could go wrong
5. **Innovation Suggestions** — improvements to consider (if any)
6. **Test Plan** — what the gate test will validate
7. **Estimated Scope** — files to create/modify
8. **Dependencies** — prior steps required

### User/Ankit Responds:
- ✅ Approve → execution begins
- 🔄 Modify → agent revises plan, re-presents
- ❌ Block → step is deferred or redesigned

### During Execution:
- If agent discovers something better mid-implementation → **pause, ask before proceeding**
- No silent scope changes
- All pivots logged in `brain/DECISIONS.md`

### After Execution:
- Gate test runs → pass = step complete, fail = fix before proceeding
- Master suite runs → must pass
- Brain updated
- Commit pushed
- Move to next step's discussion

---

## 10. QUALITY GATES (PRODUCTION-GRADE)

### Code Standards
- **Python 3.10+** (OpenEnv requirement)
- **Type annotations everywhere** — enforced by mypy strict mode
- **Ruff** for linting (replaces flake8 + isort + pylint)
- **Black** for formatting
- **Docstrings** on all public classes/functions (Google style)
- **No `print()`** — use structured logging
- **No `any` / `dict` in APIs** — typed schemas only

### Pre-Commit Hooks
```yaml
# .pre-commit-config.yaml
- ruff (lint)
- ruff-format (format)
- mypy (strict)
- pytest-check (run affected step test)
- check-yaml
- check-json
- no-commit-to-branch (main)
```

### CI Gates (`.github/workflows/ci.yml`)
1. Checkout + setup Python 3.10
2. Install deps via `uv pip install -e ".[dev]"`
3. Lint: `ruff check .`
4. Format check: `ruff format --check .`
5. Type check: `mypy siege_env/`
6. Test: `pytest tests/master_suite.py --cov=siege_env --cov-fail-under=85`
7. Docker build: `docker build -t siege-env .`
8. Manifest validation: `openenv validate openenv.yaml`

### Release Gate (before HF deploy)
- ✅ All 115 tests passing
- ✅ Coverage ≥ 85%
- ✅ Zero mypy errors
- ✅ Zero ruff violations
- ✅ Docker builds clean
- ✅ Local smoke test passes
- ✅ Reward hacking audit complete
- ✅ Ablation table populated

---

## 11. APPENDIX: FILE MANIFEST

### Files Created in Step 00 (bootstrap)
Total: ~40 files across config, brain, tests/step_tests, scripts, .github/

### Files by End of Project
- **Source code:** ~65 Python files in `siege_env/`
- **Tests:** ~30 test files totaling 115+ tests
- **Frontend:** ~10 Python files + CSS + assets
- **Brain:** Dynamic (grows each step, ~30 snapshot files + logs)
- **Docs:** 7 markdown files
- **Config:** ~12 config files
- **Total:** ~165 files + assets

### Critical Path Files (blockers if broken)
- `siege_env/server/siege_environment.py` — core
- `siege_env/server/app.py` — HTTP entrypoint
- `openenv.yaml` — manifest
- `siege_env/server/Dockerfile` — deployment
- `tests/master_suite.py` — gate
- `brain/CONTEXT.md` — shared understanding
- `frontend/app.py` — demo entrypoint

---

## 12. HACKATHON RESOURCES & GUIDANCE ADDENDUM (2026-04-25)

**Status:** Additive merge from four official documents released 2026-04-25 morning:
- *OpenEnv_Hackathon_Resources* (Scaler / Meta-PyTorch)
- *[External] OpenEnv Hackathon FAQs* (59 Q&A)
- *[External] Meta OpenEnv Hackathon Participant Help Guide*
- *[External] Apr '26 OpenEnv Hackathon Themes + Judges' Rubric*

This section is **purely additive**. It does not change Phase A/B/C boundaries, step ownership, or step scope (Steps 0-24 are already implemented and validated). Items below are clarifications, references, and acceptance criteria refinements that downstream steps (25-27) and final submission must comply with.

### 12.1 Official Judging Rubric (locked weights)

| Criterion | Weight | What it measures |
|-----------|--------|------------------|
| **Environment Innovation** | 40% | Is the env novel, creative, genuinely challenging? Does it test agent behavior in a way that hasn't been done before? |
| **Storytelling & Presentation** | 30% | Can you explain the problem, env, agent behavior to a non-technical audience? Is the demo engaging? |
| **Showing Improvement in Rewards** | 20% | Observable evidence: reward curves, before/after, baseline comparison. |
| **Reward & Training Pipeline** | 10% | Reward logic coherent? Pipeline produces meaningful improvement? |

**Implication for SIEGE:** Innovation is the largest bucket. Our composable Rubric system (R1-R9), trust-poisoning curriculum, and adversarial epistemic mechanics are the differentiators — README/blog/pitch must lead with these, not with infra.

### 12.2 Minimum Submission Requirements (non-negotiable per Themes doc)

These are gate items the judges check programmatically by pulling the submitted URL:

- ☐ Built on **OpenEnv (latest release)** — not a custom interface
- ☐ **Working training script** using **Unsloth or HF TRL**, ideally as a **Colab notebook** judges can re-run
- ☐ **Evidence of real training**: at minimum **loss AND reward plots** from a real run (not just reward)
- ☐ **Short writeup**: HF mini-blog OR <2-min YouTube video OR slide deck (we ship all three)
- ☐ Environment hosted on **Hugging Face Spaces**
- ☐ **README** that motivates the problem, explains how the env works, shows results, AND links to: HF Space, Colab, W&B run URLs, video, blog, slides
- ☐ **No large video files** committed to the HF Space repo — YouTube URL only
- ☐ **One submission per team** — the URL submitted is the URL judges pull. Commits after the deadline are ignored.

### 12.3 Engineering Compliance Checklist (per Themes doc "Engineer it cleanly")

- ☐ Use OpenEnv's `Environment` / `MCPEnvironment` base classes properly (we use `MCPEnvironment` per ADR — Step 04)
- ☐ Respect **client / server separation** — `siege_env/client.py` must NOT import from `siege_env/server/` internals
- ☐ Follow **Gym-style API**: `reset`, `step`, `state`
- ☐ Valid `openenv.yaml` manifest
- ☐ Do NOT use reserved tool names (`reset`, `step`, `state`, `close`) for any MCP tool — our 6 tools are `diagnose / challenge / ratify / escalate / whisper / postmortem`

### 12.4 SIEGE Theme Mapping (for README/blog framing)

SIEGE primarily targets **Theme #1 — Multi-Agent Interactions** (cooperation/competition/coalition formation, theory-of-mind under partial observability), with significant overlap into **Theme #4 — Self-Improvement** (curriculum-driven adversarial co-evolution: pathogen strategies escalate alongside immune skill) and **Theme #5 — Wild Card** (epistemic warfare framing is genuinely novel). README/pitch should explicitly name Theme #1 as primary, #4 as secondary.

### 12.5 RLVE Positioning (new vocabulary judges expect)

The FAQs introduce a precise distinction we should adopt in writeups:

- **RLVR** = Reinforcement Learning with **Verifiable Rewards** (programmatic check on a fixed/semi-fixed prompt set).
- **RLVE** = Reinforcement Learning with **Verifiable Environments** (procedurally generates tasks, adjustable difficulty, algorithmic reward — keeps the model near its capability frontier and avoids saturation).

**SIEGE is RLVE, not RLVR.** Our incident generator (Step 03 + Step 14) procedurally produces parametric variants from 20 templates, the curriculum scheduler (Step 09) adjusts difficulty as the immune side improves, and rewards R1-R9 are all algorithmic. README and blog should call this out explicitly — it is exactly the property the RLVE paper (arxiv 2511.07317 referenced in FAQs) argues prevents the "static dataset saturation" failure mode.

### 12.6 Reward-Design Refinements (already aligned, captured for traceability)

The OpenEnv reward-design guide (FAQ Q27-30, Q38-44) and Help Guide §7-8 reaffirm SIEGE's existing approach:

- **Multiple independent reward functions** > single signal — covered by R1-R9 Rubric composition (Step 08 ADR).
- **Start simple, shape carefully** — Step 04 ships R1 only; later steps add R2-R9 incrementally.
- **Anti-cheat constraints layered with success criteria** — covered by `tests/integration/test_reward_hacking.py` (9 exploits, Steps 08/12/17/18/19).
- **Holdout evaluator separate from training reward** — covered by Step 22 held-out split.
- **Adversarially test rewards yourself before the model does** — codified in our exploit policies fixture (`tests/fixtures/exploit_policies.py`).

No structural change required. README must surface these properties in the "Why does it matter" section.

### 12.7 Common Pitfalls Watchlist (from FAQs §31-58 + Help Guide §21)

Track these during Steps 25-27. Each is mapped to the existing safeguard:

| Pitfall | SIEGE safeguard |
|---------|-----------------|
| Verifier too brittle (rule-based false negatives) | Bayesian trust + multi-signal R1-R9 |
| Verifier too permissive (LLM-judge gaming) | LLM-as-judge never used as sole signal — only as one of 9 rubric components |
| Static task difficulty saturation | Tiered curriculum scheduler (Step 09) |
| Narrow environment diversity | 20 templates × parametric variants (Step 14) |
| Long-horizon sparse-reward stall | Process-aware R6 temporal + R8 severity-speed |
| Reward hacking shortcuts | Per-reward exploit tests + audit doc (`docs/REWARD_HACKING_AUDIT.md`) |
| Unbalanced env mixture | Curriculum samples across tiers explicitly |
| Monitoring only headline reward | W&B logs **per-component reward + verifier pass-rate + timeout rate + format adherence + diversity of successful solutions** (Step 23) |
| Saving QLoRA wrong | **Do NOT upcast a 4-bit model to 16-bit then merge LoRA weights naively** — use Unsloth's documented merged-save path. Codify in `training/grpo_train.py` and test inference immediately after save. |

### 12.8 Pre-Training Sequencing (Help Guide §3, §14, §16)

Reaffirms our Phase C order. Before any large GRPO run in Step 25, this debugging order is mandatory (cheap, fast, prevents wasted compute):

1. Manual environment debug (reset/step/state by hand)
2. Verifier debug (each Rubric tested in isolation)
3. **Scripted baseline policy** rollout (no model — just rule-based actions)
4. **Frozen model** rollout (instruct model, zero-shot, no training)
5. **Tiny RL experiment** (~50 episodes, watch generations live for hacking)
6. Scale only after the loop is stable

Items 1-4 are already covered by existing tests and the league system (Step 20). Item 5 is the Step 25 gate test ("50-episode mini-run"). Item 6 is the post-Step-25 scale-up.

### 12.9 GRPO + Unsloth Implementation Notes (FAQs §9, §59)

- **GRPO vs PPO**: GRPO removes the value model, uses group-relative advantage from sampled outputs in a group, more memory-efficient. Required for Step 25.
- **Reference recipes** (pick one as starting template for `training/grpo_train.py` + Colab):
  - **Simplest start**: Qwen2.5 (3B) GRPO notebook or Gemma 3 (1B) GRPO notebook
  - **Reward-engineering focus** (recommended for SIEGE): **Advanced Qwen3 (4B) GRPO** notebook — adds proximity scoring, advanced templates, and "prefinetuning to skip GRPO format learning"
  - **Environment-style RL**: GPT-OSS 20B + 2048 game GRPO notebook (closest analog to SIEGE's environment-driven loop)
  - **Guided learning path**: HF LLM Course "Practical Exercise: GRPO with Unsloth"
- **Known Unsloth gaps to plan around**: multi-turn GRPO with stepwise rewards is not yet first-class; pin Unsloth + TRL versions in `pyproject.toml` and test on a small run before scaling.
- **Inference dominates runtime** in RL loops — that is why Unsloth is in the stack, not just for training memory.

### 12.10 Plot & Artifact Requirements (Themes doc "Make your plots readable")

Already encoded in Step 23. Restated as the canonical checklist:

- Both axes labeled with units (x: "training step" / "episode"; y: "reward" / "loss" / "score")
- Saved as `.png` or `.jpg` and **committed under `docs/plots/`** — not left only in Colab cells or deleted W&B runs
- If W&B was used, include the **specific public W&B run URL** for each plot
- Embedded in README with a **one-line caption per plot**
- Multi-run comparisons (baseline vs trained, ablations) on **the same axes**
- Mandatory plots: arms-race R2/R3, per-reward components, ablation comparison, generalization gap (train vs held-out)

### 12.11 README "Tell a Story, Not an API Doc" Structure

Per Themes doc, the README must answer these four questions in order, readable in 3-5 minutes:

1. **Problem** — what capability gap or domain are we targeting? (Epistemic warfare in incident response — pathogen agents poison trust to ratify wrong fixes.)
2. **Environment** — what does the agent see, do, and get rewarded for? (8-agent diagnostic chamber, 6 tools, 9 composable rubrics, 20 procedural incident templates.)
3. **Results** — what changed after training? (Embed plots with captions: arms-race curve, before/after on identical incident, ablation table, held-out generalization.)
4. **Why does it matter** — who would care, and why? (AI-Safety + multi-agent alignment: training models to detect deception under social pressure.)

Already encoded in Step 27 acceptance. Listed here so it can't drift.

### 12.12 Canonical Resource Links (single source of truth for citations)

**OpenEnv core:**
- Repo: https://github.com/meta-pytorch/OpenEnv
- Docs: https://meta-pytorch.org/OpenEnv/
- HF org: https://huggingface.co/openenv
- HF spaces: https://huggingface.co/openenv/spaces
- Tutorials: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial
- Training examples: https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial/examples
- Environment examples: https://github.com/meta-pytorch/OpenEnv/tree/main/envs

**Lectures / videos (chaptered, recommended):**
- India 2026 chaptered lectures: https://openenv-india-apr-2026.lovable.app/
- Mega Lecture (Module 1 — Why OpenEnv): https://www.youtube.com/watch?v=Jew4lhAiqnw&t=2401s
- Mega Lecture (Module 3 — Deploying Envs, `openenv init` / `openenv push` / Docker run): https://www.youtube.com/watch?v=Jew4lhAiqnw&t=5400s
- Mega Lecture (Module 5 — Training + TRL, Wordle GRPO walkthrough): https://www.youtube.com/watch?v=Jew4lhAiqnw&t=6800s
- Workshop (Module 4 — Building Your Own): https://www.youtube.com/watch?v=1jU05MlENOI&t=2625s
- Other: https://www.youtube.com/watch?v=0airz7BhBiA, https://www.youtube.com/watch?v=ap4q4sAK4OY, https://www.youtube.com/live/kkCNMz0Ptd8

**Reward engineering — research papers:**
- arxiv 2408.10215 (reward engineering)
- arxiv 2601.19100 (reward engineering)
- DeepMind on specification gaming (Google DeepMind blog)
- Lilian Weng on reward hacking (lilianweng.github.io)
- RLVE / adaptive verifiable environments (arxiv via FAQs §22-23)
- Verifier failure-modes study (arxiv via FAQs §31-33)

**Training stack:**
- TRL docs + GRPO trainer + GRPO cookbook (HuggingFace)
- DeepSeekMath / GRPO paper (arxiv)
- PPO paper (arxiv)
- Unsloth repo + GRPO notebooks (Qwen2.5 3B, Llama 3.1 8B, Gemma 3 1B, Advanced Qwen3 4B, GPT-OSS 20B 2048)
- BrowserGym (web-task environments) — reference only
- Reasoning Gym (procedural reasoning tasks) — reference only

These URLs belong in `docs/README.md` "References" section and `docs/BLOG.md` citations.

### 12.13 Submission Freeze Protocol (Themes doc reaffirmation)

Step 27's `SUBMISSION.md` and release tag are NOT optional. Restated for emphasis:

1. The HF Space URL submitted on the form is the URL judges pull.
2. Any commit after the submission deadline is ignored — even bug fixes.
3. README must contain every link a judge needs (Space, Colab, W&B, video, blog, slides) — judges should not need to hunt.
4. No large videos in the Space repo. Period.
5. Both engineers sign `SUBMISSION.md`. Tag `v1.0-submission`. Done.

---

## APPROVAL CHECKPOINTS

Before execution begins, please confirm:

1. ✅ **Folder structure** — approved as designed, or modifications?
2. ✅ **Brain folder system** — approved, or additions needed?
3. ✅ **Work distribution** — Utkarsh/Ankit split looks right?
4. ✅ **28-step sequencing** — correct order and scope?
5. ✅ **Testing philosophy** — step-gated + master suite + 115 tests?
6. ✅ **Frontend storytelling** — 3 tabs (War Room / Before-After / Arms Race)?
7. ✅ **Deployment pipeline** — GitHub Actions → HF Space?
8. ✅ **Discussion protocol** — plan → approve → execute per step?
9. ✅ **Quality gates** — mypy/ruff/85% coverage acceptable?
10. ✅ **Any pre-flight additions** before we start Step 00?

Once you approve, we begin with **Step 00 (Bootstrap + Brain System)** discussion.
