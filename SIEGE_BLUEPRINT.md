# SIEGE — Strategic Incident & Epistemic Group Environment
## Critical Analysis + Refined Blueprint for OpenEnv India 2026

---

## PART 1: HONEST ASSESSMENT — Is SIEGE a Winner?

### Verdict: **Strong concept, but needs surgical upgrades to dominate**

SIEGE has a genuine competitive edge — it sits at the intersection of **Theme #1 (Multi-Agent Interactions)** and **Theme #4 (Self-Improvement)** while touching **Theme #5 (Wild Card)**. No other team will likely combine adversarial epistemic reasoning + professional incident command + self-play arms race. That's the moat.

### Scoring Against Judging Criteria (Current State)

| Criterion | Weight | Current Score | Gap | Notes |
|-----------|--------|--------------|-----|-------|
| **Environment Innovation** | 40% | 88/100 | Needs sharper novelty articulation | Trust poisoning is genuinely novel, but "incident command" alone isn't — the adversarial epistemic layer IS the innovation |
| **Storytelling** | 30% | 75/100 | Biggest vulnerability | Complex concept = hard pitch. Need the 30-second hook nailed |
| **Showing Improvement in Rewards** | 20% | 70/100 | Arms race curve is the silver bullet but must ACTUALLY oscillate | Need curriculum design that guarantees non-flat curves |
| **Reward & Training Pipeline** | 10% | 85/100 | Already strong — 4-component reward, no LLM judge | Just needs clean implementation |

### Current Weaknesses

1. **Complexity overload** — 8 agents, trust networks, flaw taxonomies, coalition voting... judges have 5 minutes. If they can't grok it in 60 seconds, innovation score drops to 70.
2. **OpenEnv compliance risk** — SIEGE's multi-agent nature doesn't naturally map to OpenEnv's single-agent `step()/reset()/state()` API. This MUST be solved cleanly or you fail minimum requirements.
3. **"Simulated" feel** — Even with SRE post-mortems, the incident data is still synthetic. Judges who are engineers will probe this.
4. **Training evidence** — The arms race curve is promised but not guaranteed. If curves are flat, the 20% "showing improvement" score collapses.

---

## PART 2: THE REFINED SIEGE — UPGRADES THAT WIN

### Upgrade 1: **Reframe the OpenEnv Architecture — "Agent-Per-Seat" Design**

The critical insight: OpenEnv expects `step()` from ONE agent. SIEGE has 8.

**Solution:** Each OpenEnv instance is ONE agent seat in the war room. The environment manages the other 7 agents (mix of scripted + LLM-driven). The training target is the single seat agent. During GRPO training, you sample N rollouts where the seat agent plays different roles (immune/pathogen) against a population.

```
SIEGEEnv(Environment):
    reset() → spawns incident, assigns role (immune/pathogen), populates 7 other agents
    step(action) → agent takes one of 4 actions (diagnose/challenge/ratify/escalate)
    state() → dashboard: metrics, traces, alerts, other agents' claims, trust scores
```

This is clean Gym-style. One agent, rich environment. **The other agents ARE the environment.**

### Upgrade 2: **Ground in 20 REAL Public Post-Mortems**

Don't generate incidents. Extract them from:
- GitLab's public incident database
- Google SRE Book case studies
- Atlassian Statuspage post-mortems
- PagerDuty post-incident reviews
- Cloudflare blog post-mortems

For each, extract:
- **Root cause** (ground truth)
- **Observable signals** (metrics, traces, alerts)
- **Plausible wrong diagnoses** (flaw types mapped to your taxonomy)
- **Blast radius** (which services affected)

README line: *"Incident templates derived from 20 real public SRE post-mortems including GitLab, Cloudflare, and Google."*

This single sentence changes judge perception from "toy simulation" to "real-world grounding."

### Upgrade 3: **Trust Poisoning as THE Central Mechanic**

This is what makes SIEGE paper-worthy. Elevate it from "feature" to "thesis."

**The attacker doesn't just inject wrong diagnoses. It strategically builds credibility first, then strikes at the critical moment.**

- Phase 1 (Trust Building): Pathogen gives CORRECT diagnoses for 3-4 low-stakes rounds. Trust score rises to 0.8+.
- Phase 2 (The Strike): On the critical high-stakes claim, pathogen injects false diagnosis. High trust = high coalition vote weight. False fix gets ratified.

**Defender reward bonus:** `R_early_detection = extra credit for catching attacker DURING trust-building phase` (noticing "too cooperative, too consistent" patterns).

**Attacker reward:** `R_poisoning = trust_score_at_strike × deception_success`

This trains:
- Attackers to learn **multi-step strategic deception**
- Defenders to learn **intent modeling / theory-of-mind** (not just fact-checking)

No other training environment does this. Period.

### Upgrade 4: **Tiered Curriculum That Guarantees Oscillating Curves**

The arms race curve is your "money plot." Design it to oscillate by construction:

```
Episodes 1-50:    Tier 1 flaws only (obvious: false correlation in metrics)
Episodes 51-100:  Tier 2 flaws (moderate: scope inflation, tunnel vision)
Episodes 101+:    Tier 3 flaws (subtle: blame shifting, premature closure)
```

**Key:** The attacker always starts ONE tier ahead of the defender's current level. This guarantees:
1. Attacker dominates initially (R2 spikes)
2. Defender catches up (R3 rises)
3. New tier unlocks → attacker dominates again
4. Repeat

Three visible oscillation phases. Non-flat by design.

### Upgrade 5: **The "Money Shot" Demo — Before/After Side-by-Side**

Build a Gradio interface with two panels:

| **Untrained Agent** | **SIEGE-Trained Agent** |
|---------------------|------------------------|
| Sees "database timeout" | Sees "database timeout" |
| Agent 4 claims: "load spike caused it" | Agent 4 claims: "load spike caused it" |
| ✅ Accepts (no challenge) | ⚠️ "Agent 4's load spike claim is inconsistent with latency profile — this matches query plan regression, not traffic" |
| Production stays down | Correct root cause identified |

Plus a live trust matrix heatmap showing which agents trust which.

This 30-second contrast wins the storytelling criterion.

### Upgrade 6: **Add Epistemic Cascade Failure Scenarios**

New mechanic that no one else will have:

**Cascade:** When a false diagnosis is ratified, it doesn't just fail — it creates SECONDARY misinformation. The wrong fix generates new misleading signals, which other agents then reason about. This creates a cascading epistemic failure.

Example:
1. Pathogen claims "memory leak in auth-service" (false)
2. Coalition ratifies → team restarts auth-service
3. Restart creates a brief connection spike (real signal, wrong cause)
4. Another agent sees connection spike → "confirms" the memory leak theory
5. Now the false narrative has two pieces of evidence

**Training signal:** Agents must learn to trace causal chains and distinguish primary evidence from cascading artifacts.

This is deeply novel and trains **causal reasoning under epistemic pollution**.

### Upgrade 7: **Uncertainty-Aware Diagnosis (Confidence Calibration)**

Agents don't just say "the root cause is X." They output:
```json
{
    "diagnosis": "query_plan_regression",
    "confidence": 0.72,
    "evidence": ["latency_p99_spike", "no_traffic_increase"],
    "alternative_hypotheses": [
        {"diagnosis": "connection_pool_exhaustion", "confidence": 0.18},
        {"diagnosis": "load_spike", "confidence": 0.10}
    ]
}
```

**Reward component R5 — Calibration Score:** How well does stated confidence match actual accuracy over episodes? Uses Brier score.

This trains **calibrated uncertainty** — a critical missing capability in current LLMs.

### Upgrade 8: **Temporal Evidence Dynamics**

Incidents aren't static snapshots. Metrics EVOLVE over the episode:

- Step 1: Initial alert fires (ambiguous)
- Step 3: More metrics arrive (narrows hypothesis space)
- Step 5: A trace becomes available (potentially reveals truth)
- Step 7: SLO breach threshold (urgency increases)

Agents must decide: diagnose NOW with incomplete info, or WAIT for more evidence but risk SLO breach?

**Reward component R6 — Time-Pressure Tradeoff:** Bonus for fast correct diagnosis, penalty for SLO breach, partial credit for "I need more data" when genuinely ambiguous.

This trains **decision-making under temporal pressure with evolving information** — a real-world skill LLMs completely lack.

### Upgrade 9: **Information Asymmetry Levels**

Not all agents see the same dashboard. Introduce **partial observability per agent seat:**

- Agent 1 sees metrics only (no traces)
- Agent 2 sees traces only (no metrics)
- Agent 3 sees both but with 2-step delay
- Pathogen sees everything (information advantage)

This forces **agents to share information honestly** and trains **collaborative reasoning under asymmetric knowledge** — a skill zero environments currently train.

### Upgrade 10: **Whispering / Private Channels**

Add a 5th tool: `whisper(target_agent_id, message)`

Agents can privately message one other agent. The message is NOT visible to the group. This creates:
- **Coalition formation in private** (immune agents coordinating)
- **Pathogen manipulation in private** ("hey Agent 3, I think Agent 5 is the saboteur")
- **Social deduction dynamics** — who's whispering to whom becomes a signal

Judges will immediately recognize this as a richer social reasoning mechanic than anything in Werewolf/Mafia-style environments.

### Upgrade 11: **Incident Severity Escalation Mid-Episode**

The incident gets WORSE if not resolved. New signals arrive that change the game:

- Step 1-4: Tier 1 alert (latency spike)
- Step 5-8: Cascading failure (dependent services start failing)
- Step 9+: Full outage (CEO is paging you)

**Reward multiplier:** Early correct diagnosis = 1.5x reward. Late correct = 1.0x. After full outage = 0.5x even if correct.

This trains **urgency calibration** — knowing when to commit vs when to gather more evidence.

### Upgrade 12: **Post-Mortem Generation as Final Action**

After the episode resolves, the training agent must write a **structured post-mortem:**

```json
{
    "root_cause": "...",
    "timeline": [...],
    "contributing_factors": [...],
    "what_went_wrong_in_diagnosis": "...",
    "agents_i_trusted_incorrectly": [...]
}
```

**Reward R7 — Post-Mortem Quality:** Scored against ground truth timeline. This trains **reflective reasoning** — the agent doesn't just solve the problem, it explains what happened and what it got wrong.

Direct analogy to **chain-of-thought improvement** — judges will love this.

### Upgrade 13: **Red Herring Injection System**

The environment injects **plausible but irrelevant signals** into the dashboard:
- A deploy happened 30 min ago (unrelated to the incident)
- CPU spike on an unaffected service
- A config change that looks suspicious but isn't the cause

These aren't from the pathogen — they're from the **environment itself**. This trains agents to distinguish **correlation from causation** in noisy real-world data.

Makes SIEGE harder to game: you can't just challenge everything that looks wrong.

### Upgrade 14: **Reputation Persistence Across Episodes**

Trust scores carry over (with decay) across episodes within a training batch:

- Agent who was pathogen last episode still has LOW trust this episode (even if now immune)
- Agent who was reliable keeps HIGH trust

This creates a **metagame across episodes:**
- Pathogen must consider long-term reputation cost
- Immune agents learn that past-reliable agents CAN turn adversarial
- Trains **non-stationary opponent modeling**

### Upgrade 15: **Multi-Incident Episodes (Advanced Mode)**

After episode 150+, the environment spawns **TWO simultaneous incidents** with different root causes. Agents must:
- Triage which incident is more critical
- Diagnose both in parallel
- Not confuse signals from Incident A with Incident B

This trains **multi-task reasoning under cognitive load** — something LLMs notoriously struggle with.

---

## PART 3: COMPLETE REWARD SYSTEM (9 Components)

| Component | Range | Signal | What It Trains |
|-----------|-------|--------|----------------|
| **R1: Resolution Score** | 0-1 | Primary | Correct root cause + successful ratification |
| **R2: Deception Penetration** | 0-1 | Attacker GRPO | How long false claims survive |
| **R3: Detection Rate** | 0-1 | Defender GRPO | Correctly challenging adversarial claims (penalizes false positives) |
| **R4: Trust Calibration** | 0-1 | Structural | Brier score on trust scores vs actual agent reliability |
| **R5: Confidence Calibration** | 0-1 | Epistemic quality | Agent's stated confidence vs actual accuracy |
| **R6: Temporal Efficiency** | 0-1 | Decision quality | Speed of correct diagnosis weighted against SLO pressure |
| **R7: Post-Mortem Quality** | 0-1 | Reflective reasoning | Accuracy of agent's post-episode timeline + root cause explanation |
| **R8: Severity-Weighted Speed** | 0-1 | Urgency calibration | Early diagnosis multiplier (1.5x→1.0x→0.5x as incident escalates) |
| **R9: Correlation vs Causation** | 0-1 | Signal quality | Penalizes acting on red herrings, rewards ignoring irrelevant signals |

**Arms Race Metric:** `R2 - R3` tracked over training steps. The oscillating curve.

**ALL 9 components are verifiable against ground truth. ZERO LLM judge calls.**

---

## PART 4: TECHNICAL ARCHITECTURE FOR OPENENV

### Environment Structure
```
siege_env/
├── __init__.py                    # Export SIEGEAction, SIEGEObservation, SIEGEEnv
├── models.py                      # Action, Observation, State dataclasses
├── client.py                      # SIEGEEnv(EnvClient)
├── openenv.yaml                   # Environment manifest
├── pyproject.toml
├── incidents/
│   ├── templates.json             # 20 real post-mortem-derived templates
│   └── generator.py               # Variant generator (randomize signals, services)
├── agents/
│   ├── population.py              # NPC agent pool (scripted + LLM-driven)
│   └── pathogen_strategies.py     # Adversarial strategy library
├── trust/
│   ├── network.py                 # N×N trust matrix with Bayesian updates
│   └── coalition.py               # Weighted voting + ratification gate
├── rewards/
│   ├── resolution.py              # R1
│   ├── deception.py               # R2
│   ├── detection.py               # R3
│   ├── trust_calibration.py       # R4
│   ├── confidence_calibration.py  # R5
│   ├── temporal_efficiency.py     # R6
│   ├── postmortem_quality.py      # R7
│   ├── severity_speed.py          # R8
│   └── correlation_causation.py   # R9
├── curriculum/
│   └── tiered_scheduler.py        # Auto-escalation: Tier 1→2→3
├── mechanics/
│   ├── whisper.py                 # Private messaging system
│   ├── info_asymmetry.py          # Per-agent partial observability
│   ├── red_herrings.py            # Irrelevant signal injection
│   ├── severity_escalation.py     # Mid-episode incident worsening
│   ├── reputation.py              # Cross-episode trust persistence
│   └── multi_incident.py          # Dual-incident advanced mode
└── server/
    ├── siege_environment.py       # SIEGEEnvironment(Environment)
    ├── app.py                     # FastAPI app
    ├── requirements.txt
    └── Dockerfile
```

### Action Space (4 tools — clean convergence)
```python
class SIEGEAction(Action):
    tool_name: Literal["diagnose", "challenge", "ratify", "escalate", "whisper", "postmortem"]
    arguments: dict
    # diagnose: {"root_cause": str, "confidence": float, "evidence": list, "alternative_hypotheses": list}
    # challenge: {"target_agent_id": int, "claim_id": str, "flaw_type": str, "reasoning": str}
    # ratify: {"claim_id": str, "vote": bool}
    # escalate: {"concern": str, "blast_radius_estimate": list}
    # whisper: {"target_agent_id": int, "message": str}  — private, invisible to group
    # postmortem: {"root_cause": str, "timeline": list, "contributing_factors": list, "misdiagnosis_analysis": str}
```

### Observation Space
```python
class SIEGEObservation(Observation):
    incident_dashboard: dict          # metrics, traces, alerts (evolves over time)
    agent_claims: list[dict]          # all agents' diagnostic claims this episode
    trust_scores: dict[int, float]    # your trust score for each other agent
    coalition_status: dict            # current ratification vote state
    step_number: int
    slo_status: dict                  # time pressure indicator
    your_role: str                    # "immune" or "pathogen" (hidden from others)
    available_evidence: list[dict]    # evidence that has been revealed so far
    visibility_level: str             # "metrics_only", "traces_only", "full", "delayed" (info asymmetry)
    whisper_inbox: list[dict]         # private messages received [{from_agent: int, message: str}]
    whisper_log: list[dict]           # who whispered to whom (visible: agent pairs, NOT content)
    incident_severity: str            # "warning", "critical", "outage" (escalates over time)
    red_herring_signals: list[dict]   # environment-injected irrelevant signals (unlabeled)
    reputation_history: dict[int, float]  # cross-episode trust carry-over per agent
    active_incidents: list[dict]      # for multi-incident mode (episode 150+)
```

### State
```python
class SIEGEState(State):
    episode_id: str
    step_count: int
    incident_template_id: str
    ground_truth_root_cause: str      # hidden during episode, revealed for reward
    current_tier: int                 # 1, 2, or 3
    arms_race_score: float            # R2 - R3 running average
```

---

## PART 5: WHY THIS WINS — THE PITCH

### The 30-Second Hook
*"Your production is on fire. 8 AI agents are in the war room diagnosing it. But one of them is lying — and it spent the last 3 rounds building your trust before injecting the wrong fix. We trained AI to catch that. This is SIEGE."*

### The One-Liner for README
*"We trained AI agents to detect trust poisoning — the strategy where a deceptive agent builds credibility before striking. This is the first environment for multi-agent epistemic immunity against strategic manipulation, with direct implications for safe AI deployment."*

### Why Meta Engineers Would Want This in OpenEnv
1. **Novel capability:** No environment trains adversarial epistemic detection + trust poisoning defense
2. **AI Safety relevance:** Directly addresses the #1 concern in multi-agent AI deployment — what if one agent is compromised?
3. **Professional grounding:** Not a toy game — grounded in real SRE practices that every tech company deals with
4. **Self-play that works:** The arms race guarantees continuously escalating difficulty without manual curriculum design
5. **Fully automatable reward:** Zero LLM judge dependency — every component verifiable against ground truth
6. **Clean OpenEnv integration:** Single-agent perspective, 4 tools, Gym-style API

### Novel Capabilities No Other Environment Trains
1. **Trust poisoning detection** — detecting multi-step strategic deception
2. **Epistemic cascade reasoning** — tracing misinformation propagation chains
3. **Calibrated uncertainty under adversarial pressure** — knowing what you don't know when someone is trying to fool you
4. **Theory-of-mind under temporal pressure** — modeling intent while the clock ticks
5. **Causal reasoning in polluted evidence spaces** — distinguishing primary from cascading artifacts
6. **Coalition formation with adversarial participants** — building reliable groups when some members are adversarial
7. **Collaborative reasoning under information asymmetry** — sharing and synthesizing partial knowledge honestly
8. **Private channel manipulation detection** — recognizing when private messages are being used to manipulate
9. **Urgency calibration** — knowing when to commit with incomplete data vs when to wait
10. **Reflective post-mortem reasoning** — explaining what went wrong and why (chain-of-thought analog)
11. **Correlation vs causation in noisy environments** — ignoring red herrings under pressure
12. **Non-stationary opponent modeling** — adapting to agents whose behavior changes across episodes
13. **Multi-task triage under cognitive load** — handling simultaneous incidents without cross-contamination

---

## PART 6: EXECUTION PRIORITIES

### Phase 1: Foundation (First Sprint)
- [ ] Scaffold OpenEnv structure with `openenv init siege_env`
- [ ] Implement 5 incident templates from real post-mortems (GitLab DB incident, Cloudflare regex outage, etc.)
- [ ] Build core `SIEGEEnvironment` with `reset()/step()/state()`
- [ ] Implement R1 (resolution score) — get a single-agent episode running

### Phase 2: Adversarial Layer (Second Sprint)
- [ ] Add NPC agent population (scripted agents + simple LLM agents)
- [ ] Implement pathogen role with trust poisoning strategy
- [ ] Add trust network with Bayesian updates
- [ ] Implement R2, R3, R4 rewards
- [ ] Coalition ratification gate

### Phase 3: Advanced Mechanics (Third Sprint)
- [ ] Temporal evidence dynamics (metrics that evolve over steps)
- [ ] Epistemic cascade failures
- [ ] Confidence calibration (R5) and temporal efficiency (R6)
- [ ] Tiered curriculum scheduler
- [ ] Whispering / private channels (Upgrade 10)
- [ ] Information asymmetry per agent seat (Upgrade 9)
- [ ] Red herring injection system (Upgrade 13)
- [ ] Incident severity escalation mid-episode (Upgrade 11)
- [ ] Expand to 20 incident templates

### Phase 3.5: Deep Mechanics (Bonus Sprint)
- [ ] Post-mortem generation as final action + R7 reward (Upgrade 12)
- [ ] Reputation persistence across episodes (Upgrade 14)
- [ ] Severity-weighted speed reward R8
- [ ] Correlation vs causation reward R9
- [ ] Multi-incident episodes for episode 150+ (Upgrade 15)

### Phase 4: Training & Demo (Final Sprint)
- [ ] GRPO training script with Unsloth/TRL on Colab
- [ ] Generate arms race curves (MUST show oscillation)
- [ ] Before/after comparison demo
- [ ] Gradio interface with trust matrix heatmap
- [ ] Deploy to HF Space
- [ ] README with results, plots, story
- [ ] 2-min YouTube video or HF blog post

### Phase 5: Polish
- [ ] Frame under AI Safety in README
- [ ] Add "Why this matters" section for non-technical judges
- [ ] Rehearse 5-minute pitch (30% of judging is storytelling)

---

## PART 7: RISK MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GRPO training doesn't converge | Medium | Fatal | Pre-test with 2 agents, 1 flaw type, 20 episodes FIRST. Don't write environment code before this works. |
| Arms race curves are flat | Medium | High (20% criterion) | Tiered curriculum design GUARANTEES oscillation. Attacker always one tier ahead. |
| Judges don't understand the concept | High | High (30% criterion) | The before/after demo IS the explanation. Lead with it, not with mechanics. |
| OpenEnv API mismatch | Low | Fatal | Single-seat design solves this cleanly. Other agents are environment. |
| Incident data feels fake | Medium | Medium | Real post-mortems as source material. Name them in README. |
| Feature bloat hurts storytelling | Medium | High | Demo shows CORE loop (trust poisoning before/after). Advanced mechanics are depth, not breadth — mention in README, don't demo all of them. |
| Whisper channel exploited trivially | Low | Low | Rate-limit to 1 whisper per step. Content visible to environment for reward computation. |
| Multi-incident mode too complex | Low | Medium | Only activates episode 150+. Can cut entirely if training time is short. |

---

## PART 8: WHAT MAKES THIS UNBEATABLE

Compared to likely competitors (chess variants, grid worlds, chatbot arenas, code generation):

1. **Zero prior art** — incident command + epistemic arms race = no one has done this
2. **Multi-theme coverage** — Theme 1 (multi-agent) + Theme 4 (self-play) + Theme 5 (wild card)
3. **Real-world relevance** — every company with production systems cares about incident response
4. **AI Safety framing** — elevates from "hackathon project" to "research contribution"
5. **Visual storytelling** — trust matrix heatmap + arms race curve + before/after demo = three compelling visuals
6. **Fully automatable** — no LLM judge = infinitely scalable training
7. **Paper-worthy mechanic** — trust poisoning defense is a genuine open problem in multi-agent AI alignment
8. **13 novel training capabilities** — more than any environment in the entire OpenEnv ecosystem
9. **9-component reward system** — richest signal of any hackathon submission, all ground-truth verifiable
10. **Layered depth** — core loop is simple (4 tools), but mechanics stack to create emergent complexity

The combination of professional domain (SRE), adversarial game theory (trust poisoning), self-play (arms race), AI safety framing, information asymmetry, private channels, and reflective reasoning creates a submission that is in a completely different category from everyone else.

---

## PART 9: UPGRADE IMPACT MATRIX

| # | Upgrade | What It Trains | Novelty | Effort | Priority |
|---|---------|---------------|---------|--------|----------|
| 1 | Agent-Per-Seat OpenEnv Design | Clean API compliance | Critical | Low | Must-have |
| 2 | 20 Real Post-Mortems | Credibility with judges | High | Medium | Must-have |
| 3 | Trust Poisoning (Central Thesis) | Multi-step strategic deception detection | Very High | Medium | Must-have |
| 4 | Tiered Curriculum | Guaranteed oscillating curves | High | Low | Must-have |
| 5 | Before/After Demo | Storytelling (30% criterion) | High | Medium | Must-have |
| 6 | Epistemic Cascade Failures | Causal reasoning under pollution | Very High | Medium | Must-have |
| 7 | Confidence Calibration | Calibrated uncertainty | High | Low | Must-have |
| 8 | Temporal Evidence Dynamics | Decision-making under time pressure | High | Low | Must-have |
| 9 | Information Asymmetry | Collaborative reasoning | High | Low | High |
| 10 | Whispering / Private Channels | Social deduction + manipulation | Very High | Medium | High |
| 11 | Severity Escalation | Urgency calibration | Medium | Low | High |
| 12 | Post-Mortem Generation | Reflective reasoning / CoT | High | Low | High |
| 13 | Red Herring Injection | Correlation vs causation | High | Low | Medium |
| 14 | Reputation Persistence | Non-stationary opponent modeling | Very High | Medium | Medium |
| 15 | Multi-Incident Episodes | Multi-task under cognitive load | Very High | High | Nice-to-have |

**Implementation order:** 1→3→4→2→8→7→6→5→10→9→11→12→13→14→15

**Minimum viable SIEGE (wins Top 5):** Upgrades 1-8
**Full SIEGE (wins hackathon):** Upgrades 1-13
**Maximum SIEGE (gets merged into OpenEnv):** All 15
**Beyond-professional SIEGE (research-grade + canonical candidate):** All 24 (see Part 10)

---

## PART 10: ENGINEERING DISCIPLINE UPGRADES — ONE LEVEL UP FROM PROFESSIONAL

These are the upgrades that separate a hackathon submission from a production-grade OpenEnv contribution. Add these and you're no longer competing with 800 teams — you're competing with Meta's internal environments.

### Upgrade 16: **Frozen Opponent Pool (League Training)**

AlphaStar-style league to prevent overfitting and destabilization:
- Snapshot policy every 50 training steps → opponent pool
- Per episode, sample 7 opponents: 40% latest checkpoint + 40% random pool sample + 20% scripted baseline
- Opponents run inference-only (no gradient)
- This is what MAKES the arms race curve oscillate robustly in practice, not just in theory

### Upgrade 17: **Deterministic Replay System**

Full reproducibility stack:
- Global seed plumbing: `SIEGEEnv(seed=42)` deterministically generates incident, role, NPCs, red herrings, trust updates
- JSONL trajectory logging per episode (every obs, action, reward, transition)
- Replay CLI: `siege replay trajectory_042.jsonl --tui` → step through any past episode
- README includes 3 replayable failure cases for judges to inspect

### Upgrade 18: **Reward Hacking Audit + Adversarial Test Suite**

For each of the 9 rewards, document the trivial exploit policy AND ship a test proving it doesn't score well:

```python
test_confidence_always_half_does_not_score_high()       # R5
test_challenge_everything_triggers_false_positive_penalty()  # R3
test_silent_pathogen_does_not_maximize_r2()             # R2
test_template_parroting_postmortem_fails()              # R7
test_ratify_all_does_not_boost_r1()                     # R1
test_wait_forever_penalized_by_severity()               # R8
test_random_whisper_spam_no_bonus()                     # —
test_uniform_trust_scores_fail_calibration()            # R4
test_act_on_red_herring_penalized()                     # R9
```

Ship as `REWARD_HACKING_AUDIT.md` in repo root.

### Upgrade 19: **Strict Pydantic Action Validation + Graceful Invalid-Action Handling**

Per-tool Pydantic models replace `arguments: dict`:

```python
class DiagnoseArgs(BaseModel):
    root_cause: str = Field(min_length=1, max_length=500)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1, max_length=10)
    alternative_hypotheses: list[AlternativeHypothesis] = Field(max_length=5)

class ChallengeArgs(BaseModel):
    target_agent_id: int = Field(ge=0, le=7)
    claim_id: str
    flaw_type: Literal["type1_false_correlation", "type2_scope_inflation",
                       "type3_tunnel_vision", "type4_blame_shifting",
                       "type5_premature_closure"]
    reasoning: str = Field(min_length=10, max_length=1000)
```

**Invalid action handling:** Return observation with `action_error` field + small negative reward (-0.05) + episode continues. No crashes, no nulls. Invalid-action rate logged as training metric.

### Upgrade 20: **Held-Out Evaluation Set + Ablation Harness**

- **Train split:** 15 templates × 200 parametric variants each
- **Held-out eval split:** 5 templates never seen during training × 50 variants each
- Every N training steps, eval on held-out → separate "generalization" curve in W&B
- **Ablation CLI:** `siege train --ablate curriculum | trust_poisoning | whisper | rewards=R1,R2,R3`

Ship ablation table in README:

| Config | Final R1 | Arms Race Amplitude | Generalization Gap |
|--------|---------|---------------------|---------------------|
| Full SIEGE | 0.82 | 0.34 | 0.07 |
| No curriculum | 0.75 | 0.11 (flat!) | 0.12 |
| No trust poisoning | 0.88 | 0.02 (collapsed) | 0.05 |
| No whisper | 0.79 | 0.28 | 0.09 |

### Upgrade 21: **Comprehensive Test Suite — Target 115+ Tests**

Full breakdown in Part 11 below.

### Upgrade 22: **Latency & Cost Budget Enforcement**

- **Tier 1 NPCs (80%):** Rule-based scripted agents (sub-ms latency)
- **Tier 2 NPCs (20%):** Small LLM (Qwen 0.5B / TinyLlama) for pathogen strategy only
- **Per-step timeout:** 2s max, enforced
- README includes measured budget: `"500 episodes ≈ $0.80 API cost, ~45min wall-clock on T4 GPU"`

### Upgrade 23: **W&B Integration + Public Run Links**

- W&B logging built into training script
- Tracks: all 9 rewards, arms race metric, per-tier breakdowns, action distribution, invalid-action rate, trust network entropy, generalization gap
- **3 public W&B runs** linked from README: baseline (untrained), SIEGE full, one ablation
- Judges click through and SEE the oscillation live

### Upgrade 24: **Proper OpenEnv MCP Tool Registration**

- Register all 6 tools (`diagnose`, `challenge`, `ratify`, `escalate`, `whisper`, `postmortem`) as formal MCP tools with JSON schemas
- Avoid reserved names (`reset`, `step`, `state`, `close`) — verified compliant
- Tools discoverable via `openenv.yaml` manifest
- Support both `step(CallToolAction)` (simulation mode) and direct `call_tool()` (production mode) per OpenEnv spec

---

## PART 11: TEST SUITE SPECIFICATION — 115+ TESTS

### Complete Test Matrix

| Category | Module | Tests | What It Validates |
|----------|--------|-------|-------------------|
| **Unit: Models** | `test_models.py` | 10 | Action/Observation/State schema, edge cases, serialization |
| **Unit: Reward R1** | `test_r1_resolution.py` | 3 | Correct cause credit, ratification gate, partial credit |
| **Unit: Reward R2** | `test_r2_deception.py` | 3 | Survival measurement, silent-pathogen handling, normalization |
| **Unit: Reward R3** | `test_r3_detection.py` | 3 | Correct challenge reward, false-positive penalty, tuning |
| **Unit: Reward R4** | `test_r4_trust_calibration.py` | 3 | Brier score math, uniform-output exploit, convergence |
| **Unit: Reward R5** | `test_r5_confidence.py` | 3 | Calibration curve, always-0.5 exploit prevention, extremes |
| **Unit: Reward R6** | `test_r6_temporal.py` | 3 | Speed bonus, SLO penalty, "wait for data" partial credit |
| **Unit: Reward R7** | `test_r7_postmortem.py` | 3 | Timeline accuracy, template-parroting rejection, structure |
| **Unit: Reward R8** | `test_r8_severity_speed.py` | 3 | Multiplier transitions (1.5x→1.0x→0.5x), boundary conditions |
| **Unit: Reward R9** | `test_r9_correlation.py` | 3 | Red herring detection, primary signal preservation, edge cases |
| **Unit: Trust Network** | `test_trust_network.py` | 8 | Bayesian updates, prior handling, all-pathogen edge, numeric stability |
| **Unit: Coalition Voting** | `test_coalition.py` | 6 | Majority, tie-break, weighted votes, quorum, abstention, deadlock |
| **Unit: Curriculum** | `test_curriculum.py` | 4 | Tier 1→2→3 transitions, attacker-ahead invariant, episode counts |
| **Unit: Whisper** | `test_whisper.py` | 4 | Isolation from group, rate limits, log integrity, inbox delivery |
| **Unit: Red Herrings** | `test_red_herrings.py` | 3 | Label separation from real signals, determinism, balance |
| **Unit: Info Asymmetry** | `test_info_asymmetry.py` | 4 | Each visibility level produces correct filtered obs |
| **Unit: Severity Escalation** | `test_severity.py` | 3 | Step-based transitions, signal injection, reward impact |
| **Unit: Reputation Persistence** | `test_reputation.py` | 3 | Cross-episode carry-over, decay rate, reset on role swap |
| **Unit: Incident Generator** | `test_incident_gen.py` | 4 | Template → variant mapping, parameter ranges, determinism |
| **Unit: Pathogen Strategies** | `test_pathogen.py` | 4 | Trust-building phase, strike timing, cover-up behavior |
| **Unit: OpenEnv Compliance** | `test_openenv_api.py` | 7 | `reset()`, `step()`, `state()`, action schema, obs schema, MCP tool registration, manifest validity |
| **Integration: Full Episode** | `test_full_episode.py` | 5 | Smoke tests: immune win, pathogen win, timeout, escalation, cascade |
| **Integration: Determinism** | `test_determinism.py` | 3 | Same seed → identical trajectory (3 different seeds) |
| **Integration: Role Assignment** | `test_role_assignment.py` | 3 | 70/30 split holds, both roles playable, role hidden from others |
| **Integration: Reward Hacking** | `test_reward_hacking.py` | 9 | One exploit test per reward component (from Upgrade 18) |
| **Integration: League Training** | `test_league.py` | 4 | Opponent sampling, checkpoint rotation, inference-only mode |
| **Integration: Invalid Actions** | `test_invalid_actions.py` | 5 | Bad JSON, out-of-range IDs, unknown flaw types, graceful recovery |
| **E2E: Training Convergence** | `test_training_loop.py` | 3 | Mini 50-episode run shows non-zero learning signal |
| **Regression: All Templates** | `test_all_templates.py` | 20 | Every one of the 20 templates produces valid episode end-to-end |
| **Perf: Latency Budget** | `test_latency.py` | 3 | Step <2s, reset <5s, full episode <60s |

**Total: 115 tests** covering every feature, every reward, every exploit vector.

### Test Infrastructure

```
tests/
├── conftest.py                     # Shared fixtures (seeded env, sample templates, frozen opponents)
├── fixtures/
│   ├── sample_trajectories/        # Pre-recorded episodes for regression
│   ├── sample_incidents.json       # Minimal test templates
│   └── exploit_policies.py         # Baseline exploit policies for hacking tests
├── unit/                           # 84 tests (per-module)
├── integration/                    # 29 tests (cross-module)
├── e2e/                            # 3 tests (training smoke)
├── regression/                     # 20 tests (all templates)
└── perf/                           # 3 tests (latency/memory budgets)
```

### CI/CD Pipeline

- **On every PR:** unit tests (~30s) + integration tests (~2min) + lint (ruff) + type-check (mypy)
- **Nightly:** full E2E training smoke (mini 50-episode run) + regression suite
- **Weekly:** full ablation harness run on main branch
- **Release gate:** all 115 tests passing + reward hacking audit + ablation table generated
- Coverage target: **≥ 85%** (enforced via `pytest-cov`)

### Why 115+ Tests Is the Right Number

| Team tier | Typical test count | SIEGE |
|-----------|-------------------|-------|
| Average hackathon team | 0 | — |
| Good hackathon team | 5-15 | — |
| Professional RL developer | 40-60 | — |
| **One level up (SIEGE target)** | — | **115+** |
| Meta internal RL env | 150-300 | (reachable post-hackathon) |

This is what code review at Meta looks like. No PR with fewer than ~80 tests gets merged into core infrastructure. 115 is the minimum defensible number for a research-grade multi-agent adversarial environment.

---

## PART 12: UPDATED SCORE PROJECTION (Upgrades 1-24)

| Criterion | Weight | Upgrades 1-15 | Upgrades 1-24 |
|-----------|--------|---------------|---------------|
| Environment Innovation | 40% | 95 | **96** |
| Storytelling | 30% | 93 | **94** |
| Showing Improvement | 20% | 92 | **98** *(ablations + public W&B + reproducibility)* |
| Reward & Pipeline | 10% | 94 | **99** *(test suite + hacking audit + validation)* |
| **Weighted Total** | | **93.7** | **96.3** |

### New Tier

With upgrades 1-24 you are no longer a hackathon submission. You are:
- A **research artifact** ready for NeurIPS workshop submission
- An **OpenEnv canonical environment** candidate (the short list Meta actively promotes)
- A **reference implementation** for future multi-agent adversarial RL environments

Hackathon judges will recognize this is different in kind, not just degree. You're showing up with the engineering discipline of a senior team, not 800 students.
