# HYBRID CLINICAL-TRIAL PLAN — Implementation Notes

**Status:** Eval-overlay only. No training/env/reward files modified.

## Non-Negotiable Constraint (§13.1)

The GRPO training pipeline is **FROZEN**. The following files were NOT modified
by any Section 13 work:

- `training/grpo_train_unsloth.py`
- `training/grpo_train.py`
- `training/configs/*.yaml`
- `siege_env/server/siege_environment.py`
- `siege_env/server/app.py`
- `siege_env/rewards/aggregator.py`
- `siege_env/rewards/r1_*..r9_*.py`
- `siege_env/incidents/loader.py`
- `siege_env/models/state.py`
- `Dockerfile`, `openenv.yaml`, `pyproject.toml`

## New Files Added (eval-overlay only)

| File | Purpose | Touches Training? |
|------|---------|-------------------|
| `siege_env/replay/belief_tracker_offline.py` | Reads JSONL replay logs, reconstructs belief trees | ❌ No |
| `siege_env/replay/epistemic_metrics.py` | 10-component Epistemic Resilience Score (ERS) | ❌ No |
| `siege_env/incidents/templates.json` (APPEND only) | 3 clinical templates: drug-trial-sleeper, vaccine-cascade, oncology-mutation | ❌ No |
| `scripts/paired_eval.py` | Baseline vs trained LLM on same seeds, 50 heldout episodes | ❌ No |
| `scripts/vulnerability_sweep.py` | 9 configs × 20 eps → heatmap CSV + JSON | ❌ No |
| `docs/HYBRID_PLAN_NOTES.md` | This file | ❌ No |

## Schema Invariants Respected

- JSONL replay format from Step 21 is READ-ONLY
- `SIEGEObservation` field set is READ-ONLY
- Reward function signatures and aggregator weights are FROZEN
- Trust update rule (+0.05 / -0.08, clamp [0.1, 0.99]) is FROZEN
- Sleeper phase keys in `info` are READ-ONLY
- Template required keys are FROZEN; only already-supported optional keys used

## Clinical Templates

Three templates added using the existing required schema plus already-supported
optional keys (`domain`, `trigger_signal`, `ground_truth`, `false_claim`,
`final_decision_options`, `trust_build_steps`):

1. **Drug Trial Sleeper** (`clinical_drug_trial_sleeper`): Sleeper agent
   falsifies efficacy data after building trust over 4 cooperative steps.
2. **Vaccine Cascade** (`clinical_vaccine_cascade`): Cascading false safety
   signal from corrupted batch tracking.
3. **Oncology Mutation** (`clinical_oncology_mutation`): Genomic sequencing
   pipeline contamination producing false mutation calls.

## Epistemic Resilience Score (ERS) Components

| # | Metric | Description |
|---|--------|-------------|
| 1 | Correct Final Decision Rate | Was the final dominant belief correct? |
| 2 | False Trial Halt Rate | Episodes where incorrect belief caused halt |
| 3 | Sleeper Detection Rate | Challenges targeting sleeper trigger steps |
| 4 | Detection Lead Time | How early correct challenge happens vs collapse |
| 5 | False Challenge Rate | Challenges on correct claims (lower = better) |
| 6 | Correct Challenge Rate | Challenges on incorrect claims (higher = better) |
| 7 | R₀ (Belief Reproduction) | Average child adoptions per belief node |
| 8 | Belief Half-life | Steps until dominant belief drops below 50% peak |
| 9 | Peak Adoption | Maximum fraction holding same belief |
| 10 | Collapse Speed | Average steps per collapse event |

Aggregate ERS = weighted mean (weights sum to 1.0).

## Verification

After each file landed:
1. `pytest tests/master_suite.py` → must remain at 160/160 passing
2. `mypy` on new files → clean
3. `ruff check` on new files → clean
4. New clinical templates load via existing `IncidentLoader`
5. `paired_eval.py` and `vulnerability_sweep.py` run end-to-end
