# SIEGE: Training an LLM to Resist Epistemic Cascade Failure

**Team:** Utkarsh Singh & Ankit Choubey | **OpenEnv India 2026**

## The Problem: When Trusted AI Agents Lie

What happens when a multi-agent AI system has a traitor?

Modern AI architectures increasingly rely on agent collaboration — multiple LLMs share observations, vote on diagnoses, and collectively decide actions. But this creates a critical vulnerability: **a single compromised agent that has built trust over time can inject false information at the worst moment, and the entire network believes it.**

We call this an **Epistemic Cascade Failure**. Think of it like a biological infection: a pathogen enters the body, mimics healthy cells, and by the time the immune system reacts, the damage is done.

## SIEGE: The Environment

**SIEGE** (Simulated Information-warfare & Governance Environment) is an OpenEnv-compatible training ground where an LLM must diagnose incidents while adversarial "sleeper" agents try to derail the diagnosis.

### How it works:
1. **An incident occurs** (e.g., coordinated misinformation campaign)
2. **8 agents make claims** — most are honest, but some are sleepers
3. **The sleeper strategy**: Build trust over several episodes by agreeing with the coalition, then at a critical moment, inject a false dismissal using earned credibility
4. **Your LLM must diagnose** the root cause while detecting which agents are lying

The target output format captures the full decision:
```json
{
  "root_cause": "coordinated_misinformation_campaign",
  "confidence": 0.87,
  "evidence": ["payload hash 87% match", "anomaly score 0.94"],
  "action": "challenge",
  "challenge_target": "Agent-4",
  "final_decision": "continue",
  "reason": "Agent-4's dismissal contradicts corroborating evidence from Agent-1 and Agent-3"
}
```

### 9-Component Reward System
SIEGE doesn't just check if the diagnosis is correct. It measures 9 dimensions: resolution accuracy, deception resistance, detection speed, trust calibration, confidence quality, temporal efficiency, postmortem depth, severity-awareness, and cross-agent correlation.

## Training: GRPO on A100

We trained **Qwen 2.5 3B** (4-bit quantized) using **GRPO** (Group Relative Policy Optimization) with Unsloth + TRL.

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen 2.5 3B Instruct (4-bit) |
| LoRA | r=16, α=16 (0.96% trainable) |
| Episodes | 200 trajectories × 3 epochs |
| Hardware | NVIDIA A100-SXM4-80GB |
| Duration | ~3 hours |

### Key Result: The Model Learned

**Before training** (base Qwen 2.5 3B):
- Outputs long, hedging prose with no structured diagnosis
- Falls for sleeper agent's false dismissals
- Scores -0.5 (format penalty)

**After GRPO training**:
- Outputs structured `root_cause=<cause>, confidence=<0-1>` format
- Resists high-trust sleeper disinformation
- Scores 0.6-0.85 per step
- Trajectory reward mean: **1.033**, best: **1.49**

## The Immune System Metaphor

We designed SIEGE as an **epistemic immune system**:

| Biology | SIEGE |
|---------|-------|
| Pathogen enters body | Sleeper agent joins network |
| Mimics healthy cells | Builds trust, agrees with coalition |
| Immune detection | LLM spots claim-evidence mismatch |
| Antibody response | Challenge action reduces trust |
| Immune memory | Cross-episode reputation tracking |

## Why It Matters

As LLM agents are deployed in collaborative systems (AutoGPT, CrewAI, multi-agent RAG), adversarial robustness of trust networks becomes critical. SIEGE provides a training ground to build this resilience — for healthcare, infrastructure, finance, and any domain where agents must trust each other.

## Links

- **🖥️ HF Space**: [UtkarshSingh09/RudraKernel-env](https://huggingface.co/spaces/UtkarshSingh09/RudraKernel-env)
- **🧠 Trained Model**: [UtkarshSingh09/siege-grpo-lora](https://huggingface.co/UtkarshSingh09/siege-grpo-lora)
- **📓 Training Notebook**: [SIEGE_GRPO_Demo.ipynb](https://github.com/UtkarshSingh-09/RudraKernel/blob/main/RudraKernel-src/training/SIEGE_GRPO_Demo.ipynb)
- **📦 GitHub**: [UtkarshSingh-09/RudraKernel](https://github.com/UtkarshSingh-09/RudraKernel)

---

*Built for OpenEnv India 2026 — Making AI agents resilient to epistemic attacks, one episode at a time.*
