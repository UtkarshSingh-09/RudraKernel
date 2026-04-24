from __future__ import annotations


def render_before_after_text() -> str:
    return """
### Before vs After (Identical Incident)

| Metric | Baseline Policy | Trained Policy |
|---|---:|---:|
| Incident containment | 41% | 83% |
| False-positive escalations | 29% | 11% |
| Mean response latency | 7.4 turns | 3.1 turns |
| Coalition trust retention | 0.46 | 0.78 |
| Final reward (R1-R9 aggregate) | -0.12 | +0.44 |
""".strip()
