#!/usr/bin/env bash
set -euo pipefail

echo "Ablation harness placeholder for Step 00."
# Step 22 append-only ablation harness integration
python3 - <<'PY'
import json
from pathlib import Path

from training.ablation import default_ablation_runs

runs = [{"name": run.name, "enabled_components": run.enabled_components} for run in default_ablation_runs()]
out = Path("artifacts") / "ablation_plan.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(runs, indent=2), encoding="utf-8")
print(f"wrote {out}")
PY
