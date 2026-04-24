"""Gate tests for Step 22 — Held-Out Eval + Ablation Harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from training.ablation import default_ablation_runs
from training.heldout_split import build_split


def test_heldout_split_integrity() -> None:
    ids = [f"t{i}" for i in range(20)]
    split = build_split(ids, seed=11, heldout_fraction=0.2)
    assert len(split["heldout"]) == 4
    assert len(set(split["train"]).intersection(split["heldout"])) == 0


def test_ablation_default_runs_available() -> None:
    runs = default_ablation_runs()
    assert len(runs) >= 3
    assert {r.name for r in runs}.issuperset({"base", "no_curriculum", "no_trust_poisoning"})


def test_generate_ablations_script_writes_plan() -> None:
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", "scripts/generate_ablations.sh"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    path = root / "artifacts" / "ablation_plan.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and len(payload) >= 3
