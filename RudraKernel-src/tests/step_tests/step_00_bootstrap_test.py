from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_required_structure_exists() -> None:
    required_paths = [
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / ".pre-commit-config.yaml",
        ROOT / "pyproject.toml",
        ROOT / "Makefile",
        ROOT / "brain" / "tools" / "update_brain.py",
        ROOT / "brain" / "tools" / "compile_master_code.py",
        ROOT / "tests" / "step_tests" / "step_00_bootstrap_test.py",
    ]
    for path in required_paths:
        assert path.exists(), f"Missing required bootstrap path: {path}"


def test_ci_config_parses() -> None:
    ci_file = ROOT / ".github" / "workflows" / "ci.yml"
    data = yaml.safe_load(ci_file.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "jobs" in data


def test_make_test_all_runs_cleanly() -> None:
    if os.getenv("SIEGE_BOOTSTRAP_SELFTEST") == "1":
        return

    env = os.environ.copy()
    env["SIEGE_BOOTSTRAP_SELFTEST"] = "1"
    result = subprocess.run(
        ["make", "test-all"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


def test_update_brain_creates_snapshot() -> None:
    tracked_files = {
        ROOT / "brain" / "MASTER_CODE.md": (ROOT / "brain" / "MASTER_CODE.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "CHANGELOG.md": (ROOT / "brain" / "CHANGELOG.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "CONTEXT.md": (ROOT / "brain" / "CONTEXT.md").read_text(
            encoding="utf-8"
        ),
        ROOT / "brain" / "ROADMAP_STATUS.md": (
            ROOT / "brain" / "ROADMAP_STATUS.md"
        ).read_text(encoding="utf-8"),
    }
    before = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
    try:
        result = subprocess.run(
            [
                "python3",
                "brain/tools/update_brain.py",
                "--step",
                "00",
                "--title",
                "Bootstrap",
                "--owner",
                "Utkarsh",
                "--reviewer",
                "Ankit",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        after = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
        assert len(after) >= len(before) + 1
    finally:
        after = set((ROOT / "brain" / "snapshots").glob("step_00_*.json"))
        for snapshot_path in after - before:
            snapshot_path.unlink(missing_ok=True)
        for file_path, original_contents in tracked_files.items():
            file_path.write_text(original_contents, encoding="utf-8")
