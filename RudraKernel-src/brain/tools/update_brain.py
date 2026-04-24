#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from compile_master_code import main as compile_master

ROOT = Path(__file__).resolve().parents[2]
BRAIN = ROOT / "brain"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def snapshot_files() -> dict[str, str]:
    tracked: dict[str, str] = {}
    for folder in [ROOT / "siege_env", ROOT / "tests", ROOT / "brain", ROOT / "scripts"]:
        if not folder.exists():
            continue
        for file_path in folder.rglob("*"):
            if file_path.is_file() and ".git" not in file_path.parts:
                tracked[str(file_path.relative_to(ROOT))] = file_sha256(file_path)
    return tracked


def append_changelog(
    step: str,
    title: str,
    owner: str,
    reviewer: str,
    snapshot_name: str,
    gate_test: str,
    master_suite: str,
    coverage: float,
) -> None:
    changelog = BRAIN / "CHANGELOG.md"
    now = utc_now().isoformat()
    entry = (
        f"\n## {now} - Step {step}: {title}\n"
        f"**Owner:** {owner} | **Reviewer:** {reviewer}\n"
        f"**Gate test:** {gate_test}\n"
        f"**Master suite:** {master_suite}\n"
        f"**Coverage:** {coverage:.1f}%\n"
        f"**Brain snapshot:** brain/snapshots/{snapshot_name}\n"
    )
    existing = changelog.read_text(encoding="utf-8") if changelog.exists() else "# CHANGELOG\n"
    changelog.write_text(existing + entry, encoding="utf-8")


def update_context(step: str, title: str, gate_test: str, master_suite: str, coverage: float) -> None:
    context = BRAIN / "CONTEXT.md"
    now = utc_now().isoformat()
    context.write_text(
        "# CURRENT CONTEXT\n\n"
        f"- Updated: {now}\n"
        f"- Latest step touched: {step} - {title}\n"
        f"- Current status: Step {step} in sync with the repository\n"
        f"- Gate test: {gate_test}\n"
        f"- Master suite: {master_suite}\n"
        f"- Coverage: {coverage:.1f}%\n",
        encoding="utf-8",
    )


def _row_for_step(
    step: str,
    title: str,
    owner: str,
    gate_test: str,
    completed: str,
) -> str:
    status = "✅" if gate_test.lower() != "pending" else "In Progress"
    gate_label = "PASS" if gate_test.lower() != "pending" else "Pending"
    return f"| {step} | {title} | {owner} | {status} | {gate_label} | {completed} |"


def update_roadmap(step: str, title: str, owner: str, gate_test: str) -> None:
    roadmap = BRAIN / "ROADMAP_STATUS.md"
    now = utc_now().date().isoformat()
    header = [
        "| Step | Title | Owner | Status | Gate Test | Completed |",
        "|------|-------|-------|--------|-----------|-----------|",
    ]
    existing_rows: dict[int, str] = {}
    if roadmap.exists():
        for line in roadmap.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| ") or line.startswith("| Step ") or line.startswith("|------"):
                continue
            parts = [part.strip() for part in line.strip("|").split("|")]
            if len(parts) != 6:
                continue
            try:
                existing_rows[int(parts[0])] = line
            except ValueError:
                continue

    existing_rows[int(step)] = _row_for_step(
        step=step,
        title=title,
        owner=owner,
        gate_test=gate_test,
        completed=now,
    )
    rows = [existing_rows[index] for index in sorted(existing_rows)]
    roadmap.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def write_snapshot(
    step: str,
    title: str,
    owner: str,
    reviewer: str,
    gate_test: str,
    master_suite: str,
    coverage: float,
) -> str:
    now = utc_now().isoformat()
    safe_time = now.replace(":", "-")
    name = f"step_{step}_{safe_time}.json"
    payload = {
        "step": int(step),
        "step_title": title,
        "completed_at": now,
        "owner": owner,
        "reviewer": reviewer,
        "files_snapshot": snapshot_files(),
        "test_results": {
            "gate_test": gate_test,
            "master_suite": master_suite,
            "coverage_pct": coverage,
        },
        "next_step": int(step) + 1,
    }
    out_path = BRAIN / "snapshots" / name
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update brain artifacts for the current step.")
    parser.add_argument("--step", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--gate-test", default="pending")
    parser.add_argument("--master-suite", default="pending")
    parser.add_argument("--coverage", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compile_master()
    snapshot_name = write_snapshot(
        args.step,
        args.title,
        args.owner,
        args.reviewer,
        args.gate_test,
        args.master_suite,
        args.coverage,
    )
    append_changelog(
        args.step,
        args.title,
        args.owner,
        args.reviewer,
        snapshot_name,
        args.gate_test,
        args.master_suite,
        args.coverage,
    )
    update_context(args.step, args.title, args.gate_test, args.master_suite, args.coverage)
    update_roadmap(args.step, args.title, args.owner, args.gate_test)


if __name__ == "__main__":
    main()
