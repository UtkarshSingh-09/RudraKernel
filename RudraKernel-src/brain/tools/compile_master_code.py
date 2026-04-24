#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "brain" / "MASTER_CODE.md"


def iter_python_files() -> list[Path]:
    candidates = []
    for base in [ROOT / "siege_env", ROOT / "tests", ROOT / "scripts"]:
        if base.exists():
            candidates.extend(sorted(base.rglob("*.py")))
    return candidates


def compile_master_code() -> str:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    files = iter_python_files()
    lines: list[str] = [
        f"# MASTER CODE - Last Updated: {now}",
        "",
        f"# Files Tracked: {len(files)}",
        "",
    ]
    for file_path in files:
        rel = file_path.relative_to(ROOT)
        mtime = datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc).replace(
            microsecond=0
        )
        lines.append(f"## {rel} (last modified: {mtime.isoformat()})")
        lines.append("```python")
        lines.append(file_path.read_text(encoding="utf-8"))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT.write_text(compile_master_code(), encoding="utf-8")


if __name__ == "__main__":
    main()