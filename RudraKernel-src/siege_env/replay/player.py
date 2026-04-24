"""Replay CLI player for SIEGE Step 21."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def replay_file(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay SIEGE JSONL trajectory")
    parser.add_argument("--file", required=True, help="Path to replay jsonl")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    events = replay_file(Path(args.file))
    print(f"replay_events={len(events)}")


if __name__ == "__main__":
    main()
