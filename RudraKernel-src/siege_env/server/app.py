"""FastAPI server scaffold for SIEGE Step 01."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from siege_env.models import SIEGEAction
from siege_env.server.siege_environment import SIEGEEnvironment

app = FastAPI(title="SIEGE Environment", version="0.1.0")
env = SIEGEEnvironment(seed=7)
training_lock = threading.Lock()
training_process: subprocess.Popen[bytes] | None = None
training_started_at: float | None = None
training_last_exit_code: int | None = None
project_root = Path(__file__).resolve().parents[2]


def _resolve_workspace_root() -> Path:
    """Find the directory that contains the training module."""
    candidates = [project_root, project_root / "RudraKernel-src"]
    for candidate in candidates:
        if (candidate / "training/grpo_train_unsloth.py").exists():
            return candidate
    return project_root


workspace_root = _resolve_workspace_root()
runtime_output_dir = Path(os.getenv("TRAIN_OUTPUT_DIR", "/tmp/rudra_unsloth"))
training_log_path = runtime_output_dir / "train.log"
training_metrics_path = runtime_output_dir / "metrics.json"
training_config_path = workspace_root / "training/configs/a100_grpo.yaml"
training_script_path = workspace_root / "scripts/run_training_a100.sh"


class StepRequest(BaseModel):
    action: SIEGEAction | dict


def _authorize_train(x_train_key: str | None) -> None:
    """Protect training endpoints when TRAIN_API_KEY is configured."""
    expected_key = os.getenv("TRAIN_API_KEY")
    if expected_key and x_train_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid training API key")


def _training_snapshot() -> dict[str, object]:
    """Build a status payload for the current training process state."""
    if training_process is None:
        return {
            "status": "idle",
            "pid": None,
            "started_at": None,
            "runtime_seconds": 0.0,
            "last_exit_code": training_last_exit_code,
            "log_path": str(training_log_path),
        }

    return_code = training_process.poll()
    if return_code is None:
        return {
            "status": "running",
            "pid": training_process.pid,
            "started_at": training_started_at,
            "runtime_seconds": round(time.time() - (training_started_at or time.time()), 2),
            "last_exit_code": training_last_exit_code,
            "log_path": str(training_log_path),
        }

    return {
        "status": "finished",
        "pid": training_process.pid,
        "started_at": training_started_at,
        "runtime_seconds": round(time.time() - (training_started_at or time.time()), 2),
        "last_exit_code": return_code,
        "log_path": str(training_log_path),
    }


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness endpoint used for local and container smoke tests."""
    return {"status": "ok"}


@app.get("/env/reset")
def reset() -> dict[str, object]:
    """Reset the minimal environment and return the starting observation."""
    observation = env.reset()
    return {"observation": observation.to_dict()}


@app.post("/env/step")
def step(payload: StepRequest) -> dict[str, object]:
    """Execute one environment step with a validated action payload."""
    observation, reward, done, info = env.step(payload.action)
    return {
        "observation": observation.to_dict(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/env/state")
def state() -> dict[str, object]:
    """Return the current internal environment state snapshot."""
    return {"state": env.state().to_dict()}


@app.post("/train/start")
def train_start(x_train_key: str | None = Header(default=None)) -> dict[str, object]:
    """Start A100 training in the background via the training shell script."""
    global training_process, training_started_at, training_last_exit_code
    try:
        _authorize_train(x_train_key)

        with training_lock:
            if training_process is not None and training_process.poll() is None:
                return _training_snapshot()

            if not training_script_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"Training script not found: {training_script_path}",
                )

            training_log_path.parent.mkdir(parents=True, exist_ok=True)
            with training_log_path.open("ab") as handle:
                child_env = os.environ.copy()
                child_env["TRAIN_OUTPUT_DIR"] = str(runtime_output_dir)
                training_process = subprocess.Popen(
                    ["bash", str(training_script_path)],
                    cwd=str(workspace_root),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=child_env,
                )

            training_started_at = time.time()
            training_last_exit_code = None
            return _training_snapshot()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - fallback for container/runtime edge cases
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start training process ({type(exc).__name__}): {exc}",
        ) from exc


@app.get("/train/status")
def train_status(x_train_key: str | None = Header(default=None)) -> dict[str, object]:
    """Return training process status (idle/running/finished)."""
    global training_last_exit_code
    _authorize_train(x_train_key)

    with training_lock:
        snapshot = _training_snapshot()
        if snapshot["status"] == "finished" and training_last_exit_code is None:
            training_last_exit_code = int(snapshot["last_exit_code"])
        return snapshot


@app.get("/train/logs")
def train_logs(lines: int = 200, x_train_key: str | None = Header(default=None)) -> dict[str, object]:
    """Return the last N lines from the training log file."""
    _authorize_train(x_train_key)
    safe_lines = max(1, min(lines, 2000))

    if not training_log_path.exists():
        return {"log_path": str(training_log_path), "lines": []}

    all_lines = training_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "log_path": str(training_log_path),
        "lines": all_lines[-safe_lines:],
    }


@app.get("/train/result")
def train_result(x_train_key: str | None = Header(default=None)) -> dict[str, object]:
    """Return parsed metrics from the latest completed training run."""
    _authorize_train(x_train_key)

    if not training_metrics_path.exists():
        return {
            "available": False,
            "metrics_path": str(training_metrics_path),
            "status": _training_snapshot(),
        }

    try:
        metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not parse metrics file at {training_metrics_path}: {exc}",
        ) from exc

    return {
        "available": True,
        "metrics_path": str(training_metrics_path),
        "metrics": metrics,
    }
