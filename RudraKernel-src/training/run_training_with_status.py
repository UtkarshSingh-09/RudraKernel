"""Wrapper that runs GRPO training + a status HTTP server on port 8000.

HF Spaces requires a web server on app_port (8000) or it kills the container.
This script starts a simple status page so the Space stays alive while training.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

TRAINING_STATUS = {
    "state": "starting",
    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "message": "Initializing training...",
    "log_lines": [],
}
LOG_LOCK = threading.Lock()
MAX_LOG_LINES = 200


class StatusHandler(BaseHTTPRequestHandler):
    """Serves a live training status page on port 8000."""

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        with LOG_LOCK:
            state = TRAINING_STATUS["state"]
            message = TRAINING_STATUS["message"]
            logs = list(TRAINING_STATUS["log_lines"])
            started = TRAINING_STATUS["started_at"]

        color = {"starting": "#f0ad4e", "running": "#5bc0de", "done": "#5cb85c", "error": "#d9534f"}.get(state, "#999")
        logs_html = "\n".join(f"<div>{line}</div>" for line in logs[-100:])

        html = f"""<!DOCTYPE html>
<html><head>
<title>SIEGE GRPO Training</title>
<meta http-equiv="refresh" content="10">
<style>
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'Courier New', monospace; padding: 20px; }}
  h1 {{ color: #00d4ff; }}
  .status {{ padding: 10px; border-radius: 8px; background: #16213e; margin: 10px 0; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px; background: {color}; color: #fff; font-weight: bold; }}
  .logs {{ background: #0f0f23; padding: 15px; border-radius: 8px; max-height: 600px; overflow-y: auto; font-size: 13px; line-height: 1.6; }}
</style>
</head><body>
<h1>🛡️ SIEGE GRPO Training</h1>
<div class="status">
  <span class="badge">{state.upper()}</span>
  <span style="margin-left:10px">{message}</span>
  <br><small>Started: {started}</small>
</div>
<h3>Live Logs (last 100 lines):</h3>
<div class="logs">{logs_html if logs_html else "<em>Waiting for output...</em>"}</div>
</body></html>"""
        self.wfile.write(html.encode())

    def log_message(self, format: str, *args: object) -> None:
        pass  # Suppress HTTP access logs


def run_status_server() -> None:
    """Run status HTTP server on port 8000."""
    server = HTTPServer(("0.0.0.0", 8000), StatusHandler)
    print("[STATUS] Status server listening on http://0.0.0.0:8000", flush=True)
    server.serve_forever()


def run_training() -> None:
    """Run the GRPO training script and stream output."""
    global TRAINING_STATUS

    with LOG_LOCK:
        TRAINING_STATUS["state"] = "running"
        TRAINING_STATUS["message"] = "GRPO training in progress..."

    cmd = [
        sys.executable, "-u", "-m", "training.grpo_train_unsloth",
        "--config", "training/configs/a100_grpo_microcheck.yaml",
    ]

    print(f"[TRAIN] Running: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        print(line, flush=True)
        with LOG_LOCK:
            TRAINING_STATUS["log_lines"].append(line)
            if len(TRAINING_STATUS["log_lines"]) > MAX_LOG_LINES:
                TRAINING_STATUS["log_lines"] = TRAINING_STATUS["log_lines"][-MAX_LOG_LINES:]
            # Update message with latest meaningful line
            if "✓" in line or "INFO" in line or "loss" in line.lower():
                TRAINING_STATUS["message"] = line[:120]

    exit_code = proc.wait()

    with LOG_LOCK:
        if exit_code == 0:
            TRAINING_STATUS["state"] = "done"
            TRAINING_STATUS["message"] = "✓ Training completed successfully!"
        else:
            TRAINING_STATUS["state"] = "error"
            TRAINING_STATUS["message"] = f"Training failed with exit code {exit_code}"

    print(f"[TRAIN] Process exited with code {exit_code}", flush=True)


def main() -> None:
    print("=" * 60, flush=True)
    print("  SIEGE GRPO Training Runner", flush=True)
    print("  Status page: http://0.0.0.0:8000", flush=True)
    print("=" * 60, flush=True)

    # Start status server in background thread
    server_thread = threading.Thread(target=run_status_server, daemon=True)
    server_thread.start()

    # Give server a moment to bind
    time.sleep(1)

    # Run training (blocks until complete)
    run_training()

    # Keep server alive so judges can see results
    print("[STATUS] Training finished. Status page still available.", flush=True)
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
