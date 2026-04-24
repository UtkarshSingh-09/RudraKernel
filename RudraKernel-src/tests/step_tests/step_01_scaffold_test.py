from __future__ import annotations

import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    info = subprocess.run(
        ["docker", "info"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return info.returncode == 0


def test_import_and_health_route() -> None:
    from siege_env.server.app import app

    routes = {route.path for route in app.routes}
    assert "/health" in routes


def test_openenv_manifest_has_required_keys() -> None:
    manifest = ROOT / "openenv.yaml"
    assert manifest.exists()
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    assert data["name"] == "siege_env"
    assert data["runtime"]["entrypoint"] == "siege_env.server.app:app"
    assert data["runtime"]["healthcheck"] == "/health"


@pytest.mark.skipif(not _docker_ready(), reason="Docker daemon not available")
def test_docker_build_succeeds() -> None:
    result = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            "siege_env/server/Dockerfile",
            "-t",
            "siege-step01-test",
            ".",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr


@pytest.mark.skipif(not _docker_ready(), reason="Docker daemon not available")
def test_container_health_endpoint() -> None:
    port = _free_port()
    run = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-d",
            "-p",
            f"{port}:8000",
            "siege-step01-test",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + "\n" + run.stderr
    container_id = run.stdout.strip()
    try:
        url = f"http://127.0.0.1:{port}/health"
        deadline = time.time() + 20
        last_error = ""
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    body = response.read().decode("utf-8")
                    assert response.status == 200
                    assert "ok" in body
                    return
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                last_error = str(exc)
                time.sleep(0.5)
        raise AssertionError(f"Container health endpoint did not become ready: {last_error}")
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
