"""Unit tests for OpenEnv-facing API contracts."""

from __future__ import annotations

from siege_env.client import SIEGEEnv
from siege_env.server.app import app, reset, state


def test_server_exposes_core_routes() -> None:
    routes = {route.path for route in app.routes}
    assert "/health" in routes
    assert "/env/reset" in routes
    assert "/env/step" in routes
    assert "/env/state" in routes
    assert "/train/start" in routes
    assert "/train/status" in routes
    assert "/train/logs" in routes
    assert "/train/result" in routes


def test_reset_and_state_payload_shape() -> None:
    reset_payload = reset()
    assert "observation" in reset_payload

    state_payload = state()
    assert "state" in state_payload
    assert "episode_id" in state_payload["state"]


def test_client_class_exists_with_openenv_compatible_shape() -> None:
    env = SIEGEEnv(base_url="http://127.0.0.1:8000")
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "state")
