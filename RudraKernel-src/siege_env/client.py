"""Client wrapper for interacting with the SIEGE server.

This module provides SIEGEEnv, a small EnvClient-compatible wrapper used by
training/evaluation code. It supports both:
1) OpenEnv EnvClient inheritance when openenv is installed.
2) A local fallback implementation for environments without openenv.
"""

from __future__ import annotations

import json
from typing import Any
from urllib import request

from siege_env.models import SIEGEAction, SIEGEObservation, SIEGEState

try:
    from openenv import EnvClient
except Exception:  # pragma: no cover

    class EnvClient:  # type: ignore[no-redef]
        """Fallback base when openenv is not installed."""


class SIEGEEnv(EnvClient):
    """HTTP client for SIEGE reset/step/state endpoints."""

    def __init__(self, *, base_url: str = "http://127.0.0.1:8000", timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def reset(self) -> SIEGEObservation:
        with request.urlopen(f"{self._base_url}/env/reset", timeout=self._timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return SIEGEObservation.from_dict(payload["observation"])

    def step(
        self, action: SIEGEAction | dict[str, Any]
    ) -> tuple[SIEGEObservation, float, bool, dict[str, Any]]:
        action_payload = (
            action.model_dump(mode="json") if isinstance(action, SIEGEAction) else dict(action)
        )
        raw = json.dumps({"action": action_payload}).encode("utf-8")
        req = request.Request(
            f"{self._base_url}/env/step",
            data=raw,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self._timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        observation = SIEGEObservation.from_dict(payload["observation"])
        reward = float(payload["reward"])
        done = bool(payload["done"])
        info = dict(payload.get("info", {}))
        return observation, reward, done, info

    def state(self) -> SIEGEState:
        with request.urlopen(f"{self._base_url}/env/state", timeout=self._timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return SIEGEState.from_dict(payload["state"])
