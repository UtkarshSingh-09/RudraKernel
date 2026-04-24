"""FastAPI server scaffold for SIEGE Step 01."""

from __future__ import annotations

from fastapi import FastAPI

from siege_env.server.siege_environment import SIEGEEnvironment


app = FastAPI(title="SIEGE Environment", version="0.1.0")
env = SIEGEEnvironment(seed=7)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness endpoint used for local and container smoke tests."""
    return {"status": "ok"}


@app.get("/env/reset")
def reset() -> dict[str, object]:
    """Reset the minimal environment and return the starting observation."""
    observation = env.reset()
    return {"observation": observation.to_dict()}
