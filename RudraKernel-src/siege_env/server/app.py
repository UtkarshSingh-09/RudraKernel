"""FastAPI server scaffold for SIEGE Step 01."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from siege_env.models import SIEGEAction
from siege_env.server.siege_environment import SIEGEEnvironment

app = FastAPI(title="SIEGE Environment", version="0.1.0")
env = SIEGEEnvironment(seed=7)


class StepRequest(BaseModel):
    action: SIEGEAction | dict


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
