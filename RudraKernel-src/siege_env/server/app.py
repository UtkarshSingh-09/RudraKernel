"""FastAPI server scaffold for SIEGE Step 01."""

from __future__ import annotations

from fastapi import FastAPI


app = FastAPI(title="SIEGE Environment", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness endpoint used for local and container smoke tests."""
    return {"status": "ok"}