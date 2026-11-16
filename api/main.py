"""FastAPI application entry point for ReviewLens.

Provides endpoints for review deception analysis and service health checks.
"""

from fastapi import FastAPI
from ml.config import MODEL_VERSION

app = FastAPI(
    title="ReviewLens API",
    description="AI service for detecting potentially fake product reviews.",
    version=MODEL_VERSION,
)


@app.get("/health")
def health_check() -> dict:
    """Return service health status and loaded model version."""
    return {"status": "ok", "model_version": MODEL_VERSION}

