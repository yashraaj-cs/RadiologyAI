"""FastAPI application for RadiologyAI Copilot."""

from fastapi import FastAPI

# Initialize FastAPI application
app = FastAPI(
    title="RadiologyAI Copilot API",
    description="Chest X-ray report-drafting assistant for radiologist review.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint confirming API service status."""
    return {"status": "ok"}
