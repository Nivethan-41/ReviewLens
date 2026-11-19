"""FastAPI application entry point for ReviewLens.

Provides endpoints for review deception analysis and service health checks,
with CORS support, lifespan model management, and request timing instrumentation.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
)
from ml.config import (
    CORS_ALLOWED_ORIGIN_REGEX,
    CORS_ALLOWED_ORIGINS,
    MODEL_PATH,
    MODEL_VERSION,
)
from ml.predict import ModelService

# Configure logger
logger = logging.getLogger("api.main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle by initializing ModelService once at startup.

    Args:
        app: FastAPI application instance.
    """
    try:
        app.state.model_service = ModelService(MODEL_PATH)
        logger.info("ModelService initialized successfully with artifact at '%s'.", MODEL_PATH)
    except FileNotFoundError as exc:
        logger.warning(
            "Model artifact not found at startup (%s). "
            "Inference requests will return 503 until `make train` is executed.",
            exc,
        )
        app.state.model_service = None
    except Exception as exc:
        logger.error("Failed to initialize ModelService at startup: %s", exc)
        app.state.model_service = None

    yield


app = FastAPI(
    title="ReviewLens API",
    description="AI service for detecting potentially fake and deceptive product reviews.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_origin_regex=CORS_ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Middleware to measure and log HTTP request duration in milliseconds.

    Args:
        request: Incoming FastAPI Request.
        call_next: Request handler callback.

    Returns:
        Response from downstream handler.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    logger.info(
        "%s %s finished with status %d in %.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
)
def health_check() -> HealthResponse:
    """Return service health status and loaded model version."""
    return HealthResponse(status="ok", model_version=MODEL_VERSION)


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze reviews for deception and calculate reliability score",
)
def analyze_reviews(payload: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    """Analyze a collection of reviews for deceptive patterns and compute reliability score.

    Args:
        payload: Validated AnalyzeRequest containing list of reviews.
        request: FastAPI Request instance providing access to app.state.

    Returns:
        AnalyzeResponse with review predictions, product summary, and processing duration.

    Raises:
        HTTPException: 503 if model is missing, 422 if review validation fails.
    """
    start_time = time.perf_counter()

    service: Optional[ModelService] = getattr(request.app.state, "model_service", None)
    if service is None:
        # Check if model was trained after startup
        if MODEL_PATH.exists():
            try:
                service = ModelService(MODEL_PATH)
                request.app.state.model_service = service
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Model artifact failed to load: {exc}. Please run `make train`.",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model file is missing. Please run `make train` to train the model.",
            )

    try:
        raw_reviews = [
            {
                "id": rev.id,
                "text": rev.text,
                "rating": rev.rating,
                "date": rev.date,
                "has_images": getattr(rev, "has_images", False),
                "has_video": getattr(rev, "has_video", False),
                "verified_purchase": getattr(rev, "verified_purchase", False),
            }
            for rev in payload.reviews
        ]
        result = service.analyze(raw_reviews, seller_info=getattr(payload, "seller_info", None))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
            if hasattr(status, "HTTP_422_UNPROCESSABLE_ENTITY")
            else 422,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Inference error during review analysis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference error: {exc}",
        )

    processing_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    logger.info(
        "Analysis complete: evaluated %d reviews in %.2f ms (page_url=%s)",
        len(payload.reviews),
        processing_ms,
        payload.page_url,
    )

    return AnalyzeResponse(
        results=result["results"],
        summary=result["summary"],
        processing_ms=processing_ms,
    )
