"""
NFL AI/ML Platform - FastAPI Backend
Advanced web application for NFL touchdown prediction with RAG system
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import uvicorn
import logging
import os
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.api import api_router
from app.core.ml_pipeline import MLPipeline
from app.core.rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_ENVS = [
    "DATABASE_URL",
    "EMBEDDING_MODEL",
]

MODEL_ARTIFACT_ENVS = {
    "xgboost": {"model": "XGBOOST_MODEL_PATH", "scaler": None},
    "tensorflow": {
        "model": "TENSORFLOW_MODEL_PATH",
        "scaler": "TENSORFLOW_SCALER_PATH",
    },
    "pytorch": {"model": "PYTORCH_MODEL_PATH", "scaler": None},
}

def _resolve_setting(name: str) -> Optional[str]:
    """Resolve an environment variable or fall back to settings."""

    value = os.getenv(name)
    if value:
        return value
    return getattr(settings, name, None)


def _check_required_env() -> None:
    missing = [key for key in REQUIRED_ENVS if not _resolve_setting(key)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _check_model_artifacts() -> None:
    critical_missing: list[str] = []
    optional_warnings: list[str] = []

    for name, env_keys in MODEL_ARTIFACT_ENVS.items():
        model_env = env_keys["model"]
        model_path_str = _resolve_setting(model_env)
        if not model_path_str:
            if name == "tensorflow":
                critical_missing.append(f"{model_env} (not configured)")
            else:
                optional_warnings.append(f"{name} model path not configured; skipping")
            continue

        model_path = Path(model_path_str)
        if not model_path.exists():
            if name == "tensorflow":
                critical_missing.append(str(model_path))
            else:
                optional_warnings.append(f"{name} model missing at {model_path}")

        scaler_env = env_keys.get("scaler")
        if not scaler_env:
            continue

        scaler_path_str = _resolve_setting(scaler_env)
        if not scaler_path_str:
            if name == "tensorflow":
                critical_missing.append(f"{scaler_env} (not configured)")
            else:
                optional_warnings.append(f"{name} scaler not configured")
            continue

        scaler_path = Path(scaler_path_str)
        if not scaler_path.exists():
            if name == "tensorflow":
                critical_missing.append(str(scaler_path))
            else:
                optional_warnings.append(f"{name} scaler missing at {scaler_path}")

    for message in optional_warnings:
        logger.info(message)

    if critical_missing and not settings.TEST_MODE:
        raise RuntimeError(
            "Missing TensorFlow model artifacts: " + ", ".join(critical_missing)
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting NFL AI/ML Platform...")
    
    # Initialize database
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")
    
    # Deployment guards
    try:
        _check_required_env()
        _check_model_artifacts()
    except Exception as e:
        logger.error(f"Startup guard failed: {e}")
        raise

    if settings.TEST_MODE:
        logger.info("Test mode enabled; skipping ML and RAG initialisation.")
        app.state.ml_pipeline = object()
        app.state.rag_system = object()
        yield
        logger.info("Test mode shutdown complete.")
        return
    
    # Initialize ML pipeline
    app.state.ml_pipeline = MLPipeline()
    await app.state.ml_pipeline.initialize()
    logger.info("ML Pipeline initialized")
    
    # Initialize RAG system
    app.state.rag_system = RAGSystem()
    await app.state.rag_system.initialize()
    logger.info("RAG System initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NFL AI/ML Platform...")

# Create FastAPI app
app = FastAPI(
    title="NFL AI/ML Platform",
    description="Advanced NFL touchdown prediction with AI/ML and RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ml_pipeline": "ready" if getattr(app.state, "ml_pipeline", None) else "loading",
        "rag_system": "ready" if getattr(app.state, "rag_system", None) else "loading"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NFL AI/ML Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Dependency to get ML pipeline
def get_ml_pipeline() -> MLPipeline:
    mlp = getattr(app.state, "ml_pipeline", None)
    if mlp is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Pipeline not initialized"
        )
    return mlp

# Dependency to get RAG system
def get_rag_system() -> RAGSystem:
    rag = getattr(app.state, "rag_system", None)
    if rag is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System not initialized"
        )
    return rag

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
