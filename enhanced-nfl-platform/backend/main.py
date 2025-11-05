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
from typing import Optional, Any
import random

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.api import api_router
from app.core.ml_pipeline import MLPipeline
from app.core.rag_system import RAGSystem


class StubMLPipeline:
    """Lightweight pipeline used when TEST_MODE=true."""

    feature_columns = [
        "age",
        "experience",
        "games_played",
        "targets",
        "touchdowns",
    ]

    async def initialize(self) -> None:  # parity with real pipeline
        return None

    async def predict(self, features: dict[str, Any], model_name: Optional[str] = None) -> dict[str, Any]:
        age = float(features.get("age", 0) or 0)
        experience = float(features.get("experience", 0) or 0)
        targets = float(features.get("targets", 0) or 0)
        touchdowns = float(features.get("touchdowns", 0) or 0)

        signal = (targets * 0.02) + (touchdowns * 0.1) + (experience * 0.05)
        base_confidence = 0.45 + min(0.4, signal)
        confidence = round(min(0.95, max(0.35, base_confidence + random.uniform(-0.05, 0.05))), 2)
        prediction = confidence >= 0.55

        return {
            "prediction": prediction,
            "confidence": confidence,
            "model_used": model_name or "stub-ensemble",
        }

    async def get_model_performance(self) -> dict[str, dict[str, float]]:
        return {
            "ensemble": {"accuracy": 0.91, "f1_score": 0.89},
            "xgboost": {"accuracy": 0.88, "f1_score": 0.86},
            "tensorflow": {"accuracy": 0.9, "f1_score": 0.87},
        }


class StubRAGSystem:
    """Minimal RAG system replacement for demo/test mode."""

    async def initialize(self) -> None:
        return None

    async def query(self, question: str) -> dict[str, Any]:
        return {
            "answer": "Stub response – RAG is disabled in test mode.",
            "sources": [],
            "question": question,
        }

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
        logger.info("Test mode enabled; using stub ML pipeline and RAG components.")
        stub_pipeline = StubMLPipeline()
        await stub_pipeline.initialize()
        stub_rag = StubRAGSystem()
        await stub_rag.initialize()
        app.state.ml_pipeline = stub_pipeline
        app.state.rag_system = stub_rag
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
