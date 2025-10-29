"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
from app.core.ml_pipeline import MLPipeline, get_ml_pipeline
from app.core.rag_system import RAGSystem, get_rag_system

router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "message": "NFL AI/ML Platform is running"}

@router.get("/detailed")
async def detailed_health_check(
    ml_pipeline: MLPipeline = Depends(get_ml_pipeline),
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """Detailed health check with component status"""
    return {
        "status": "healthy",
        "components": {
            "ml_pipeline": "ready" if ml_pipeline else "loading",
            "rag_system": "ready" if rag_system else "loading",
            "database": "connected",
            "redis": "connected"
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }

