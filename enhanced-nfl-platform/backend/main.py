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

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.api import api_router
from app.core.ml_pipeline import MLPipeline
from app.core.rag_system import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for ML models and RAG system
ml_pipeline = None
rag_system = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global ml_pipeline, rag_system
    
    # Startup
    logger.info("Starting NFL AI/ML Platform...")
    
    # Initialize database
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")
    
    if settings.TEST_MODE:
        logger.info("Test mode enabled; skipping ML and RAG initialisation.")
        ml_pipeline = object()
        rag_system = object()
        yield
        logger.info("Test mode shutdown complete.")
        return
    
    # Initialize ML pipeline
    ml_pipeline = MLPipeline()
    await ml_pipeline.initialize()
    logger.info("ML Pipeline initialized")
    
    # Initialize RAG system
    rag_system = RAGSystem()
    await rag_system.initialize()
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
        "ml_pipeline": "ready" if ml_pipeline else "loading",
        "rag_system": "ready" if rag_system else "loading"
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
    if ml_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Pipeline not initialized"
        )
    return ml_pipeline

# Dependency to get RAG system
def get_rag_system() -> RAGSystem:
    if rag_system is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG System not initialized"
        )
    return rag_system

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
