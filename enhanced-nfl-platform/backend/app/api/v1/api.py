"""
API v1 router configuration
"""

from fastapi import APIRouter
from app.api.v1.endpoints import players, predictions, analytics, rag, health

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(players.router, prefix="/players", tags=["players"])
api_router.include_router(predictions.router, prefix="/predictions", tags=["predictions"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])

