"""
Predictions API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.core.ml_pipeline import MLPipeline, get_ml_pipeline
from app.schemas.prediction import PredictionCreate, PredictionResponse, PredictionRequest
from app.services.prediction_service import PredictionService

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
async def make_prediction(
    prediction_request: PredictionRequest,
    db: Session = Depends(get_db),
    ml_pipeline: MLPipeline = Depends(get_ml_pipeline)
):
    """Make a touchdown prediction for a player"""
    service = PredictionService(db, ml_pipeline)
    return await service.make_prediction(prediction_request)

@router.get("/", response_model=List[PredictionResponse])
async def get_predictions(
    player_id: Optional[int] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get prediction history"""
    service = PredictionService(db)
    return await service.get_predictions(player_id=player_id, limit=limit)

@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get a specific prediction"""
    service = PredictionService(db)
    prediction = await service.get_prediction(prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction

@router.get("/player/{player_id}/accuracy")
async def get_prediction_accuracy(player_id: int, db: Session = Depends(get_db)):
    """Get prediction accuracy for a specific player"""
    service = PredictionService(db)
    accuracy = await service.get_prediction_accuracy(player_id)
    if accuracy is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return accuracy

@router.get("/model/performance")
async def get_model_performance(ml_pipeline: MLPipeline = Depends(get_ml_pipeline)):
    """Get performance metrics for all ML models"""
    return await ml_pipeline.get_model_performance()

@router.post("/batch")
async def make_batch_predictions(
    player_ids: List[int],
    db: Session = Depends(get_db),
    ml_pipeline: MLPipeline = Depends(get_ml_pipeline)
):
    """Make predictions for multiple players"""
    service = PredictionService(db, ml_pipeline)
    return await service.make_batch_predictions(player_ids)

