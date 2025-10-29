"""
Prediction Pydantic schemas
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    player_id: int
    features: Dict[str, Any]
    model_name: Optional[str] = "ensemble"

class PredictionCreate(BaseModel):
    player_id: int
    game_id: Optional[int] = None
    prediction: bool
    confidence: float
    features_used: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    created_by: Optional[str] = None

class PredictionResponse(BaseModel):
    id: int
    player_id: int
    game_id: Optional[int] = None
    prediction: bool
    confidence: float
    features_used: Optional[Dict[str, Any]] = None
    actual_result: Optional[bool] = None
    model_used: Optional[str] = None
    created_at: datetime
    created_by: Optional[str] = None

    class Config:
        from_attributes = True

