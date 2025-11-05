"""
Prediction Pydantic schemas
"""

from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    player_id: int
    features: Dict[str, Any]
    model_name: Optional[str] = "ensemble"

class PredictionCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    player_id: int
    game_id: Optional[int] = None
    prediction: bool
    confidence: float
    features_used: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None
    created_by: Optional[str] = None

class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
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

