"""
Prediction service for business logic
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.database import Prediction
from app.schemas.prediction import PredictionCreate, PredictionResponse, PredictionRequest
from app.core.ml_pipeline import MLPipeline

class PredictionService:
    def __init__(self, db: Session, ml_pipeline: Optional[MLPipeline] = None):
        self.db = db
        self.ml_pipeline = ml_pipeline

    async def make_prediction(self, prediction_request: PredictionRequest) -> PredictionResponse:
        """Make a touchdown prediction"""
        if not self.ml_pipeline:
            raise ValueError("ML pipeline not available")
        
        # Use ML pipeline to make prediction
        result = await self.ml_pipeline.predict(
            features=prediction_request.features,
            model_name=prediction_request.model_name
        )
        
        # Create prediction record
        prediction_data = PredictionCreate(
            player_id=prediction_request.player_id,
            prediction=bool(result['prediction']),
            confidence=result['confidence'],
            features_used=prediction_request.features,
            model_used=result['model_used'],
            created_by="api_user"
        )
        
        db_prediction = Prediction(**prediction_data.dict())
        self.db.add(db_prediction)
        self.db.commit()
        self.db.refresh(db_prediction)
        
        return PredictionResponse.from_orm(db_prediction)

    async def get_predictions(
        self, 
        player_id: Optional[int] = None, 
        limit: int = 100
    ) -> List[PredictionResponse]:
        """Get prediction history"""
        query = self.db.query(Prediction)
        
        if player_id:
            query = query.filter(Prediction.player_id == player_id)
        
        predictions = query.order_by(Prediction.created_at.desc()).limit(limit).all()
        return [PredictionResponse.from_orm(pred) for pred in predictions]

    async def get_prediction(self, prediction_id: int) -> Optional[PredictionResponse]:
        """Get a specific prediction"""
        prediction = self.db.query(Prediction).filter(Prediction.id == prediction_id).first()
        return PredictionResponse.from_orm(prediction) if prediction else None

    async def get_prediction_accuracy(self, player_id: int) -> Optional[dict]:
        """Get prediction accuracy for a specific player"""
        predictions = self.db.query(Prediction).filter(
            Prediction.player_id == player_id,
            Prediction.actual_result.isnot(None)
        ).all()
        
        if not predictions:
            return None
        
        total = len(predictions)
        correct = sum(1 for p in predictions if p.prediction == p.actual_result)
        accuracy = correct / total if total > 0 else 0
        
        return {
            "player_id": player_id,
            "total_predictions": total,
            "correct_predictions": correct,
            "accuracy": accuracy
        }

    async def make_batch_predictions(self, player_ids: List[int]) -> List[PredictionResponse]:
        """Make predictions for multiple players"""
        if not self.ml_pipeline:
            raise ValueError("ML pipeline not available")
        
        predictions = []
        for player_id in player_ids:
            # This would typically load player features from database
            # For now, using placeholder features
            features = {
                "passing_yards_roll3": 250.0,
                "td_passes_roll3": 1.5,
                "passes_attempted_roll3": 35.0,
                "age": 28,
                "experience": 5,
                "height": 74,
                "weight": 220
            }
            
            prediction_request = PredictionRequest(
                player_id=player_id,
                features=features
            )
            
            prediction = await self.make_prediction(prediction_request)
            predictions.append(prediction)
        
        return predictions

