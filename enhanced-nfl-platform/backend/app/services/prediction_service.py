"""
Prediction service for business logic
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.database import Prediction
from app.schemas.prediction import PredictionCreate, PredictionResponse, PredictionRequest
from app.core.ml_pipeline import MLPipeline
import anyio

class PredictionService:
    def __init__(self, db: Session, ml_pipeline: Optional[MLPipeline] = None):
        self.db = db
        self.ml_pipeline = ml_pipeline

    async def make_prediction(self, prediction_request: PredictionRequest) -> PredictionResponse:
        """Make a touchdown prediction"""
        if not self.ml_pipeline:
            raise ValueError("ML pipeline not available")
        
        # Validate feature completeness against pipeline requirements
        required_columns = getattr(self.ml_pipeline, "feature_columns", [])
        if required_columns:
            required = set(required_columns)
            provided = set(prediction_request.features.keys())
            missing = required - provided
            if missing:
                raise ValueError(
                    f"Missing required feature(s): {', '.join(sorted(missing))}"
                )
        
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
        
        db_prediction = Prediction(**prediction_data.model_dump())
        
        def _persist(pred: Prediction) -> Prediction:
            self.db.add(pred)
            self.db.commit()
            self.db.refresh(pred)
            return pred
        
        db_prediction = await anyio.to_thread.run_sync(_persist, db_prediction)
        
        return PredictionResponse.model_validate(db_prediction)

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
        return [PredictionResponse.model_validate(pred) for pred in predictions]

    async def get_prediction(self, prediction_id: int) -> Optional[PredictionResponse]:
        """Get a specific prediction"""
        prediction = self.db.query(Prediction).filter(Prediction.id == prediction_id).first()
        return PredictionResponse.model_validate(prediction) if prediction else None

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
            # TODO: Replace with real feature builder that pulls from DB
            # Build a complete feature vector with sensible defaults
            features = {col: 0.0 for col in self.ml_pipeline.feature_columns}
            # Optionally, override a few commonly available fields to avoid all-zeros
            for k, v in {
                "age": 28.0,
                "experience": 5.0,
            }.items():
                if k in features:
                    features[k] = v
            
            prediction_request = PredictionRequest(
                player_id=player_id,
                features=features
            )
            
            prediction = await self.make_prediction(prediction_request)
            predictions.append(prediction)
        
        return predictions

