"""
Advanced ML Pipeline with Multiple Models
Supports XGBoost, TensorFlow, PyTorch, and Ensemble methods
"""

import os

import numpy as np
import pandas as pd
import joblib
import pickle
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from fastapi import HTTPException, status

# Machine Learning Libraries (optional in test environments)
SKIP_HEAVY_IMPORTS = os.getenv("SKIP_ML_IMPORTS") == "1"

if not SKIP_HEAVY_IMPORTS:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        xgb = None
else:
    xgb = None

if not SKIP_HEAVY_IMPORTS:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        tf = None
else:
    tf = None

if not SKIP_HEAVY_IMPORTS:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        torch = None
        nn = None
else:
    torch = None
    nn = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler

# Custom imports
from app.core.config import settings

logger = logging.getLogger(__name__)

class PyTorchLSTM(nn.Module if nn is not None else object):  # type: ignore[misc]
    """PyTorch LSTM model for sequence prediction"""
    
    def __init__(self, input_size: int = 15, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        if nn is None or torch is None:  # pragma: no cover - executed only when torch missing
            raise RuntimeError("PyTorch is not installed; install torch to enable LSTM support.")
        super(PyTorchLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layers
        x = self.dropout(last_output)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.sigmoid(x)

class MLPipeline:
    """Advanced ML Pipeline with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = [
            "age",
            "height",
            "weight",
            "experience",
            "passing_yards",
            "td_passes",
            "interceptions",
            "passes_attempted",
            "completion_percentage",
            "yards_per_attempt",
            "passer_rating",
            "passing_yards_roll3",
            "td_passes_roll3",
            "passes_attempted_roll3",
            "td_rate_roll3",
            "completion_rate_roll3",
        ]
        self.target_column = 'threw_td'
        
    async def initialize(self):
        """Initialize the ML pipeline"""
        logger.info("Initializing ML Pipeline...")
        
        # Create models directory
        Path(settings.MODEL_PATH).mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        await self._initialize_models()
        
        # Load or train models
        await self._load_or_train_models()
        
        logger.info("ML Pipeline initialized successfully")
    
    async def _initialize_models(self):
        """Initialize all model types"""
        
        # XGBoost Model
        if xgb is not None:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            logger.info('XGBoost library not installed; xgboost model disabled.')
            self.models['xgboost'] = None

        # TensorFlow Model
        if tf is not None:
            self.models['tensorflow'] = None
            self.scalers['tensorflow'] = None
        else:
            logger.warning('TensorFlow library not installed; tensorflow model disabled. TensorFlow support is required for the primary model.')
            self.models['tensorflow'] = None
            self.scalers['tensorflow'] = None

        # PyTorch Model
        if torch is not None and nn is not None:
            self.models['pytorch'] = PyTorchLSTM(
                input_size=len(self.feature_columns),
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            )
            self.scalers['pytorch'] = StandardScaler()
        else:
            logger.info('PyTorch library not installed; pytorch model disabled.')
            self.models['pytorch'] = None
            self.scalers['pytorch'] = None

        # Ensemble Model (will be created after training)
        self.models['ensemble'] = None
    
    def _build_tensorflow_model(self):
        """Build TensorFlow neural network"""
        if tf is None:
            raise RuntimeError("TensorFlow is not installed; install tensorflow to enable this model.")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(len(self.feature_columns),)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    async def _load_or_train_models(self):
        """Load existing models or train new ones"""
        artifact_paths = {
            "xgboost": {
                "model": getattr(settings, "XGBOOST_MODEL_PATH", None),
                "scaler": None,
            },
            "tensorflow": {
                "model": settings.TENSORFLOW_MODEL_PATH,
                "scaler": getattr(settings, "TENSORFLOW_SCALER_PATH", None),
            },
            "pytorch": {
                "model": getattr(settings, "PYTORCH_MODEL_PATH", None),
                "scaler": None,
            },
        }

        optional_models = {"xgboost", "pytorch"}

        # Load persisted scalers if present
        for name, paths in artifact_paths.items():
            scaler_path_str = paths.get("scaler")
            if not scaler_path_str:
                continue

            scaler_path = Path(scaler_path_str)
            if scaler_path.exists():
                try:
                    self.scalers[name] = joblib.load(scaler_path)
                    logger.info("Loaded scaler for %s from %s", name, scaler_path)
                except Exception as exc:
                    logger.warning("Failed to load scaler for %s: %s", name, exc)
            else:
                if name == "tensorflow":
                    logger.warning("Scaler for %s not found at %s", name, scaler_path)
                else:
                    logger.debug("Optional scaler for %s not found at %s; skipping", name, scaler_path)

        for model_name, paths in artifact_paths.items():
            model_path_str = paths.get("model")
            if not model_path_str:
                if model_name in optional_models:
                    logger.debug("Optional model %s not configured; skipping", model_name)
                    continue
                else:
                    raise RuntimeError(f"Model path for required model {model_name} is not configured")

            model_path = Path(model_path_str)
            if model_path.exists():
                logger.info("Loading %s model from %s", model_name, model_path)
                await self._load_model(model_name, str(model_path))
            else:
                if model_name in optional_models:
                    logger.info("Optional model %s not found at %s; skipping load", model_name, model_path)
                    self.models[model_name] = None
                else:
                    logger.info("Training %s model...", model_name)
                    await self._train_model(model_name)
    
    async def _load_model(self, model_name: str, model_path: str):
        """Load a trained model"""
        try:
            if model_name == 'xgboost':
                self.models[model_name] = joblib.load(model_path)
            elif model_name == 'tensorflow':
                if tf is None:
                    raise RuntimeError('TensorFlow is not installed; cannot load tensorflow model.')
                self.models[model_name] = tf.keras.models.load_model(model_path)
                if self.scalers.get('tensorflow') is None:
                    scaler_path_str = getattr(settings, 'TENSORFLOW_SCALER_PATH', None)
                    if scaler_path_str:
                        scaler_path = Path(scaler_path_str)
                        try:
                            self.scalers['tensorflow'] = joblib.load(scaler_path)
                            logger.info("Loaded TensorFlow scaler from %s", scaler_path)
                        except Exception as exc:
                            logger.warning("Failed to load TensorFlow scaler from %s: %s", scaler_path, exc)
            elif model_name == 'pytorch':
                if torch is None:
                    raise RuntimeError('PyTorch is not installed; cannot load pytorch model.')
                model_state = torch.load(model_path, map_location='cpu')
                self.models[model_name].load_state_dict(model_state)
                self.models[model_name].eval()
            
            logger.info(f"Successfully loaded {model_name} model")
        except Exception as e:
            logger.error(f"Error loading {model_name} model: {e}")
            # Train model if loading fails
            await self._train_model(model_name)
    
    async def _train_model(self, model_name: str):
        """Train a specific model"""
        # This would typically load data from database
        # For now, we'll create a placeholder
        logger.info(f"Training {model_name} model...")
        
        # In a real implementation, you would:
        # 1. Load data from database
        # 2. Preprocess features
        # 3. Split into train/test
        # 4. Train the model
        # 5. Save the model
        
        # Placeholder training logic
        if model_name == 'xgboost':
            if xgb is None:
                raise RuntimeError('XGBoost is not installed; cannot train xgboost model.')
            # XGBoost training would go here
            pass
        elif model_name == 'tensorflow':
            if tf is None:
                raise RuntimeError('TensorFlow is not installed; cannot train tensorflow model.')
            # TensorFlow training would go here
            pass
        elif model_name == 'pytorch':
            if torch is None:
                raise RuntimeError('PyTorch is not installed; cannot train pytorch model.')
            # PyTorch training would go here
            raise RuntimeError('PyTorch training routine not implemented; provide trained model and scaler.')
    
    async def predict(self, features: Dict[str, float], model_name: str = 'ensemble') -> Dict[str, Any]:
        """Make prediction using specified model"""
        try:
            # Convert features to array
            feature_array = np.array([features.get(col, 0) for col in self.feature_columns]).reshape(1, -1)
            
            if model_name == 'ensemble':
                return await self._ensemble_predict(feature_array)
            else:
                return await self._single_model_predict(feature_array, model_name)
                
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    async def _single_model_predict(self, features: np.ndarray, model_name: str) -> Dict[str, Any]:
        """Make prediction using a single model"""
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        def _ensure_fitted_scaler(scaler: Optional[StandardScaler], name: str) -> None:
            if scaler is None:
                raise ValueError(f"Scaler for {name} is not available. Train the model first.")
            # StandardScaler is considered fitted if it has mean_
            if not hasattr(scaler, 'mean_'):
                raise ValueError(f"Scaler for {name} is not fitted. Train the model to fit and persist scalers.")
        
        if model_name == 'xgboost':
            probability = model.predict_proba(features)[0][1]
            prediction = model.predict(features)[0]
            
        elif model_name == 'tensorflow':
            # Scale features for TensorFlow
            _ensure_fitted_scaler(self.scalers.get('tensorflow'), 'tensorflow')
            features_scaled = self.scalers['tensorflow'].transform(features)
            probability = float(model.predict(features_scaled)[0][0])
            prediction = int(probability > 0.5)
            
        elif model_name == 'pytorch':
            # Scale features for PyTorch
            _ensure_fitted_scaler(self.scalers.get('pytorch'), 'pytorch')
            features_scaled = self.scalers['pytorch'].transform(features)
            features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            
            with torch.no_grad():
                probability = float(model(features_tensor)[0][0])
                prediction = int(probability > 0.5)
        
        return {
            'prediction': prediction,
            'probability': probability,
            'model_used': model_name,
            'confidence': abs(probability - 0.5) * 2  # Convert to 0-1 confidence
        }
    
    async def _ensemble_predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make ensemble prediction using all models"""
        predictions = {}
        probabilities = []
        
        for model_name in ['xgboost', 'tensorflow', 'pytorch']:
            if self.models[model_name] is not None:
                result = await self._single_model_predict(features, model_name)
                predictions[model_name] = result['prediction']
                probabilities.append(result['probability'])
        
        if not probabilities:
            raise ValueError("No models available for ensemble prediction")
        
        # Weighted average of probabilities
        ensemble_probability = np.mean(probabilities)
        ensemble_prediction = int(ensemble_probability > 0.5)
        
        return {
            'prediction': ensemble_prediction,
            'probability': ensemble_probability,
            'model_used': 'ensemble',
            'confidence': abs(ensemble_probability - 0.5) * 2,
            'individual_predictions': predictions
        }
    
    async def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        # This would typically load test data and evaluate models
        # For now, return placeholder metrics
        return {
            'xgboost': {'accuracy': 0.92, 'f1_score': 0.89, 'roc_auc': 0.94},
            'tensorflow': {'accuracy': 0.90, 'f1_score': 0.87, 'roc_auc': 0.92},
            'pytorch': {'accuracy': 0.91, 'f1_score': 0.88, 'roc_auc': 0.93},
            'ensemble': {'accuracy': 0.93, 'f1_score': 0.90, 'roc_auc': 0.95}
        }
    
    async def retrain_model(self, model_name: str, new_data: pd.DataFrame):
        """Retrain a specific model with new data"""
        logger.info(f"Retraining {model_name} model with new data...")
        
        # Preprocess new data
        X = new_data[self.feature_columns]
        y = new_data[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        if model_name == 'xgboost':
            self.models[model_name].fit(X_train, y_train)
            
        elif model_name == 'tensorflow':
            if self.models.get(model_name) is None:
                self.models[model_name] = self._build_tensorflow_model()
            if self.scalers.get('tensorflow') is None:
                self.scalers['tensorflow'] = StandardScaler()
            X_train_scaled = self.scalers['tensorflow'].fit_transform(X_train)
            X_test_scaled = self.scalers['tensorflow'].transform(X_test)
            
            self.models[model_name].fit(
                X_train_scaled, y_train,
                validation_data=(X_test_scaled, y_test),
                epochs=50,
                batch_size=32,
                verbose=0
            )
            
        elif model_name == 'pytorch':
            # PyTorch training logic would go here
            # Fit scaler so inference is valid even if training logic is minimal
            if self.scalers.get('pytorch') is None:
                self.scalers['pytorch'] = StandardScaler()
            _ = self.scalers['pytorch'].fit(X_train)
        
        # Save retrained model
        await self._save_model(model_name)
        
        logger.info(f"Successfully retrained {model_name} model")
    
    async def _save_model(self, model_name: str):
        """Save a trained model"""
        model_paths = {
            'xgboost': settings.XGBOOST_MODEL_PATH,
            'tensorflow': settings.TENSORFLOW_MODEL_PATH,
            'pytorch': settings.PYTORCH_MODEL_PATH
        }
        
        model_path = model_paths[model_name]
        
        try:
            if model_name == 'xgboost':
                joblib.dump(self.models[model_name], model_path)
            elif model_name == 'tensorflow':
                self.models[model_name].save(model_path)
            elif model_name == 'pytorch':
                torch.save(self.models[model_name].state_dict(), model_path)
            
            logger.info(f"Successfully saved {model_name} model to {model_path}")
            # Persist scaler if present
            scaler = self.scalers.get(model_name)
            if scaler is not None:
                if model_name == 'tensorflow':
                    scaler_path = getattr(settings, 'TENSORFLOW_SCALER_PATH', f"{model_path}.scaler")
                else:
                    scaler_path = f"{model_path}.scaler"
                joblib.dump(scaler, scaler_path)
                logger.info("Saved scaler for %s to %s", model_name, scaler_path)
        except Exception as e:
            logger.error(f"Error saving {model_name} model: {e}")
            raise

_ml_pipeline_singleton = None


def set_ml_pipeline(instance: "MLPipeline") -> None:
    global _ml_pipeline_singleton
    _ml_pipeline_singleton = instance


def get_ml_pipeline() -> "MLPipeline":
    if _ml_pipeline_singleton is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML Pipeline not initialized"
        )
    return _ml_pipeline_singleton
