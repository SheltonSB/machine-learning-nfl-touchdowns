"""
Configuration settings for the NFL AI/ML Platform
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "NFL AI/ML Platform"
    VERSION: str = "1.0.0"
    
    # CORS Settings
    ALLOWED_HOSTS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Database Settings
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/nfl_platform"
    REDIS_URL: str = "redis://localhost:6379"
    
    # JWT Settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ML Model Settings
    MODEL_PATH: str = "models/"
    XGBOOST_MODEL_PATH: str = "models/xgboost_model.pkl"
    TENSORFLOW_MODEL_PATH: str = "models/qb_td_model.keras"
    TENSORFLOW_SCALER_PATH: str = "models/feature_scaler.pkl"
    TENSORFLOW_METRICS_PATH: str = "models/training_metrics.json"
    PYTORCH_MODEL_PATH: str = "models/pytorch_model.pth"
    
    # RAG System Settings
    VECTOR_DB_URL: str = "your-pinecone-url"
    VECTOR_DB_API_KEY: str = "your-pinecone-api-key"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "microsoft/DialoGPT-medium"
    
    # Test / runtime toggles
    TEST_MODE: bool = False

    # AWS Settings (for production)
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET: Optional[str] = None
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
# Create settings instance
settings = Settings()
