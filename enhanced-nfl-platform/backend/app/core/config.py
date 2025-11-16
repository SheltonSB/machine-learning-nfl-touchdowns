"""
Configuration settings for the NFL AI/ML Platform
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os
from urllib.parse import quote_plus

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
    # For AWS RDS: Use format: postgresql://user:password@rds-endpoint:5432/dbname
    # For AWS RDS MySQL: Use format: mysql+pymysql://user:password@rds-endpoint:3306/dbname
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/nfl_platform"
    
    # AWS RDS specific settings (optional, can be used instead of DATABASE_URL)
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_NAME: Optional[str] = None
    DB_ENGINE: str = "postgresql"  # postgresql or mysql
    DB_USE_SSL: bool = True  # Use SSL for AWS RDS connections
    
    # Legacy MySQL settings (for backward compatibility)
    MYSQL_HOST: Optional[str] = None
    MYSQL_PORT: Optional[int] = None
    MYSQL_USER: Optional[str] = None
    MYSQL_PASSWORD: Optional[str] = None
    MYSQL_DATABASE: Optional[str] = None
    
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
    
    # AWS RDS Settings
    RDS_ENDPOINT: Optional[str] = None  # RDS instance endpoint
    RDS_CA_CERT_PATH: Optional[str] = None  # Path to RDS CA certificate (optional)
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    def get_database_url(self) -> str:
        """
        Get database URL, constructing it from individual components if DATABASE_URL is not set.
        Prioritizes DB_* settings, then MYSQL_* settings, then RDS_ENDPOINT, then DATABASE_URL.
        """
        # If individual DB components are provided, construct URL
        if self.DB_HOST and self.DB_USER and self.DB_PASSWORD and self.DB_NAME:
            port = self.DB_PORT or (5432 if self.DB_ENGINE == "postgresql" else 3306)
            engine = self.DB_ENGINE
            if engine == "mysql":
                engine = "mysql+pymysql"
            # URL encode password to handle special characters
            encoded_password = quote_plus(self.DB_PASSWORD)
            return f"{engine}://{self.DB_USER}:{encoded_password}@{self.DB_HOST}:{port}/{self.DB_NAME}"
        
        # Fallback to legacy MySQL settings
        if self.MYSQL_HOST and self.MYSQL_USER and self.MYSQL_PASSWORD and self.MYSQL_DATABASE:
            port = self.MYSQL_PORT or 3306
            encoded_password = quote_plus(self.MYSQL_PASSWORD)
            return f"mysql+pymysql://{self.MYSQL_USER}:{encoded_password}@{self.MYSQL_HOST}:{port}/{self.MYSQL_DATABASE}"
        
        # Use RDS_ENDPOINT if provided (assumes PostgreSQL unless specified)
        if self.RDS_ENDPOINT and self.DB_USER and self.DB_PASSWORD and self.DB_NAME:
            port = self.DB_PORT or 5432
            engine = self.DB_ENGINE
            if engine == "mysql":
                engine = "mysql+pymysql"
            encoded_password = quote_plus(self.DB_PASSWORD)
            return f"{engine}://{self.DB_USER}:{encoded_password}@{self.RDS_ENDPOINT}:{port}/{self.DB_NAME}"
        
        # Default to DATABASE_URL
        return self.DATABASE_URL
    
    def get_database_connect_args(self) -> dict:
        """Get database connection arguments, including SSL for AWS RDS"""
        connect_args = {}
        database_url = self.get_database_url()
        
        # For SQLite
        if "sqlite" in database_url.lower():
            connect_args["check_same_thread"] = False
            return connect_args
        
        # Check if this is an RDS connection
        is_rds = (
            self.RDS_ENDPOINT is not None or
            (self.DB_HOST and "rds.amazonaws.com" in self.DB_HOST.lower()) or
            "rds.amazonaws.com" in database_url.lower()
        )
        
        # Enable SSL for RDS connections if DB_USE_SSL is True
        if self.DB_USE_SSL and is_rds:
            if "postgresql" in database_url.lower():
                # PostgreSQL SSL configuration
                connect_args["sslmode"] = "require"
                if self.RDS_CA_CERT_PATH and os.path.exists(self.RDS_CA_CERT_PATH):
                    connect_args["sslcert"] = self.RDS_CA_CERT_PATH
                    connect_args["sslrootcert"] = self.RDS_CA_CERT_PATH
            elif "mysql" in database_url.lower():
                # MySQL SSL configuration for PyMySQL
                ssl_config = {}
                if self.RDS_CA_CERT_PATH and os.path.exists(self.RDS_CA_CERT_PATH):
                    ssl_config["ssl_ca"] = self.RDS_CA_CERT_PATH
                # For PyMySQL, SSL is enabled when ssl dictionary is provided
                # Empty dict enables SSL with default settings
                connect_args["ssl"] = ssl_config
        
        return connect_args
    
# Create settings instance
settings = Settings()
