"""
NFL QB Touchdown Predictor - Configuration

Centralized configuration for the project.

Author: Shelton Bumhe
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"
APP_DIR = PROJECT_ROOT / "app"

# Database configuration
DATABASE_NAME = "nfl_data.db"
DATABASE_PATH = PROJECT_ROOT / DATABASE_NAME

# Data processing configuration
CHUNK_SIZE = 1000  # Number of rows to process at once when loading CSV files
ROLLING_WINDOW = 3  # Number of games to include in rolling averages

# Model configuration
MODEL_FILENAME = "qb_td_model.pkl"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME

# Feature configuration
FEATURE_COLUMNS = [
    # Player characteristics
    'age', 'height', 'weight', 'experience',
    
    # Current game stats
    'passing_yards', 'td_passes', 'interceptions', 'passes_attempted',
    'completion_percentage', 'yards_per_attempt', 'passer_rating',
    
    # Rolling averages (excluding current game)
    'passing_yards_roll3', 'td_passes_roll3', 'passes_attempted_roll3',
    'td_rate_roll3', 'completion_rate_roll3',
    
    # Target variable
    'threw_td'
]

# Required CSV files
REQUIRED_CSV_FILES = [
    "Basic_Stats.csv",
    "Game_Logs_Quarterback.csv", 
    "Career_Stats_Passing.csv"
]

# Streamlit app configuration
STREAMLIT_CONFIG = {
    "page_title": "🏈 NFL QB Touchdown Predictor",
    "page_icon": "🏈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Validation thresholds
VALIDATION_THRESHOLDS = {
    "min_year": 2000,
    "max_year": 2030,
    "min_week": 1,
    "max_week": 18,
    "min_passing_yards": 0,
    "max_passing_yards": 600,
    "min_td_passes": 0,
    "max_td_passes": 10,
    "min_age": 18,
    "max_age": 50,
    "min_height": 60,  # inches
    "max_height": 85,  # inches
    "min_weight": 150,  # lbs
    "max_weight": 350,  # lbs
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}

# Database table schemas
TABLE_SCHEMAS = {
    "basic_stats": """
        CREATE TABLE IF NOT EXISTS basic_stats (
            player_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            height REAL,
            weight REAL,
            experience REAL,
            position TEXT
        )
    """,
    
    "game_logs": """
        CREATE TABLE IF NOT EXISTS game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            name TEXT,
            year INTEGER,
            week INTEGER,
            season TEXT,
            team TEXT,
            opponent TEXT,
            position TEXT,
            game_date TEXT,
            home_away TEXT,
            result TEXT,
            FOREIGN KEY (player_id) REFERENCES basic_stats (player_id)
        )
    """,
    
    "qb_stats": """
        CREATE TABLE IF NOT EXISTS qb_stats (
            game_log_id INTEGER PRIMARY KEY,
            passing_yards INTEGER,
            td_passes INTEGER,
            interceptions INTEGER,
            passes_attempted INTEGER,
            passes_completed INTEGER,
            completion_percentage REAL,
            yards_per_attempt REAL,
            passer_rating REAL,
            threw_td INTEGER,
            FOREIGN KEY (game_log_id) REFERENCES game_logs (id)
        )
    """,
    
    "career_stats": """
        CREATE TABLE IF NOT EXISTS career_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            name TEXT,
            year INTEGER,
            team TEXT,
            position TEXT,
            games_played INTEGER,
            games_started INTEGER,
            FOREIGN KEY (player_id) REFERENCES basic_stats (player_id)
        )
    """,
    
    "qb_career_passing": """
        CREATE TABLE IF NOT EXISTS qb_career_passing (
            career_id INTEGER PRIMARY KEY,
            passing_yards INTEGER,
            td_passes INTEGER,
            interceptions INTEGER,
            attempts INTEGER,
            completions INTEGER,
            completion_percentage REAL,
            yards_per_attempt REAL,
            passer_rating REAL,
            FOREIGN KEY (career_id) REFERENCES career_stats (id)
        )
    """,
    
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id TEXT,
            game_date TEXT,
            opponent TEXT,
            prediction INTEGER,
            confidence REAL,
            features_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (player_id) REFERENCES basic_stats (player_id)
        )
    """
}

# Model performance targets
MODEL_TARGETS = {
    "min_accuracy": 0.80,
    "min_f1_score": 0.75,
    "min_roc_auc": 0.85
}

# App features configuration
APP_FEATURES = {
    "max_recent_games": 10,
    "default_recent_games": 3,
    "prediction_history_limit": 50,
    "sample_data_limit": 10
}

# Environment variables
ENV_VARS = {
    "DATABASE_PATH": os.getenv("NFL_DATABASE_PATH", str(DATABASE_PATH)),
    "LOG_LEVEL": os.getenv("NFL_LOG_LEVEL", "INFO"),
    "DEBUG_MODE": os.getenv("NFL_DEBUG_MODE", "False").lower() == "true"
}

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        SRC_DIR,
        APP_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_config_summary():
    """Get a summary of the current configuration."""
    return {
        "project_root": str(PROJECT_ROOT),
        "database_path": str(DATABASE_PATH),
        "data_directories": {
            "raw": str(RAW_DATA_DIR),
            "processed": str(PROCESSED_DATA_DIR)
        },
        "model_path": str(MODEL_PATH),
        "feature_count": len(FEATURE_COLUMNS) - 1,  # Exclude target
        "rolling_window": ROLLING_WINDOW,
        "validation_thresholds": VALIDATION_THRESHOLDS,
        "model_targets": MODEL_TARGETS
    }

# Ensure directories exist when module is imported
ensure_directories() 