from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import asyncio
import logging
import os
import json
import requests
import pandas as pd
import numpy as np
import hashlib
import aiohttp
import uvicorn
import random
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import time
import psutil

# Import our advanced AI system
from advanced_ai_system import AdvancedRAGSystem, AdvancedMLPipeline, AdvancedAIConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MySQL Database Configuration
MYSQL_USERNAME = "root"
MYSQL_PASSWORD = "NewStrongPassword!123"
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DATABASE = "nfl_ai"

# Create MySQL connection string
DATABASE_URL = f"mysql+pymysql://{MYSQL_USERNAME}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
NFL_API_KEY = os.getenv("NFL_API_KEY", "")

# Create FastAPI app
app = FastAPI(
    title="NFL AI/ML Platform - Ultimate Production",
    description="The most advanced NFL AI platform with fine-tuning, temperature control, and ensemble ML",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)

# Serve static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

# Database setup
Base = declarative_base()
engine = None
SessionLocal = None
DATABASE_AVAILABLE = False

try:
    engine = create_engine(DATABASE_URL, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    
    DATABASE_AVAILABLE = True
    logger.info("MySQL database connected successfully")
    
except Exception as e:
    logger.error(f"MySQL connection failed: {e}")
    DATABASE_AVAILABLE = False

# Initialize Advanced AI Systems
rag_system = None
ml_pipeline = None
ai_config = AdvancedAIConfig()

# Pydantic models
class Player(BaseModel):
    id: int
    player_id: str
    name: str
    position: str
    team: str
    age: Optional[int] = None
    height: Optional[str] = None
    weight: Optional[int] = None
    experience: Optional[int] = None
    stats: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    player_id: int
    features: Dict[str, Any]
    model_name: Optional[str] = "advanced_ensemble"
    confidence_threshold: Optional[float] = 0.7

class RAGQuery(BaseModel):
    question: str
    mode: Optional[str] = "balanced"  # creative, balanced, precise, conservative
    use_real_data: Optional[bool] = True
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None

class RAGResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    mode: str
    temperature: float
    top_k: int
    top_p: float
    response_time: float
    model_used: str
    sources: Optional[List[str]] = None
    data_freshness: Optional[str] = None

class PredictionResponse(BaseModel):
    id: int
    player_id: int
    prediction: bool
    confidence: float
    model_used: str
    reasoning: Optional[str] = None
    created_at: str
    features_used: List[str]
    model_breakdown: Optional[Dict[str, bool]] = None
    response_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    components: Dict[str, str]
    database_status: str
    rag_status: str
    ml_status: str
    performance_metrics: Dict[str, Any]

class SystemStats(BaseModel):
    platform: str
    version: str
    status: str
    database: Dict[str, Any]
    rag_system: Dict[str, Any]
    ml_pipeline: Dict[str, Any]
    performance: Dict[str, Any]
    features: Dict[str, bool]

# Database dependency
def get_db():
    if DATABASE_AVAILABLE and SessionLocal:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        raise HTTPException(status_code=500, detail="Database not available")

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
    def record_request(self, response_time: float, success: bool = True):
        self.request_count += 1
        self.total_response_time += response_time
        if not success:
            self.error_count += 1
    
    def get_metrics(self):
        uptime = time.time() - self.start_time
        avg_response_time = self.total_response_time / max(self.request_count, 1)
        error_rate = self.error_count / max(self.request_count, 1)
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "avg_response_time": avg_response_time,
            "error_rate": error_rate,
            "requests_per_second": self.request_count / max(uptime, 1),
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent
        }

performance_monitor = PerformanceMonitor()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend"""
    try:
        with open("../frontend/ultimate.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>NFL AI Platform</title></head>
            <body>
                <h1>NFL AI Platform - Ultimate Production</h1>
                <p>API is running! Visit <a href="/docs">/docs</a> for API documentation.</p>
            </body>
        </html>
        """)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with performance metrics"""
    start_time = time.time()
    
    try:
        # Check database
        db_status = "connected" if DATABASE_AVAILABLE else "disconnected"
        
        # Check RAG system
        rag_status = "active" if rag_system and rag_system.is_initialized else "initializing"
        
        # Check ML pipeline
        ml_status = "trained" if ml_pipeline and ml_pipeline.is_trained else "training"
        
        # Get performance metrics
        performance_metrics = performance_monitor.get_metrics()
        
        # Add AI-specific metrics
        if rag_system:
            performance_metrics.update(rag_system.get_performance_metrics())
        if ml_pipeline:
            performance_metrics.update(ml_pipeline.get_performance_metrics())
        
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time)
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            components={
                "api": "online",
                "mysql_database": db_status,
                "advanced_rag": rag_status,
                "advanced_ml": ml_status,
                "performance_monitor": "active"
            },
            database_status=db_status,
            rag_status=rag_status,
            ml_status=ml_status,
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail="Health check failed")

@app.post("/api/v1/rag/query", response_model=RAGResponse)
async def query_advanced_rag(query: RAGQuery, background_tasks: BackgroundTasks, db = Depends(get_db)):
    """Advanced RAG query with fine-tuning and temperature control"""
    start_time = time.time()
    
    try:
        if not rag_system or not rag_system.is_initialized:
            raise HTTPException(status_code=503, detail="RAG system not ready")
        
        # Use custom parameters if provided
        if query.temperature is not None or query.top_k is not None or query.top_p is not None:
            # Override AI config temporarily
            original_temp = ai_config.TEMPERATURE_SETTINGS.get(query.mode, 0.7)
            original_top_k = ai_config.TOP_K_SETTINGS.get(query.mode, 20)
            original_top_p = ai_config.TOP_P_SETTINGS.get(query.mode, 0.85)
            
            if query.temperature is not None:
                ai_config.TEMPERATURE_SETTINGS[query.mode] = query.temperature
            if query.top_k is not None:
                ai_config.TOP_K_SETTINGS[query.mode] = query.top_k
            if query.top_p is not None:
                ai_config.TOP_P_SETTINGS[query.mode] = query.top_p
        
        # Query the advanced RAG system
        result = await rag_system.query(query.question, query.mode)
        
        # Restore original settings
        if query.temperature is not None or query.top_k is not None or query.top_p is not None:
            ai_config.TEMPERATURE_SETTINGS[query.mode] = original_temp
            ai_config.TOP_K_SETTINGS[query.mode] = original_top_k
            ai_config.TOP_P_SETTINGS[query.mode] = original_top_p
        
        # Store query in database
        background_tasks.add_task(store_rag_query, query, result, db)
        
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time)
        
        return RAGResponse(
            question=query.question,
            answer=result["answer"],
            confidence=result["confidence"],
            mode=query.mode,
            temperature=result.get("temperature", ai_config.TEMPERATURE_SETTINGS[query.mode]),
            top_k=result.get("top_k", ai_config.TOP_K_SETTINGS[query.mode]),
            top_p=result.get("top_p", ai_config.TOP_P_SETTINGS[query.mode]),
            response_time=response_time,
            model_used=result.get("model_used", "advanced_rag"),
            sources=result.get("sources", []),
            data_freshness=result.get("data_freshness", "real_database")
        )
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail="Error processing query")

@app.post("/api/v1/predictions", response_model=PredictionResponse, status_code=201)
async def create_advanced_prediction(prediction_data: PredictionRequest, background_tasks: BackgroundTasks, db = Depends(get_db)):
    """Create advanced prediction using ensemble ML models"""
    start_time = time.time()
    
    try:
        if not ml_pipeline or not ml_pipeline.is_trained:
            raise HTTPException(status_code=503, detail="ML pipeline not ready")
        
        # Make prediction using advanced ML pipeline
        result = ml_pipeline.predict(prediction_data.player_id, prediction_data.features)
        
        # Check confidence threshold
        if result["confidence"] < prediction_data.confidence_threshold:
            result["prediction"] = False  # Conservative approach for low confidence
        
        prediction = PredictionResponse(
            id=hash(f"{prediction_data.player_id}_{datetime.now()}") % 1000000,
            player_id=prediction_data.player_id,
            prediction=result["prediction"],
            confidence=result["confidence"],
            model_used=result["model_used"],
            reasoning=result["reasoning"],
            created_at=datetime.now().isoformat(),
            features_used=result["features_used"],
            model_breakdown=result.get("model_breakdown"),
            response_time=result.get("response_time", 0.0)
        )
        
        # Store prediction in database
        background_tasks.add_task(store_prediction, prediction_data, result, db)
        
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        performance_monitor.record_request(time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail="Error creating prediction")

@app.get("/api/v1/stats", response_model=SystemStats)
async def get_system_stats():
    """Get comprehensive system statistics"""
    try:
        # Get database record counts
        record_counts = {}
        if DATABASE_AVAILABLE:
            with engine.connect() as conn:
                tables = [
                    "basic_stats", "career_stats_passing", "career_stats_rushing", 
                    "career_stats_receiving", "game_logs_quarterback", 
                    "game_logs_runningback", "game_logs_wide_receiver"
                ]
                
                for table in tables:
                    try:
                        result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.fetchone()[0]
                        record_counts[table] = count
                    except:
                        record_counts[table] = 0
        
        # Get performance metrics
        performance_metrics = performance_monitor.get_metrics()
        
        # Add AI-specific metrics
        if rag_system:
            performance_metrics.update(rag_system.get_performance_metrics())
        if ml_pipeline:
            performance_metrics.update(ml_pipeline.get_performance_metrics())
        
        return SystemStats(
            platform="NFL AI/ML Platform - Ultimate Production",
            version="5.0.0",
            status="live",
            database={
                "type": "MySQL",
                "username": MYSQL_USERNAME,
                "host": MYSQL_HOST,
                "port": MYSQL_PORT,
                "database": MYSQL_DATABASE,
                "status": "connected" if DATABASE_AVAILABLE else "disconnected",
                "real_time": True,
                "record_counts": record_counts,
                "total_records": sum(record_counts.values())
            },
            rag_system={
                "status": "active" if rag_system and rag_system.is_initialized else "initializing",
                "collection_size": "real_player_data" if rag_system and rag_system.is_initialized else "0",
                "data_sources": ["Real NFL Database", "Advanced RAG", "MySQL Storage"],
                "features": ["fine_tuning", "temperature_control", "top_k_sampling", "probability_distribution"]
            },
            ml_pipeline={
                "status": "trained" if ml_pipeline and ml_pipeline.is_trained else "training",
                "models": list(ml_pipeline.models.keys()) if ml_pipeline and ml_pipeline.models else ["fallback"],
                "accuracy": performance_metrics.get("accuracy", "advanced_ensemble"),
                "features": ["ensemble_learning", "hyperparameter_optimization", "feature_engineering"]
            },
            performance=performance_metrics,
            features={
                "real_database": DATABASE_AVAILABLE,
                "advanced_rag": rag_system is not None,
                "advanced_ml": ml_pipeline is not None,
                "fine_tuning": True,
                "temperature_control": True,
                "top_k_sampling": True,
                "probability_distribution": True,
                "ensemble_learning": True,
                "hyperparameter_optimization": True,
                "performance_monitoring": True,
                "live_data": True,
                "api_integration": True,
                "comprehensive_knowledge": True,
                "data_persistence": True,
                "csv_data_loaded": True
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve statistics")

@app.get("/api/v1/players")
async def get_players(limit: int = 100, position: Optional[str] = None, db = Depends(get_db)):
    """Get players from database with filtering"""
    try:
        query = """
            SELECT 
                bs.player_id,
                bs.name,
                bs.position,
                bs.team,
                bs.age,
                bs.height,
                bs.weight,
                bs.experience,
                csp.passing_yards,
                csp.passing_tds,
                csp.completion_pct,
                csp.passer_rating
            FROM basic_stats bs
            LEFT JOIN career_stats_passing csp ON bs.player_id = csp.player_id
        """
        
        params = {}
        if position:
            query += " WHERE bs.position = :position"
            params["position"] = position
        
        query += " ORDER BY bs.name LIMIT :limit"
        params["limit"] = limit
        
        result = db.execute(text(query), params).fetchall()
        
        players = []
        for row in result:
            players.append({
                "player_id": row[0],
                "name": row[1],
                "position": row[2],
                "team": row[3],
                "age": row[4],
                "height": row[5],
                "weight": row[6],
                "experience": row[7],
                "passing_yards": row[8],
                "passing_tds": row[9],
                "completion_pct": row[10],
                "passer_rating": row[11]
            })
        
        return {"players": players, "count": len(players)}
        
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        raise HTTPException(status_code=500, detail="Error fetching players")

@app.get("/api/v1/ai/performance")
async def get_ai_performance():
    """Get AI system performance metrics"""
    try:
        metrics = {}
        
        if rag_system:
            metrics["rag"] = rag_system.get_performance_metrics()
        
        if ml_pipeline:
            metrics["ml"] = ml_pipeline.get_performance_metrics()
        
        metrics["system"] = performance_monitor.get_metrics()
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting AI performance: {e}")
        raise HTTPException(status_code=500, detail="Error getting performance metrics")

# Background tasks
async def store_rag_query(query: RAGQuery, result: Dict, db):
    """Store RAG query in database"""
    try:
        db.execute(text("""
            INSERT INTO rag_queries (question, answer, confidence, model_used, sources, data_freshness)
            VALUES (:question, :answer, :confidence, :model_used, :sources, :data_freshness)
        """), {
            "question": query.question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "model_used": result.get("model_used", "advanced_rag"),
            "sources": json.dumps(result.get("sources", [])),
            "data_freshness": result.get("data_freshness", "real_database")
        })
        db.commit()
        logger.info("RAG query stored in database")
    except Exception as e:
        logger.error(f"Error storing RAG query: {e}")

async def store_prediction(prediction_data: PredictionRequest, result: Dict, db):
    """Store prediction in database"""
    try:
        db.execute(text("""
            INSERT INTO predictions (player_id, prediction, confidence, model_used, features, reasoning)
            VALUES (:player_id, :prediction, :confidence, :model_used, :features, :reasoning)
        """), {
            "player_id": prediction_data.player_id,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "model_used": result["model_used"],
            "features": json.dumps(prediction_data.features),
            "reasoning": result["reasoning"]
        })
        db.commit()
        logger.info("Prediction stored in database")
    except Exception as e:
        logger.error(f"Error storing prediction: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize all systems on startup"""
    global rag_system, ml_pipeline
    
    logger.info("Starting NFL AI/ML Platform - Ultimate Production")
    
    try:
        # Initialize Advanced RAG System
        logger.info("Initializing Advanced RAG System...")
        rag_system = AdvancedRAGSystem(DATABASE_URL)
        await rag_system.initialize()
        
        # Initialize Advanced ML Pipeline
        logger.info("Initializing Advanced ML Pipeline...")
        ml_pipeline = AdvancedMLPipeline(DATABASE_URL)
        await ml_pipeline.initialize()
        
        logger.info("Ultimate Production platform ready")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
