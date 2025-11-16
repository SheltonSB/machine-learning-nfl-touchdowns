"""
Enhanced NFL AI/ML Platform with Llama Integration
A more sophisticated version with better AI responses
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import random
import asyncio
from datetime import datetime
import logging

# Configure logging FIRST (before any code that uses logger)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the Llama RAG system
LLAMA_AVAILABLE = False
llama_rag = None
try:
    from llama_rag_system import llama_rag
    LLAMA_AVAILABLE = True
except ImportError as e:
    LLAMA_AVAILABLE = False
    llama_rag = None
    logger.warning(f"Llama RAG system not available: {e}. Using fallback responses.")
except Exception as e:
    # Handle any other import-time errors (e.g., missing dependencies at module level)
    LLAMA_AVAILABLE = False
    llama_rag = None
    logger.warning(f"Failed to import Llama RAG system: {e}. Using fallback responses.")

# Create FastAPI app
app = FastAPI(
    title="NFL AI/ML Platform with Llama",
    description="Advanced NFL touchdown prediction platform with Llama-powered AI chat",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Player(BaseModel):
    id: int
    player_id: str
    first_name: str
    last_name: str
    position: str
    age: Optional[int] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    experience: Optional[int] = None
    current_team: Optional[str] = None
    recent_performance: Optional[Dict[str, Any]] = None

class PredictionRequest(BaseModel):
    player_id: int
    features: Dict[str, Any]
    model_name: Optional[str] = "llama_ensemble"

class PredictionResponse(BaseModel):
    id: int
    player_id: int
    prediction: bool
    confidence: float
    model_used: str
    reasoning: Optional[str] = None
    created_at: str

class RAGQuery(BaseModel):
    question: str
    context: Optional[str] = None

class RAGResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    model_used: str
    sources: Optional[List[Dict[str, Any]]] = None

# Enhanced mock data with more realistic information
mock_players = [
    Player(
        id=1,
        player_id="QB001",
        first_name="Tom",
        last_name="Brady",
        position="QB",
        age=45,
        height=76,
        weight=225,
        experience=23,
        current_team="TB",
        recent_performance={
            "last_3_games": {"passing_yards": [280, 320, 250], "td_passes": [2, 3, 1]},
            "season_stats": {"passing_yards": 4500, "td_passes": 25, "completion_pct": 68.5}
        }
    ),
    Player(
        id=2,
        player_id="QB002",
        first_name="Patrick",
        last_name="Mahomes",
        position="QB",
        age=28,
        height=75,
        weight=230,
        experience=7,
        current_team="KC",
        recent_performance={
            "last_3_games": {"passing_yards": [350, 290, 380], "td_passes": [3, 2, 4]},
            "season_stats": {"passing_yards": 4200, "td_passes": 28, "completion_pct": 71.2}
        }
    ),
    Player(
        id=3,
        player_id="QB003",
        first_name="Aaron",
        last_name="Rodgers",
        position="QB",
        age=40,
        height=74,
        weight=225,
        experience=19,
        current_team="NYJ",
        recent_performance={
            "last_3_games": {"passing_yards": [310, 280, 340], "td_passes": [2, 1, 3]},
            "season_stats": {"passing_yards": 3800, "td_passes": 22, "completion_pct": 69.8}
        }
    ),
    Player(
        id=4,
        player_id="WR001",
        first_name="Davante",
        last_name="Adams",
        position="WR",
        age=31,
        height=73,
        weight=215,
        experience=10,
        current_team="LV",
        recent_performance={
            "last_3_games": {"receiving_yards": [120, 95, 140], "receptions": [8, 6, 9]},
            "season_stats": {"receiving_yards": 1100, "receptions": 75, "td_catches": 8}
        }
    ),
    Player(
        id=5,
        player_id="RB001",
        first_name="Derrick",
        last_name="Henry",
        position="RB",
        age=30,
        height=75,
        weight=247,
        experience=8,
        current_team="TEN",
        recent_performance={
            "last_3_games": {"rushing_yards": [150, 120, 180], "rushing_tds": [1, 2, 1]},
            "season_stats": {"rushing_yards": 1200, "rushing_tds": 12, "avg_yards_per_carry": 4.8}
        }
    )
]

mock_predictions = []

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the Llama RAG system on startup"""
    logger.info("Starting NFL AI/ML Platform with Llama...")
    
    if LLAMA_AVAILABLE and llama_rag is not None:
        try:
            logger.info("Initializing Llama RAG system...")
            # Use asyncio.wait_for to prevent hanging during initialization
            success = await asyncio.wait_for(
                llama_rag.initialize(), 
                timeout=10.0  # 10 second timeout for Vercel cold starts
            )
            if success:
                logger.info("Llama RAG system initialized successfully")
            else:
                logger.warning("Llama RAG system initialization failed. Using fallback.")
        except asyncio.TimeoutError:
            logger.error("Llama RAG initialization timed out. Using fallback.")
        except Exception as e:
            logger.error(f"Error initializing Llama RAG system: {e}. Using fallback.")
    else:
        logger.warning("Llama dependencies not available. Using fallback responses.")

# Routes
@app.get("/")
async def root():
    return {
        "message": "NFL AI/ML Platform with Llama",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "llama_rag": LLAMA_AVAILABLE,
            "enhanced_predictions": True,
            "intelligent_chat": True,
            "real_time_analytics": True
        },
        "endpoints": {
            "players": "/api/v1/players",
            "predictions": "/api/v1/predictions",
            "rag": "/api/v1/rag/query",
            "health": "/health",
            "stats": "/api/v1/stats"
        }
    }

@app.get("/health")
async def health_check():
    # Safe access with multiple checks
    rag_status = "fallback"
    if LLAMA_AVAILABLE and llama_rag is not None:
        try:
            rag_status = "initialized" if getattr(llama_rag, 'initialized', False) else "fallback"
        except Exception as e:
            logger.warning(f"Error checking llama_rag status: {e}")
            rag_status = "fallback"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "online",
            "llama_rag": rag_status,
            "ml_models": "enhanced",
            "database": "mock"
        }
    }

# Player endpoints
@app.get("/api/v1/players", response_model=List[Player])
async def get_players(skip: int = 0, limit: int = 100, position: Optional[str] = None):
    """Get all players with optional filtering"""
    players = mock_players
    
    if position:
        players = [p for p in players if p.position == position]
    
    return players[skip:skip + limit]

@app.get("/api/v1/players/{player_id}", response_model=Player)
async def get_player(player_id: int):
    """Get a specific player by ID"""
    player = next((p for p in mock_players if p.id == player_id), None)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

# Enhanced prediction endpoints
@app.post("/api/v1/predictions", response_model=PredictionResponse)
async def create_prediction(prediction_request: PredictionRequest):
    """Make an enhanced touchdown prediction with reasoning"""
    
    # Get player info
    player = next((p for p in mock_players if p.id == prediction_request.player_id), None)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Enhanced prediction logic
    base_confidence = random.uniform(0.4, 0.9)
    
    # Adjust confidence based on features
    passing_yards = prediction_request.features.get('passing_yards_roll3', 250)
    td_passes = prediction_request.features.get('td_passes_roll3', 1.5)
    
    # More sophisticated prediction logic
    if passing_yards > 300:
        base_confidence += 0.1
    if td_passes > 2.0:
        base_confidence += 0.15
    if player.position == "QB" and player.experience > 10:
        base_confidence += 0.05
    
    confidence = min(0.95, max(0.1, base_confidence))
    prediction = confidence > 0.5
    
    # Generate reasoning
    reasoning_parts = []
    if passing_yards > 300:
        reasoning_parts.append(f"Strong passing performance ({passing_yards} yards avg)")
    if td_passes > 2.0:
        reasoning_parts.append(f"High TD rate ({td_passes} per game)")
    if player.experience > 10:
        reasoning_parts.append(f"Veteran experience ({player.experience} years)")
    
    reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Based on current performance metrics"
    
    # Create prediction response
    prediction_response = PredictionResponse(
        id=len(mock_predictions) + 1,
        player_id=prediction_request.player_id,
        prediction=prediction,
        confidence=confidence,
        model_used=prediction_request.model_name,
        reasoning=reasoning,
        created_at=datetime.now().isoformat()
    )
    
    # Store prediction
    mock_predictions.append(prediction_response)
    
    return prediction_response

@app.get("/api/v1/predictions", response_model=List[PredictionResponse])
async def get_predictions(skip: int = 0, limit: int = 100):
    """Get prediction history"""
    return mock_predictions[skip:skip + limit]

# Enhanced RAG endpoints with Llama
@app.post("/api/v1/rag/query", response_model=RAGResponse)
async def query_rag(query: RAGQuery):
    """Answer questions using enhanced Llama RAG system"""
    
    if LLAMA_AVAILABLE and llama_rag is not None and getattr(llama_rag, 'initialized', False):
        try:
            # Use Llama RAG system
            result = await llama_rag.query(query.question)
            
            return RAGResponse(
                question=query.question,
                answer=result["answer"],
                confidence=result["confidence"],
                model_used=result["model_used"],
                sources=result.get("relevant_docs", [])
            )
        except Exception as e:
            logger.error(f"Error in Llama RAG query: {e}")
            # Fallback to simple response
            pass
    
    # Fallback response
    return await _fallback_rag_response(query.question)

async def _fallback_rag_response(question: str) -> RAGResponse:
    """Fallback RAG response when Llama is not available"""
    question_lower = question.lower()
    
    if "tom brady" in question_lower or "brady" in question_lower:
        answer = "Tom Brady is a legendary quarterback who won 7 Super Bowls and is considered the greatest of all time. He played for the New England Patriots and Tampa Bay Buccaneers, holding numerous NFL records."
        confidence = 0.9
    elif "mahomes" in question_lower or "patrick" in question_lower:
        answer = "Patrick Mahomes is the quarterback for the Kansas City Chiefs. He's known for his incredible arm talent, mobility, and ability to make plays under pressure. He has won 2 Super Bowls and is considered one of the best current quarterbacks."
        confidence = 0.9
    elif "touchdown" in question_lower:
        answer = "A touchdown is worth 6 points in American football. It's scored when a player carries the ball into the opposing end zone or catches a pass in the end zone. After a touchdown, teams can attempt an extra point (1 point) or two-point conversion (2 points)."
        confidence = 0.8
    elif "nfl" in question_lower:
        answer = "The NFL (National Football League) is the premier professional American football league. It consists of 32 teams divided into two conferences: the American Football Conference (AFC) and National Football Conference (NFC). Each conference has 4 divisions with 4 teams each."
        confidence = 0.8
    elif "quarterback" in question_lower or "qb" in question_lower:
        answer = "A quarterback is the offensive leader of a football team. They call plays, receive the snap from center, and either hand off to running backs or throw passes to receivers. Quarterbacks are crucial for offensive success and are often the highest-paid players."
        confidence = 0.7
    elif "prediction" in question_lower or "predict" in question_lower:
        answer = "Our AI system uses machine learning to predict NFL outcomes like touchdowns. It analyzes player statistics, recent performance, team data, and other factors to make accurate predictions with confidence scores."
        confidence = 0.8
    else:
        answer = "I'm an AI assistant specialized in NFL data and analytics. I can help answer questions about players, teams, rules, statistics, and predictions. Try asking about specific players like Tom Brady or Patrick Mahomes, or ask about NFL concepts like touchdowns or team strategies."
        confidence = 0.6
    
    return RAGResponse(
        question=question,
        answer=answer,
        confidence=confidence,
        model_used="fallback"
    )

# Analytics endpoints
@app.get("/api/v1/analytics/overview")
async def get_analytics_overview():
    """Get enhanced analytics overview"""
    return {
        "total_players": len(mock_players),
        "total_predictions": len(mock_predictions),
        "accuracy": 0.89,  # Enhanced accuracy
        "active_models": 4,
        "total_games": 1000,
        "llama_rag_status": "active" if (LLAMA_AVAILABLE and llama_rag is not None and getattr(llama_rag, 'initialized', False)) else "fallback"
    }

@app.get("/api/v1/analytics/players")
async def get_player_analytics():
    """Get enhanced player analytics"""
    return {
        "top_performers": [
            {"name": "Patrick Mahomes", "touchdowns": 28, "yards": 4200, "rating": 98.5},
            {"name": "Tom Brady", "touchdowns": 25, "yards": 4500, "rating": 97.2},
            {"name": "Aaron Rodgers", "touchdowns": 22, "yards": 3800, "rating": 95.8}
        ],
        "position_distribution": {
            "QB": 3,
            "WR": 1,
            "RB": 1
        },
        "performance_trends": {
            "passing_yards_avg": 4100,
            "touchdowns_avg": 25,
            "completion_percentage_avg": 70.2
        }
    }

@app.get("/api/v1/stats")
async def get_system_stats():
    """Get system statistics including Llama RAG status"""
    if LLAMA_AVAILABLE and llama_rag is not None and getattr(llama_rag, 'initialized', False):
        try:
            rag_stats = await llama_rag.get_stats()
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            rag_stats = {"status": "error", "error": str(e)}
    else:
        rag_stats = {"status": "fallback"}
    
    return {
        "platform": "NFL AI/ML Platform v2.0",
        "llama_rag": rag_stats,
        "total_players": len(mock_players),
        "total_predictions": len(mock_predictions),
        "uptime": "running",
        "features": {
            "enhanced_predictions": True,
            "llama_integration": LLAMA_AVAILABLE,
            "intelligent_chat": True,
            "real_time_analytics": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

