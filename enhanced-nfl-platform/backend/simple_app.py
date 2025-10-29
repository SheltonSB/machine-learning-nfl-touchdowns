"""
Simple NFL AI/ML Platform Demo
A minimal version that can run without heavy dependencies
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import random
from datetime import datetime

# Create FastAPI app
app = FastAPI(
    title="NFL AI/ML Platform",
    description="A simple demo of the NFL touchdown prediction platform",
    version="1.0.0"
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

class PredictionRequest(BaseModel):
    player_id: int
    features: Dict[str, Any]
    model_name: Optional[str] = "ensemble"

class PredictionResponse(BaseModel):
    id: int
    player_id: int
    prediction: bool
    confidence: float
    model_used: str
    created_at: str

class RAGQuery(BaseModel):
    question: str

class RAGResponse(BaseModel):
    question: str
    answer: str
    confidence: float

# Mock data
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
        current_team="TB"
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
        current_team="KC"
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
        current_team="NYJ"
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
        current_team="LV"
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
        current_team="TEN"
    )
]

mock_predictions = []

# Routes
@app.get("/")
async def root():
    return {
        "message": "🏈 NFL AI/ML Platform",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "players": "/api/v1/players",
            "predictions": "/api/v1/predictions",
            "rag": "/api/v1/rag/query",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "online",
            "ml_models": "simulated",
            "database": "mock"
        }
    }

# Player endpoints
@app.get("/api/v1/players", response_model=List[Player])
async def get_players(skip: int = 0, limit: int = 100):
    """Get all players with pagination"""
    return mock_players[skip:skip + limit]

@app.get("/api/v1/players/{player_id}", response_model=Player)
async def get_player(player_id: int):
    """Get a specific player by ID"""
    player = next((p for p in mock_players if p.id == player_id), None)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

# Prediction endpoints
@app.post("/api/v1/predictions", response_model=PredictionResponse)
async def create_prediction(prediction_request: PredictionRequest):
    """Make a touchdown prediction"""
    
    # Simulate ML prediction
    confidence = random.uniform(0.3, 0.95)
    prediction = confidence > 0.5
    
    # Create prediction response
    prediction_response = PredictionResponse(
        id=len(mock_predictions) + 1,
        player_id=prediction_request.player_id,
        prediction=prediction,
        confidence=confidence,
        model_used=prediction_request.model_name,
        created_at=datetime.now().isoformat()
    )
    
    # Store prediction
    mock_predictions.append(prediction_response)
    
    return prediction_response

@app.get("/api/v1/predictions", response_model=List[PredictionResponse])
async def get_predictions(skip: int = 0, limit: int = 100):
    """Get prediction history"""
    return mock_predictions[skip:skip + limit]

# RAG endpoints
@app.post("/api/v1/rag/query", response_model=RAGResponse)
async def query_rag(query: RAGQuery):
    """Answer questions using simulated RAG"""
    
    # Simple keyword-based responses
    question_lower = query.question.lower()
    
    if "tom brady" in question_lower or "brady" in question_lower:
        answer = "Tom Brady is a legendary quarterback who played for the New England Patriots and Tampa Bay Buccaneers. He won 7 Super Bowls and is considered one of the greatest quarterbacks of all time."
        confidence = 0.9
    elif "mahomes" in question_lower or "patrick" in question_lower:
        answer = "Patrick Mahomes is the quarterback for the Kansas City Chiefs. He's known for his incredible arm talent, mobility, and ability to make plays under pressure. He won Super Bowl LIV and LVII."
        confidence = 0.9
    elif "touchdown" in question_lower:
        answer = "A touchdown is worth 6 points in football. It's scored when a player carries the ball into the opposing end zone or catches a pass in the end zone. After a touchdown, teams can attempt an extra point or two-point conversion."
        confidence = 0.8
    elif "nfl" in question_lower:
        answer = "The NFL (National Football League) is the premier professional American football league. It consists of 32 teams divided into two conferences: the American Football Conference (AFC) and National Football Conference (NFC)."
        confidence = 0.8
    elif "quarterback" in question_lower or "qb" in question_lower:
        answer = "A quarterback is the offensive leader of a football team. They call plays, receive the snap from center, and either hand off to running backs or throw passes to receivers. Quarterbacks are crucial for offensive success."
        confidence = 0.7
    else:
        answer = "I'm a simulated AI assistant for NFL data. I can help answer questions about players, teams, rules, and statistics. Try asking about specific players like Tom Brady or Patrick Mahomes, or ask about NFL rules and concepts."
        confidence = 0.5
    
    return RAGResponse(
        question=query.question,
        answer=answer,
        confidence=confidence
    )

# Analytics endpoints
@app.get("/api/v1/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview"""
    return {
        "total_players": len(mock_players),
        "total_predictions": len(mock_predictions),
        "accuracy": 0.87,
        "active_models": 3,
        "total_games": 1000
    }

@app.get("/api/v1/analytics/players")
async def get_player_analytics():
    """Get player analytics"""
    return {
        "top_performers": [
            {"name": "Tom Brady", "touchdowns": 45, "yards": 4500},
            {"name": "Patrick Mahomes", "touchdowns": 42, "yards": 4200},
            {"name": "Aaron Rodgers", "touchdowns": 40, "yards": 4000}
        ],
        "position_distribution": {
            "QB": 1,
            "WR": 1,
            "RB": 1
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

