"""
Database models for the NFL AI/ML Platform
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base

class Player(Base):
    """Player model"""
    __tablename__ = "players"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(String(50), unique=True, index=True, nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    position = Column(String(10), nullable=False)
    age = Column(Integer)
    height = Column(Integer)  # inches
    weight = Column(Integer)  # pounds
    experience = Column(Integer)
    current_team = Column(String(10))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    game_logs = relationship("GameLog", back_populates="player")
    predictions = relationship("Prediction", back_populates="player")

class Team(Base):
    """Team model"""
    __tablename__ = "teams"
    
    id = Column(Integer, primary_key=True, index=True)
    team_code = Column(String(10), unique=True, index=True, nullable=False)
    team_name = Column(String(100), nullable=False)
    city = Column(String(100))
    conference = Column(String(10))
    division = Column(String(20))

class Game(Base):
    """Game model"""
    __tablename__ = "games"
    
    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=False)
    game_date = Column(DateTime(timezone=True), nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"))
    away_team_id = Column(Integer, ForeignKey("teams.id"))
    home_score = Column(Integer)
    away_score = Column(Integer)
    weather_conditions = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])
    game_logs = relationship("GameLog", back_populates="game")

class GameLog(Base):
    """Game log model"""
    __tablename__ = "game_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"))
    opponent_team_id = Column(Integer, ForeignKey("teams.id"))
    
    # Passing stats
    passing_yards = Column(Integer, default=0)
    td_passes = Column(Integer, default=0)
    interceptions = Column(Integer, default=0)
    passes_attempted = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    completion_percentage = Column(Float)
    yards_per_attempt = Column(Float)
    passer_rating = Column(Float)
    
    # Rushing stats
    rushing_yards = Column(Integer, default=0)
    rushing_attempts = Column(Integer, default=0)
    fumbles = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    player = relationship("Player", back_populates="game_logs")
    game = relationship("Game", back_populates="game_logs")
    team = relationship("Team", foreign_keys=[team_id])
    opponent_team = relationship("Team", foreign_keys=[opponent_team_id])

class Prediction(Base):
    """Prediction model"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    game_id = Column(Integer, ForeignKey("games.id"))
    prediction = Column(Boolean, nullable=False)
    confidence = Column(Float, nullable=False)
    features_used = Column(JSON)
    actual_result = Column(Boolean)
    model_used = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(100))
    
    # Relationships
    player = relationship("Player", back_populates="predictions")
    game = relationship("Game")

class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True))

