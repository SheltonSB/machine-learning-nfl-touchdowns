"""
Players API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.models.database import Player
from app.schemas.player import PlayerCreate, PlayerUpdate, PlayerResponse
from app.services.player_service import PlayerService

router = APIRouter()

@router.get("/", response_model=List[PlayerResponse])
async def get_players(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    position: Optional[str] = Query(None),
    team: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get list of players with optional filtering"""
    service = PlayerService(db)
    return await service.get_players(
        skip=skip,
        limit=limit,
        position=position,
        team=team,
        search=search
    )

@router.get("/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: int, db: Session = Depends(get_db)):
    """Get a specific player by ID"""
    service = PlayerService(db)
    player = await service.get_player(player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

@router.post("/", response_model=PlayerResponse)
async def create_player(player: PlayerCreate, db: Session = Depends(get_db)):
    """Create a new player"""
    service = PlayerService(db)
    return await service.create_player(player)

@router.put("/{player_id}", response_model=PlayerResponse)
async def update_player(
    player_id: int,
    player_update: PlayerUpdate,
    db: Session = Depends(get_db)
):
    """Update a player"""
    service = PlayerService(db)
    player = await service.update_player(player_id, player_update)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

@router.delete("/{player_id}")
async def delete_player(player_id: int, db: Session = Depends(get_db)):
    """Delete a player"""
    service = PlayerService(db)
    success = await service.delete_player(player_id)
    if not success:
        raise HTTPException(status_code=404, detail="Player not found")
    return {"message": "Player deleted successfully"}

@router.get("/{player_id}/stats")
async def get_player_stats(player_id: int, db: Session = Depends(get_db)):
    """Get player statistics"""
    service = PlayerService(db)
    stats = await service.get_player_stats(player_id)
    if not stats:
        raise HTTPException(status_code=404, detail="Player not found")
    return stats

@router.get("/{player_id}/recent-games")
async def get_recent_games(
    player_id: int,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """Get player's recent games"""
    service = PlayerService(db)
    games = await service.get_recent_games(player_id, limit)
    if not games:
        raise HTTPException(status_code=404, detail="Player not found")
    return games

