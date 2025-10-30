"""
Player service for business logic
"""

from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.database import Player
from app.schemas.player import PlayerCreate, PlayerUpdate, PlayerResponse

class PlayerService:
    def __init__(self, db: Session):
        self.db = db

    async def get_players(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        position: Optional[str] = None,
        team: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[PlayerResponse]:
        """Get players with optional filtering"""
        query = self.db.query(Player)
        
        if position:
            query = query.filter(Player.position == position)
        if team:
            query = query.filter(Player.current_team == team)
        if search:
            query = query.filter(
                (Player.first_name.ilike(f"%{search}%")) |
                (Player.last_name.ilike(f"%{search}%"))
            )
        
        players = query.offset(skip).limit(limit).all()
        return [PlayerResponse.from_orm(player) for player in players]

    async def get_player(self, player_id: int) -> Optional[PlayerResponse]:
        """Get a specific player by ID"""
        player = self.db.query(Player).filter(Player.id == player_id).first()
        return PlayerResponse.from_orm(player) if player else None

    async def create_player(self, player: PlayerCreate) -> PlayerResponse:
        """Create a new player"""
        db_player = Player(**player.dict())
        self.db.add(db_player)
        self.db.commit()
        self.db.refresh(db_player)
        return PlayerResponse.from_orm(db_player)

    async def update_player(self, player_id: int, player_update: PlayerUpdate) -> Optional[PlayerResponse]:
        """Update a player"""
        db_player = self.db.query(Player).filter(Player.id == player_id).first()
        if not db_player:
            return None
        
        update_data = player_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_player, field, value)
        
        self.db.commit()
        self.db.refresh(db_player)
        return PlayerResponse.from_orm(db_player)

    async def delete_player(self, player_id: int) -> bool:
        """Delete a player"""
        db_player = self.db.query(Player).filter(Player.id == player_id).first()
        if not db_player:
            return False
        
        self.db.delete(db_player)
        self.db.commit()
        return True

    async def get_player_stats(self, player_id: int) -> Optional[dict]:
        """Get player statistics"""
        player = self.db.query(Player).filter(Player.id == player_id).first()
        if not player:
            return None
        
        # This would typically include more complex statistics
        return {
            "player_id": player.player_id,
            "name": f"{player.first_name} {player.last_name}",
            "position": player.position,
            "team": player.current_team,
            "age": player.age,
            "experience": player.experience
        }

    async def get_recent_games(self, player_id: int, limit: int = 10) -> List[dict]:
        """Get player's recent games"""
        from app.models.database import GameLog
        
        games = self.db.query(GameLog).filter(
            GameLog.player_id == player_id
        ).order_by(GameLog.created_at.desc()).limit(limit).all()
        
        return [
            {
                "game_id": game.id,
                "passing_yards": game.passing_yards,
                "td_passes": game.td_passes,
                "interceptions": game.interceptions,
                "completion_percentage": game.completion_percentage,
                "created_at": game.created_at
            }
            for game in games
        ]

