"""
Player Pydantic schemas
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class PlayerBase(BaseModel):
    player_id: str
    first_name: str
    last_name: str
    position: str
    age: Optional[int] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    experience: Optional[int] = None
    current_team: Optional[str] = None

class PlayerCreate(PlayerBase):
    pass

class PlayerUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    position: Optional[str] = None
    age: Optional[int] = None
    height: Optional[int] = None
    weight: Optional[int] = None
    experience: Optional[int] = None
    current_team: Optional[str] = None

class PlayerResponse(PlayerBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

