"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for tests
    redis = None

from typing import Generator

from app.core.config import settings

# PostgreSQL Database
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis for caching
if redis is not None:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
else:
    class _RedisStub:
        def __getattr__(self, name):
            raise RuntimeError("Redis library not installed; install redis or set REDIS_URL appropriately")

    redis_client = _RedisStub()

def get_db() -> Generator[Session, None, None]:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_redis():
    """Get Redis client"""
    return redis_client

