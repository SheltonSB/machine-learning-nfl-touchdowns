"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for tests
    redis = None

from typing import Generator
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Get database URL and connection args from settings
database_url = settings.get_database_url()
connect_args = settings.get_database_connect_args()

# Determine pool class based on database type
# Use QueuePool for PostgreSQL/MySQL (better for production/RDS)
# Use StaticPool for SQLite
if "sqlite" in database_url.lower():
    poolclass = StaticPool
    pool_pre_ping = False
else:
    # Use QueuePool for RDS/PostgreSQL/MySQL with connection pooling
    poolclass = QueuePool
    pool_pre_ping = True  # Verify connections before using (important for RDS)

# Create database engine
engine = create_engine(
    database_url,
    poolclass=poolclass,
    connect_args=connect_args,
    pool_pre_ping=pool_pre_ping,
    # Connection pool settings for RDS
    pool_size=10 if poolclass == QueuePool else None,
    max_overflow=20 if poolclass == QueuePool else None,
    pool_recycle=3600,  # Recycle connections after 1 hour (important for RDS)
    echo=False  # Set to True for SQL query logging
)

logger.info(f"Database engine created: {database_url.split('@')[1] if '@' in database_url else 'local database'}")

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

