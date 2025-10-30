import os

import pytest
import sys

# Ensure backend package is importable for tests.
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Ensure lightweight configuration for tests before importing application modules.
os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("SKIP_ML_IMPORTS", "1")
os.environ.setdefault("SKIP_RAG_IMPORTS", "1")

from app.core.database import Base, engine, SessionLocal  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from main import app  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_database():
    """Recreate database schema for every test to ensure isolation."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture
def db_session():
    """Provide a transactional database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client():
    """FastAPI test client with application lifespan management."""
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
