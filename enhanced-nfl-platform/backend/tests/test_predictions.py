from contextlib import contextmanager

import pytest

from app.core.ml_pipeline import get_ml_pipeline
from app.models.database import Player, Prediction
from main import app


class StubPipeline:
    def __init__(self, *, should_fail: bool = False):
        self.should_fail = should_fail

    async def predict(self, features, model_name="ensemble"):
        if self.should_fail:
            raise RuntimeError("pipeline failure")
        return {
            "prediction": 1,
            "probability": 0.87,
            "confidence": 0.74,
            "model_used": model_name,
        }

    async def get_model_performance(self):
        return {"ensemble": {"accuracy": 0.95}}


@contextmanager
def override_pipeline(pipeline):
    app.dependency_overrides[get_ml_pipeline] = lambda: pipeline
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_ml_pipeline, None)


def _create_player(session) -> Player:
    player = Player(
        player_id="test-player-1",
        first_name="Test",
        last_name="Quarterback",
        position="QB",
        age=28,
        height=75,
        weight=225,
        experience=6,
        current_team="TEST",
    )
    session.add(player)
    session.commit()
    session.refresh(player)
    return player


def test_make_prediction_happy_path(client, db_session):
    player = _create_player(db_session)
    pipeline = StubPipeline()
    with override_pipeline(pipeline):
        response = client.post(
            "/api/v1/predictions",
            json={
                "player_id": player.id,
                "features": {
                    "passing_yards_roll3": 250,
                    "td_passes_roll3": 1.6,
                    "passes_attempted_roll3": 34,
                    "age": 28,
                    "experience": 6,
                    "height": 75,
                    "weight": 225,
                },
                "model_name": "ensemble",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["player_id"] == player.id
    assert payload["confidence"] == pytest.approx(0.74, rel=1e-3)
    assert payload["prediction"] is True
    assert payload["model_used"] == "ensemble"

    stored = db_session.query(Prediction).all()
    assert len(stored) == 1
    assert stored[0].player_id == player.id


def test_make_prediction_pipeline_error(client, db_session):
    player = _create_player(db_session)
    pipeline = StubPipeline(should_fail=True)
    with override_pipeline(pipeline):
        response = client.post(
            "/api/v1/predictions",
            json={"player_id": player.id, "features": {}, "model_name": "ensemble"},
        )

    assert response.status_code == 500


def test_get_predictions_returns_history(client, db_session):
    player = _create_player(db_session)
    record = Prediction(
        player_id=player.id,
        prediction=True,
        confidence=0.8,
        features_used={"sample": 1},
        model_used="ensemble",
        created_by="unit-test",
    )
    db_session.add(record)
    db_session.commit()

    response = client.get("/api/v1/predictions", params={"limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["player_id"] == player.id
