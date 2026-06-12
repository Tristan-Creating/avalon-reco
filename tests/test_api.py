"""API tests with a real (tiny) artifact behind a TestClient."""

import sys
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="module")
def client(tmp_path_factory, request):
    """Boot the API against an artifact built from the tiny university."""
    tiny = request.getfixturevalue("tiny_university")
    from avalon_reco.model import CourseRecommender

    interactions, students, courses = tiny
    model = CourseRecommender.fit(interactions, students, courses, top_n_neighbors=8)

    model_path = tmp_path_factory.mktemp("models") / "latest.joblib"
    joblib.dump(model.to_artifact(), model_path)

    import os
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["ADMIN_TOKEN"] = "test-token"
    os.environ.pop("AVALON_DSN", None)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
    import main as api_main  # imported once; reloading would re-register Prometheus metrics

    with TestClient(api_main.app) as test_client:
        yield test_client


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["courses"] == 8


class TestRecommendations:
    def test_known_student(self, client):
        resp = client.get("/recommendations/3?k=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["student_id"] == 3
        assert len(body["recommendations"]) == 3
        assert body["recommendations"][0]["course_id"] == 103
        assert body["recommendations"][0]["strategy"] == "collaborative"

    def test_unknown_student_cold_start(self, client):
        resp = client.get("/recommendations/424242")
        assert resp.status_code == 200
        assert resp.json()["recommendations"][0]["strategy"] == "popular_level"

    def test_k_validation(self, client):
        assert client.get("/recommendations/3?k=0").status_code == 422
        assert client.get("/recommendations/3?k=100").status_code == 422

    def test_non_integer_student_rejected(self, client):
        assert client.get("/recommendations/bob").status_code == 422


class TestOps:
    def test_metrics_exposed(self, client):
        client.get("/recommendations/3")
        resp = client.get("/metrics/")
        assert resp.status_code == 200
        assert "avalon_reco_requests_total" in resp.text
        assert "avalon_reco_model_loaded" in resp.text

    def test_reload_requires_token(self, client):
        assert client.post("/admin/reload").status_code == 403

    def test_reload_with_token(self, client):
        resp = client.post("/admin/reload", headers={"X-Admin-Token": "test-token"})
        assert resp.status_code == 200
        assert resp.json()["reloaded"] is True
