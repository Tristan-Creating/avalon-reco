"""Course recommendation API for the Avalon intranet.

Endpoints:
    GET  /health                          liveness/readiness + model version
    GET  /recommendations/{student_id}    top-k course recommendations
    GET  /metrics                         Prometheus metrics
    POST /admin/reload                    hot-swap to the newest model artifact
                                          (X-Admin-Token header)

The model artifact is self-contained; the warehouse is only queried when an
unknown student needs cold-start context, and the API degrades gracefully
without it.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Header, HTTPException, Query
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from avalon_reco.model import CourseRecommender  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/latest.joblib"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "avalon_admin")
AVALON_DSN = os.getenv("AVALON_DSN", "")

REQUESTS = Counter(
    "avalon_reco_requests_total", "Recommendation requests", ["strategy"]
)
LATENCY = Histogram(
    "avalon_reco_latency_seconds", "Recommendation latency",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
)
MODEL_INFO = Gauge("avalon_reco_model_loaded", "1 when a model is loaded", ["trained_at"])

state: dict = {"model": None}


def load_model() -> None:
    artifact = joblib.load(MODEL_PATH)
    model = CourseRecommender.from_artifact(artifact)
    state["model"] = model
    MODEL_INFO.clear()
    MODEL_INFO.labels(trained_at=model.trained_at).set(1)


def lookup_student_info(student_id: int):
    """Cold-start context from the warehouse; None if unavailable."""
    if not AVALON_DSN:
        return None
    try:
        import psycopg2

        with psycopg2.connect(AVALON_DSN, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT faculty, program_level FROM warehouse.dim_student WHERE student_key = %s",
                (student_id,),
            )
            row = cur.fetchone()
            return (row[0], row[1]) if row else None
    except Exception:
        return None  # cold-start degrades to level-wide popularity


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Avalon Course Recommendations",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


class RecommendationOut(BaseModel):
    course_id: int
    score: float
    code: str
    title: str
    faculty: str
    level: str
    strategy: str


class RecommendationsResponse(BaseModel):
    student_id: int
    k: int
    model_trained_at: str
    recommendations: list[RecommendationOut]


@app.get("/health")
def health():
    model: CourseRecommender | None = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ok",
        "model_trained_at": model.trained_at,
        "students": len(model.histories),
        "courses": int(len(model.course_ids)),
    }


@app.get("/recommendations/{student_id}", response_model=RecommendationsResponse)
def recommendations(student_id: int, k: int = Query(default=10, ge=1, le=50)):
    model: CourseRecommender | None = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    info = None
    if student_id not in model.histories:
        info = lookup_student_info(student_id)

    with LATENCY.time():
        recs = model.recommend(student_id, k=k, student_info=info)

    if not recs:
        raise HTTPException(status_code=404, detail="no recommendations available")

    REQUESTS.labels(strategy=recs[0].strategy).inc()
    return RecommendationsResponse(
        student_id=student_id,
        k=k,
        model_trained_at=model.trained_at,
        recommendations=[RecommendationOut(**vars(r)) for r in recs],
    )


@app.post("/admin/reload")
def reload_model(x_admin_token: str = Header(default="")):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="bad token")
    before = state["model"].trained_at if state["model"] else None
    load_model()
    return {"reloaded": True, "previous": before, "current": state["model"].trained_at}
