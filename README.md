# Avalon University — Course Recommendation Service (Block 4)

End-to-end ML system recommending courses to Avalon students: trained on the
warehouse built by Blocks 2-3, served by FastAPI, retrained on a
champion/challenger gate, monitored for drift with Evidently and for serving
health with Prometheus/Grafana.

## The ML problem

**Business goal** — help students discover relevant electives and next
courses at enrollment time, on the intranet.

**Formulation** — implicit-feedback recommendation. Enrollments are positive
signals (no ratings exist); the model is **item-item collaborative filtering**
(cosine similarity on the student × course matrix, top-50 neighbors), with two
domain constraints baked in: never recommend a course already taken, never
recommend outside the student's academic level. Cold start degrades through
faculty popularity → level popularity, and is *labelled as such* in responses
(`strategy` field) — the API never pretends a fallback is personalization.

**Why not deep learning** — 5k students × 200 courses trains in seconds,
is fully explainable ("students who took X also took Y"), and its GDPR/DPIA
story (Block 1) is clean: no sensitive features, only enrollment behavior.

**Evaluation** — temporal split (train < 2025, test on 2025: exactly the
enrollments streamed by the Block 3 pipeline), measured with precision@10,
recall@10, hit rate and catalog coverage, against a popularity-per-level
baseline the model must beat. Metrics are versioned next to every artifact
(`models/registry.json`).

## Quickstart

```bash
pip install -r requirements-dev.txt

# unit + API tests (no infrastructure needed)
pytest tests/ -v

# train against the warehouse (Blocks 2-3 up & seeded)
PYTHONPATH=src python -m avalon_reco.train --dsn postgresql://avalon:avalon_dev_password@localhost:5432/avalon

# serve (docker, on the platform network)
docker compose up -d --build
curl localhost:8000/recommendations/42?k=5
```

## Repository layout (assignment structure)

```
├── notebooks/            # EDA on the warehouse (executed, with outputs)
├── src/avalon_reco/      # data.py, model.py, evaluate.py, train.py
├── tests/                # model + API tests; CI warehouse seeder
├── models/               # artifacts + metrics + registry.json (gitignored)
├── api/main.py           # FastAPI serving, Prometheus metrics, hot reload
├── retrain/              # champion/challenger script + Airflow DAG
├── monitoring/           # Evidently drift report → HTML + ops.ml_drift
├── k8s/                  # Deployment + Service + HPA (probes, non-root)
├── Dockerfile            # slim serving image, non-root, healthcheck
└── .github/workflows/    # CI: lint → tests → training smoke test → image
```

## MLOps loop

```
   warehouse (Blocks 2-3, grows via Kafka stream)
        │ train
        ▼
   models/model-<stamp>.joblib + metrics ──▶ registry.json
        │ promote (only if challenger ≥ champion − 0.005 on precision@10)
        ▼
   models/latest.joblib ──▶ POST /admin/reload ──▶ API serves new model
        ▲                                              │ /metrics
   retrain/retrain.py (weekly Airflow DAG)        Prometheus → Grafana
        ▲ alerts                                       │
   monitoring/drift_report.py (Evidently) ◀────────────┘
```

- **CI/CD** (`.github/workflows/ci.yml`): ruff + pytest → end-to-end training
  smoke test against a disposable Postgres → on main, build & push the
  serving image to GHCR with the freshly trained artifact baked in.
- **Retraining** (`retrain/retrain.py`): exit 0 = promoted, exit 2 = quality
  gate held the champion. The Airflow DAG treats exit 2 as success-without-
  promotion, not failure.
- **Drift** (`monitoring/drift_report.py`): compares the live stream window
  against the training reference on the features the model actually uses;
  writes an Evidently HTML report and a row in `ops.ml_drift` for Grafana.
- **Serving metrics**: request counts by strategy (personalized vs cold-start
  share is itself a drift signal), latency histogram, loaded-model gauge.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness/readiness; model version & corpus size |
| `GET /recommendations/{student_id}?k=10` | top-k, with scores, course metadata and the serving `strategy` |
| `GET /metrics` | Prometheus exposition |
| `POST /admin/reload` | hot-swap to the new `latest.joblib` (`X-Admin-Token`) |
| `GET /docs` | OpenAPI / Swagger UI |

## Governance tie-in (Block 1)

Recommending courses is profiling under GDPR. The governance plan's DPIA
covers it: only enrollment behavior is used (no grades, no special-category
data), recommendations are suggestions with no automated legal effect
(Art. 22), the model is explainable, and training data is synthetic in all
non-production environments.
