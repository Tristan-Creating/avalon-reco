# Serving image: FastAPI + the trained model artifact.
# Build context = repo root; the artifact must exist (train first or let CI do it).

FROM python:3.12-slim AS base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY api/ api/
COPY models/latest.joblib models/latest.joblib

ENV MODEL_PATH=/app/models/latest.joblib
EXPOSE 8000

# non-root, as the governance plan would insist
RUN useradd --uid 10001 --no-create-home reco
USER reco

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
