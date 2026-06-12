#!/usr/bin/env python3
"""Train, evaluate and version the course recommender.

    python -m avalon_reco.train --dsn postgresql://... --models-dir models/

Produces in --models-dir:
    model-<UTC timestamp>.joblib    the artifact
    metrics-<UTC timestamp>.json    evaluation report (model vs baseline)
    latest.joblib                   copy of the newest promoted artifact
    registry.json                   append-only training history
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

from .data import load_frames, temporal_split
from .evaluate import evaluate, popularity_baseline
from .model import CourseRecommender

DEFAULT_DSN = os.getenv(
    "AVALON_DSN", "postgresql://avalon:avalon_dev_password@localhost:5432/avalon"
)


def train_and_evaluate(dsn: str, test_year: int, k: int) -> tuple[CourseRecommender, dict]:
    interactions, students, courses = load_frames(dsn)
    train_df, test_df = temporal_split(interactions, test_year)
    print(f"interactions: {len(interactions)} total → {len(train_df)} train / {len(test_df)} test (year {test_year})")

    eval_model = CourseRecommender.fit(train_df, students, courses)
    model_metrics = evaluate(eval_model, test_df, k=k)
    baseline_metrics = popularity_baseline(train_df, test_df, students, courses, k=k)

    # the shipped artifact learns from ALL data (including the test year)
    final_model = CourseRecommender.fit(interactions, students, courses)

    report = {
        "trained_at": final_model.trained_at,
        "test_year": test_year,
        "training_rows": len(interactions),
        "students": len(final_model.histories),
        "courses": len(final_model.course_ids),
        "model": model_metrics,
        "baseline_popularity": baseline_metrics,
        "lift_precision": round(
            model_metrics["precision_at_k"] / max(baseline_metrics["precision_at_k"], 1e-9), 2
        ),
    }
    return final_model, report


def save(model: CourseRecommender, report: dict, models_dir: Path, promote: bool) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_path = models_dir / f"model-{stamp}.joblib"
    joblib.dump(model.to_artifact(), model_path, compress=3)
    (models_dir / f"metrics-{stamp}.json").write_text(json.dumps(report, indent=2))

    registry_path = models_dir / "registry.json"
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else []
    registry.append({"model": model_path.name, "promoted": promote, **report})
    registry_path.write_text(json.dumps(registry, indent=2))

    if promote:
        shutil.copy2(model_path, models_dir / "latest.joblib")
    return model_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--no-promote", action="store_true", help="save but do not update latest.joblib")
    args = parser.parse_args()

    model, report = train_and_evaluate(args.dsn, args.test_year, args.k)
    print(json.dumps(report, indent=2))

    if report["model"]["precision_at_k"] <= report["baseline_popularity"]["precision_at_k"]:
        print("WARNING: model does not beat the popularity baseline", file=sys.stderr)

    path = save(model, report, args.models_dir, promote=not args.no_promote)
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
