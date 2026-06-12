"""Offline evaluation: temporal split, ranking metrics, popularity baseline.

The model must beat the popularity baseline to be promoted — quality is a
gate, not a slide.
"""

from __future__ import annotations

import pandas as pd

from .model import CourseRecommender


def evaluate(
    model: CourseRecommender,
    test: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Precision/recall@k, hit rate and catalog coverage on held-out data.

    Only students with training history are scored (cold-start quality is a
    separate, deliberate strategy — measuring it as CF would be misleading).
    """
    test_by_student = test.groupby("student_id")["course_id"].agg(set)
    scored = recommended_total = hits_total = relevant_total = hitrate_hits = 0
    distinct_recommended: set[int] = set()

    for student_id, actual in test_by_student.items():
        if student_id not in model.histories:
            continue
        recs = [r.course_id for r in model.recommend(student_id, k=k)]
        if not recs:
            continue
        hits = len(set(recs) & actual)
        scored += 1
        hits_total += hits
        recommended_total += len(recs)
        relevant_total += len(actual)
        hitrate_hits += 1 if hits > 0 else 0
        distinct_recommended.update(recs)

    return {
        "k": k,
        "students_scored": scored,
        "precision_at_k": round(hits_total / max(recommended_total, 1), 4),
        "recall_at_k": round(hits_total / max(relevant_total, 1), 4),
        "hit_rate": round(hitrate_hits / max(scored, 1), 4),
        "catalog_coverage": round(len(distinct_recommended) / len(model.course_ids), 4),
    }


def popularity_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    students: pd.DataFrame,
    courses: pd.DataFrame,
    k: int = 10,
) -> dict:
    """Same metrics for 'recommend the k most popular courses of your level'."""
    level_of = students.set_index("student_id")["level"]
    enriched = train.merge(courses[["course_id", "level"]], on="course_id")
    top_by_level = {
        lvl: grp["course_id"].value_counts().index.tolist()
        for lvl, grp in enriched.groupby("level")
    }
    seen_by_student = train.groupby("student_id")["course_id"].agg(set)

    test_by_student = test.groupby("student_id")["course_id"].agg(set)
    scored = recommended_total = hits_total = relevant_total = hitrate_hits = 0

    for student_id, actual in test_by_student.items():
        if student_id not in seen_by_student.index or student_id not in level_of.index:
            continue
        seen = seen_by_student[student_id]
        pool = top_by_level.get(level_of[student_id], [])
        recs = [c for c in pool if c not in seen][:k]
        if not recs:
            continue
        hits = len(set(recs) & actual)
        scored += 1
        hits_total += hits
        recommended_total += len(recs)
        relevant_total += len(actual)
        hitrate_hits += 1 if hits > 0 else 0

    return {
        "k": k,
        "students_scored": scored,
        "precision_at_k": round(hits_total / max(recommended_total, 1), 4),
        "recall_at_k": round(hits_total / max(relevant_total, 1), 4),
        "hit_rate": round(hitrate_hits / max(scored, 1), 4),
    }
